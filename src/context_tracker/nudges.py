"""Real-time session nudge engine.

Evaluates session state against configurable thresholds and returns
actionable warnings that the UserPromptSubmit hook prints to stderr.

Primary data source: the hook trace JSONL file (~/.claude/context-trace/),
which is written in real-time during a session. Falls back to the SQLite DB
for metrics that require transcript parsing (e.g., actual token counts).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from context_tracker.nudge_config import NUDGE_DEFAULTS
from context_tracker.storage import DEFAULT_TRACE_DIR

logger = logging.getLogger(__name__)

MODEL_CONTEXT_WINDOW = 1_000_000
CHARS_PER_TOKEN_EST = 4


@dataclass
class Nudge:
    """A single nudge/warning for the user."""

    code: str  # e.g. "CONTEXT_THRESHOLD"
    severity: str  # "info" | "warning" | "critical"
    message: str  # one-line human-readable


def _read_trace_events(session_id: str, trace_dir: Path) -> list[dict[str, Any]]:
    """Read raw JSON events from the hook trace JSONL (fast, no DB needed)."""
    trace_file = trace_dir / f"{session_id}.jsonl"
    if not trace_file.exists():
        return []
    events: list[dict[str, Any]] = []
    with open(trace_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _estimate_context_from_events(events: list[dict[str, Any]]) -> int:
    """Estimate cumulative context size in tokens from hook events.

    Sums input_payload_chars + output_payload_chars for all tool use events,
    plus prompt lengths. Divides by ~4 chars/token for a rough estimate.
    This is a running total (not accounting for compaction resets).
    """
    total_chars = 0
    for ev in events:
        event_type = ev.get("event", "")
        if event_type == "post_tool_use":
            total_chars += ev.get("input_payload_chars", 0)
            total_chars += ev.get("output_payload_chars", 0)
        elif event_type == "user_prompt":
            total_chars += ev.get("prompt_length_chars", 0)
        elif event_type == "post_compact":
            total_chars = total_chars // 3
    return total_chars // CHARS_PER_TOKEN_EST


def _estimate_cost_from_events(events: list[dict[str, Any]]) -> float:
    """Rough cost estimate from hook events.

    Counts API round-trips (user_prompt events) and estimates
    cost based on average context size growth.
    """
    total_chars = 0
    total_cost = 0.0
    for ev in events:
        event_type = ev.get("event", "")
        if event_type == "post_tool_use":
            total_chars += ev.get("input_payload_chars", 0)
            total_chars += ev.get("output_payload_chars", 0)
        elif event_type == "user_prompt":
            context_tokens = total_chars // CHARS_PER_TOKEN_EST
            total_cost += context_tokens * 3.0 / 1_000_000
        elif event_type == "post_compact":
            total_chars = total_chars // 3
    return total_cost


def evaluate_nudges(
    session_id: str,
    db_path: Path | None = None,
    config: dict[str, Any] | None = None,
    trace_dir: Path | None = None,
) -> list[Nudge]:
    """Evaluate all nudge rules against current session state.

    Reads the hook trace JSONL for real-time data. Falls back to the
    SQLite DB for pre-computed metrics when available.
    """
    if trace_dir is None:
        trace_dir = DEFAULT_TRACE_DIR

    cfg: dict[str, Any] = dict(NUDGE_DEFAULTS)
    if config:
        cfg.update(config)

    if not cfg.get("enabled", True):
        return []

    nudges: list[Nudge] = []

    events = _read_trace_events(session_id, trace_dir)

    # Try DB first for accurate metrics, fall back to trace estimates
    db_peak_tokens: int | None = None
    db_total_cost: float | None = None

    if db_path is None:
        from context_tracker.db import DEFAULT_DB_PATH

        db_path = DEFAULT_DB_PATH

    if db_path.exists():
        try:
            from context_tracker.db import SessionRecord, get_engine, get_session_factory

            engine = get_engine(db_path)
            factory = get_session_factory(engine)
            with factory() as db:
                session: SessionRecord | None = db.get(SessionRecord, session_id)
                if session is not None:
                    db_peak_tokens = int(session.peak_context_tokens or 0)
                    db_total_cost = float(session.total_cost_usd or 0.0)
        except Exception:
            logger.debug("DB fallback failed", exc_info=True)

    if not events and db_peak_tokens is None:
        return []

    # --- CONTEXT_THRESHOLD ---
    _raw_pct = cfg.get("context_threshold_pct", 60)
    threshold_pct = int(_raw_pct) if _raw_pct is not None else 60
    peak = db_peak_tokens if db_peak_tokens else _estimate_context_from_events(events)
    threshold_tokens = MODEL_CONTEXT_WINDOW * threshold_pct // 100
    if peak > threshold_tokens:
        pct = round(peak / MODEL_CONTEXT_WINDOW * 100)
        severity = "critical" if pct >= 80 else "warning"
        nudges.append(
            Nudge(
                code="CONTEXT_THRESHOLD",
                severity=severity,
                message=f"⚠️ Context at {pct}% of 1M limit — consider compacting or starting a new session",
            )
        )

    # --- COST_WARNING ---
    _raw_cost = cfg.get("cost_warning_usd", 10.0)
    cost_threshold = float(_raw_cost) if _raw_cost is not None else 10.0
    total_cost = db_total_cost if db_total_cost is not None else _estimate_cost_from_events(events)
    if total_cost > cost_threshold:
        nudges.append(
            Nudge(
                code="COST_WARNING",
                severity="warning",
                message=f"\U0001f4b0 Session cost: ${total_cost:.2f} — splitting would save on future API calls",
            )
        )

    # --- REPEATED_READS (always from real-time trace) ---
    _raw_repeat = cfg.get("repeated_read_count", 3)
    repeat_count = int(_raw_repeat) if _raw_repeat is not None else 3
    tool_events = [ev for ev in events if ev.get("event") == "post_tool_use"]
    tool_names = [ev.get("tool_name", "") for ev in tool_events[-50:]]
    counts = Counter(tool_names)
    for tool_name, count in counts.most_common():
        if count >= repeat_count and tool_name:
            nudges.append(
                Nudge(
                    code="REPEATED_READS",
                    severity="info",
                    message=f"\U0001f4a1 {tool_name} used {count} times recently — consider pinning relevant sections",
                )
            )
            break

    return nudges
