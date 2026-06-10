"""Real-time session nudge engine.

Evaluates session state against configurable thresholds and returns
actionable warnings that the UserPromptSubmit hook prints to stderr.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from context_tracker.db import (
    DEFAULT_DB_PATH,
    HookEventRecord,
    SessionRecord,
    get_engine,
    get_session_factory,
)
from context_tracker.nudge_config import NUDGE_DEFAULTS

logger = logging.getLogger(__name__)

# 1M token model context window (standard Claude Code limit)
MODEL_CONTEXT_WINDOW = 1_000_000


@dataclass
class Nudge:
    """A single nudge/warning for the user."""

    code: str  # e.g. "CONTEXT_THRESHOLD"
    severity: str  # "info" | "warning" | "critical"
    message: str  # one-line human-readable


def evaluate_nudges(
    session_id: str,
    db_path: Path | None = None,
    config: dict[str, Any] | None = None,
) -> list[Nudge]:
    """Evaluate all nudge rules against current session state.

    Queries the SQLite DB for session-level aggregates and recent hook events.
    Returns a list of Nudge objects (possibly empty for healthy sessions).
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    cfg: dict[str, Any] = dict(NUDGE_DEFAULTS)
    if config:
        cfg.update(config)

    if not cfg.get("enabled", True):
        return []

    nudges: list[Nudge] = []

    try:
        engine = get_engine(db_path)
        factory = get_session_factory(engine)
    except Exception:
        logger.debug("Could not connect to DB at %s", db_path)
        return []

    try:
        with factory() as db:
            session: SessionRecord | None = db.get(SessionRecord, session_id)
            if session is None:
                return []

            # --- CONTEXT_THRESHOLD ---
            _raw_pct = cfg.get("context_threshold_pct", 60)
            threshold_pct = int(_raw_pct) if _raw_pct is not None else 60
            peak = session.peak_context_tokens or 0
            threshold_tokens = MODEL_CONTEXT_WINDOW * threshold_pct // 100
            if peak > threshold_tokens:
                pct = round(peak / MODEL_CONTEXT_WINDOW * 100)
                severity = "critical" if pct >= 80 else "warning"
                nudges.append(
                    Nudge(
                        code="CONTEXT_THRESHOLD",
                        severity=severity,
                        message=(f"⚠️ Context at {pct}% of 1M limit — consider compacting or starting a new session"),
                    )
                )

            # --- COST_WARNING ---
            _raw_cost = cfg.get("cost_warning_usd", 10.0)
            cost_threshold = float(_raw_cost) if _raw_cost is not None else 10.0
            total_cost = session.total_cost_usd or 0.0
            if total_cost > cost_threshold:
                nudges.append(
                    Nudge(
                        code="COST_WARNING",
                        severity="warning",
                        message=(
                            f"\U0001f4b0 Session cost: ${total_cost:.2f} — splitting would save on future API calls"
                        ),
                    )
                )

            # --- REPEATED_READS ---
            _raw_repeat = cfg.get("repeated_read_count", 3)
            repeat_count = int(_raw_repeat) if _raw_repeat is not None else 3
            recent_events = (
                db.query(HookEventRecord)
                .filter_by(session_id=session_id, event_type="post_tool_use")
                .order_by(HookEventRecord.id.desc())
                .limit(50)
                .all()
            )

            tool_names = [e.tool_name for e in recent_events if e.tool_name]
            counts = Counter(tool_names)
            for tool_name, count in counts.most_common():
                if count >= repeat_count:
                    nudges.append(
                        Nudge(
                            code="REPEATED_READS",
                            severity="info",
                            message=(
                                f"\U0001f4a1 {tool_name} read {count} times recently "
                                f"— consider pinning relevant sections"
                            ),
                        )
                    )
                    break  # Only report the top offender
    except Exception:
        logger.debug("Error evaluating nudges for session %s", session_id, exc_info=True)
        return []

    return nudges
