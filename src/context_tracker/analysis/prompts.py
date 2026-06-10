"""Prompt pattern detection — correlate prompt specificity with efficiency.

Pure-regex heuristics to score how specific a user prompt is, then correlate
with resolution cost (API calls, tokens, tool failures) to surface
actionable "be more specific" guidance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sqlalchemy.orm import Session as DbSession

from context_tracker.analysis.config import PRICING
from context_tracker.db import ApiCallRecord, HookEventRecord, TurnRecord

# ---------------------------------------------------------------------------
# Specificity signal patterns
# ---------------------------------------------------------------------------

# file_path: foo/bar.py, src/module.py:123, ./dir/file.ts
_FILE_PATH_RE = re.compile(
    r"(?:^|[\s\"'`(])(?:\.?/)?(?:[\w.-]+/)+[\w.-]+\.[\w]+(?::\d+)?",
    re.MULTILINE,
)

# line_number: :123, line 45, L45
_LINE_NUMBER_RE = re.compile(
    r"(?::\d{1,6}|[Ll]ine\s+\d+|L\d+)\b",
)

# function_name: functionName(), snake_case_func(), ClassName (PascalCase)
_FUNCTION_NAME_RE = re.compile(
    r"\b(?:[a-z_][a-zA-Z0-9_]*\(\)|[A-Z][a-zA-Z0-9]*(?=[.\s,;:)\]]))",
)

# error_message: quoted strings with error-like words
_ERROR_MESSAGE_RE = re.compile(
    r"""(?:["'`])(?:[^"'`]*(?:error|exception|traceback|failed|"""
    r"""undefined|not found|cannot|could not|unexpected|invalid)[^"'`]*)(?:["'`])""",
    re.IGNORECASE,
)

# constraint_language: "don't", "without changing", "keep the", "only modify"
_CONSTRAINT_RE = re.compile(
    r"\b(?:don'?t|do not|without changing|keep the|only modify|only change|"
    r"must not|should not|leave .* as.is|preserve the|avoid)\b",
    re.IGNORECASE,
)

# Signal definitions: (name, regex, weight)
_SIGNALS: list[tuple[str, re.Pattern[str], float]] = [
    ("file_path", _FILE_PATH_RE, 0.25),
    ("line_number", _LINE_NUMBER_RE, 0.15),
    ("function_name", _FUNCTION_NAME_RE, 0.15),
    ("error_message", _ERROR_MESSAGE_RE, 0.15),
    ("constraint_language", _CONSTRAINT_RE, 0.15),
]

_LENGTH_THRESHOLD = 100
_LENGTH_WEIGHT = 0.15


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PromptAnalysis:
    """Result of analysing a single user prompt."""

    turn_number: int
    prompt_preview: str
    specificity_score: float  # 0.0 - 1.0
    signals: list[str] = field(default_factory=list)  # which signals detected
    resolution_turns: int = 0  # how many API calls to resolve this turn
    resolution_cost: float = 0.0  # USD cost for this turn
    tool_failures: int = 0  # tool errors in this turn range


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------


def classify_prompt(text: str) -> tuple[float, list[str]]:
    """Score prompt specificity based on heuristic signals.

    Returns (score, signal_names) where score is 0.0 - 1.0.
    """
    if not text or not text.strip():
        return 0.0, []

    score = 0.0
    signals: list[str] = []

    for name, pattern, weight in _SIGNALS:
        if pattern.search(text):
            score += weight
            signals.append(name)

    # Length signal
    if len(text) > _LENGTH_THRESHOLD:
        score += _LENGTH_WEIGHT
        signals.append("length_signal")

    # Normalize to 0.0-1.0 (cap at 1.0)
    score = min(score, 1.0)

    return round(score, 4), signals


# ---------------------------------------------------------------------------
# Session-level analysis with resolution tracking
# ---------------------------------------------------------------------------


def _compute_api_call_cost(
    api_call: ApiCallRecord,
    model: str = "_default",
) -> float:
    """Compute USD cost for a single ApiCallRecord."""
    rates = PRICING.get(model, PRICING["_default"])
    return float(
        int(api_call.input_tokens or 0) * rates["input"] / 1_000_000
        + int(api_call.output_tokens or 0) * rates["output"] / 1_000_000
        + int(api_call.cache_read or 0) * rates["cache_read"] / 1_000_000
        + int(api_call.cache_creation or 0) * rates["cache_create"] / 1_000_000
    )


def analyze_session_prompts(
    session_id: str,
    db_session: DbSession,
    model: str = "_default",
) -> list[PromptAnalysis]:
    """Analyse all user prompts in a session with resolution tracking.

    For each turn:
    - Score the prompt_preview for specificity
    - Count API calls (resolution_turns)
    - Count tool failures (HookEventRecords with event_type containing 'failure')
    - Sum cost across the turn's API calls
    """
    turns = db_session.query(TurnRecord).filter_by(session_id=session_id).order_by(TurnRecord.turn_number).all()

    if not turns:
        return []

    # Pre-fetch all API calls for the session, indexed by call_index
    api_calls = (
        db_session.query(ApiCallRecord).filter_by(session_id=session_id).order_by(ApiCallRecord.call_index).all()
    )
    api_call_by_index: dict[int, ApiCallRecord] = {int(ac.call_index): ac for ac in api_calls}

    # Pre-fetch hook events that indicate tool failures
    hook_events = db_session.query(HookEventRecord).filter_by(session_id=session_id).all()

    results: list[PromptAnalysis] = []

    for turn in turns:
        prompt_text = str(turn.prompt_preview or "")
        score, signals = classify_prompt(prompt_text)

        # Compute resolution metrics from the turn's API call range
        first_call = int(turn.first_api_call) if turn.first_api_call is not None else None
        last_call = int(turn.last_api_call) if turn.last_api_call is not None else None
        resolution_turns = 0
        resolution_cost = 0.0

        if first_call is not None and last_call is not None:
            resolution_turns = last_call - first_call + 1
            for idx in range(first_call, last_call + 1):
                ac = api_call_by_index.get(idx)
                if ac:
                    resolution_cost += _compute_api_call_cost(ac, model)

        # Count tool failures in this turn's range
        tool_failures = 0
        for he in hook_events:
            if he.event_type and "failure" in he.event_type.lower():
                # Hook events don't have a turn_number, but they have metadata.
                # Use a simple heuristic: count all failures for now.
                # A more precise approach would correlate timestamps.
                tool_failures += 1

        # For tool failures, divide total by number of turns as a rough per-turn estimate
        # unless we can correlate more precisely
        per_turn_failures = 0
        if tool_failures > 0 and len(turns) > 0:
            # Better approach: count error_length > 0 hook events
            # For now, just attribute failures to turns proportionally
            per_turn_failures = 0  # Will be refined below

        # Refined: count hook events with error_length > 0
        turn_failures = 0
        for he in hook_events:
            if he.error_length and he.error_length > 0:
                turn_failures += 1
        # Distribute proportionally (rough)
        if len(turns) > 0 and turn_failures > 0:
            per_turn_failures = max(0, round(turn_failures / len(turns)))

        results.append(
            PromptAnalysis(
                turn_number=int(turn.turn_number),
                prompt_preview=prompt_text,
                specificity_score=score,
                signals=signals,
                resolution_turns=max(resolution_turns, 1),
                resolution_cost=round(resolution_cost, 6),
                tool_failures=per_turn_failures,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Aggregate stats for the API response
# ---------------------------------------------------------------------------


def compute_aggregate_stats(analyses: list[PromptAnalysis]) -> dict:
    """Compute aggregate stats: avg score, avg resolution cost by quartile."""
    if not analyses:
        return {
            "total_prompts": 0,
            "avg_specificity": 0.0,
            "specific_prompts": {"count": 0, "avg_cost": 0.0, "avg_resolution_turns": 0.0},
            "moderate_prompts": {"count": 0, "avg_cost": 0.0, "avg_resolution_turns": 0.0},
            "vague_prompts": {"count": 0, "avg_cost": 0.0, "avg_resolution_turns": 0.0},
        }

    specific = [a for a in analyses if a.specificity_score > 0.6]
    moderate = [a for a in analyses if 0.3 <= a.specificity_score <= 0.6]
    vague = [a for a in analyses if a.specificity_score < 0.3]

    def _avg(items: list[PromptAnalysis], attr: str) -> float:
        if not items:
            return 0.0
        return round(float(sum(getattr(a, attr) for a in items)) / len(items), 6)

    return {
        "total_prompts": len(analyses),
        "avg_specificity": round(sum(a.specificity_score for a in analyses) / len(analyses), 4),
        "specific_prompts": {
            "count": len(specific),
            "avg_cost": _avg(specific, "resolution_cost"),
            "avg_resolution_turns": _avg(specific, "resolution_turns"),
        },
        "moderate_prompts": {
            "count": len(moderate),
            "avg_cost": _avg(moderate, "resolution_cost"),
            "avg_resolution_turns": _avg(moderate, "resolution_turns"),
        },
        "vague_prompts": {
            "count": len(vague),
            "avg_cost": _avg(vague, "resolution_cost"),
            "avg_resolution_turns": _avg(vague, "resolution_turns"),
        },
    }
