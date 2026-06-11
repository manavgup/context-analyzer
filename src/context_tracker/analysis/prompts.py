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
from context_tracker.db import ApiCallRecord, TurnRecord

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

# function_name: func(), snake_case(), CamelCase identifiers, `backticked`
_FUNCTION_NAME_RE = re.compile(
    r"(?:"
    r"\b[a-z_][a-zA-Z0-9_]*\(\)"  # snake_case() or camelCase() calls
    r"|\b[A-Z][a-zA-Z0-9_]*\.\w+"  # Qualified access: ClassName.method
    r"|\b[A-Z][a-zA-Z0-9]*[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*\b"  # CamelCase with internal cap
    r"|`[a-zA-Z_][a-zA-Z0-9_.]*`"  # `backticked` identifiers
    r"|\b[A-Z][a-zA-Z0-9]*\(\)"  # PascalCase() function call
    r")",
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
    - Sum cost across the turn's API calls

    Note: tool_failures is always 0 per-prompt because failures are a
    session-level metric that cannot be accurately attributed to individual
    prompts without timestamp/API-call range correlation.
    """
    turns = db_session.query(TurnRecord).filter_by(session_id=session_id).order_by(TurnRecord.turn_number).all()

    if not turns:
        return []

    # Pre-fetch all API calls for the session, indexed by call_index
    api_calls = (
        db_session.query(ApiCallRecord).filter_by(session_id=session_id).order_by(ApiCallRecord.call_index).all()
    )
    api_call_by_index: dict[int, ApiCallRecord] = {int(ac.call_index): ac for ac in api_calls}

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

        # Tool failures are a session-level metric and cannot be accurately
        # attributed to individual prompts without timestamp correlation.
        # Set to 0 at the per-prompt level; session summary should report
        # the aggregate instead.

        results.append(
            PromptAnalysis(
                turn_number=int(turn.turn_number),
                prompt_preview=prompt_text,
                specificity_score=score,
                signals=signals,
                resolution_turns=resolution_turns,
                resolution_cost=round(resolution_cost, 6),
                tool_failures=0,
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
