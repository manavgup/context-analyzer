"""Post-session optimization report generator.

Analyzes a completed session from the SQLite database and produces
an OptimizationReport identifying token waste and recommending
session splits.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field

from sqlalchemy.orm import Session as DbSession

from context_tracker.db import (
    ApiCallRecord,
    BlockRecord,
    HookEventRecord,
    SessionRecord,
)

# Pricing per million tokens (USD) — must match ingest.py cost model.
COST_PER_M_INPUT = 15.0
COST_PER_M_OUTPUT = 75.0
COST_PER_M_CACHE_READ = 1.875
COST_PER_M_CACHE_CREATION = 18.75


def _tokens_to_cost(tokens: int) -> float:
    """Convert *input* token count to estimated cost in USD.

    Uses the full input-token rate. For split analysis the per-call
    helper ``_api_call_cost`` is preferred because it accounts for
    output and cache pricing too.
    """
    return tokens * COST_PER_M_INPUT / 1_000_000


def _api_call_cost(call: ApiCallRecord) -> float:
    """Return the real cost of a single API call using all pricing tiers."""
    return (
        int(call.input_tokens or 0) * COST_PER_M_INPUT / 1_000_000
        + int(call.output_tokens or 0) * COST_PER_M_OUTPUT / 1_000_000
        + int(call.cache_read or 0) * COST_PER_M_CACHE_READ / 1_000_000
        + int(call.cache_creation or 0) * COST_PER_M_CACHE_CREATION / 1_000_000
    )


@dataclass
class WasteItem:
    """A single category of detected token waste."""

    category: str  # "stale_content", "repeated_reads", "failed_retries", "oversized_output"
    description: str
    tokens: int
    estimated_cost: float
    suggestion: str


@dataclass
class SplitRecommendation:
    """Recommendation to split the session at a specific API call."""

    split_at_turn: int
    current_cost: float
    projected_cost: float
    savings: float
    reason: str


@dataclass
class OptimizationReport:
    """Full optimization report for a session."""

    session_id: str
    total_cost: float
    total_turns: int
    total_api_calls: int
    peak_context: int
    waste_items: list[WasteItem] = field(default_factory=list)
    total_waste_tokens: int = 0
    total_waste_cost: float = 0.0
    split_recommendation: SplitRecommendation | None = None

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        d = asdict(self)
        return d


def _detect_stale_content(
    session_id: str,
    db: DbSession,
) -> WasteItem | None:
    """Detect blocks that lived in context for an excessively long time.

    Blocks where (effective_exit - enter_turn) > 50 API calls are considered
    to have occupied context too long.  For blocks still present at session
    end (exit_turn IS NULL), the session's total_api_calls (or the max
    call_index) is used as the effective exit turn.
    """
    # Determine the effective session-end turn for blocks that never exited.
    session_rec = db.get(SessionRecord, session_id)
    max_turn: int = int(session_rec.total_api_calls or 0) if session_rec else 0

    blocks = (
        db.query(BlockRecord)
        .filter(
            BlockRecord.session_id == session_id,
            BlockRecord.enter_turn.isnot(None),
        )
        .all()
    )

    stale_tokens = 0
    stale_count = 0
    for block in blocks:
        effective_exit = int(block.exit_turn) if block.exit_turn is not None else max_turn
        lifespan = effective_exit - int(block.enter_turn or 0)
        if lifespan > 50:
            stale_tokens += int(block.tokens or 0)
            stale_count += 1

    if stale_count == 0:
        return None

    return WasteItem(
        category="stale_content",
        description=f"{stale_count} blocks lingered in context for 50+ API calls",
        tokens=stale_tokens,
        estimated_cost=_tokens_to_cost(stale_tokens),
        suggestion="Use /compact more frequently or start a new session after completing a sub-task",
    )


def _detect_repeated_reads(
    session_id: str,
    db: DbSession,
) -> WasteItem | None:
    """Detect files read more than twice via tool_result blocks.

    Groups BlockRecords by label (file path) where block_type contains
    'tool_result'. If the same label appears > 2 times, the excess reads
    are waste.
    """
    blocks = (
        db.query(BlockRecord)
        .filter(
            BlockRecord.session_id == session_id,
            BlockRecord.block_type.contains("tool_result"),
            BlockRecord.label.isnot(None),
        )
        .all()
    )

    label_groups: Counter[str] = Counter()
    # Store (enter_turn, id, tokens) tuples so we can sort chronologically.
    label_reads: dict[str, list[tuple[int, int, int]]] = {}
    for block in blocks:
        label = str(block.label or "")
        if not label:
            continue
        label_groups[label] += 1
        if label not in label_reads:
            label_reads[label] = []
        label_reads[label].append((int(block.enter_turn or 0), int(block.id or 0), int(block.tokens or 0)))

    waste_tokens = 0
    repeated_count = 0
    for label, count in label_groups.items():
        if count > 2:
            excess = count - 2
            repeated_count += excess
            # Sort by chronological order (enter_turn, then id as tie-breaker).
            # The first 2 reads are legitimate; excess reads after that are waste.
            chronological = sorted(label_reads[label], key=lambda t: (t[0], t[1]))
            waste_tokens += sum(t[2] for t in chronological[2:])

    if repeated_count == 0:
        return None

    return WasteItem(
        category="repeated_reads",
        description=f"{repeated_count} excess file reads across {sum(1 for c in label_groups.values() if c > 2)} files",
        tokens=waste_tokens,
        estimated_cost=_tokens_to_cost(waste_tokens),
        suggestion="Avoid re-reading files that have not changed. Use Edit instead of Read+Write",
    )


def _detect_failed_retries(
    session_id: str,
    db: DbSession,
) -> WasteItem | None:
    """Detect token waste from failed tool invocations.

    Counts HookEventRecords with event_type containing 'failure'
    and sums their error_length as a proxy for wasted tokens.
    """
    events = (
        db.query(HookEventRecord)
        .filter(
            HookEventRecord.session_id == session_id,
            HookEventRecord.event_type.contains("failure"),
        )
        .all()
    )

    if not events:
        return None

    waste_tokens = sum(int(e.error_length or 0) for e in events)

    return WasteItem(
        category="failed_retries",
        description=f"{len(events)} tool failures consumed context with error output",
        tokens=waste_tokens,
        estimated_cost=_tokens_to_cost(waste_tokens),
        suggestion="Investigate recurring tool failures; each retry adds error text to context",
    )


def _detect_oversized_output(
    session_id: str,
    db: DbSession,
) -> WasteItem | None:
    """Detect tool results that are excessively large (>30K tokens).

    For each oversized block, the waste is the portion above 30K.
    """
    threshold = 30_000
    blocks = (
        db.query(BlockRecord)
        .filter(
            BlockRecord.session_id == session_id,
            BlockRecord.block_type.contains("tool_result"),
            BlockRecord.tokens > threshold,
        )
        .all()
    )

    if not blocks:
        return None

    waste_tokens = sum(int(block.tokens or 0) - threshold for block in blocks)

    return WasteItem(
        category="oversized_output",
        description=f"{len(blocks)} tool results exceeded 30K tokens",
        tokens=waste_tokens,
        estimated_cost=_tokens_to_cost(waste_tokens),
        suggestion="Use targeted reads (offset/limit) or summarize large outputs before they enter context",
    )


def _compute_split_recommendation(
    session_id: str,
    db: DbSession,
) -> SplitRecommendation | None:
    """Analyze API call sequence and recommend an optimal session split point.

    Walks API call records and evaluates the total cost with and without
    a split.  Uses the same multi-tier cost model as ``ingest.py``
    (input, output, cache_read, cache_creation rates) so that the
    numbers displayed here are consistent with ``SessionRecord.total_cost_usd``.

    The "current cost" is taken from the session's actual ``total_cost_usd``
    when available, falling back to a per-call sum.

    Only recommends a split if savings > 20%.
    """
    api_calls = (
        db.query(ApiCallRecord).filter(ApiCallRecord.session_id == session_id).order_by(ApiCallRecord.call_index).all()
    )

    if len(api_calls) < 4:
        return None

    # Use the session's real total_cost_usd when available so
    # "Current cost" matches what the user sees elsewhere.
    session_rec = db.get(SessionRecord, session_id)
    actual_cost = float(session_rec.total_cost_usd or 0.0) if session_rec else 0.0

    # Compute current total cost from per-call data (same formula as ingest.py).
    computed_cost = sum(_api_call_cost(c) for c in api_calls)
    current_cost = actual_cost if actual_cost > 0 else computed_cost

    if current_cost <= 0:
        return None

    # Walk possible split points and find the one with maximum savings.
    # After a split, the second session starts with a smaller context,
    # so the input_tokens for calls after the split would be reduced.
    # We model the reduction as: each call after the split pays only
    # its own new tokens (output_tokens of previous call) plus a base
    # context size (the context at the split point divided by a growth factor).
    best_savings = 0.0
    best_split = -1
    best_projected = current_cost

    for i in range(2, len(api_calls) - 1):
        # Cost before split: actual per-call costs up to split
        cost_before = sum(_api_call_cost(c) for c in api_calls[:i])

        # Context at split point
        context_at_split = int(api_calls[i - 1].input_tokens or 0)

        # Cost after split: model reduced context
        # After starting fresh, the initial context is much smaller.
        # Estimate: each call after split starts from a base of ~10% of
        # the context at split point (system prompt + essential context),
        # then grows by output_tokens.
        base_context = max(context_at_split * 0.1, 5000)
        cost_after_input = 0.0
        running_context = base_context
        for c in api_calls[i:]:
            cost_after_input += running_context
            running_context += int(c.output_tokens or 0)

        # For the second session, output / cache costs stay roughly the
        # same — only the input portion shrinks.
        cost_after_output = sum(
            int(c.output_tokens or 0) * COST_PER_M_OUTPUT / 1_000_000
            + int(c.cache_read or 0) * COST_PER_M_CACHE_READ / 1_000_000
            + int(c.cache_creation or 0) * COST_PER_M_CACHE_CREATION / 1_000_000
            for c in api_calls[i:]
        )

        projected_total = cost_before + cost_after_input * COST_PER_M_INPUT / 1_000_000 + cost_after_output
        savings = current_cost - projected_total

        if savings > best_savings:
            best_savings = savings
            best_split = i
            best_projected = projected_total

    # Only recommend if savings exceed 20%
    if best_split < 0 or best_savings / current_cost < 0.20:
        return None

    return SplitRecommendation(
        split_at_turn=best_split,
        current_cost=round(current_cost, 4),
        projected_cost=round(best_projected, 4),
        savings=round(best_savings, 4),
        reason=(
            f"Splitting at API call {best_split} would reduce total cost "
            f"by {best_savings / current_cost:.0%} (${best_savings:.4f})"
        ),
    )


def generate_report(session_id: str, db_session: DbSession) -> OptimizationReport:
    """Analyze a completed session and generate optimization suggestions.

    Args:
        session_id: The session to analyze.
        db_session: An open SQLAlchemy session.

    Returns:
        An OptimizationReport with waste items and split recommendation.

    Raises:
        ValueError: If the session is not found in the database.
    """
    session_rec = db_session.get(SessionRecord, session_id)
    if session_rec is None:
        raise ValueError(f"Session {session_id!r} not found")

    report = OptimizationReport(
        session_id=session_id,
        total_cost=float(session_rec.total_cost_usd or 0.0),
        total_turns=int(session_rec.total_turns or 0),
        total_api_calls=int(session_rec.total_api_calls or 0),
        peak_context=int(session_rec.peak_context_tokens or 0),
    )

    # Run all waste detectors
    detectors = [
        _detect_stale_content,
        _detect_repeated_reads,
        _detect_failed_retries,
        _detect_oversized_output,
    ]
    for detector in detectors:
        item = detector(session_id, db_session)
        if item is not None:
            report.waste_items.append(item)

    # Compute totals
    report.total_waste_tokens = sum(w.tokens for w in report.waste_items)
    report.total_waste_cost = sum(w.estimated_cost for w in report.waste_items)

    # Compute split recommendation
    report.split_recommendation = _compute_split_recommendation(session_id, db_session)

    return report
