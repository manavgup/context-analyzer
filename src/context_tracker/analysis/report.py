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

# Approximate cost per million input tokens (USD).
COST_PER_M_INPUT = 3.0


def _tokens_to_cost(tokens: int) -> float:
    """Convert token count to estimated cost in USD."""
    return tokens * COST_PER_M_INPUT / 1_000_000


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

    Blocks where exit_turn is set and (exit_turn - enter_turn) > 50 API calls
    are considered to have occupied context for too long.
    """
    blocks = (
        db.query(BlockRecord)
        .filter(
            BlockRecord.session_id == session_id,
            BlockRecord.exit_turn.isnot(None),
        )
        .all()
    )

    stale_tokens = 0
    stale_count = 0
    for block in blocks:
        lifespan = int(block.exit_turn or 0) - int(block.enter_turn or 0)
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
    label_tokens: dict[str, list[int]] = {}
    for block in blocks:
        label = str(block.label or "")
        if not label:
            continue
        label_groups[label] += 1
        if label not in label_tokens:
            label_tokens[label] = []
        label_tokens[label].append(int(block.tokens or 0))

    waste_tokens = 0
    repeated_count = 0
    for label, count in label_groups.items():
        if count > 2:
            excess = count - 2
            repeated_count += excess
            # Sum the tokens for excess reads (the oldest ones)
            sorted_tokens = sorted(label_tokens[label])
            waste_tokens += sum(sorted_tokens[:excess])

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
    a split. The cost model accounts for context growth: larger contexts
    cost more per call because input tokens scale.

    Only recommends a split if savings > 20%.
    """
    api_calls = (
        db.query(ApiCallRecord).filter(ApiCallRecord.session_id == session_id).order_by(ApiCallRecord.call_index).all()
    )

    if len(api_calls) < 4:
        return None

    # Compute current total cost (simple sum of input tokens)
    total_input = sum(int(c.input_tokens or 0) for c in api_calls)
    current_cost = _tokens_to_cost(total_input)

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
        # Cost before split: sum of actual input tokens up to split
        cost_before = sum(int(c.input_tokens or 0) for c in api_calls[:i])

        # Context at split point
        context_at_split = int(api_calls[i - 1].input_tokens or 0)

        # Cost after split: model reduced context
        # After starting fresh, the initial context is much smaller.
        # Estimate: each call after split starts from a base of ~10% of
        # the context at split point (system prompt + essential context),
        # then grows by output_tokens.
        base_context = max(context_at_split * 0.1, 5000)
        cost_after = 0.0
        running_context = base_context
        for c in api_calls[i:]:
            cost_after += running_context
            running_context += int(c.output_tokens or 0)

        projected_total = _tokens_to_cost(cost_before) + _tokens_to_cost(int(cost_after))
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
            f"Splitting at API call {best_split} would reduce total input cost "
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
