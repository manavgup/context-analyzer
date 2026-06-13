"""Context freshness analysis — recommend optimal compaction timing.

Categorises stale blocks into four buckets and computes a compact-readiness
score (0-100) that combines staleness ratio with context pressure.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy.orm import Session as DbSession

from context_tracker.analysis.config import PRICING
from context_tracker.db import BlockRecord, HookEventRecord, SessionRecord


@dataclass
class StaleBlock:
    block_id: str
    label: str
    category: str  # "aged_out", "superseded", "failed_output", "redundant"
    tokens: int


@dataclass
class FreshnessReport:
    total_tokens: int
    active_tokens: int
    stale_tokens: int
    stale_breakdown: dict[str, int]  # category -> token count
    compact_readiness_score: int  # 0-100
    safe_to_drop: list[StaleBlock]
    estimated_savings_per_call: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _stale_block_from(b: BlockRecord, category: str) -> StaleBlock:
    """Helper to build a StaleBlock from a BlockRecord (resolves Column types)."""
    return StaleBlock(
        block_id=str(b.block_id),
        label=str(b.label or ""),
        category=category,
        tokens=int(b.tokens or 0),
    )


def _detect_aged_out(
    blocks: list[BlockRecord],
    latest_call: int,
) -> list[StaleBlock]:
    """Blocks that have aged out of usefulness.

    Two sub-cases:
    1. Block has exit_turn set and exited more than 10 calls ago.
    2. Block is still present (exit_turn IS NULL) but entered > 50 API calls
       ago -- these are long-lingering blocks likely stale.
    """
    results: list[StaleBlock] = []
    for b in blocks:
        entered = int(b.enter_turn or 0)
        age = latest_call - entered

        if b.exit_turn is not None:
            # Already exited -- if it left long ago, it's aged out
            calls_since_exit = latest_call - int(b.exit_turn)
            if calls_since_exit > 10:
                results.append(_stale_block_from(b, "aged_out"))
        elif age > 50:
            # Still present but very old
            results.append(_stale_block_from(b, "aged_out"))
    return results


def _detect_superseded(blocks: list[BlockRecord]) -> list[StaleBlock]:
    """Blocks superseded by a newer block with the same label (same file read twice).

    Only a present (non-exited) newer block can supersede an older block that is
    still present.  An exited newer block must NOT supersede a present older block
    -- that would incorrectly mark the only current copy as stale.
    """
    # Group blocks by label, keeping only those with a non-empty label
    by_label: dict[str, list[BlockRecord]] = defaultdict(list)
    for b in blocks:
        if b.label:
            by_label[str(b.label)].append(b)

    results: list[StaleBlock] = []
    for _label, group in by_label.items():
        if len(group) < 2:
            continue
        # Sort by enter_turn; check each older block against the newest
        sorted_group = sorted(group, key=lambda x: int(x.enter_turn or 0))
        newest = sorted_group[-1]
        newest_is_present = newest.exit_turn is None

        for older in sorted_group[:-1]:
            older_is_present = older.exit_turn is None
            # An exited newer block must not supersede a present older block
            if older_is_present and not newest_is_present:
                continue
            results.append(_stale_block_from(older, "superseded"))
    return results


def _detect_failed_output(
    blocks: list[BlockRecord],
    hook_events: list[HookEventRecord],
) -> list[StaleBlock]:
    """Tool result blocks associated with post_tool_use failures."""
    # Build set of tool_use_ids that had failures
    failure_tool_ids: set[str] = set()
    for h in hook_events:
        if "failure" in str(h.event_type or ""):
            if h.tool_use_id:
                failure_tool_ids.add(str(h.tool_use_id))

    results: list[StaleBlock] = []
    for b in blocks:
        if "tool_result" in str(b.block_type or ""):
            # Check if block_id or label contains a tool_use_id that failed
            # The block_id format is like "t5-tool_result-toolu_01ABC"
            bid_parts = str(b.block_id).split("-", 2)
            tool_use_id_from_bid = bid_parts[2] if len(bid_parts) > 2 else ""
            if tool_use_id_from_bid in failure_tool_ids:
                results.append(_stale_block_from(b, "failed_output"))
    return results


def _detect_redundant(blocks: list[BlockRecord]) -> list[StaleBlock]:
    """Blocks with identical or near-identical labels in the same session."""
    # Group by label -- if 3+ blocks share the exact same label, the
    # extras beyond the first two are redundant.
    by_label: dict[str, list[BlockRecord]] = defaultdict(list)
    for b in blocks:
        if b.label:
            by_label[str(b.label)].append(b)

    results: list[StaleBlock] = []
    for _label, group in by_label.items():
        if len(group) < 3:
            continue
        # Sort by enter_turn; mark all but the last two as redundant
        sorted_group = sorted(group, key=lambda x: int(x.enter_turn or 0))
        for redundant_block in sorted_group[:-2]:
            results.append(_stale_block_from(redundant_block, "redundant"))
    return results


def _compute_readiness(stale_tokens: int, total_tokens: int) -> int:
    """Compact readiness score: 0-100.

    Formula:
        staleness_ratio = stale_tokens / max(total_tokens, 1)
        context_pressure = total_tokens / 1_000_000
        readiness = min(100, int((0.6 * staleness_ratio + 0.4 * context_pressure) * 100))
    """
    staleness_ratio = stale_tokens / max(total_tokens, 1)
    context_pressure = total_tokens / 1_000_000
    raw = (0.6 * staleness_ratio + 0.4 * context_pressure) * 100
    return min(100, int(raw))


def _estimate_savings(stale_tokens: int, model: str | None) -> float:
    """Estimate per-call savings in USD from dropping stale tokens."""
    pricing = PRICING.get(model or "", PRICING["_default"])
    # Savings come from not paying cache-read cost for these tokens each call
    return stale_tokens * pricing["cache_read"] / 1e6


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_freshness(
    session_id: str,
    turn: int | None,
    db_session: DbSession,
) -> FreshnessReport:
    """Analyze context freshness at a given turn (or latest if None).

    Args:
        session_id: The session to analyze.
        turn: API call index to analyze at, or None for the latest.
        db_session: SQLAlchemy database session.

    Returns:
        FreshnessReport with breakdown and readiness score.
    """
    # Load session record for model info
    session_rec: SessionRecord | None = db_session.get(SessionRecord, session_id)

    model = session_rec.model if session_rec else None

    # Load all blocks for the session
    all_blocks: list[BlockRecord] = db_session.query(BlockRecord).filter_by(session_id=session_id).all()

    if not all_blocks:
        return FreshnessReport(
            total_tokens=0,
            active_tokens=0,
            stale_tokens=0,
            stale_breakdown={},
            compact_readiness_score=0,
            safe_to_drop=[],
            estimated_savings_per_call=0.0,
        )

    # Determine the latest call index
    max_enter = max(int(b.enter_turn or 0) for b in all_blocks)
    max_exit = (
        max(int(b.exit_turn or 0) for b in all_blocks if b.exit_turn is not None)
        if any(b.exit_turn is not None for b in all_blocks)
        else 0
    )
    latest_call: int = turn if turn is not None else max(max_enter, max_exit)

    # Filter blocks relevant at the target turn
    # A block is "present" if: enter_turn <= turn AND (exit_turn IS NULL OR exit_turn > turn)
    present_blocks: list[BlockRecord] = []
    exited_blocks: list[BlockRecord] = []
    for b in all_blocks:
        entered = int(b.enter_turn or 0)
        if entered > latest_call:
            continue
        if b.exit_turn is not None and int(b.exit_turn) <= latest_call:
            exited_blocks.append(b)
        else:
            present_blocks.append(b)

    # Combine present + recently exited for analysis
    analysis_blocks = present_blocks + exited_blocks

    # Load hook events for failure detection
    hook_events: list[HookEventRecord] = db_session.query(HookEventRecord).filter_by(session_id=session_id).all()

    # Detect stale blocks by category
    aged = _detect_aged_out(analysis_blocks, latest_call)
    superseded = _detect_superseded(analysis_blocks)
    failed = _detect_failed_output(analysis_blocks, hook_events)
    redundant = _detect_redundant(analysis_blocks)

    # De-duplicate: a block should only appear in one category
    # Priority: superseded > failed_output > aged_out > redundant
    seen_ids: set[str] = set()
    all_stale: list[StaleBlock] = []
    for stale_group in [superseded, failed, aged, redundant]:
        for sb in stale_group:
            if sb.block_id not in seen_ids:
                seen_ids.add(sb.block_id)
                all_stale.append(sb)

    # Compute totals
    present_ids = {str(b.block_id) for b in present_blocks}
    total_tokens: int = sum(int(b.tokens or 0) for b in present_blocks)
    stale_tokens: int = sum(sb.tokens for sb in all_stale if sb.block_id in present_ids)
    active_tokens: int = total_tokens - stale_tokens

    # Breakdown by category
    stale_breakdown: dict[str, int] = defaultdict(int)
    for sb in all_stale:
        # Only count tokens for blocks still present
        if sb.block_id in present_ids:
            stale_breakdown[sb.category] += sb.tokens

    readiness = _compute_readiness(stale_tokens, total_tokens)
    savings = _estimate_savings(stale_tokens, str(model) if model else None)

    # Only recommend dropping blocks that are still present -- exited blocks
    # are already gone and cannot be acted upon.
    safe_to_drop = sorted(
        [sb for sb in all_stale if sb.block_id in present_ids],
        key=lambda s: s.tokens,
        reverse=True,
    )

    return FreshnessReport(
        total_tokens=total_tokens,
        active_tokens=active_tokens,
        stale_tokens=stale_tokens,
        stale_breakdown=dict(stale_breakdown),
        compact_readiness_score=readiness,
        safe_to_drop=safe_to_drop,
        estimated_savings_per_call=round(savings, 6),
    )
