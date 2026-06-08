"""Session health scoring and recommendations."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from context_tracker.analysis.config import (
    MODEL_CONTEXT_WINDOWS,
    PRICING,
    HealthConfig,
    StalenessConfig,
)
from context_tracker.analysis.models import (
    ApiCall,
    BlockType,
    ContentStore,
    ContextBlock,
    ConversationTurn,
    TurnSnapshot,
)
from context_tracker.analysis.staleness import (
    compute_staleness,
    detect_superseded,
    detect_task_boundaries,
)


@dataclass
class HealthSignals:
    turn_number: int
    dead_weight_ratio: float
    context_utilization: float
    cache_efficiency: float
    cache_efficiency_trend: float  # Normalized 0-1: 0=stable, 1=declining fast
    repeated_reads: dict[str, int]  # resource → unchanged read count (rolling window)
    error_rate: float
    error_rate_spike: float  # max(0, current/max(avg,0.01) - 1.0)
    output_inflation: float  # Normalized 0-1
    edit_churn: list[str]  # Evidence only
    compaction_count: int
    cost_this_turn: float
    cost_cumulative: float


@dataclass
class AttentionLossSignal:
    signal_type: str
    severity: str  # info, warning, critical
    description: str
    turn: int
    resource: str | None = None
    evidence: dict = field(default_factory=dict)


@dataclass
class SessionRecommendation:
    urgency_score: float
    recommendation: str
    reasons: list[str]
    recoverable_tokens: int
    recoverable_blocks: int
    top_stale_block_ids: list[str]
    confidence: str  # "high" or "low"


def compute_turn_cost(api_call: ApiCall, model: str) -> float:
    """Compute cost for a single API call."""
    rates = PRICING.get(model, PRICING["_default"])
    return (
        api_call.input_tokens * rates["input"] / 1_000_000
        + api_call.output_tokens * rates["output"] / 1_000_000
        + api_call.cache_read_tokens * rates["cache_read"] / 1_000_000
        + api_call.cache_creation_tokens * rates["cache_create"] / 1_000_000
    )


def compute_urgency(signals: HealthSignals, config: HealthConfig) -> float:
    """Compute urgency score from health signals. Returns 0.0 to 1.0."""
    repeated_count = len([r for r, c in signals.repeated_reads.items() if c >= config.repeated_read_warning])

    score = (
        signals.dead_weight_ratio * config.weight_dead_weight
        + signals.context_utilization * config.weight_utilization
        + signals.cache_efficiency_trend * config.weight_cache
        + signals.output_inflation * config.weight_output_inflation
        + min(1.0, repeated_count / 5) * config.weight_repeated
        + min(1.0, signals.error_rate_spike) * config.weight_errors
    )
    return min(1.0, max(0.0, score))


def classify_recommendation(urgency_score: float, config: HealthConfig) -> str:
    """Map urgency score to recommendation label."""
    if urgency_score < config.threshold_healthy:
        return "healthy"
    if urgency_score < config.threshold_degrading:
        return "degrading"
    if urgency_score < config.threshold_recommend_new:
        return "recommend_new_session"
    return "urgent"


def build_health_signals(
    turns: list[ConversationTurn],
    snapshots: list[TurnSnapshot],
    block_registry: dict[str, ContextBlock],
    content_store: ContentStore,
    model: str,
    config: HealthConfig | None = None,
    staleness_config: StalenessConfig | None = None,
) -> HealthSignals:
    """Populate HealthSignals from reconstructed session data.

    Args:
        turns: Conversation turns from reconstruct_session.
        snapshots: Per-turn snapshots from reconstruct_session.
        block_registry: block_id -> ContextBlock mapping.
        content_store: ContentStore with block content.
        model: Model identifier string.
        config: Health configuration (uses defaults if None).
        staleness_config: Staleness configuration (uses defaults if None).
    """
    if config is None:
        config = HealthConfig()
    if staleness_config is None:
        staleness_config = StalenessConfig()

    model_window = MODEL_CONTEXT_WINDOWS.get(model, config.model_context_window)

    # If peak context exceeds the mapped window, the session is on a larger
    # variant (e.g. transcript says "claude-opus-4-6" but it's actually the
    # 1M variant).  Check for a "[1m]" entry or fall back to 1M.
    peak_actual = max(
        (s.actual_context_tokens for s in snapshots if s.actual_context_tokens > 0),
        default=0,
    )
    if peak_actual > model_window:
        larger_key = model + "[1m]"
        model_window = MODEL_CONTEXT_WINDOWS.get(larger_key, 1_000_000)

    # --- Find the final non-compaction snapshot ---
    final_snap = None
    for snap in reversed(snapshots):
        if not snap.compaction_detected:
            final_snap = snap
            break
    if final_snap is None and snapshots:
        final_snap = snapshots[-1]

    # --- Dead weight ratio ---
    # Score all blocks at the final snapshot
    task_boundaries = detect_task_boundaries(turns, staleness_config)
    resource_last_used: dict[str, int] = {}
    blocks_seen: list[ContextBlock] = []
    assistant_texts_by_turn: dict[int, list[str]] = {}
    for bid, block in block_registry.items():
        if block.block_type == BlockType.ASSISTANT_TEXT:
            tn = block.turn_entered
            if tn not in assistant_texts_by_turn:
                assistant_texts_by_turn[tn] = []
            text = content_store.get_content(bid)
            if text:
                assistant_texts_by_turn[tn].append(text)

    for snap in snapshots:
        for bid in snap.blocks_entered_ids:
            blk: ContextBlock | None = block_registry.get(bid)
            if blk:
                blocks_seen.append(blk)
                if blk.resource:
                    resource_last_used[blk.resource] = snap.turn_number

    superseded = detect_superseded(blocks_seen)

    dead_weight_tokens = 0
    actual_context_tokens = final_snap.actual_context_tokens if final_snap else 0
    last_turn = final_snap.turn_number if final_snap else 0
    superseded_count = 0

    if final_snap:
        for bid in final_snap.block_ids:
            block_or_none: ContextBlock | None = block_registry.get(bid)
            if not block_or_none:
                continue
            block = block_or_none
            messages_since: list[str] = []
            for t in range(block.turn_entered + 1, last_turn + 1):
                messages_since.extend(assistant_texts_by_turn.get(t, []))
            score_val, label = compute_staleness(
                block=block,
                current_turn=last_turn,
                config=staleness_config,
                resource_last_used=resource_last_used,
                messages_since_block=messages_since,
                active_resources=set(resource_last_used.keys()),
                task_boundaries=task_boundaries,
                superseded_map=superseded,
            )
            if label == "dead_weight":
                dead_weight_tokens += block.size_tokens_est
            if bid in superseded:
                superseded_count += 1

    dead_weight_ratio = dead_weight_tokens / actual_context_tokens if actual_context_tokens > 0 else 0.0

    # --- Context utilization ---
    peak_context = max((s.actual_context_tokens for s in snapshots), default=0)
    context_utilization = peak_context / model_window if model_window > 0 else 0.0

    # --- Cache efficiency + trend ---
    window = config.cache_trend_window
    cache_efficiencies: list[float] = []
    for snap in snapshots:
        total_in = snap.cache_read_tokens + snap.input_tokens + snap.cache_creation_tokens
        if total_in > 0:
            cache_efficiencies.append(snap.cache_read_tokens / total_in)
        else:
            cache_efficiencies.append(0.0)

    recent = cache_efficiencies[-window:] if cache_efficiencies else []
    cache_efficiency = sum(recent) / len(recent) if recent else 0.0

    # Trend: compare first half to second half of recent window
    cache_efficiency_trend = 0.0
    if len(recent) >= 4:
        mid = len(recent) // 2
        first_half = sum(recent[:mid]) / mid
        second_half = sum(recent[mid:]) / (len(recent) - mid)
        if first_half > 0.01:
            decline = max(0.0, first_half - second_half) / first_half
            cache_efficiency_trend = min(1.0, decline)

    # --- Repeated reads ---
    read_window = config.repeated_read_rolling_window
    # Use snapshots (which have blocks_entered_ids populated) instead of
    # ApiCall.blocks_entered (which is never populated by reconstruct_session)
    recent_snap_start = max(0, len(snapshots) - read_window)
    recent_snaps = snapshots[recent_snap_start:]
    resource_read_counts: dict[str, int] = defaultdict(int)
    for snap in recent_snaps:
        for bid in snap.blocks_entered_ids:
            read_blk: ContextBlock | None = block_registry.get(bid)
            if (
                read_blk
                and read_blk.block_type == BlockType.TOOL_RESULT
                and read_blk.resource
                and read_blk.tool_name in ("Read", "Glob", "Grep", None)
            ):
                resource_read_counts[read_blk.resource] += 1
    repeated_reads = {r: c for r, c in resource_read_counts.items() if c >= 2}

    # --- Error rate ---
    total_tool_results = 0
    error_blocks = 0
    for block in block_registry.values():
        if block.block_type == BlockType.TOOL_RESULT:
            total_tool_results += 1
            if block.is_error:
                error_blocks += 1
    error_rate = error_blocks / total_tool_results if total_tool_results > 0 else 0.0

    # Error rate spike: compare recent error rate to overall average
    error_rate_spike = 0.0
    if len(turns) > 5:
        recent_turn_nums = {t.turn_number for t in turns[-5:]}
        recent_errors = sum(
            1
            for b in block_registry.values()
            if b.block_type == BlockType.TOOL_RESULT and b.is_error and b.turn_entered in recent_turn_nums
        )
        recent_total = sum(
            1
            for b in block_registry.values()
            if b.block_type == BlockType.TOOL_RESULT and b.turn_entered in recent_turn_nums
        )
        if recent_total > 0:
            recent_err = recent_errors / recent_total
            avg_err = max(error_rate, 0.01)
            error_rate_spike = max(0.0, recent_err / avg_err - 1.0)

    # --- Output inflation ---
    output_tokens_list = [s.output_tokens for s in snapshots if s.output_tokens > 0]
    output_inflation = 0.0
    if len(output_tokens_list) > 3:
        avg_output = sum(output_tokens_list) / len(output_tokens_list)
        recent_output = sum(output_tokens_list[-3:]) / 3
        if avg_output > 0:
            inflation = max(0.0, recent_output / avg_output - 1.0)
            output_inflation = min(1.0, inflation)

    # --- Edit churn (evidence only) ---
    # Use snapshots (blocks_entered_ids) since ApiCall.blocks_entered is unpopulated
    edit_churn: list[str] = []
    edit_window = config.edit_churn_window
    recent_edits: dict[str, int] = defaultdict(int)
    edit_snap_start = max(0, len(snapshots) - edit_window)
    for snap in snapshots[edit_snap_start:]:
        for bid in snap.blocks_entered_ids:
            edit_blk: ContextBlock | None = block_registry.get(bid)
            if edit_blk and edit_blk.tool_name in ("Edit", "Write") and edit_blk.resource:
                recent_edits[edit_blk.resource] += 1
    edit_churn = [r for r, c in recent_edits.items() if c >= 3]

    # --- Compaction count ---
    compaction_count = sum(1 for s in snapshots if s.compaction_detected)

    # --- Cost ---
    cost_cumulative = 0.0
    cost_this_turn = 0.0
    for turn in turns:
        for api_call in turn.api_calls:
            cost = compute_turn_cost(api_call, model)
            cost_cumulative += cost
    if turns:
        last_turn_obj = turns[-1]
        for api_call in last_turn_obj.api_calls:
            cost_this_turn += compute_turn_cost(api_call, model)

    return HealthSignals(
        turn_number=last_turn,
        dead_weight_ratio=round(dead_weight_ratio, 4),
        context_utilization=round(context_utilization, 4),
        cache_efficiency=round(cache_efficiency, 4),
        cache_efficiency_trend=round(cache_efficiency_trend, 4),
        repeated_reads=repeated_reads,
        error_rate=round(error_rate, 4),
        error_rate_spike=round(error_rate_spike, 4),
        output_inflation=round(output_inflation, 4),
        edit_churn=edit_churn,
        compaction_count=compaction_count,
        cost_this_turn=round(cost_this_turn, 6),
        cost_cumulative=round(cost_cumulative, 6),
    )


def generate_recommendations(
    signals: HealthSignals,
    block_registry: dict[str, ContextBlock],
    snapshots: list[TurnSnapshot],
    config: HealthConfig | None = None,
    staleness_config: StalenessConfig | None = None,
) -> list[dict]:
    """Generate actionable recommendations from health signals.

    Returns a list of recommendation dicts, each with:
        priority, code, title, detail, action, tokens_recoverable, target_turn
    """
    if config is None:
        config = HealthConfig()
    if staleness_config is None:
        staleness_config = StalenessConfig()

    recs: list[dict] = []
    last_turn_idx = len(snapshots) - 1 if snapshots else 0

    # HIGH_DEAD_WEIGHT
    if signals.dead_weight_ratio > 0.30:
        recoverable = 0
        if snapshots:
            final_snap = snapshots[-1]
            recoverable = int(final_snap.actual_context_tokens * signals.dead_weight_ratio)
        priority = "critical" if signals.dead_weight_ratio > 0.50 else "warning"
        recs.append(
            {
                "priority": priority,
                "code": "HIGH_DEAD_WEIGHT",
                "title": "High dead weight",
                "detail": (f"{signals.dead_weight_ratio:.0%} of context is dead weight ({recoverable:,} tokens)."),
                "action": ("Start a new session or use /compact to reclaim stale context."),
                "tokens_recoverable": recoverable,
                "target_turn": last_turn_idx,
            }
        )

    # CONTEXT_NEAR_LIMIT
    if signals.context_utilization > 0.75:
        priority = "critical" if signals.context_utilization > 0.90 else "warning"
        recs.append(
            {
                "priority": priority,
                "code": "CONTEXT_NEAR_LIMIT",
                "title": "Context near limit",
                "detail": (f"Context utilization at {signals.context_utilization:.0%} of model window."),
                "action": ("Consider starting a new session to avoid auto-compaction."),
                "tokens_recoverable": 0,
                "target_turn": last_turn_idx,
            }
        )

    # REPEATED_READS
    heavy_reads = {r: c for r, c in signals.repeated_reads.items() if c >= config.repeated_read_warning}
    if heavy_reads:
        top_resource = max(heavy_reads, key=lambda r: heavy_reads[r])
        top_count = heavy_reads[top_resource]
        priority = "critical" if top_count >= config.repeated_read_critical else "warning"
        # Estimate recoverable: size of the superseded copies
        recoverable = 0
        for block in block_registry.values():
            if block.block_type == BlockType.TOOL_RESULT and block.resource in heavy_reads:
                recoverable += block.size_tokens_est
        # Subtract one copy per resource (the active one stays)
        active_copy_est = recoverable // max(1, sum(heavy_reads.values()))
        recoverable = max(0, recoverable - active_copy_est * len(heavy_reads))

        # Find turn with highest dead_weight_ratio, fallback to last turn
        repeated_target = last_turn_idx
        max_dw = -1.0
        for idx, snap in enumerate(snapshots):
            dw_tokens = 0
            for bid in snap.block_ids:
                blk = block_registry.get(bid)
                if blk and blk.resource and blk.resource in heavy_reads:
                    dw_tokens += blk.size_tokens_est
            if snap.actual_context_tokens > 0:
                ratio = dw_tokens / snap.actual_context_tokens
                if ratio > max_dw:
                    max_dw = ratio
                    repeated_target = idx

        recs.append(
            {
                "priority": priority,
                "code": "REPEATED_READS",
                "title": "Repeated file reads",
                "detail": (
                    f"{len(heavy_reads)} file(s) read 3+ times. Top: {top_resource.split('/')[-1]} ({top_count}x)."
                ),
                "action": ("Avoid re-reading files that haven't changed. Use Edit instead of Read+Write."),
                "tokens_recoverable": recoverable,
                "target_turn": repeated_target,
            }
        )

    # CACHE_CHURN
    if signals.cache_efficiency_trend > 0.3:
        priority = "warning" if signals.cache_efficiency_trend < 0.6 else "critical"
        recs.append(
            {
                "priority": priority,
                "code": "CACHE_CHURN",
                "title": "Cache hit rate declining",
                "detail": (
                    f"Cache efficiency dropped by {signals.cache_efficiency_trend:.0%} "
                    f"over recent turns (current: {signals.cache_efficiency:.0%})."
                ),
                "action": ("Context is changing too fast for the cache. Group related work together."),
                "tokens_recoverable": 0,
                "target_turn": last_turn_idx,
            }
        )

    # HIGH_ERROR_RATE
    if signals.error_rate > 0.15:
        priority = "warning" if signals.error_rate < 0.30 else "critical"
        recs.append(
            {
                "priority": priority,
                "code": "HIGH_ERROR_RATE",
                "title": "High tool error rate",
                "detail": (f"{signals.error_rate:.0%} of tool results are errors."),
                "action": (
                    "Investigate failing tool calls. Errors consume context without contributing useful information."
                ),
                "tokens_recoverable": 0,
                "target_turn": last_turn_idx,
            }
        )

    # OUTPUT_INFLATION
    if signals.output_inflation > 0.5:
        priority = "info" if signals.output_inflation < 0.8 else "warning"
        recs.append(
            {
                "priority": priority,
                "code": "OUTPUT_INFLATION",
                "title": "Response inflation",
                "detail": (f"Recent outputs are {signals.output_inflation:.0%} larger than session average."),
                "action": ("Larger outputs fill context faster. Consider more targeted prompts."),
                "tokens_recoverable": 0,
                "target_turn": last_turn_idx,
            }
        )

    # SUPERSEDED_READS
    if snapshots:
        blocks_list = list(block_registry.values())
        superseded_map = detect_superseded(blocks_list)
        # Count superseded blocks still in the final snapshot
        final_block_ids = set(snapshots[-1].block_ids) if snapshots else set()
        superseded_in_context = [bid for bid in superseded_map if bid in final_block_ids]
        if len(superseded_in_context) > 3:
            recoverable = sum(
                block_registry[bid].size_tokens_est for bid in superseded_in_context if bid in block_registry
            )
            recs.append(
                {
                    "priority": "warning",
                    "code": "SUPERSEDED_READS",
                    "title": "Superseded file reads",
                    "detail": (
                        f"{len(superseded_in_context)} blocks have been superseded by newer reads of the same file."
                    ),
                    "action": ("These old file contents waste context. A compaction would remove them."),
                    "tokens_recoverable": recoverable,
                    "target_turn": last_turn_idx,
                }
            )

    # Sort: critical first, then warning, then info
    priority_order = {"critical": 0, "warning": 1, "info": 2}
    recs.sort(key=lambda r: priority_order.get(r["priority"], 3))

    return recs
