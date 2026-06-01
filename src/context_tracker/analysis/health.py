"""Session health scoring and recommendations."""

from __future__ import annotations

from dataclasses import dataclass, field

from context_tracker.analysis.config import PRICING, HealthConfig
from context_tracker.analysis.models import ApiCall


@dataclass
class HealthSignals:
    turn_number: int
    dead_weight_ratio: float
    context_utilization: float
    cache_efficiency: float
    cache_efficiency_trend: float   # Normalized 0-1: 0=stable, 1=declining fast
    repeated_reads: dict[str, int]  # resource → unchanged read count (rolling window)
    error_rate: float
    error_rate_spike: float         # max(0, current/max(avg,0.01) - 1.0)
    output_inflation: float         # Normalized 0-1
    edit_churn: list[str]           # Evidence only
    compaction_count: int
    cost_this_turn: float
    cost_cumulative: float


@dataclass
class AttentionLossSignal:
    signal_type: str
    severity: str       # info, warning, critical
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
    confidence: str     # "high" or "low"


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
