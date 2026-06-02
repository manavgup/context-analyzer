"""Tests for session health scoring and recommendations."""

from context_tracker.analysis.config import HealthConfig
from context_tracker.analysis.health import (
    HealthSignals,
    classify_recommendation,
    compute_turn_cost,
    compute_urgency,
)
from context_tracker.analysis.models import ApiCall


def test_compute_urgency_healthy():
    config = HealthConfig()
    signals = HealthSignals(
        turn_number=50,
        dead_weight_ratio=0.1,
        context_utilization=0.2,
        cache_efficiency=0.97,
        cache_efficiency_trend=0.1,
        repeated_reads={},
        error_rate=0.01,
        error_rate_spike=0.0,
        output_inflation=0.0,
        edit_churn=[],
        compaction_count=0,
        cost_this_turn=0.02,
        cost_cumulative=1.50,
    )
    score = compute_urgency(signals, config)
    assert score < 0.3
    rec = classify_recommendation(score, config)
    assert rec == "healthy"


def test_compute_urgency_degrading():
    config = HealthConfig()
    signals = HealthSignals(
        turn_number=150,
        dead_weight_ratio=0.45,
        context_utilization=0.5,
        cache_efficiency=0.90,
        cache_efficiency_trend=0.4,
        repeated_reads={"server.py": 3, "models.py": 4},
        error_rate=0.05,
        error_rate_spike=0.5,
        output_inflation=0.3,
        edit_churn=["hooks.py"],
        compaction_count=1,
        cost_this_turn=0.03,
        cost_cumulative=4.50,
    )
    score = compute_urgency(signals, config)
    assert 0.3 <= score < 0.7
    rec = classify_recommendation(score, config)
    assert rec in ("degrading", "recommend_new_session")


def test_compute_urgency_urgent():
    config = HealthConfig()
    signals = HealthSignals(
        turn_number=300,
        dead_weight_ratio=0.7,
        context_utilization=0.85,
        cache_efficiency=0.80,
        cache_efficiency_trend=0.8,
        repeated_reads={"a.py": 5, "b.py": 4, "c.py": 3, "d.py": 3, "e.py": 3},
        error_rate=0.15,
        error_rate_spike=1.0,
        output_inflation=0.8,
        edit_churn=["a.py", "b.py"],
        compaction_count=3,
        cost_this_turn=0.05,
        cost_cumulative=8.00,
    )
    score = compute_urgency(signals, config)
    assert score >= 0.5
    rec = classify_recommendation(score, config)
    assert rec in ("recommend_new_session", "urgent")


def test_compute_turn_cost():
    api_call = ApiCall(
        api_call_index=0,
        conversation_turn=1,
        input_tokens=100,
        output_tokens=500,
        cache_read_tokens=40000,
        cache_creation_tokens=1000,
    )
    cost = compute_turn_cost(api_call, "claude-opus-4-6")
    assert cost > 0
    # cache_read: 40000 * 1.875 / 1M = 0.075
    # cache_create: 1000 * 18.75 / 1M = 0.01875
    # output: 500 * 75 / 1M = 0.0375
    # input: 100 * 15 / 1M = 0.0015
    expected = 0.075 + 0.01875 + 0.0375 + 0.0015
    assert abs(cost - expected) < 0.001


def test_classify_recommendation_thresholds():
    config = HealthConfig()
    assert classify_recommendation(0.0, config) == "healthy"
    assert classify_recommendation(0.29, config) == "healthy"
    assert classify_recommendation(0.3, config) == "degrading"
    assert classify_recommendation(0.49, config) == "degrading"
    assert classify_recommendation(0.5, config) == "recommend_new_session"
    assert classify_recommendation(0.69, config) == "recommend_new_session"
    assert classify_recommendation(0.7, config) == "urgent"
    assert classify_recommendation(1.0, config) == "urgent"
