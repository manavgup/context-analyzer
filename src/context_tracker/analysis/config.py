"""Configurable thresholds for staleness detection and health scoring.

All defaults are labeled as uncalibrated — to be tuned against real sessions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class StalenessConfig:
    decay_window: int = 10
    resource_window: int = 10
    reference_scan_window: int = 15
    task_boundary_time_gap: int = 10     # Minutes
    task_boundary_overlap: float = 0.2
    min_prompt_length_for_boundary: int = 20


@dataclass
class HealthConfig:
    model_context_window: int = 200_000
    weight_dead_weight: float = 0.35
    weight_utilization: float = 0.25
    weight_cache: float = 0.15
    weight_output_inflation: float = 0.10
    weight_repeated: float = 0.10
    weight_errors: float = 0.05
    threshold_healthy: float = 0.3
    threshold_degrading: float = 0.5
    threshold_recommend_new: float = 0.7
    repeated_read_warning: int = 3
    repeated_read_critical: int = 5
    repeated_read_rolling_window: int = 20
    edit_churn_window: int = 5
    error_spike_multiplier: float = 2.0
    output_inflation_multiplier: float = 1.5
    cache_trend_window: int = 10


MODEL_CONTEXT_WINDOWS = {
    "claude-opus-4-6": 200_000,
    "claude-opus-4-6[1m]": 1_000_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5": 200_000,
}

# Pricing per million tokens
PRICING = {
    "claude-opus-4-6": {
        "input": 15.0, "output": 75.0,
        "cache_read": 1.875, "cache_create": 18.75,
    },
    "claude-opus-4-6[1m]": {
        "input": 15.0, "output": 75.0,
        "cache_read": 1.875, "cache_create": 18.75,
    },
    "claude-sonnet-4-6": {
        "input": 3.0, "output": 15.0,
        "cache_read": 0.375, "cache_create": 3.75,
    },
    "_default": {
        "input": 15.0, "output": 75.0,
        "cache_read": 1.875, "cache_create": 18.75,
    },
}


def load_config(
    config_path: Path | None = None,
) -> tuple[StalenessConfig, HealthConfig]:
    """Load config from JSON file, falling back to defaults."""
    staleness = StalenessConfig()
    health = HealthConfig()

    if config_path is None:
        config_path = Path.home() / ".claude" / "context-analyzer.json"

    if not config_path.exists():
        return staleness, health

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return staleness, health

    staleness_data = data.get("staleness", {})
    for key, value in staleness_data.items():
        if hasattr(staleness, key):
            setattr(staleness, key, value)

    health_data = data.get("health", {})
    for key, value in health_data.items():
        if hasattr(health, key):
            setattr(health, key, value)

    return staleness, health
