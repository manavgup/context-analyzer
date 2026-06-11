"""Tests for analysis/config.py — load_config function."""

import json

from context_tracker.analysis.config import HealthConfig, StalenessConfig, load_config


def test_load_config_defaults(tmp_path):
    """When config file doesn't exist, use defaults."""
    config_path = tmp_path / "nonexistent.json"
    staleness, health = load_config(config_path)
    assert isinstance(staleness, StalenessConfig)
    assert isinstance(health, HealthConfig)
    assert staleness.decay_window == 10
    assert health.model_context_window == 200_000


def test_load_config_default_path(tmp_path, monkeypatch):
    """When config_path is None, uses ~/.claude/context-analyzer.json."""
    staleness, health = load_config(None)
    # Should return defaults since the file likely doesn't exist
    assert isinstance(staleness, StalenessConfig)
    assert isinstance(health, HealthConfig)


def test_load_config_with_overrides(tmp_path):
    """Load config with custom values."""
    config_path = tmp_path / "config.json"
    config_data = {
        "staleness": {
            "decay_window": 20,
            "resource_window": 15,
        },
        "health": {
            "model_context_window": 500000,
            "weight_dead_weight": 0.50,
            "threshold_healthy": 0.2,
        },
    }
    config_path.write_text(json.dumps(config_data), encoding="utf-8")

    staleness, health = load_config(config_path)
    assert staleness.decay_window == 20
    assert staleness.resource_window == 15
    assert staleness.reference_scan_window == 15  # default unchanged
    assert health.model_context_window == 500000
    assert health.weight_dead_weight == 0.50
    assert health.threshold_healthy == 0.2


def test_load_config_ignores_unknown_keys(tmp_path):
    """Unknown keys in config should be silently ignored."""
    config_path = tmp_path / "config.json"
    config_data = {
        "staleness": {
            "decay_window": 25,
            "nonexistent_key": "should_be_ignored",
        },
        "health": {
            "also_nonexistent": 999,
        },
    }
    config_path.write_text(json.dumps(config_data), encoding="utf-8")

    staleness, health = load_config(config_path)
    assert staleness.decay_window == 25
    assert not hasattr(staleness, "nonexistent_key")


def test_load_config_malformed_json(tmp_path):
    """Malformed JSON should return defaults."""
    config_path = tmp_path / "config.json"
    config_path.write_text("not valid json {{{", encoding="utf-8")

    staleness, health = load_config(config_path)
    assert staleness.decay_window == 10  # default
    assert health.model_context_window == 200_000  # default


def test_load_config_empty_sections(tmp_path):
    """Config with empty sections should use defaults."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"staleness": {}, "health": {}}), encoding="utf-8")

    staleness, health = load_config(config_path)
    assert staleness.decay_window == 10
    assert health.model_context_window == 200_000
