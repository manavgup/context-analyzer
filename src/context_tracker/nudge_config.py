"""Default configuration for real-time session nudges."""

from __future__ import annotations

NUDGE_DEFAULTS: dict[str, object] = {
    "enabled": True,
    "context_window": 1_000_000,
    "context_threshold_pct": 60,
    "cost_warning_usd": 10.0,
    "repeated_read_count": 3,
    "cooldown_turns": 5,
}
