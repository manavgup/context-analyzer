"""Tests for staleness detection engine."""

from context_tracker.analysis.models import (
    BlockType,
    ContextBlock,
)
from context_tracker.analysis.staleness import (
    detect_superseded,
    compute_staleness,
    label_staleness,
    base_decay,
)
from context_tracker.analysis.config import StalenessConfig


def _block(block_id: str, turn: int, resource: str | None = None,
           resource_type: str | None = None, block_type: BlockType = BlockType.TOOL_RESULT,
           is_pinned: bool = False) -> ContextBlock:
    return ContextBlock(
        block_id=block_id, turn_entered=turn, api_call_entered=0,
        epoch_entered=0, block_type=block_type, resource=resource,
        resource_type=resource_type, size_chars=1000, size_tokens_est=250,
        content_hash=f"hash-{block_id}", is_pinned=is_pinned,
    )


def test_detect_superseded():
    blocks = [
        _block("b1", turn=5, resource="/src/server.py", resource_type="file"),
        _block("b2", turn=20, resource="/src/server.py", resource_type="file"),
        _block("b3", turn=10, resource="/src/models.py", resource_type="file"),
    ]
    superseded = detect_superseded(blocks)
    assert superseded == {"b1": "b2"}  # b1 superseded by b2
    assert "b3" not in superseded  # Only one read, not superseded


def test_superseded_block_is_dead_weight():
    block = _block("b1", turn=5, resource="/src/server.py", resource_type="file")
    config = StalenessConfig()
    score, label = compute_staleness(
        block=block,
        current_turn=25,
        config=config,
        resource_last_used={"/src/server.py": 20},
        messages_since_block=[],
        active_resources={"/src/server.py"},
        task_boundaries=[],
        superseded_map={"b1": "b2"},
    )
    assert score == 0.9
    assert label == "dead_weight"


def test_pinned_block_always_fresh():
    block = _block("sys", turn=1, is_pinned=True, block_type=BlockType.SYSTEM)
    config = StalenessConfig()
    score, label = compute_staleness(
        block=block, current_turn=300, config=config,
        resource_last_used={}, messages_since_block=[],
        active_resources=set(), task_boundaries=[],
        superseded_map={},
    )
    assert score == 0.0
    assert label == "pinned"


def test_fresh_block_is_active():
    block = _block("b1", turn=10, resource="/a.py", resource_type="file")
    config = StalenessConfig()
    score, label = compute_staleness(
        block=block, current_turn=11, config=config,
        resource_last_used={"/a.py": 10}, messages_since_block=[],
        active_resources={"/a.py"}, task_boundaries=[],
        superseded_map={},
    )
    assert score < 0.3
    assert label == "active"


def test_old_unreferenced_block_is_stale():
    block = _block("b1", turn=5, resource="/old.py", resource_type="file")
    config = StalenessConfig(decay_window=10)
    score, label = compute_staleness(
        block=block, current_turn=50, config=config,
        resource_last_used={},  # Never used again
        messages_since_block=[],  # Never referenced
        active_resources=set(),
        task_boundaries=[],
        superseded_map={},
    )
    assert score > 0.6
    assert label in ("stale", "dead_weight")


def test_base_decay():
    assert base_decay(0, 10) == 0.0
    assert base_decay(2, 10) == 0.0
    assert 0 < base_decay(5, 10) < 0.5
    assert base_decay(10, 10) == 0.5
    assert base_decay(30, 10) > 0.5
    assert base_decay(100, 10) <= 1.0


def test_label_staleness():
    assert label_staleness(0.0) == "active"
    assert label_staleness(0.29) == "active"
    assert label_staleness(0.3) == "warm"
    assert label_staleness(0.59) == "warm"
    assert label_staleness(0.6) == "stale"
    assert label_staleness(0.79) == "stale"
    assert label_staleness(0.8) == "dead_weight"
    assert label_staleness(1.0) == "dead_weight"
