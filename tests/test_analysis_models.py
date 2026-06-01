"""Tests for analysis data models."""

from context_tracker.analysis.models import (
    BlockType,
    ContentBlock,
    ContextBlock,
    ContextEpoch,
    BlockStateAtTurn,
    ApiCall,
    ConversationTurn,
    TurnSnapshot,
    ContentStore,
    DataQualityWarning,
)


def test_context_block_is_frozen():
    block = ContextBlock(
        block_id="toolu_01",
        turn_entered=1,
        api_call_entered=0,
        epoch_entered=0,
        block_type=BlockType.TOOL_RESULT,
        resource="/src/server.py",
        resource_type="file",
        size_chars=5000,
        size_tokens_est=1250,
        content_hash="abc123",
        tool_name="Read",
        tool_use_id="toolu_01",
    )
    assert block.block_id == "toolu_01"
    assert block.is_pinned is False

    # Frozen — should raise on mutation
    try:
        block.size_chars = 9999
        assert False, "Should have raised"
    except AttributeError:
        pass


def test_block_type_enum():
    assert BlockType.TOOL_RESULT.value == "tool_result"
    assert BlockType.COMPACTION_SUMMARY.value == "compaction_summary"


def test_content_store_roundtrip():
    store = ContentStore()
    store.add("block-1", "Hello world, this is a long content string for testing purposes.")
    assert store.get_content("block-1") == "Hello world, this is a long content string for testing purposes."
    assert store.get_preview("block-1", max_chars=11) == "Hello world"
    assert store.get_content("nonexistent") == ""


def test_block_state_at_turn():
    state = BlockStateAtTurn(
        block_id="b1",
        staleness_score=0.75,
        staleness_label="stale",
        is_superseded=False,
        superseded_by=None,
    )
    assert state.staleness_label == "stale"
    assert state.is_superseded is False


def test_context_epoch():
    epoch = ContextEpoch(
        epoch_number=1,
        started_at_turn=50,
        compaction_summary_size=1500,
        blocks_before_compaction=120,
    )
    assert epoch.epoch_number == 1


def test_conversation_turn():
    turn = ConversationTurn(
        turn_number=1,
        timestamp="2026-06-01T10:00:00Z",
        user_prompt_text="Fix the bug in server.py",
        api_calls=[],
        epoch=0,
    )
    assert turn.turn_number == 1
    assert turn.user_prompt_text == "Fix the bug in server.py"


def test_turn_snapshot():
    snap = TurnSnapshot(
        turn_number=5,
        timestamp="2026-06-01T10:05:00Z",
        epoch=0,
        block_ids=["b1", "b2", "b3"],
        block_states=[],
        blocks_entered_ids=["b3"],
        blocks_exited_ids=[],
        total_tokens_est=45000,
        input_tokens=44000,
        output_tokens=500,
        cache_read_tokens=40000,
        cache_creation_tokens=1000,
        compaction_detected=False,
        api_call_count=2,
    )
    assert len(snap.block_ids) == 3
    assert snap.api_call_count == 2


def test_data_quality_warning():
    w = DataQualityWarning(
        line_number=42,
        warning_type="malformed_json",
        description="Could not parse JSON",
    )
    assert w.line_number == 42


def test_content_block():
    cb = ContentBlock(
        block_type="tool_use",
        content='{"file_path": "/src/server.py"}',
        size_chars=30,
        tool_use_id="toolu_01",
        tool_name="Read",
        tool_input={"file_path": "/src/server.py"},
    )
    assert cb.tool_name == "Read"
    assert cb.is_error is False
