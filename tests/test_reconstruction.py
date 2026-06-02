"""Tests for context window reconstruction."""


from context_tracker.analysis.reconstruction import (
    extract_resource,
    group_into_turns,
    reconstruct_session,
    _extract_bash_program,
    _detect_compactions_from_api,
)
from context_tracker.transcript_parser import TranscriptMessage
from context_tracker.analysis.models import (
    ApiCall,
    BlockType,
    ContentBlock as CB,
    ConversationTurn,
)


def _make_user_msg(seq: int, text: str, ts: str = "T") -> TranscriptMessage:
    return TranscriptMessage(
        message_id=f"u{seq}",
        sequence_index=seq,
        entry_type="user",
        timestamp=ts,
        session_id="s1",
        content_blocks=[CB(block_type="text", content=text, size_chars=len(text))],
    )


def _make_assistant_msg(
    seq: int, text: str, tool_uses: list[tuple[str, str, dict]] | None = None,
    input_tokens: int = 100, output_tokens: int = 50, ts: str = "T",
) -> TranscriptMessage:
    blocks = [CB(block_type="text", content=text, size_chars=len(text))]
    if tool_uses:
        for tu_id, name, inp in tool_uses:
            inp_str = str(inp)
            blocks.append(CB(
                block_type="tool_use", content=inp_str, size_chars=len(inp_str),
                tool_use_id=tu_id, tool_name=name, tool_input=inp,
            ))
    return TranscriptMessage(
        message_id=f"a{seq}",
        sequence_index=seq,
        entry_type="assistant",
        timestamp=ts,
        session_id="s1",
        content_blocks=blocks,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        stop_reason="end_turn",
        model="claude-opus-4-6",
    )


def _make_tool_result_msg(
    seq: int, tool_use_id: str, content: str, ts: str = "T",
) -> TranscriptMessage:
    return TranscriptMessage(
        message_id=f"tr{seq}",
        sequence_index=seq,
        entry_type="user",
        timestamp=ts,
        session_id="s1",
        content_blocks=[CB(
            block_type="tool_result", content=content, size_chars=len(content),
            tool_use_id=tool_use_id,
        )],
    )


def test_extract_resource_file():
    res, rtype = extract_resource("Read", {"file_path": "/src/server.py"})
    assert res == "/src/server.py"
    assert rtype == "file"


def test_extract_resource_bash():
    res, rtype = extract_resource("Bash", {"command": "cd /project && pytest tests/ -v"})
    assert res == "pytest"
    assert rtype == "command"


def test_extract_bash_program():
    assert _extract_bash_program("pytest tests/ -v") == "pytest"
    assert _extract_bash_program("cd /foo && npm test") == "npm"
    assert _extract_bash_program("env FOO=1 python3 main.py") == "python3"
    assert _extract_bash_program("uv run pytest") == "pytest"
    assert _extract_bash_program("cat foo | grep bar") == "cat"
    assert _extract_bash_program("") == ""


def test_extract_resource_grep():
    res, rtype = extract_resource("Grep", {"pattern": "import", "path": "/src"})
    assert res == "import@/src"
    assert rtype == "pattern"


def test_extract_resource_unknown_tool():
    res, rtype = extract_resource("CustomTool", {"foo": "bar"})
    assert res is None
    assert rtype is None


def test_group_into_turns():
    messages = [
        _make_user_msg(0, "Fix the bug"),
        _make_assistant_msg(1, "Let me read", tool_uses=[("t1", "Read", {"file_path": "/a.py"})]),
        _make_tool_result_msg(2, "t1", "def foo(): pass"),
        _make_assistant_msg(3, "Fixed it"),
        _make_user_msg(4, "Now add tests"),
        _make_assistant_msg(5, "Sure"),
    ]
    turns = group_into_turns(messages)
    assert len(turns) == 2
    assert turns[0].turn_number == 1
    assert turns[0].user_prompt_text == "Fix the bug"
    assert turns[1].turn_number == 2
    assert turns[1].user_prompt_text == "Now add tests"


def test_group_into_turns_tool_result_continues_turn():
    """User messages with only tool_result blocks continue the current turn."""
    messages = [
        _make_user_msg(0, "Read two files"),
        _make_assistant_msg(1, "Reading", tool_uses=[
            ("t1", "Read", {"file_path": "/a.py"}),
            ("t2", "Read", {"file_path": "/b.py"}),
        ]),
        _make_tool_result_msg(2, "t1", "file a content"),
        _make_tool_result_msg(3, "t2", "file b content"),
        _make_assistant_msg(4, "Done"),
    ]
    turns = group_into_turns(messages)
    assert len(turns) == 1


def test_reconstruct_session_basic():
    messages = [
        _make_user_msg(0, "Fix bug", ts="2026-06-01T10:00:00Z"),
        _make_assistant_msg(1, "Reading file", ts="2026-06-01T10:00:05Z",
                           tool_uses=[("t1", "Read", {"file_path": "/src/server.py"})],
                           input_tokens=30000, output_tokens=200),
        _make_tool_result_msg(2, "t1", "def main(): pass\n" * 100, ts="2026-06-01T10:00:06Z"),
        _make_assistant_msg(3, "Found the bug, fixing now", ts="2026-06-01T10:00:10Z",
                           input_tokens=32000, output_tokens=300),
    ]
    turns, snapshots, content_store, epochs, warnings, block_registry = reconstruct_session(messages, hook_events=[])
    assert len(turns) == 1
    assert len(snapshots) == 1
    assert len(epochs) == 1  # Epoch 0
    assert epochs[0].epoch_number == 0
    # Should have blocks: user_prompt, assistant_text, tool_use, tool_result, assistant_text
    assert len(snapshots[0].block_ids) >= 4
    # Content store should have content for each block
    for bid in snapshots[0].block_ids:
        assert content_store.has(bid)


def test_reconstruct_session_actual_context_tokens():
    """TurnSnapshot carries actual_context_tokens from API ground truth."""
    messages = [
        _make_user_msg(0, "Fix bug", ts="2026-06-01T10:00:00Z"),
        _make_assistant_msg(
            1, "Reading file", ts="2026-06-01T10:00:05Z",
            input_tokens=1000, output_tokens=200,
        ),
    ]
    turns, snapshots, *_ = reconstruct_session(messages, hook_events=[])
    assert len(snapshots) == 1
    # actual_context_tokens = input + cache_read + cache_create = 1000 + 0 + 0
    assert snapshots[0].actual_context_tokens == 1000


def _make_assistant_msg_with_cache(
    seq: int, text: str,
    input_tokens: int = 0, output_tokens: int = 50,
    cache_read: int = 0, cache_create: int = 0,
    ts: str = "T",
) -> TranscriptMessage:
    """Helper to create an assistant message with explicit cache token counts."""
    return TranscriptMessage(
        message_id=f"a{seq}",
        sequence_index=seq,
        entry_type="assistant",
        timestamp=ts,
        session_id="s1",
        content_blocks=[CB(block_type="text", content=text, size_chars=len(text))],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_creation_tokens=cache_create,
        stop_reason="end_turn",
        model="claude-opus-4-6",
    )


def test_detect_compactions_from_api_no_compaction():
    """No compaction: cache_create stays small after first call."""
    turns = [
        ConversationTurn(turn_number=1, timestamp="T1", user_prompt_text="t1",
                         api_calls=[ApiCall(api_call_index=0, conversation_turn=1,
                                           input_tokens=3, cache_read_tokens=0,
                                           cache_creation_tokens=30000)]),
        ConversationTurn(turn_number=2, timestamp="T2", user_prompt_text="t2",
                         api_calls=[ApiCall(api_call_index=1, conversation_turn=2,
                                           input_tokens=100, cache_read_tokens=30000,
                                           cache_creation_tokens=5000)]),
        ConversationTurn(turn_number=3, timestamp="T3", user_prompt_text="t3",
                         api_calls=[ApiCall(api_call_index=2, conversation_turn=3,
                                           input_tokens=200, cache_read_tokens=35000,
                                           cache_creation_tokens=3000)]),
    ]
    result = _detect_compactions_from_api(turns)
    assert result == []


def test_detect_compactions_from_api_detects_spike():
    """Compaction detected when cache_create > 50% of total after first call."""
    turns = [
        ConversationTurn(turn_number=1, timestamp="T1", user_prompt_text="t1",
                         api_calls=[ApiCall(api_call_index=0, conversation_turn=1,
                                           input_tokens=3, cache_read_tokens=0,
                                           cache_creation_tokens=30000)]),
        ConversationTurn(turn_number=2, timestamp="T2", user_prompt_text="t2",
                         api_calls=[ApiCall(api_call_index=1, conversation_turn=2,
                                           input_tokens=100, cache_read_tokens=50000,
                                           cache_creation_tokens=2000)]),
        # Turn 3: compaction — cache_create is 100% of total
        ConversationTurn(turn_number=3, timestamp="T3", user_prompt_text="t3",
                         api_calls=[ApiCall(api_call_index=2, conversation_turn=3,
                                           input_tokens=3, cache_read_tokens=0,
                                           cache_creation_tokens=60000)]),
        ConversationTurn(turn_number=4, timestamp="T4", user_prompt_text="t4",
                         api_calls=[ApiCall(api_call_index=3, conversation_turn=4,
                                           input_tokens=100, cache_read_tokens=60000,
                                           cache_creation_tokens=1000)]),
    ]
    result = _detect_compactions_from_api(turns)
    assert result == [3]


def test_detect_compactions_from_api_multiple():
    """Multiple compaction events detected."""
    turns = [
        ConversationTurn(turn_number=1, timestamp="T1", user_prompt_text="t1",
                         api_calls=[ApiCall(api_call_index=0, conversation_turn=1,
                                           input_tokens=3, cache_creation_tokens=30000)]),
        # Compaction at turn 2
        ConversationTurn(turn_number=2, timestamp="T2", user_prompt_text="t2",
                         api_calls=[ApiCall(api_call_index=1, conversation_turn=2,
                                           input_tokens=3, cache_creation_tokens=50000)]),
        ConversationTurn(turn_number=3, timestamp="T3", user_prompt_text="t3",
                         api_calls=[ApiCall(api_call_index=2, conversation_turn=3,
                                           input_tokens=100, cache_read_tokens=50000,
                                           cache_creation_tokens=2000)]),
        # Compaction at turn 4
        ConversationTurn(turn_number=4, timestamp="T4", user_prompt_text="t4",
                         api_calls=[ApiCall(api_call_index=3, conversation_turn=4,
                                           input_tokens=3, cache_creation_tokens=80000)]),
    ]
    result = _detect_compactions_from_api(turns)
    assert result == [2, 4]


def test_detect_compactions_from_api_skips_first_call():
    """First API call always has 100% cache_create and is not a compaction."""
    turns = [
        ConversationTurn(turn_number=1, timestamp="T1", user_prompt_text="t1",
                         api_calls=[ApiCall(api_call_index=0, conversation_turn=1,
                                           input_tokens=0, cache_read_tokens=0,
                                           cache_creation_tokens=30000)]),
    ]
    result = _detect_compactions_from_api(turns)
    assert result == []


def test_reconstruct_session_compaction_creates_epochs():
    """Compaction events create new epochs and compaction summary blocks."""
    messages = [
        # Turn 1: initial (cache_create 100% — not compaction, first call)
        _make_user_msg(0, "First task", ts="2026-06-01T10:00:00Z"),
        _make_assistant_msg_with_cache(
            1, "Working on it", ts="2026-06-01T10:00:05Z",
            input_tokens=3, cache_create=30000, output_tokens=200,
        ),
        # Turn 2: normal (cache_read dominant)
        _make_user_msg(2, "Continue", ts="2026-06-01T10:01:00Z"),
        _make_assistant_msg_with_cache(
            3, "Reading more", ts="2026-06-01T10:01:05Z",
            input_tokens=100, cache_read=30000, cache_create=5000, output_tokens=150,
        ),
        # Turn 3: COMPACTION (cache_create 100% of total)
        _make_user_msg(4, "New approach", ts="2026-06-01T10:05:00Z"),
        _make_assistant_msg_with_cache(
            5, "After compaction", ts="2026-06-01T10:05:05Z",
            input_tokens=3, cache_read=0, cache_create=50000, output_tokens=200,
        ),
        # Turn 4: normal post-compaction
        _make_user_msg(6, "Keep going", ts="2026-06-01T10:06:00Z"),
        _make_assistant_msg_with_cache(
            7, "Continuing", ts="2026-06-01T10:06:05Z",
            input_tokens=100, cache_read=50000, cache_create=2000, output_tokens=100,
        ),
    ]
    turns, snapshots, content_store, epochs, warnings, block_registry = (
        reconstruct_session(messages, hook_events=[])
    )

    assert len(turns) == 4
    assert len(snapshots) == 4

    # Should have at least 2 epochs: 0 and 1 (compaction at turn 3)
    assert len(epochs) >= 2
    assert epochs[0].epoch_number == 0
    assert epochs[0].started_at_turn == 1
    assert epochs[1].epoch_number == 1
    assert epochs[1].started_at_turn == 3

    # The compaction epoch should have blocks_before_compaction > 0
    assert epochs[1].blocks_before_compaction > 0

    # Snapshot at turn 3 should have compaction_detected
    snap_3 = snapshots[2]
    assert snap_3.compaction_detected is True

    # Snapshot at turn 3 should NOT contain blocks from turns 1-2
    # (except pinned system blocks)
    for bid in snap_3.block_ids:
        block = block_registry[bid]
        if not block.is_pinned and block.block_type != BlockType.COMPACTION_SUMMARY:
            assert block.turn_entered >= 3, (
                f"Block {bid} (type={block.block_type}, turn_entered={block.turn_entered}) "
                f"should not be in context after compaction at turn 3"
            )

    # There should be a compaction summary block in context at turn 3
    summary_blocks = [
        bid for bid in snap_3.block_ids
        if block_registry[bid].block_type == BlockType.COMPACTION_SUMMARY
    ]
    assert len(summary_blocks) >= 1

    # actual_context_tokens should be populated
    assert snap_3.actual_context_tokens > 0


def test_reconstruct_session_blocks_after_compaction():
    """After compaction, only current-epoch blocks and pinned blocks are in context."""
    messages = [
        _make_user_msg(0, "Task one", ts="2026-06-01T10:00:00Z"),
        _make_assistant_msg_with_cache(
            1, "Initial response", ts="2026-06-01T10:00:05Z",
            input_tokens=3, cache_create=20000, output_tokens=100,
        ),
        _make_user_msg(2, "Continue task", ts="2026-06-01T10:01:00Z"),
        _make_assistant_msg_with_cache(
            3, "More work", ts="2026-06-01T10:01:05Z",
            input_tokens=100, cache_read=20000, cache_create=3000, output_tokens=100,
        ),
        # Compaction
        _make_user_msg(4, "New task after compact", ts="2026-06-01T10:05:00Z"),
        _make_assistant_msg_with_cache(
            5, "Post compaction work", ts="2026-06-01T10:05:05Z",
            input_tokens=3, cache_read=0, cache_create=40000, output_tokens=100,
        ),
    ]
    turns, snapshots, content_store, epochs, warnings, block_registry = (
        reconstruct_session(messages, hook_events=[])
    )

    # Pre-compaction snapshot (turn 2) should have more blocks
    snap_pre = snapshots[1]
    # Post-compaction snapshot (turn 3) should have fewer non-summary blocks
    snap_post = snapshots[2]

    pre_non_pinned = [
        bid for bid in snap_pre.block_ids
        if not block_registry[bid].is_pinned
        and block_registry[bid].block_type != BlockType.COMPACTION_SUMMARY
    ]
    post_non_pinned_non_summary = [
        bid for bid in snap_post.block_ids
        if not block_registry[bid].is_pinned
        and block_registry[bid].block_type != BlockType.COMPACTION_SUMMARY
    ]

    # Post-compaction should have fewer blocks than pre-compaction (old ones evicted)
    assert len(post_non_pinned_non_summary) < len(pre_non_pinned)

    # Exited blocks should be non-empty at compaction turn
    assert len(snap_post.blocks_exited_ids) > 0
