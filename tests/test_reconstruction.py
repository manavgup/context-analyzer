"""Tests for context window reconstruction."""


from context_tracker.analysis.reconstruction import (
    extract_resource,
    group_into_turns,
    reconstruct_session,
    _extract_bash_program,
)
from context_tracker.transcript_parser import TranscriptMessage
from context_tracker.analysis.models import ContentBlock as CB


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
    turns, snapshots, content_store, epochs, warnings = reconstruct_session(messages, hook_events=[])
    assert len(turns) == 1
    assert len(snapshots) == 1
    assert len(epochs) == 1  # Epoch 0
    assert epochs[0].epoch_number == 0
    # Should have blocks: user_prompt, assistant_text, tool_use, tool_result, assistant_text
    assert len(snapshots[0].block_ids) >= 4
    # Content store should have content for each block
    for bid in snapshots[0].block_ids:
        assert content_store.has(bid)
