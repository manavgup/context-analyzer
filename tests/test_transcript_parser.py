"""Tests for raw transcript parser."""

import json
from pathlib import Path

from context_tracker.transcript_parser import (
    parse_raw_transcript,
    TranscriptMessage,
)


def _write_transcript(path: Path, entries: list[dict]) -> None:
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def test_parse_user_text_message(tmp_path):
    path = tmp_path / "session.jsonl"
    _write_transcript(path, [
        {
            "type": "user",
            "uuid": "u1",
            "sessionId": "sess-1",
            "timestamp": "2026-06-01T10:00:00Z",
            "message": {"role": "user", "content": "Fix the bug in server.py"},
        },
    ])
    messages, warnings = parse_raw_transcript(path)
    assert len(messages) == 1
    assert messages[0].entry_type == "user"
    assert messages[0].session_id == "sess-1"
    assert messages[0].timestamp == "2026-06-01T10:00:00Z"
    assert len(messages[0].content_blocks) == 1
    assert messages[0].content_blocks[0].block_type == "text"
    assert messages[0].content_blocks[0].content == "Fix the bug in server.py"
    assert len(warnings) == 0


def test_parse_assistant_with_tool_use(tmp_path):
    path = tmp_path / "session.jsonl"
    _write_transcript(path, [
        {
            "type": "assistant",
            "uuid": "a1",
            "sessionId": "sess-1",
            "timestamp": "2026-06-01T10:01:00Z",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-6",
                "stop_reason": "tool_use",
                "content": [
                    {"type": "text", "text": "Let me read that file."},
                    {"type": "tool_use", "id": "toolu_01", "name": "Read",
                     "input": {"file_path": "/src/server.py"}},
                ],
                "usage": {
                    "input_tokens": 30000,
                    "output_tokens": 200,
                    "cache_read_input_tokens": 25000,
                    "cache_creation_input_tokens": 3000,
                },
            },
        },
    ])
    messages, warnings = parse_raw_transcript(path)
    assert len(messages) == 1
    m = messages[0]
    assert m.entry_type == "assistant"
    assert m.input_tokens == 30000
    assert m.output_tokens == 200
    assert m.cache_read_tokens == 25000
    assert m.cache_creation_tokens == 3000
    assert m.stop_reason == "tool_use"
    assert m.model == "claude-opus-4-6"
    assert len(m.content_blocks) == 2
    assert m.content_blocks[0].block_type == "text"
    assert m.content_blocks[1].block_type == "tool_use"
    assert m.content_blocks[1].tool_name == "Read"
    assert m.content_blocks[1].tool_use_id == "toolu_01"
    assert m.content_blocks[1].tool_input == {"file_path": "/src/server.py"}


def test_parse_user_with_tool_result(tmp_path):
    path = tmp_path / "session.jsonl"
    _write_transcript(path, [
        {
            "type": "user",
            "uuid": "u2",
            "sessionId": "sess-1",
            "timestamp": "2026-06-01T10:01:05Z",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_01",
                     "content": "def main():\n    pass\n", "is_error": False},
                ],
            },
        },
    ])
    messages, warnings = parse_raw_transcript(path)
    assert len(messages) == 1
    m = messages[0]
    assert len(m.content_blocks) == 1
    cb = m.content_blocks[0]
    assert cb.block_type == "tool_result"
    assert cb.tool_use_id == "toolu_01"
    assert cb.content == "def main():\n    pass\n"
    assert cb.is_error is False


def test_parse_skips_streaming_chunks(tmp_path):
    """Only completed API calls (stop_reason set, output_tokens > 0) are kept."""
    path = tmp_path / "session.jsonl"
    _write_transcript(path, [
        {
            "type": "assistant",
            "uuid": "a1",
            "sessionId": "sess-1",
            "timestamp": "2026-06-01T10:01:00Z",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-6",
                "stop_reason": None,
                "content": [{"type": "text", "text": "partial"}],
                "usage": {"input_tokens": 100, "output_tokens": 5,
                           "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
            },
        },
        {
            "type": "assistant",
            "uuid": "a2",
            "sessionId": "sess-1",
            "timestamp": "2026-06-01T10:01:01Z",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-6",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "complete response"}],
                "usage": {"input_tokens": 100, "output_tokens": 200,
                           "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
            },
        },
    ])
    messages, warnings = parse_raw_transcript(path)
    assert len(messages) == 1
    assert messages[0].content_blocks[0].content == "complete response"


def test_parse_malformed_lines_produce_warnings(tmp_path):
    path = tmp_path / "session.jsonl"
    with open(path, "w") as f:
        f.write("not json at all\n")
        f.write(json.dumps({
            "type": "user", "uuid": "u1", "sessionId": "s1",
            "timestamp": "2026-06-01T10:00:00Z",
            "message": {"role": "user", "content": "hello"},
        }) + "\n")
        f.write("{broken json\n")

    messages, warnings = parse_raw_transcript(path)
    assert len(messages) == 1
    assert len(warnings) == 2
    assert warnings[0].warning_type == "malformed_json"
    assert warnings[0].line_number == 1
    assert warnings[1].line_number == 3


def test_parse_system_entries_as_metadata(tmp_path):
    path = tmp_path / "session.jsonl"
    _write_transcript(path, [
        {
            "type": "system",
            "uuid": "s1",
            "sessionId": "sess-1",
            "timestamp": "2026-06-01T10:00:00Z",
            "subtype": "turn_duration",
            "content": "",
        },
    ])
    messages, warnings = parse_raw_transcript(path)
    # System entries are parsed but have no content blocks
    assert len(messages) == 1
    assert messages[0].entry_type == "system"
    assert len(messages[0].content_blocks) == 0


def test_parse_nonexistent_file(tmp_path):
    messages, warnings = parse_raw_transcript(tmp_path / "nope.jsonl")
    assert messages == []
    assert len(warnings) == 0


def test_parse_thinking_blocks(tmp_path):
    path = tmp_path / "session.jsonl"
    _write_transcript(path, [
        {
            "type": "assistant",
            "uuid": "a1",
            "sessionId": "sess-1",
            "timestamp": "2026-06-01T10:01:00Z",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-6",
                "stop_reason": "end_turn",
                "content": [
                    {"type": "thinking", "thinking": "Let me analyze this..."},
                    {"type": "text", "text": "Here is my answer."},
                ],
                "usage": {"input_tokens": 100, "output_tokens": 50,
                           "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
            },
        },
    ])
    messages, warnings = parse_raw_transcript(path)
    assert len(messages) == 1
    assert len(messages[0].content_blocks) == 2
    assert messages[0].content_blocks[0].block_type == "thinking"
    assert messages[0].content_blocks[0].content == "Let me analyze this..."


def test_sequential_indexing(tmp_path):
    path = tmp_path / "session.jsonl"
    _write_transcript(path, [
        {"type": "user", "uuid": "u1", "sessionId": "s1", "timestamp": "T1",
         "message": {"role": "user", "content": "hello"}},
        {"type": "assistant", "uuid": "a1", "sessionId": "s1", "timestamp": "T2",
         "message": {"role": "assistant", "model": "m", "stop_reason": "end_turn",
                     "content": [{"type": "text", "text": "hi"}],
                     "usage": {"input_tokens": 1, "output_tokens": 1,
                                "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}}},
        {"type": "user", "uuid": "u2", "sessionId": "s1", "timestamp": "T3",
         "message": {"role": "user", "content": "bye"}},
    ])
    messages, warnings = parse_raw_transcript(path)
    assert len(messages) == 3
    assert messages[0].sequence_index == 0
    assert messages[1].sequence_index == 1
    assert messages[2].sequence_index == 2
