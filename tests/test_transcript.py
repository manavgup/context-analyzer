import json
from pathlib import Path

from context_tracker.models import ApiTurnEvent
from context_tracker.transcript import parse_transcript


def _write_transcript(path: Path, entries: list[dict]) -> None:
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def test_parse_transcript_extracts_usage(tmp_path):
    transcript_path = tmp_path / "session.jsonl"
    _write_transcript(transcript_path, [
        {
            "type": "user",
            "message": {"role": "user", "content": "hello"},
            "sessionId": "sess-1",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-6",
                "content": [{"type": "text", "text": "Hi!"}],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 45000,
                    "output_tokens": 200,
                    "cache_read_input_tokens": 38000,
                    "cache_creation_input_tokens": 2000,
                },
            },
            "sessionId": "sess-1",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-6",
                "content": [{"type": "text", "text": "More"}],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 46000,
                    "output_tokens": 150,
                    "cache_read_input_tokens": 40000,
                    "cache_creation_input_tokens": 1000,
                },
            },
            "sessionId": "sess-1",
        },
    ])

    events = parse_transcript(transcript_path)
    assert len(events) == 2
    assert all(isinstance(e, ApiTurnEvent) for e in events)
    assert events[0].turn_number == 1
    assert events[0].input_tokens == 45000
    assert events[0].cache_read_input_tokens == 38000
    assert events[1].turn_number == 2
    assert events[1].input_tokens == 46000


def test_parse_transcript_skips_non_assistant(tmp_path):
    transcript_path = tmp_path / "session.jsonl"
    _write_transcript(transcript_path, [
        {"type": "user", "message": {"role": "user", "content": "hi"}, "sessionId": "s"},
        {"type": "file-history-snapshot", "messageId": "x", "snapshot": {}},
    ])
    events = parse_transcript(transcript_path)
    assert events == []


def test_parse_transcript_skips_missing_usage(tmp_path):
    transcript_path = tmp_path / "session.jsonl"
    _write_transcript(transcript_path, [
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-6",
                "content": [{"type": "text", "text": "Hi"}],
                "stop_reason": "end_turn",
            },
            "sessionId": "sess-1",
        },
    ])
    events = parse_transcript(transcript_path)
    assert events == []


def test_parse_transcript_deduplicates_by_stop_reason(tmp_path):
    """Claude Code emits multiple assistant entries per API call (streaming chunks).
    Only entries with a non-null stop_reason and usage.output_tokens > 0
    represent completed turns."""
    transcript_path = tmp_path / "session.jsonl"
    _write_transcript(transcript_path, [
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-6",
                "content": [{"type": "text", "text": "chunk1"}],
                "stop_reason": None,
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 10,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 50,
                },
            },
            "sessionId": "sess-1",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-6",
                "content": [{"type": "text", "text": "final"}],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 200,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 50,
                },
            },
            "sessionId": "sess-1",
        },
    ])
    events = parse_transcript(transcript_path)
    assert len(events) == 1
    assert events[0].output_tokens == 200


def test_parse_nonexistent_file(tmp_path):
    events = parse_transcript(tmp_path / "nope.jsonl")
    assert events == []
