import json

import pytest

from context_tracker.models import (
    PostCompactEvent,
    PostToolUseEvent,
    PreCompactEvent,
    SessionStartEvent,
    UserPromptEvent,
)
from context_tracker.server import (
    get_compaction_history,
    get_context_hogs,
    get_session_history,
    get_session_summary,
    get_tool_breakdown,
)
from context_tracker.storage import append_event


@pytest.fixture
def populated_session(tmp_path):
    """Create a trace directory with a realistic session."""
    trace_dir = tmp_path / "traces"
    transcript_dir = tmp_path / "transcripts"
    transcript_dir.mkdir(parents=True)

    session_id = "test-session"

    append_event(SessionStartEvent(
        session_id=session_id, source="startup", model="claude-opus-4-6",
    ), trace_dir=trace_dir)

    for i, (name, in_size, out_size) in enumerate([
        ("Read", 50, 8000),
        ("Read", 60, 12000),
        ("Bash", 200, 3000),
        ("Grep", 80, 500),
        ("Read", 45, 6000),
    ]):
        append_event(PostToolUseEvent(
            session_id=session_id,
            tool_name=name,
            input_payload_chars=in_size,
            output_payload_chars=out_size,
            tool_use_id=f"toolu_{i}",
        ), trace_dir=trace_dir)

    append_event(PreCompactEvent(
        session_id=session_id, trigger="auto",
    ), trace_dir=trace_dir)

    append_event(PostCompactEvent(
        session_id=session_id, trigger="auto", compact_summary_length=1500,
    ), trace_dir=trace_dir)

    append_event(UserPromptEvent(
        session_id=session_id, prompt_length_chars=200,
    ), trace_dir=trace_dir)

    # Write a transcript file
    transcript_path = transcript_dir / f"{session_id}.jsonl"
    turns = [
        {"type": "assistant", "sessionId": session_id, "message": {
            "role": "assistant", "model": "claude-opus-4-6", "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "hi"}],
            "usage": {"input_tokens": 30000, "output_tokens": 500,
                       "cache_read_input_tokens": 25000, "cache_creation_input_tokens": 3000},
        }},
        {"type": "assistant", "sessionId": session_id, "message": {
            "role": "assistant", "model": "claude-opus-4-6", "stop_reason": "tool_use",
            "content": [{"type": "text", "text": "let me check"}],
            "usage": {"input_tokens": 45000, "output_tokens": 800,
                       "cache_read_input_tokens": 40000, "cache_creation_input_tokens": 1000},
        }},
    ]
    with open(transcript_path, "w") as f:
        for t in turns:
            f.write(json.dumps(t) + "\n")

    return session_id, trace_dir, transcript_dir


def test_get_session_summary(populated_session):
    session_id, trace_dir, transcript_dir = populated_session
    result = get_session_summary(session_id, trace_dir=trace_dir, transcript_dir=transcript_dir)
    assert result["session_id"] == session_id
    assert result["tool_calls"] == 5
    assert result["compactions"] == 1
    assert result["api_turns"] == 2
    assert result["total_input_tokens"] == 75000
    assert result["total_output_tokens"] == 1300


def test_get_tool_breakdown(populated_session):
    session_id, trace_dir, transcript_dir = populated_session
    result = get_tool_breakdown(session_id, trace_dir=trace_dir)
    assert len(result) == 3  # Read, Bash, Grep
    # Read should be first (highest total output)
    assert result[0]["tool_name"] == "Read"
    assert result[0]["call_count"] == 3
    assert result[0]["total_output_payload_chars"] == 26000


def test_get_compaction_history(populated_session):
    session_id, trace_dir, _ = populated_session
    result = get_compaction_history(session_id, trace_dir=trace_dir)
    assert len(result) == 1
    assert result[0]["trigger"] == "auto"
    assert result[0]["summary_length"] == 1500


def test_get_context_hogs(populated_session):
    session_id, trace_dir, _ = populated_session
    result = get_context_hogs(session_id, top_n=3, trace_dir=trace_dir)
    assert len(result) == 3
    # Largest output first
    assert result[0]["tool_name"] == "Read"
    assert result[0]["output_payload_chars"] == 12000


def test_get_session_history(populated_session):
    session_id, trace_dir, transcript_dir = populated_session
    result = get_session_history(trace_dir=trace_dir, transcript_dir=transcript_dir)
    assert len(result) == 1
    assert result[0]["session_id"] == session_id
    assert result[0]["tool_calls"] == 5
