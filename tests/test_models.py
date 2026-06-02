import json

from context_tracker.models import (
    ApiTurnEvent,
    PostToolUseEvent,
    PostToolUseFailureEvent,
    SessionStartEvent,
    SubagentStopEvent,
    parse_event,
)


def test_post_tool_use_roundtrip():
    event = PostToolUseEvent(
        session_id="abc-123",
        tool_name="Read",
        input_payload_chars=142,
        output_payload_chars=8420,
        tool_use_id="toolu_01X",
    )
    line = event.to_jsonl()
    parsed = parse_event(line)
    assert isinstance(parsed, PostToolUseEvent)
    assert parsed.tool_name == "Read"
    assert parsed.output_payload_chars == 8420
    assert parsed.event == "post_tool_use"


def test_session_start_roundtrip():
    event = SessionStartEvent(
        session_id="abc-123",
        source="startup",
        model="claude-opus-4-6",
    )
    line = event.to_jsonl()
    parsed = parse_event(line)
    assert isinstance(parsed, SessionStartEvent)
    assert parsed.source == "startup"
    assert parsed.model == "claude-opus-4-6"


def test_api_turn_roundtrip():
    event = ApiTurnEvent(
        session_id="abc-123",
        turn_number=5,
        input_tokens=45200,
        output_tokens=1830,
        cache_read_input_tokens=38000,
        cache_creation_input_tokens=2100,
        model="claude-opus-4-6",
        stop_reason="end_turn",
    )
    line = event.to_jsonl()
    parsed = parse_event(line)
    assert isinstance(parsed, ApiTurnEvent)
    assert parsed.input_tokens == 45200
    assert parsed.cache_read_input_tokens == 38000


def test_parse_event_unknown_type():
    line = json.dumps({"event": "unknown_event", "session_id": "x"})
    result = parse_event(line)
    assert result is None


def test_parse_event_malformed_json():
    result = parse_event("not json at all {{{")
    assert result is None


def test_post_tool_use_failure_roundtrip():
    event = PostToolUseFailureEvent(
        session_id="abc-123",
        tool_name="Bash",
        input_payload_chars=85,
        error_length=320,
        tool_use_id="toolu_02Y",
    )
    line = event.to_jsonl()
    parsed = parse_event(line)
    assert isinstance(parsed, PostToolUseFailureEvent)
    assert parsed.error_length == 320


def test_subagent_stop_roundtrip():
    event = SubagentStopEvent(
        session_id="abc-123",
        agent_id="agent-001",
        agent_type="general-purpose",
        agent_transcript_path="/path/to/transcript.jsonl",
    )
    line = event.to_jsonl()
    parsed = parse_event(line)
    assert isinstance(parsed, SubagentStopEvent)
    assert parsed.agent_transcript_path == "/path/to/transcript.jsonl"
