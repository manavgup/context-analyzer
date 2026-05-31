import pytest

from context_tracker.models import PostToolUseEvent, SessionStartEvent
from context_tracker.storage import append_event, read_events, list_sessions


def test_append_and_read_roundtrip(tmp_path):
    trace_dir = tmp_path / "traces"
    event = PostToolUseEvent(
        session_id="sess-1",
        tool_name="Read",
        input_payload_chars=100,
        output_payload_chars=5000,
        tool_use_id="toolu_01",
    )
    append_event(event, trace_dir=trace_dir)

    events = read_events("sess-1", trace_dir=trace_dir)
    assert len(events) == 1
    assert events[0].tool_name == "Read"


def test_multiple_events_same_session(tmp_path):
    trace_dir = tmp_path / "traces"
    for i in range(5):
        event = PostToolUseEvent(
            session_id="sess-1",
            tool_name=f"Tool{i}",
            input_payload_chars=i * 10,
            output_payload_chars=i * 100,
            tool_use_id=f"toolu_{i}",
        )
        append_event(event, trace_dir=trace_dir)

    events = read_events("sess-1", trace_dir=trace_dir)
    assert len(events) == 5


def test_read_nonexistent_session(tmp_path):
    events = read_events("no-such-session", trace_dir=tmp_path)
    assert events == []


def test_malformed_lines_skipped(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir(parents=True)
    filepath = trace_dir / "sess-bad.jsonl"
    filepath.write_text(
        '{"event":"post_tool_use","session_id":"sess-bad","tool_name":"X","input_payload_chars":1,"output_payload_chars":2,"tool_use_id":"t"}\n'
        'not json\n'
        '{"event":"post_tool_use","session_id":"sess-bad","tool_name":"Y","input_payload_chars":3,"output_payload_chars":4,"tool_use_id":"u"}\n'
    )
    events = read_events("sess-bad", trace_dir=trace_dir)
    assert len(events) == 2
    assert events[0].tool_name == "X"
    assert events[1].tool_name == "Y"


def test_list_sessions(tmp_path):
    trace_dir = tmp_path / "traces"
    for sid in ["aaa", "bbb", "ccc"]:
        append_event(
            SessionStartEvent(session_id=sid, source="startup", model="test"),
            trace_dir=trace_dir,
        )
    sessions = list_sessions(trace_dir=trace_dir)
    assert set(sessions) == {"aaa", "bbb", "ccc"}


def test_creates_directory_on_first_write(tmp_path):
    trace_dir = tmp_path / "nonexistent" / "deep" / "traces"
    assert not trace_dir.exists()
    append_event(
        SessionStartEvent(session_id="new", source="startup", model="test"),
        trace_dir=trace_dir,
    )
    assert trace_dir.exists()
    events = read_events("new", trace_dir=trace_dir)
    assert len(events) == 1


def test_rejects_path_traversal(tmp_path):
    from context_tracker.storage import append_event
    from context_tracker.models import SessionStartEvent

    with pytest.raises(ValueError, match="Invalid session_id"):
        event = SessionStartEvent(session_id="../../../etc/passwd", source="startup", model="test")
        append_event(event, trace_dir=tmp_path)


def test_rejects_path_traversal_read(tmp_path):
    from context_tracker.storage import read_events

    with pytest.raises(ValueError, match="Invalid session_id"):
        read_events("../../../etc/passwd", trace_dir=tmp_path)
