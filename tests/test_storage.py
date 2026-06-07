import pytest

from context_tracker.models import PostToolUseEvent, SessionStartEvent
from context_tracker.storage import append_event, list_sessions, read_events


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
        "not json\n"
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
    sessions = list_sessions(trace_dir=trace_dir, projects_dir=tmp_path / "empty")
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


def test_list_sessions_from_projects_dir(tmp_path):
    """Sessions from transcript files in projects dir are discovered."""
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    projects_dir = tmp_path / "projects"

    # Create transcript files in project subdirectories (mimics ~/.claude/projects/<slug>/<uuid>.jsonl)
    proj1 = projects_dir / "project-one"
    proj1.mkdir(parents=True)
    (proj1 / "aaaaaaaa-1111-2222-3333-444444444444.jsonl").write_text("{}\n")

    proj2 = projects_dir / "project-two"
    proj2.mkdir(parents=True)
    (proj2 / "bbbbbbbb-1111-2222-3333-444444444444.jsonl").write_text("{}\n")

    sessions = list_sessions(trace_dir=trace_dir, projects_dir=projects_dir)
    assert set(sessions) == {
        "aaaaaaaa-1111-2222-3333-444444444444",
        "bbbbbbbb-1111-2222-3333-444444444444",
    }


def test_list_sessions_deduplicates(tmp_path):
    """Sessions present in both trace and projects dirs appear once."""
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    shared_sid = "cccccccc-1111-2222-3333-444444444444"
    (trace_dir / f"{shared_sid}.jsonl").write_text("{}\n")

    projects_dir = tmp_path / "projects" / "proj"
    projects_dir.mkdir(parents=True)
    (projects_dir / f"{shared_sid}.jsonl").write_text("{}\n")

    sessions = list_sessions(trace_dir=trace_dir, projects_dir=projects_dir)
    assert sessions.count(shared_sid) == 1


def test_list_sessions_excludes_subagents(tmp_path):
    """Subagent transcripts (agent-*.jsonl inside subagents/) are excluded."""
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    projects_dir = tmp_path / "projects" / "proj"
    projects_dir.mkdir(parents=True)

    # Real session
    (projects_dir / "dddddddd-1111-2222-3333-444444444444.jsonl").write_text("{}\n")

    # Subagent transcript — should be excluded
    sa_dir = projects_dir / "dddddddd-1111-2222-3333-444444444444" / "subagents"
    sa_dir.mkdir(parents=True)
    (sa_dir / "agent-eeeeeeee-1111-2222-3333-444444444444.jsonl").write_text("{}\n")

    sessions = list_sessions(trace_dir=trace_dir, projects_dir=projects_dir)
    assert "dddddddd-1111-2222-3333-444444444444" in sessions
    # The subagent file has "agent-" prefix so its stem won't match UUID pattern,
    # but even if it somehow did, the subagents/ filter catches it
    assert "eeeeeeee-1111-2222-3333-444444444444" not in sessions


def test_list_sessions_excludes_non_uuid(tmp_path):
    """Non-UUID filenames in projects dir are excluded."""
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    projects_dir = tmp_path / "projects" / "proj"
    projects_dir.mkdir(parents=True)

    # Valid UUID
    (projects_dir / "ffffffff-1111-2222-3333-444444444444.jsonl").write_text("{}\n")
    # Not a UUID
    (projects_dir / "random-notes.jsonl").write_text("{}\n")

    sessions = list_sessions(trace_dir=trace_dir, projects_dir=projects_dir)
    assert sessions == ["ffffffff-1111-2222-3333-444444444444"]


def test_list_sessions_sorted_by_mtime(tmp_path):
    """Sessions are sorted newest first by mtime."""
    import os
    import time

    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    projects_dir = tmp_path / "projects" / "proj"
    projects_dir.mkdir(parents=True)

    older = projects_dir / "11111111-1111-2222-3333-444444444444.jsonl"
    older.write_text("{}\n")
    os.utime(older, (1000, 1000))

    time.sleep(0.01)
    newer = projects_dir / "22222222-1111-2222-3333-444444444444.jsonl"
    newer.write_text("{}\n")
    os.utime(newer, (2000, 2000))

    sessions = list_sessions(trace_dir=trace_dir, projects_dir=projects_dir)
    assert sessions[0] == "22222222-1111-2222-3333-444444444444"
    assert sessions[1] == "11111111-1111-2222-3333-444444444444"


def test_list_sessions_missing_dirs(tmp_path):
    """Both dirs missing returns empty list."""
    sessions = list_sessions(
        trace_dir=tmp_path / "nope",
        projects_dir=tmp_path / "also-nope",
    )
    assert sessions == []


def test_rejects_path_traversal(tmp_path):
    from context_tracker.models import SessionStartEvent
    from context_tracker.storage import append_event

    with pytest.raises(ValueError, match="Invalid session_id"):
        event = SessionStartEvent(session_id="../../../etc/passwd", source="startup", model="test")
        append_event(event, trace_dir=tmp_path)


def test_rejects_path_traversal_read(tmp_path):
    from context_tracker.storage import read_events

    with pytest.raises(ValueError, match="Invalid session_id"):
        read_events("../../../etc/passwd", trace_dir=tmp_path)
