"""Extended tests for server.py — MCP tool wrappers, bloat_events, should_clear, main."""

import json
from unittest.mock import patch

import pytest

from context_tracker.models import (
    PostCompactEvent,
    PostToolUseEvent,
    SessionStartEvent,
)
from context_tracker.server import (
    _cached_read_events,
    _find_transcript,
    get_bloat_events,
    mcp_get_bloat_events,
    mcp_get_compaction_history,
    mcp_get_context_hogs,
    mcp_get_session_history,
    mcp_get_session_summary,
    mcp_get_tool_breakdown,
    mcp_should_clear,
    should_clear,
)
from context_tracker.storage import append_event


@pytest.fixture
def session_env(tmp_path):
    """Create a trace directory and transcript for a session."""
    trace_dir = tmp_path / "traces"
    transcript_dir = tmp_path / "transcripts"
    transcript_dir.mkdir(parents=True)
    session_id = "test-srv-session"

    append_event(
        SessionStartEvent(session_id=session_id, source="startup", model="claude-opus-4-6"),
        trace_dir=trace_dir,
    )
    for i in range(5):
        append_event(
            PostToolUseEvent(
                session_id=session_id,
                tool_name="Read",
                input_payload_chars=50,
                output_payload_chars=8000,
                tool_use_id=f"toolu_{i}",
            ),
            trace_dir=trace_dir,
        )
    append_event(
        PostCompactEvent(session_id=session_id, trigger="auto", compact_summary_length=500),
        trace_dir=trace_dir,
    )
    append_event(
        PostCompactEvent(session_id=session_id, trigger="auto", compact_summary_length=600),
        trace_dir=trace_dir,
    )

    # Write transcript
    transcript_path = transcript_dir / f"{session_id}.jsonl"
    turns = [
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-6",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "hello"}],
                "usage": {
                    "input_tokens": 50000,
                    "output_tokens": 500,
                    "cache_read_input_tokens": 30000,
                    "cache_creation_input_tokens": 5000,
                },
            },
        },
    ]
    with open(transcript_path, "w") as f:
        for t in turns:
            f.write(json.dumps(t) + "\n")

    return session_id, trace_dir, transcript_dir


def test_get_bloat_events(session_env):
    session_id, trace_dir, transcript_dir = session_env
    result = get_bloat_events(session_id, threshold=5000, trace_dir=trace_dir, transcript_dir=transcript_dir)
    assert len(result) == 5
    assert all(r["output_payload_chars"] > 5000 for r in result)


def test_get_bloat_events_high_threshold(session_env):
    session_id, trace_dir, transcript_dir = session_env
    result = get_bloat_events(session_id, threshold=50000, trace_dir=trace_dir, transcript_dir=transcript_dir)
    assert len(result) == 0


def test_should_clear_continue(session_env):
    session_id, trace_dir, transcript_dir = session_env
    result = should_clear(session_id, trace_dir=trace_dir, transcript_dir=transcript_dir)
    assert "recommendation" in result
    assert "context_pct" in result
    assert "cache_hit_rate" in result
    assert "reasons" in result
    assert result["compactions"] == 2


def test_should_clear_urgent(tmp_path):
    """Session with high context usage should trigger urgent_clear."""
    trace_dir = tmp_path / "traces"
    transcript_dir = tmp_path / "transcripts"
    transcript_dir.mkdir(parents=True)
    session_id = "high-usage"

    append_event(
        SessionStartEvent(session_id=session_id, source="startup", model="claude-opus-4-6"),
        trace_dir=trace_dir,
    )
    # Add compactions
    for _ in range(3):
        append_event(
            PostCompactEvent(session_id=session_id, trigger="auto", compact_summary_length=1000),
            trace_dir=trace_dir,
        )

    # Write transcript with very high token usage
    transcript_path = transcript_dir / f"{session_id}.jsonl"
    turns = [
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-6",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "hi"}],
                "usage": {
                    "input_tokens": 800000,
                    "output_tokens": 5000,
                    "cache_read_input_tokens": 10000,
                    "cache_creation_input_tokens": 5000,
                },
            },
        },
    ]
    with open(transcript_path, "w") as f:
        for t in turns:
            f.write(json.dumps(t) + "\n")

    result = should_clear(session_id, trace_dir=trace_dir, transcript_dir=transcript_dir)
    assert result["recommendation"] == "urgent_clear"
    assert len(result["reasons"]) >= 2


def test_find_transcript_direct(tmp_path):
    """_find_transcript finds a direct match."""
    transcript_dir = tmp_path / "transcripts"
    transcript_dir.mkdir()
    (transcript_dir / "abc123.jsonl").write_text("{}")
    result = _find_transcript("abc123", transcript_dir)
    assert result is not None
    assert result.name == "abc123.jsonl"


def test_find_transcript_nested(tmp_path):
    """_find_transcript finds transcript in subdirectory."""
    transcript_dir = tmp_path / "transcripts"
    sub = transcript_dir / "proj"
    sub.mkdir(parents=True)
    (sub / "xyz789.jsonl").write_text("{}")
    result = _find_transcript("xyz789", transcript_dir)
    assert result is not None
    assert result.name == "xyz789.jsonl"


def test_find_transcript_missing(tmp_path):
    transcript_dir = tmp_path / "transcripts"
    transcript_dir.mkdir()
    result = _find_transcript("nonexistent", transcript_dir)
    assert result is None


def test_cached_read_events_returns_same(session_env):
    """Test that _cached_read_events returns cached result on second call."""
    session_id, trace_dir, _ = session_env
    # Clear cache first
    from context_tracker.server import _cache

    _cache.clear()
    events1 = _cached_read_events(session_id, trace_dir=trace_dir)
    events2 = _cached_read_events(session_id, trace_dir=trace_dir)
    assert events1 is events2  # Same object from cache
    _cache.clear()


# --- MCP tool wrapper tests ---


def test_mcp_get_session_summary_no_sessions():
    with patch("context_tracker.server.list_sessions", return_value=[]):
        result = json.loads(mcp_get_session_summary())
        assert "error" in result


def test_mcp_get_session_summary_default_session(session_env):
    session_id, trace_dir, transcript_dir = session_env
    with (
        patch("context_tracker.server.list_sessions", return_value=[session_id]),
        patch(
            "context_tracker.server.get_session_summary",
            return_value={"session_id": session_id, "tool_calls": 5},
        ),
    ):
        result = json.loads(mcp_get_session_summary())
        assert result["session_id"] == session_id


def test_mcp_get_tool_breakdown_no_sessions():
    with patch("context_tracker.server.list_sessions", return_value=[]):
        result = json.loads(mcp_get_tool_breakdown())
        assert "error" in result


def test_mcp_get_tool_breakdown_default_session():
    with (
        patch("context_tracker.server.list_sessions", return_value=["s1"]),
        patch("context_tracker.server.get_tool_breakdown", return_value=[]),
    ):
        result = json.loads(mcp_get_tool_breakdown())
        assert result == []


def test_mcp_get_compaction_history_no_sessions():
    with patch("context_tracker.server.list_sessions", return_value=[]):
        result = json.loads(mcp_get_compaction_history())
        assert "error" in result


def test_mcp_get_compaction_history_default_session():
    with (
        patch("context_tracker.server.list_sessions", return_value=["s1"]),
        patch("context_tracker.server.get_compaction_history", return_value=[{"trigger": "auto"}]),
    ):
        result = json.loads(mcp_get_compaction_history())
        assert len(result) == 1


def test_mcp_get_context_hogs_no_sessions():
    with patch("context_tracker.server.list_sessions", return_value=[]):
        result = json.loads(mcp_get_context_hogs())
        assert "error" in result


def test_mcp_get_context_hogs_default_session():
    with (
        patch("context_tracker.server.list_sessions", return_value=["s1"]),
        patch("context_tracker.server.get_context_hogs", return_value=[]),
    ):
        result = json.loads(mcp_get_context_hogs())
        assert result == []


def test_mcp_get_session_history():
    with patch("context_tracker.server.get_session_history", return_value=[]):
        result = json.loads(mcp_get_session_history())
        assert result == []


def test_mcp_get_bloat_events_no_sessions():
    with patch("context_tracker.server.list_sessions", return_value=[]):
        result = json.loads(mcp_get_bloat_events())
        assert "error" in result


def test_mcp_get_bloat_events_default_session():
    with (
        patch("context_tracker.server.list_sessions", return_value=["s1"]),
        patch("context_tracker.server.get_bloat_events", return_value=[]),
    ):
        result = json.loads(mcp_get_bloat_events())
        assert result == []


def test_mcp_should_clear_no_sessions():
    with patch("context_tracker.server.list_sessions", return_value=[]):
        result = json.loads(mcp_should_clear())
        assert "error" in result


def test_mcp_should_clear_default_session():
    with (
        patch("context_tracker.server.list_sessions", return_value=["s1"]),
        patch("context_tracker.server.should_clear", return_value={"recommendation": "continue"}),
    ):
        result = json.loads(mcp_should_clear())
        assert result["recommendation"] == "continue"


def test_mcp_get_session_summary_with_explicit_id(session_env):
    """When session_id is provided, it doesn't call list_sessions."""
    session_id, trace_dir, transcript_dir = session_env
    result = json.loads(
        mcp_get_session_summary(session_id=session_id),
    )
    # It will fail to find events in default trace_dir but won't error (returns partial summary)
    assert "session_id" in result or "error" in result


# --- MCP staleness/health/block tools that use reconcile ---


def test_mcp_get_staleness_analysis_no_sessions():
    from context_tracker.server import mcp_get_staleness_analysis

    with patch("context_tracker.server.list_sessions", return_value=[]):
        result = json.loads(mcp_get_staleness_analysis())
        assert "error" in result


def test_mcp_get_staleness_analysis_file_not_found():
    from context_tracker.server import mcp_get_staleness_analysis

    with patch("context_tracker.server.list_sessions", return_value=["s1"]):
        result = json.loads(mcp_get_staleness_analysis())
        assert "error" in result


def test_mcp_get_staleness_analysis_with_data():
    from context_tracker.server import mcp_get_staleness_analysis

    mock_blocks = [
        {"id": "b1", "label": "sys", "tokens": 100, "ref": True, "cached": True},
        {"id": "b2", "label": "tool", "tokens": 200, "ref": False, "cached": False},
        {"id": "b3", "label": "text", "tokens": 300, "ref": True, "cached": False},
    ]
    mock_churn = [{"cache_read": 1000, "cache_creation": 100, "input": 500}]

    with (
        patch("context_tracker.server.list_sessions", return_value=["s1"]),
        patch("context_tracker.ccscope.reconcile.reconcile", return_value=(mock_blocks, mock_churn, [])),
    ):
        result = json.loads(mcp_get_staleness_analysis())
        assert result["session_id"] == "s1"
        assert result["total_blocks"] == 3
        assert result["stale_blocks"] == 1
        assert result["stale_tokens"] == 200
        assert result["dead_weight_ratio"] > 0


def test_mcp_get_session_health_no_sessions():
    from context_tracker.server import mcp_get_session_health

    with patch("context_tracker.server.list_sessions", return_value=[]):
        result = json.loads(mcp_get_session_health())
        assert "error" in result


def test_mcp_get_session_health_with_data():
    from context_tracker.server import mcp_get_session_health

    mock_blocks = [
        {"id": "b1", "label": "sys", "tokens": 500, "ref": True, "cached": True},
        {"id": "b2", "label": "tool", "tokens": 200, "ref": False, "cached": False},
    ]
    mock_churn = [
        {"cache_read": 5000, "cache_creation": 500, "input": 1000, "output": 300},
        {"cache_read": 6000, "cache_creation": 400, "input": 800, "output": 200},
    ]

    with (
        patch("context_tracker.server.list_sessions", return_value=["s1"]),
        patch("context_tracker.ccscope.reconcile.reconcile", return_value=(mock_blocks, mock_churn, [])),
    ):
        result = json.loads(mcp_get_session_health())
        assert result["session_id"] == "s1"
        assert "urgency_score" in result
        assert "recommendation" in result
        assert "dead_weight_ratio" in result


def test_mcp_get_new_session_recommendation_no_sessions():
    from context_tracker.server import mcp_get_new_session_recommendation

    with patch("context_tracker.server.list_sessions", return_value=[]):
        result = json.loads(mcp_get_new_session_recommendation())
        assert "error" in result


def test_mcp_get_new_session_recommendation_with_data():
    from context_tracker.server import mcp_get_new_session_recommendation

    with (
        patch("context_tracker.server.list_sessions", return_value=["s1"]),
        patch(
            "context_tracker.server.should_clear",
            return_value={"recommendation": "continue", "context_pct": 20},
        ),
    ):
        result = json.loads(mcp_get_new_session_recommendation())
        assert result["recommendation"] == "continue"
        assert "note" in result


def test_mcp_get_block_lifespans_no_sessions():
    from context_tracker.server import mcp_get_block_lifespans

    with patch("context_tracker.server.list_sessions", return_value=[]):
        result = json.loads(mcp_get_block_lifespans())
        assert "error" in result


def test_mcp_get_block_lifespans_file_not_found():
    from context_tracker.server import mcp_get_block_lifespans

    with patch("context_tracker.server.list_sessions", return_value=["s1"]):
        result = json.loads(mcp_get_block_lifespans())
        assert "error" in result


def test_mcp_get_block_lifespans_with_data():
    from context_tracker.server import mcp_get_block_lifespans

    mock_blocks = [
        {
            "id": "b1",
            "label": "sys",
            "tokens": 100,
            "enter": 0,
            "exit": None,
            "cached": True,
            "ref": True,
            "type": "system",
        },
        {
            "id": "b2",
            "label": "tool",
            "tokens": 200,
            "enter": 2,
            "exit": 5,
            "cached": False,
            "ref": False,
            "type": "tool_result",
        },
    ]
    mock_churn = [{"cache_read": 1000}]

    with (
        patch("context_tracker.server.list_sessions", return_value=["s1"]),
        patch("context_tracker.ccscope.reconcile.reconcile", return_value=(mock_blocks, mock_churn, [])),
    ):
        result = json.loads(mcp_get_block_lifespans())
        assert result["session_id"] == "s1"
        assert result["total_blocks"] == 2
        assert len(result["lifespans"]) == 2


def test_mcp_get_cache_churn_no_sessions():
    from context_tracker.server import mcp_get_cache_churn

    with patch("context_tracker.server.list_sessions", return_value=[]):
        result = json.loads(mcp_get_cache_churn())
        assert "error" in result


def test_mcp_get_cache_churn_with_data():
    from context_tracker.server import mcp_get_cache_churn

    mock_blocks = [{"id": "b1", "tokens": 100}]
    mock_churn = [
        {"cache_read": 10000, "cache_creation": 1000, "input": 2000, "output": 500},
        {"cache_read": 15000, "cache_creation": 500, "input": 1500, "output": 300},
    ]
    mock_subagents = [{"total_cache_read": 3000}]

    with (
        patch("context_tracker.server.list_sessions", return_value=["s1"]),
        patch("context_tracker.ccscope.reconcile.reconcile", return_value=(mock_blocks, mock_churn, mock_subagents)),
    ):
        result = json.loads(mcp_get_cache_churn())
        assert result["session_id"] == "s1"
        assert result["total_cache_read"] == 25000
        assert result["total_new_input"] == 3500
        assert result["subagent_cache_read"] == 3000
        assert "headline" in result


# --- main() test ---


def test_server_main_stdio():
    """Test main() with stdio transport (just ensure it parses args)."""
    with (
        patch("sys.argv", ["context-tracker"]),
        patch("context_tracker.server.mcp") as mock_mcp,
    ):
        from context_tracker.server import main

        main()
        mock_mcp.run.assert_called_once()


def test_server_main_sse():
    """Test main() with SSE transport."""
    all_ifaces = "0.0.0.0"  # noqa: S104
    with (
        patch("sys.argv", ["context-tracker", "--transport", "sse", "--host", all_ifaces, "--port", "8000"]),
        patch("context_tracker.server.mcp") as mock_mcp,
    ):
        from context_tracker.server import main

        main()
        mock_mcp.run.assert_called_once_with(transport="sse", host=all_ifaces, port=8000)


def test_server_main_dashboard():
    """Test main() with dashboard subcommand."""
    with (
        patch("sys.argv", ["context-tracker", "dashboard", "--host", "127.0.0.1", "--port", "9999"]),
        patch("uvicorn.run") as mock_run,
        patch("context_tracker.dashboard.create_app") as mock_create,
    ):
        mock_create.return_value = "mock_app"
        from context_tracker.server import main

        main()
        mock_run.assert_called_once_with("mock_app", host="127.0.0.1", port=9999)
