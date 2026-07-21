"""Extended tests for dashboard.py — covers helper functions and more endpoints."""

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from context_tracker.dashboard import (
    _detect_self_corrections,
    _extract_content_blocks_with_images,
    _extract_image_metadata,
    _flatten_entries_to_messages,
    _serve_image_source,
    _validate_session_id,
    _walk_transcript_for_range,
    create_app,
)

# --- Helper function tests ---


class TestDetectSelfCorrections:
    def test_high_confidence_error(self):
        matched, confidence, _ = _detect_self_corrections("I made an error in the previous code")
        assert matched
        assert confidence == "high"

    def test_high_confidence_wrong(self):
        matched, confidence, _ = _detect_self_corrections("that was wrong, let me fix it")
        assert matched
        assert confidence == "high"

    def test_high_confidence_accidentally(self):
        matched, confidence, _ = _detect_self_corrections("I accidentally deleted the file")
        assert matched
        assert confidence == "high"

    def test_high_confidence_fix(self):
        matched, confidence, _ = _detect_self_corrections("let me fix that")
        assert matched
        assert confidence == "high"

    def test_medium_confidence_apologize(self):
        matched, confidence, _ = _detect_self_corrections("I apologize for the confusion")
        assert matched
        assert confidence == "medium"

    def test_medium_confidence_actually(self):
        matched, confidence, _ = _detect_self_corrections("actually, I need to rethink this")
        assert matched
        assert confidence == "medium"

    def test_medium_confidence_forgot(self):
        matched, confidence, _ = _detect_self_corrections("I forgot to add the import")
        assert matched
        assert confidence == "medium"

    def test_medium_confidence_previous_approach(self):
        matched, confidence, _ = _detect_self_corrections("the previous approach didn't work")
        assert matched
        assert confidence == "medium"

    def test_no_match(self):
        matched, confidence, _ = _detect_self_corrections("This looks good, the tests pass")
        assert not matched
        assert confidence == ""


class TestValidateSessionId:
    def test_valid_id(self):
        _validate_session_id("abc-123_def")

    def test_invalid_id_injection(self):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            _validate_session_id("../etc/passwd")
        assert exc_info.value.status_code == 400

    def test_invalid_id_space(self):
        from fastapi import HTTPException

        with pytest.raises(HTTPException):
            _validate_session_id("abc def")


class TestExtractImageMetadata:
    def test_base64_source(self):
        meta = _extract_image_metadata(
            {"type": "base64", "data": "", "media_type": "image/png"},
            "image/png",
        )
        assert meta["media_type"] == "image/png"
        assert meta["width"] > 0
        assert meta["height"] > 0
        assert meta["tokens"] > 0

    def test_url_source_data_uri(self):
        meta = _extract_image_metadata(
            {"type": "url", "url": "data:image/jpeg;base64,"},
            "image/jpeg",
        )
        assert "media_type" in meta
        assert "tokens" in meta

    def test_url_source_https(self):
        meta = _extract_image_metadata(
            {"type": "url", "url": "https://example.com/image.png"},
            "image/png",
        )
        # For http(s) URLs, dimensions are fallback
        assert meta["width"] == 1024
        assert meta["height"] == 1024


class TestServeImageSource:
    def test_base64_source(self):
        result = _serve_image_source({"type": "base64", "media_type": "image/png", "data": "abc123"})
        assert "data_uri" in result
        assert result["data_uri"] == "data:image/png;base64,abc123"

    def test_url_source_data(self):
        result = _serve_image_source({"type": "url", "url": "data:image/png;base64,abc"})
        assert result["data_uri"] == "data:image/png;base64,abc"

    def test_url_source_http(self):
        result = _serve_image_source({"type": "url", "url": "https://example.com/img.png"})
        assert result["url"] == "https://example.com/img.png"

    def test_unknown_source_with_data(self):
        result = _serve_image_source({"type": "unknown", "media_type": "image/png", "data": "xyz"})
        assert "data_uri" in result

    def test_unknown_source_no_data(self):
        result = _serve_image_source({"type": "unknown"})
        assert "error" in result


class TestExtractContentBlocks:
    def test_string_content(self):
        msgs = _extract_content_blocks_with_images("Hello world", "user", "2026-01-01")
        assert len(msgs) == 1
        assert msgs[0]["type"] == "user"
        assert msgs[0]["content"] == "Hello world"

    def test_assistant_string(self):
        msgs = _extract_content_blocks_with_images("I will help", "assistant", "2026-01-01")
        assert len(msgs) == 1
        assert msgs[0]["type"] == "assistant"

    def test_empty_string(self):
        msgs = _extract_content_blocks_with_images("", "user", "2026-01-01")
        assert len(msgs) == 0

    def test_non_list_non_string(self):
        msgs = _extract_content_blocks_with_images(123, "user", "2026-01-01")
        assert len(msgs) == 0

    def test_text_block(self):
        content = [{"type": "text", "text": "Hello"}]
        msgs = _extract_content_blocks_with_images(content, "assistant", "2026-01-01")
        assert len(msgs) == 1
        assert msgs[0]["type"] == "assistant_text"

    def test_user_text_block(self):
        content = [{"type": "text", "text": "Hello"}]
        msgs = _extract_content_blocks_with_images(content, "user", "2026-01-01")
        assert len(msgs) == 1
        assert msgs[0]["type"] == "user_text"

    def test_thinking_block(self):
        content = [{"type": "thinking", "thinking": "Let me think..."}]
        msgs = _extract_content_blocks_with_images(content, "assistant", "2026-01-01")
        assert len(msgs) == 1
        assert msgs[0]["type"] == "thinking"
        assert msgs[0]["role"] == "assistant"

    def test_tool_use_block(self):
        content = [
            {
                "type": "tool_use",
                "name": "Read",
                "id": "tu_1",
                "input": {"file_path": "/src/a.py"},
            }
        ]
        msgs = _extract_content_blocks_with_images(content, "assistant", "2026-01-01")
        assert len(msgs) == 1
        assert msgs[0]["type"] == "tool_use"
        assert msgs[0]["tool_name"] == "Read"
        assert msgs[0]["resource"] == "/src/a.py"

    def test_tool_use_bash_block(self):
        content = [
            {
                "type": "tool_use",
                "name": "Bash",
                "id": "tu_2",
                "input": {"command": "ls -la"},
            }
        ]
        msgs = _extract_content_blocks_with_images(content, "assistant", "2026-01-01")
        assert len(msgs) == 1
        assert msgs[0]["resource"] == "ls -la"

    def test_tool_result_string(self):
        content = [{"type": "tool_result", "content": "file contents", "tool_use_id": "tu_1"}]
        msgs = _extract_content_blocks_with_images(content, "user", "2026-01-01")
        assert len(msgs) == 1
        assert msgs[0]["type"] == "tool_result"

    def test_tool_result_list_with_image(self):
        content = [
            {
                "type": "tool_result",
                "content": [
                    {"type": "text", "text": "partial"},
                    {"type": "image", "source": {"type": "base64", "data": "", "media_type": "image/png"}},
                ],
                "tool_use_id": "tu_1",
            }
        ]
        msgs = _extract_content_blocks_with_images(content, "user", "2026-01-01")
        assert len(msgs) == 1
        assert "images" in msgs[0]
        assert len(msgs[0]["images"]) == 1

    def test_tool_result_with_error(self):
        content = [{"type": "tool_result", "content": "error msg", "tool_use_id": "tu_1", "is_error": True}]
        msgs = _extract_content_blocks_with_images(content, "user", "2026-01-01")
        assert msgs[0]["is_error"] is True

    def test_tool_result_non_string_content(self):
        content = [{"type": "tool_result", "content": 12345, "tool_use_id": "tu_1"}]
        msgs = _extract_content_blocks_with_images(content, "user", "2026-01-01")
        assert len(msgs) == 1
        assert msgs[0]["content"][:5] == "12345"

    def test_image_block(self):
        content = [
            {"type": "image", "source": {"type": "base64", "data": "", "media_type": "image/png"}},
        ]
        msgs = _extract_content_blocks_with_images(content, "user", "2026-01-01")
        assert len(msgs) == 1
        assert msgs[0]["type"] == "image"
        assert "images" in msgs[0]

    def test_non_dict_in_list(self):
        content = ["just a string", {"type": "text", "text": "ok"}]
        msgs = _extract_content_blocks_with_images(content, "user", "2026-01-01")
        assert len(msgs) == 1  # The string is skipped, only dict processed

    def test_truncation_flag(self):
        long_text = "x" * 10000
        content = [{"type": "text", "text": long_text}]
        msgs = _extract_content_blocks_with_images(content, "assistant", "2026-01-01")
        assert msgs[0]["is_truncated"] is True
        assert msgs[0]["size_chars"] == 10000
        assert len(msgs[0]["content"]) == 8000


class TestWalkTranscriptForRange:
    def _write_transcript(self, tmp_path, entries):
        path = tmp_path / "test.jsonl"
        with open(path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        return path

    def test_basic_walk(self, tmp_path):
        entries = [
            {"type": "user", "message": {"content": "hi"}},
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "model": "claude-opus-4-6",
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": "hello"}],
                    "usage": {"input_tokens": 1000, "output_tokens": 50},
                },
            },
        ]
        path = self._write_transcript(tmp_path, entries)
        result = _walk_transcript_for_range(path, 0, 0)
        assert len(result) == 2  # user + assistant

    def test_skip_synthetic(self, tmp_path):
        entries = [
            {
                "type": "assistant",
                "message": {
                    "model": "synthetic",
                    "stop_reason": "end_turn",
                    "content": [],
                    "usage": {"output_tokens": 10},
                },
            },
            {
                "type": "assistant",
                "message": {
                    "model": "claude-opus-4-6",
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": "real"}],
                    "usage": {"input_tokens": 100, "output_tokens": 50},
                },
            },
        ]
        path = self._write_transcript(tmp_path, entries)
        result = _walk_transcript_for_range(path, 0, 0)
        assert len(result) == 1

    def test_skip_file_history(self, tmp_path):
        entries = [
            {"type": "file-history-snapshot"},
            {"type": "last-prompt"},
            {"type": "pr-link"},
            {"type": "queue-operation"},
            {
                "type": "assistant",
                "message": {
                    "model": "claude-opus-4-6",
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 100, "output_tokens": 50},
                },
            },
        ]
        path = self._write_transcript(tmp_path, entries)
        result = _walk_transcript_for_range(path, 0, 0)
        assert len(result) == 1

    def test_range_filtering(self, tmp_path):
        entries = []
        for i in range(3):
            entries.append({"type": "user", "message": {"content": f"prompt {i}"}})
            entries.append(
                {
                    "type": "assistant",
                    "message": {
                        "model": "claude-opus-4-6",
                        "stop_reason": "end_turn",
                        "content": [{"type": "text", "text": f"response {i}"}],
                        "usage": {"input_tokens": 100, "output_tokens": 50},
                    },
                }
            )
        path = self._write_transcript(tmp_path, entries)
        # Get only call 1
        result = _walk_transcript_for_range(path, 1, 1)
        assert len(result) == 2  # user + assistant for call 1

    def test_empty_lines(self, tmp_path):
        path = tmp_path / "test.jsonl"
        with open(path, "w") as f:
            f.write("\n\n")
            f.write(
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "model": "claude-opus-4-6",
                            "stop_reason": "end_turn",
                            "content": [{"type": "text", "text": "ok"}],
                            "usage": {"input_tokens": 100, "output_tokens": 50},
                        },
                    }
                )
                + "\n"
            )
        result = _walk_transcript_for_range(path, 0, 0)
        assert len(result) == 1


class TestFlattenEntries:
    def test_flatten(self):
        entries = [
            {
                "type": "user",
                "message": {"content": "Hello"},
                "timestamp": "2026-01-01T00:00:00Z",
            },
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Hi there"}]},
                "timestamp": "2026-01-01T00:00:01Z",
            },
        ]
        msgs = _flatten_entries_to_messages(entries)
        assert len(msgs) == 2


# --- Full endpoint integration tests ---


@pytest.fixture
def rich_client(tmp_path):
    """Client with a transcript that has tool use, errors, and more."""
    transcript_dir = tmp_path / "transcripts"
    transcript_dir.mkdir(parents=True)
    trace_dir = tmp_path / "traces"
    static_dir = tmp_path / "static"
    static_dir.mkdir(parents=True)

    session_id = "rich-session-001"
    messages = [
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Fix the bug in server.py and make it work properly"},
                ],
            },
            "timestamp": "2026-06-01T10:00:00Z",
            "uuid": "u1",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I will read the file first."},
                    {"type": "tool_use", "id": "tu_1", "name": "Read", "input": {"file_path": "/src/server.py"}},
                ],
                "model": "claude-opus-4-6",
                "usage": {
                    "input_tokens": 5000,
                    "output_tokens": 200,
                    "cache_read_input_tokens": 3000,
                    "cache_creation_input_tokens": 500,
                },
                "stop_reason": "tool_use",
            },
            "timestamp": "2026-06-01T10:00:05Z",
            "uuid": "a1",
        },
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_1",
                        "content": "file content " * 100,
                    }
                ],
            },
            "timestamp": "2026-06-01T10:00:06Z",
            "uuid": "u2",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I see the issue. Let me edit the file."},
                    {
                        "type": "tool_use",
                        "id": "tu_2",
                        "name": "Edit",
                        "input": {"file_path": "/src/server.py", "old_string": "old", "new_string": "new"},
                    },
                ],
                "model": "claude-opus-4-6",
                "usage": {
                    "input_tokens": 7000,
                    "output_tokens": 300,
                    "cache_read_input_tokens": 5000,
                    "cache_creation_input_tokens": 400,
                },
                "stop_reason": "tool_use",
            },
            "timestamp": "2026-06-01T10:00:10Z",
            "uuid": "a2",
        },
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_2",
                        "content": "OK",
                    }
                ],
            },
            "timestamp": "2026-06-01T10:00:11Z",
            "uuid": "u3",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "The fix is applied."}],
                "model": "claude-opus-4-6",
                "usage": {
                    "input_tokens": 8000,
                    "output_tokens": 100,
                    "cache_read_input_tokens": 6000,
                    "cache_creation_input_tokens": 300,
                },
                "stop_reason": "end_turn",
            },
            "timestamp": "2026-06-01T10:00:15Z",
            "uuid": "a3",
        },
    ]

    transcript_path = transcript_dir / f"{session_id}.jsonl"
    with open(transcript_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")

    app = create_app(
        trace_dir=trace_dir,
        transcript_dir=transcript_dir,
        static_dir=static_dir,
        db_path=tmp_path / "test.db",
    )
    return TestClient(app), session_id


def test_get_session_health(rich_client):
    client, session_id = rich_client
    resp = client.get(f"/api/session/{session_id}/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "health_score" in data
    assert "urgency_score" in data
    assert "classification" in data
    assert "signals" in data
    assert "recommendations" in data


def test_get_session_health_not_found(rich_client):
    client, _ = rich_client
    resp = client.get("/api/session/nonexistent/health")
    assert resp.status_code == 404


def test_get_session_dead_weight(rich_client):
    client, session_id = rich_client
    resp = client.get(f"/api/session/{session_id}/dead_weight")
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert "top_blocks" in data
    assert "per_turn" in data
    assert "peak_dead_weight_pct" in data["summary"]


def test_get_session_dead_weight_not_found(rich_client):
    client, _ = rich_client
    resp = client.get("/api/session/nonexistent/dead_weight")
    assert resp.status_code == 404


def test_get_session_errors(rich_client):
    client, session_id = rich_client
    resp = client.get(f"/api/session/{session_id}/errors")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_errors" in data
    assert "total_tool_results" in data
    assert "error_rate" in data
    assert "per_turn" in data
    assert "clusters" in data
    assert "retry_patterns" in data
    assert "self_corrections" in data
    assert "recommendations" in data


def test_get_session_errors_not_found(rich_client):
    client, _ = rich_client
    resp = client.get("/api/session/nonexistent/errors")
    assert resp.status_code == 404


def test_get_session_nudges(rich_client):
    client, session_id = rich_client
    resp = client.get(f"/api/session/{session_id}/nudges")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == session_id
    assert "nudges" in data


def test_get_call_content(rich_client):
    client, session_id = rich_client
    resp = client.get(f"/api/session/{session_id}/call/0/content")
    assert resp.status_code == 200
    data = resp.json()
    assert data["call_index"] == 0
    assert "messages" in data
    assert "usage" in data


def test_get_call_content_not_found(rich_client):
    client, session_id = rich_client
    resp = client.get(f"/api/session/{session_id}/call/999/content")
    assert resp.status_code == 404


def test_get_call_content_invalid_session(rich_client):
    client, _ = rich_client
    resp = client.get("/api/session/nonexistent/call/0/content")
    assert resp.status_code == 404


def test_get_conv_turn_content(rich_client):
    client, session_id = rich_client
    resp = client.get(f"/api/session/{session_id}/conv_turn/1/content")
    assert resp.status_code == 200
    data = resp.json()
    assert data["conv_turn"] == 1
    assert "messages" in data
    assert "usage" in data
    assert "first_call" in data
    assert "last_call" in data


def test_get_conv_turn_content_not_found(rich_client):
    client, session_id = rich_client
    resp = client.get(f"/api/session/{session_id}/conv_turn/999/content")
    assert resp.status_code == 404


def test_get_conv_turn_content_no_transcript(rich_client):
    client, _ = rich_client
    resp = client.get("/api/session/nonexistent/conv_turn/1/content")
    assert resp.status_code == 404


def test_get_conv_turn_image_not_found(rich_client):
    client, session_id = rich_client
    resp = client.get(f"/api/session/{session_id}/conv_turn/1/image/0/0")
    # Should return 404 since there are no images in our test transcript
    assert resp.status_code == 404


def test_get_tool_intelligence(rich_client):
    client, session_id = rich_client
    resp = client.get(f"/api/session/{session_id}/tool-intelligence")
    assert resp.status_code == 200
    data = resp.json()
    assert "composition" in data
    assert "mcp_servers" in data
    assert "skills" in data
    assert "regular_tools" in data
    assert "agents" in data


def test_get_tool_intelligence_not_found(rich_client):
    client, _ = rich_client
    resp = client.get("/api/session/nonexistent/tool-intelligence")
    assert resp.status_code == 404


def test_serve_dashboard_fallback(rich_client):
    client, _ = rich_client
    resp = client.get("/")
    assert resp.status_code == 200


def test_serve_sessions_page(rich_client):
    client, _ = rich_client
    resp = client.get("/sessions")
    assert resp.status_code == 200


def test_serve_workflows_page(rich_client):
    client, _ = rich_client
    resp = client.get("/workflows")
    assert resp.status_code == 200


def test_blocks_json_not_found(rich_client):
    client, _ = rich_client
    resp = client.get("/blocks.json")
    assert resp.status_code == 404


def test_churn_json_not_found(rich_client):
    client, _ = rich_client
    resp = client.get("/churn.json")
    assert resp.status_code == 404


def test_meta_json_not_found(rich_client):
    client, _ = rich_client
    resp = client.get("/meta.json")
    assert resp.status_code == 404


def test_turn_map_json_not_found(rich_client):
    client, _ = rich_client
    resp = client.get("/turn_map.json")
    assert resp.status_code == 404


def test_static_json_files_when_exist(tmp_path):
    """Test that static JSON endpoints serve files when they exist."""
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "blocks.json").write_text("[]")
    (static_dir / "churn.json").write_text("[]")
    (static_dir / "meta.json").write_text("{}")
    (static_dir / "turn_map.json").write_text("[]")
    (static_dir / "dashboard-v3.html").write_text("<html>v3</html>")
    (static_dir / "sessions.html").write_text("<html>sessions</html>")
    (static_dir / "workflows.html").write_text("<html>workflows</html>")

    app = create_app(
        trace_dir=tmp_path / "traces",
        transcript_dir=tmp_path / "transcripts",
        static_dir=static_dir,
    )
    client = TestClient(app)

    assert client.get("/blocks.json").status_code == 200
    assert client.get("/churn.json").status_code == 200
    assert client.get("/meta.json").status_code == 200
    assert client.get("/turn_map.json").status_code == 200
    assert client.get("/").status_code == 200
    assert client.get("/sessions").status_code == 200
    assert client.get("/workflows").status_code == 200


def test_data_dir_serves_build_artifacts(tmp_path):
    """Artifacts in data_dir are served even when static_dir has stale copies.

    ccscope writes blocks.json etc. to a user cache dir (site-packages can be
    read-only), and the dashboard is pointed at it via data_dir.
    """
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    data_dir = tmp_path / "cache"
    data_dir.mkdir()
    (data_dir / "blocks.json").write_text('[{"id": "b1"}]')
    (data_dir / "churn.json").write_text("[]")
    (data_dir / "meta.json").write_text('{"session_id": "s1"}')
    (data_dir / "turn_map.json").write_text("[]")
    # static_dir holds a stale legacy copy — data_dir must win.
    (static_dir / "blocks.json").write_text('[{"id": "stale"}]')

    app = create_app(
        trace_dir=tmp_path / "traces",
        transcript_dir=tmp_path / "transcripts",
        static_dir=static_dir,
        data_dir=data_dir,
    )
    client = TestClient(app)

    assert client.get("/blocks.json").json() == [{"id": "b1"}]
    assert client.get("/churn.json").status_code == 200
    assert client.get("/meta.json").json() == {"session_id": "s1"}
    assert client.get("/turn_map.json").status_code == 200


def test_data_dir_falls_back_to_static_dir(tmp_path):
    """Legacy in-repo builds (artifacts next to the HTML) still work."""
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "blocks.json").write_text('[{"id": "legacy"}]')
    data_dir = tmp_path / "cache"  # nonexistent — nothing built there yet

    app = create_app(
        trace_dir=tmp_path / "traces",
        transcript_dir=tmp_path / "transcripts",
        static_dir=static_dir,
        data_dir=data_dir,
    )
    client = TestClient(app)

    assert client.get("/blocks.json").json() == [{"id": "legacy"}]
    assert client.get("/churn.json").status_code == 404


def test_serve_dashboard_v2_fallback(tmp_path):
    """Serve v2 dashboard when v3 doesn't exist."""
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "context-scope.html").write_text("<html>v2</html>")

    app = create_app(
        trace_dir=tmp_path / "traces",
        transcript_dir=tmp_path / "transcripts",
        static_dir=static_dir,
    )
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "v2" in resp.text


def test_dashboard_main():
    """Test dashboard main() entry point."""
    with (
        patch("sys.argv", ["dashboard", "--host", "127.0.0.1", "--port", "9876"]),
        patch("uvicorn.run") as mock_run,
    ):
        from context_tracker.dashboard import main

        main()
        mock_run.assert_called_once()


def test_dashboard_main_non_localhost():
    """Test dashboard main() warns about non-localhost binding."""
    all_ifaces = "0.0.0.0"  # noqa: S104
    with (
        patch("sys.argv", ["dashboard", "--host", all_ifaces, "--port", "9876"]),
        patch("uvicorn.run"),
        patch("builtins.print") as mock_print,
    ):
        from context_tracker.dashboard import main

        main()
        # Should have printed a warning about non-localhost
        warning_printed = any("WARNING" in str(call) for call in mock_print.call_args_list)
        assert warning_printed


# --- Error analysis with richer transcript ---


@pytest.fixture
def error_client(tmp_path):
    """Client with transcript that has tool errors and self-corrections."""
    transcript_dir = tmp_path / "transcripts"
    transcript_dir.mkdir(parents=True)
    trace_dir = tmp_path / "traces"
    static_dir = tmp_path / "static"
    static_dir.mkdir()

    session_id = "error-session-002"
    messages = [
        # Turn 1: user prompt
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "Read and fix the file at /src/broken.py"}],
            },
            "timestamp": "2026-06-01T10:00:00Z",
        },
        # API call 0: tool use + error result
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tu_1", "name": "Read", "input": {"file_path": "/src/broken.py"}},
                ],
                "model": "claude-opus-4-6",
                "usage": {
                    "input_tokens": 2000,
                    "output_tokens": 100,
                    "cache_read_input_tokens": 1000,
                    "cache_creation_input_tokens": 200,
                },
                "stop_reason": "tool_use",
            },
            "timestamp": "2026-06-01T10:00:01Z",
        },
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_1",
                        "content": "File not found",
                        "is_error": True,
                    },
                ],
            },
            "timestamp": "2026-06-01T10:00:02Z",
        },
        # API call 1: self-correction
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I made an error - the file path was wrong."},
                    {
                        "type": "tool_use",
                        "id": "tu_2",
                        "name": "Read",
                        "input": {"file_path": "/src/broken_fixed.py"},
                    },
                ],
                "model": "claude-opus-4-6",
                "usage": {
                    "input_tokens": 3000,
                    "output_tokens": 200,
                    "cache_read_input_tokens": 2000,
                    "cache_creation_input_tokens": 300,
                },
                "stop_reason": "tool_use",
            },
            "timestamp": "2026-06-01T10:00:03Z",
        },
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_2",
                        "content": "File not found",
                        "is_error": True,
                    },
                ],
            },
            "timestamp": "2026-06-01T10:00:04Z",
        },
        # API call 2: final success
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Found it."},
                ],
                "model": "claude-opus-4-6",
                "usage": {
                    "input_tokens": 4000,
                    "output_tokens": 50,
                    "cache_read_input_tokens": 3000,
                    "cache_creation_input_tokens": 200,
                },
                "stop_reason": "end_turn",
            },
            "timestamp": "2026-06-01T10:00:05Z",
        },
    ]

    transcript_path = transcript_dir / f"{session_id}.jsonl"
    with open(transcript_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")

    app = create_app(
        trace_dir=trace_dir,
        transcript_dir=transcript_dir,
        static_dir=static_dir,
        db_path=tmp_path / "test.db",
    )
    return TestClient(app), session_id


def test_errors_with_tool_failures(error_client):
    client, session_id = error_client
    resp = client.get(f"/api/session/{session_id}/errors")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_errors"] >= 1
    assert data["error_rate"] > 0
    assert "self_corrections" in data


def test_call_content_with_errors(error_client):
    client, session_id = error_client
    resp = client.get(f"/api/session/{session_id}/call/1/content")
    assert resp.status_code == 200
    data = resp.json()
    # Check self-correction detection in messages
    for msg in data["messages"]:
        assert "is_self_correction" in msg
        assert "is_retry" in msg


def test_session_trends(rich_client):
    client, _ = rich_client
    resp = client.get("/api/sessions/trends")
    assert resp.status_code == 200
    data = resp.json()
    assert "session_count" in data
    assert "total_cost" in data


def test_session_data_not_found(rich_client):
    client, _ = rich_client
    resp = client.get("/api/session/nonexistent/data")
    assert resp.status_code == 404


def test_subagents_endpoint(rich_client):
    client, session_id = rich_client
    resp = client.get(f"/api/session/{session_id}/subagents")
    assert resp.status_code == 200
    data = resp.json()
    assert "count" in data
    assert "subagents" in data


def test_workflows_endpoint(rich_client):
    client, session_id = rich_client
    resp = client.get(f"/api/session/{session_id}/workflows")
    assert resp.status_code == 200
    data = resp.json()
    assert "count" in data
    assert "workflows" in data
