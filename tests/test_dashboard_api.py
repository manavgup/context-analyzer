"""Tests for dashboard REST API endpoints."""


import json

import pytest
from fastapi.testclient import TestClient

from context_tracker.dashboard import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(
        trace_dir=tmp_path / "traces",
        transcript_dir=tmp_path / "transcripts",
        static_dir=tmp_path / "static",
    )
    return TestClient(app)


@pytest.fixture
def client_with_transcript(tmp_path):
    """Client with a minimal transcript file to test turn/block endpoints."""
    transcript_dir = tmp_path / "transcripts"
    transcript_dir.mkdir(parents=True)

    # Write a minimal transcript JSONL
    session_id = "test-session-123"
    messages = [
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "Fix the bug"}],
            },
            "timestamp": "2026-06-01T10:00:00Z",
            "uuid": "u1",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "I will fix it."}],
                "model": "claude-opus-4-6",
                "usage": {
                    "input_tokens": 1000,
                    "output_tokens": 50,
                    "cache_read_input_tokens": 500,
                    "cache_creation_input_tokens": 200,
                },
                "stop_reason": "end_turn",
            },
            "timestamp": "2026-06-01T10:00:05Z",
            "uuid": "a1",
        },
    ]

    transcript_path = transcript_dir / f"{session_id}.jsonl"
    with open(transcript_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")

    app = create_app(
        trace_dir=tmp_path / "traces",
        transcript_dir=transcript_dir,
        static_dir=tmp_path / "static",
    )
    return TestClient(app), session_id


def test_get_sessions_empty(client):
    resp = client.get("/api/sessions")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_session_not_found(client):
    resp = client.get("/api/session/nonexistent/summary")
    assert resp.status_code == 404


def test_health_check(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_get_turns_not_found(client):
    resp = client.get("/api/session/nonexistent/turns")
    assert resp.status_code == 404


def test_get_blocks_not_found(client):
    resp = client.get("/api/session/nonexistent/blocks")
    assert resp.status_code == 404


def test_get_turn_messages_not_found(client):
    resp = client.get("/api/session/nonexistent/turn/1/messages")
    assert resp.status_code == 404


def test_get_turns_with_data(client_with_transcript):
    client, session_id = client_with_transcript
    resp = client.get(f"/api/session/{session_id}/turns")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == session_id
    assert data["turn_count"] >= 1
    assert "turns" in data
    assert len(data["turns"]) >= 1
    # Each turn should have the expected fields
    turn = data["turns"][0]
    assert "turn" in turn
    assert "system_tokens" in turn
    assert "active_tokens" in turn
    assert "stale_tokens" in turn
    assert "total_tokens" in turn
    assert "block_count" in turn


def test_get_blocks_with_data(client_with_transcript):
    client, session_id = client_with_transcript
    resp = client.get(f"/api/session/{session_id}/blocks")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == session_id
    assert "blocks" in data
    if len(data["blocks"]) > 0:
        block = data["blocks"][0]
        assert "block_id" in block
        assert "staleness_label" in block
        assert "size_tokens_est" in block


def test_get_turn_messages_with_data(client_with_transcript):
    client, session_id = client_with_transcript
    resp = client.get(f"/api/session/{session_id}/turn/1/messages")
    assert resp.status_code == 200
    data = resp.json()
    assert data["turn"] == 1
    assert "messages" in data


def test_get_turn_messages_invalid_turn(client_with_transcript):
    client, session_id = client_with_transcript
    resp = client.get(f"/api/session/{session_id}/turn/999/messages")
    assert resp.status_code == 404
