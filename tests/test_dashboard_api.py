"""Tests for dashboard REST API endpoints."""


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
