"""Tests for the top-level CLI: context-tracker up / down (issue #77)."""

from __future__ import annotations

import json
import sys

import pytest
from fastapi.testclient import TestClient

import context_tracker.cli as cli
from context_tracker.installer import CONTEXT_TRACKER_MARKER, install_hooks

SESSION_ID = "11111111-2222-3333-4444-555555555555"

# ---------------------------------------------------------------------------
# Helpers — minimal transcript builders (same pattern as test_ingest.py)
# ---------------------------------------------------------------------------


def _make_transcript(projects_dir):
    """Write a minimal one-turn transcript discoverable by list_sessions/ingest."""
    entries = [
        {
            "type": "user",
            "uuid": "u1",
            "parentUuid": None,
            "timestamp": "2026-01-01T00:00:00.000Z",
            "message": {"content": "Hello, help me with code"},
        },
        {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "u1",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "message": {
                "model": "claude-opus-4-6",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "Sure, I can help!"}],
                "usage": {
                    "input_tokens": 5,
                    "cache_creation_input_tokens": 10000,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 20,
                },
            },
        },
    ]
    path = projects_dir / "test-project" / f"{SESSION_ID}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return path


@pytest.fixture
def env(tmp_path):
    """Isolated settings/trace/projects/db paths with one existing transcript."""
    projects_dir = tmp_path / "projects"
    _make_transcript(projects_dir)
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("{}")
    return {
        "settings_path": settings_path,
        "trace_dir": tmp_path / "traces",
        "projects_dir": projects_dir,
        "db_path": tmp_path / "analyzer.db",
    }


def _run_up(env, **kwargs):
    """Run cli.up against the isolated env with a no-op serve; returns (app, serve calls)."""
    calls = []

    def fake_serve(app, host, port):
        calls.append((host, port))

    app = cli.up(
        settings_path=env["settings_path"],
        trace_dir=env["trace_dir"],
        projects_dir=env["projects_dir"],
        db_path=env["db_path"],
        serve=fake_serve,
        **kwargs,
    )
    return app, calls


# ---------------------------------------------------------------------------
# up
# ---------------------------------------------------------------------------


def test_up_installs_hooks_ingests_and_serves(env):
    app, calls = _run_up(env)

    # Hooks were installed into settings.json
    settings = json.loads(env["settings_path"].read_text())
    assert "hooks" in settings
    commands = json.dumps(settings["hooks"])
    assert CONTEXT_TRACKER_MARKER in commands

    # The dashboard server was started with defaults
    assert calls == [("127.0.0.1", 8080)]

    # Existing transcript was ingested — the dashboard shows the session
    sessions = TestClient(app).get("/api/sessions").json()
    assert [s["session_id"] for s in sessions] == [SESSION_ID]
    assert sessions[0]["total_api_calls"] == 1


def test_up_no_hooks_still_shows_existing_transcript_data(env):
    """--no-hooks must not touch settings.json, yet the dashboard has data.

    This is the zero-setup guarantee from #77: value from existing transcripts
    in ~/.claude/projects/ even when the user declines hook installation.
    """
    app, calls = _run_up(env, install=False)

    # settings.json untouched — no hooks written
    assert json.loads(env["settings_path"].read_text()) == {}

    # Dashboard still serves the pre-existing session
    assert calls == [("127.0.0.1", 8080)]
    sessions = TestClient(app).get("/api/sessions").json()
    assert [s["session_id"] for s in sessions] == [SESSION_ID]
    assert sessions[0]["total_api_calls"] == 1


def test_up_respects_host_and_port(env):
    _, calls = _run_up(env, host="0.0.0.0", port=9999)  # noqa: S104
    assert calls == [("0.0.0.0", 9999)]  # noqa: S104


def test_up_is_idempotent_for_hooks(env):
    """Running up twice must not duplicate hook entries (reinstall semantics)."""
    _run_up(env)
    _run_up(env)

    settings = json.loads(env["settings_path"].read_text())
    for matchers in settings["hooks"].values():
        owned = [m for m in matchers if any(CONTEXT_TRACKER_MARKER in h.get("command", "") for h in m.get("hooks", []))]
        assert len(owned) == 1


# ---------------------------------------------------------------------------
# down
# ---------------------------------------------------------------------------


def test_down_removes_only_our_hooks(env):
    """down removes context-tracker entries and leaves foreign hooks untouched."""
    settings_path = env["settings_path"]
    foreign = {"matcher": "Bash", "hooks": [{"type": "command", "command": "echo mine"}]}
    settings_path.write_text(json.dumps({"hooks": {"PostToolUse": [foreign]}, "theme": "dark"}))

    install_hooks(settings_path=settings_path, hook_command="python3 -m context_tracker.hooks")
    cli.down(settings_path=settings_path)

    settings = json.loads(settings_path.read_text())
    assert CONTEXT_TRACKER_MARKER not in json.dumps(settings)
    assert settings["hooks"]["PostToolUse"] == [foreign]
    assert settings["theme"] == "dark"


def test_install_creates_backup_used_by_down_messaging(env, capsys):
    """Install backs up settings.json; down points the user at the backup."""
    settings_path = env["settings_path"]
    settings_path.write_text(json.dumps({"theme": "dark"}))

    install_hooks(settings_path=settings_path, hook_command="python3 -m context_tracker.hooks")
    backup = settings_path.with_suffix(".json.bak")
    assert json.loads(backup.read_text()) == {"theme": "dark"}

    cli.down(settings_path=settings_path)
    assert str(backup) in capsys.readouterr().out


# ---------------------------------------------------------------------------
# main() dispatch
# ---------------------------------------------------------------------------


def test_main_dispatches_up_with_flags(monkeypatch):
    recorded = {}

    def fake_up(**kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(cli, "up", fake_up)
    monkeypatch.setattr(sys, "argv", ["context-tracker", "up", "--no-hooks", "--port", "9000"])
    cli.main()

    assert recorded["install"] is False
    assert recorded["port"] == 9000
    assert recorded["host"] == "127.0.0.1"
    assert recorded["open_browser"] is False


def test_main_dispatches_down(monkeypatch):
    called = []
    monkeypatch.setattr(cli, "down", lambda **kw: called.append(kw))
    monkeypatch.setattr(sys, "argv", ["context-tracker", "down"])
    cli.main()
    assert called == [{}]


def test_main_dispatches_install_and_uninstall(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "install_hooks", lambda: calls.append("install"))
    monkeypatch.setattr(cli, "uninstall_hooks", lambda: calls.append("uninstall"))

    monkeypatch.setattr(sys, "argv", ["context-tracker", "install"])
    cli.main()
    monkeypatch.setattr(sys, "argv", ["context-tracker", "uninstall"])
    cli.main()
    assert calls == ["install", "uninstall"]


def test_main_delegates_unknown_commands_to_server(monkeypatch):
    """MCP server compatibility: no subcommand (and `dashboard`) go to server.main."""
    import context_tracker.server as server

    called = []
    monkeypatch.setattr(server, "main", lambda: called.append(True))
    monkeypatch.setattr(sys, "argv", ["context-tracker"])
    cli.main()
    assert called == [True]
