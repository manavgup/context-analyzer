"""Tests for Codex CLI rollout ingestion (issue #80, phase 1)."""

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from context_tracker.codex import (
    find_codex_rollout,
    list_codex_sessions,
    parse_codex_rollout,
)
from context_tracker.dashboard import create_app
from context_tracker.db import (
    AGENT_CLAUDE_CODE,
    AGENT_CODEX,
    ApiCallRecord,
    BlockRecord,
    SessionRecord,
    TurnRecord,
    get_engine,
    get_session_factory,
)
from context_tracker.ingest import ingest_codex_session

FIXTURE_CODEX_DIR = Path(__file__).parent / "fixtures" / "codex"
FIXTURE_SESSION_ID = "0199aaaa-bbbb-7ccc-8ddd-eeeeffff0001"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_list_codex_sessions_finds_fixture():
    assert list_codex_sessions(FIXTURE_CODEX_DIR) == [FIXTURE_SESSION_ID]


def test_list_codex_sessions_missing_dir(tmp_path):
    assert list_codex_sessions(tmp_path / "nope") == []


def test_find_codex_rollout():
    path = find_codex_rollout(FIXTURE_SESSION_ID, codex_dir=FIXTURE_CODEX_DIR)
    assert path is not None
    assert path.name.endswith(f"{FIXTURE_SESSION_ID}.jsonl")


def test_find_codex_rollout_unknown_session():
    assert find_codex_rollout("0199ffff-0000-7000-8000-000000000000", codex_dir=FIXTURE_CODEX_DIR) is None


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


@pytest.fixture
def parsed():
    rollout = find_codex_rollout(FIXTURE_SESSION_ID, codex_dir=FIXTURE_CODEX_DIR)
    return parse_codex_rollout(rollout)


def test_parse_meta(parsed):
    assert parsed["session_id"] == FIXTURE_SESSION_ID
    assert parsed["model"] == "gpt-5-codex"
    assert parsed["cwd"] == "/home/dev/example-project"
    assert parsed["cli_version"] == "0.41.0"
    assert parsed["started_at"] == "2026-01-05T10:00:00.000Z"
    assert parsed["ended_at"] > parsed["started_at"]


def test_parse_churn(parsed):
    churn = parsed["churn"]
    # Three token_count events with usage info; info:null ones are skipped.
    assert len(churn) == 3
    # input = input_tokens - cached_input_tokens; cache_read = cached.
    assert churn[0] == {"turn": 0, "cache_read": 3000, "cache_creation": 0, "input": 2000, "output": 120}
    assert churn[1]["cache_read"] == 4800
    assert churn[1]["input"] == 1700
    assert churn[2]["output"] == 340


def test_parse_blocks(parsed):
    blocks = parsed["blocks"]
    by_type: dict[str, list] = {}
    for b in blocks:
        by_type.setdefault(b["type"], []).append(b)

    # Pinned instructions block from session_meta.
    assert len(by_type["system"]) == 1
    assert by_type["system"][0]["cached"] is True
    assert by_type["system"][0]["enter"] == 0

    # 2 meta user messages + 2 real prompts.
    assert len(by_type["user"]) == 4
    meta = [b for b in by_type["user"] if "meta" in b["label"]]
    prompts = [b for b in by_type["user"] if b["label"] == "user prompt"]
    assert len(meta) == 2
    assert len(prompts) == 2
    assert prompts[0]["content"].startswith("Add a retry helper")

    assert len(by_type["thinking"]) == 1
    assert len(by_type["tool_call"]) == 2
    assert len(by_type["tool_result"]) == 2
    assert len(by_type["assistant"]) == 2

    # tool_call/tool_result share the call_id-based suffix and get labels.
    tc = by_type["tool_call"][0]
    assert tc["id"].endswith("call_0001")
    assert tc["label"].startswith("shell ")
    tr = by_type["tool_result"][0]
    assert tr["id"].endswith("call_0001")
    assert tr["label"] == "shell → result"

    # Every block has a positive token estimate and an enter turn.
    for b in blocks:
        assert b["tokens"] >= 1
        assert b["enter"] is not None


def test_parse_block_enter_turns(parsed):
    blocks = parsed["blocks"]
    # Second-turn blocks enter at API call 2.
    second_prompt = next(b for b in blocks if b["content"].startswith("Now add a unit test"))
    assert second_prompt["enter"] == 2
    second_asst = next(b for b in blocks if b["content"].startswith("Added test_retry"))
    assert second_asst["enter"] == 2


def test_parse_turn_map(parsed):
    turn_map = parsed["turn_map"]
    assert len(turn_map) == 2
    assert turn_map[0]["conv_turn"] == 1
    assert turn_map[0]["first_call"] == 0
    assert turn_map[0]["last_call"] == 1
    assert turn_map[0]["user_prompt"].startswith("Add a retry helper")
    assert turn_map[1]["first_call"] == 2
    assert turn_map[1]["last_call"] == 2


def test_parser_tolerates_unknown_and_malformed(tmp_path):
    """Unknown record types, unknown payload types, and bad JSON are skipped."""
    rollout = tmp_path / "rollout-2026-01-01T00-00-00-0199cccc-0000-7000-8000-000000000abc.jsonl"
    lines = [
        "garbage not json",
        json.dumps({"type": "totally_new_thing", "payload": {"x": 1}}),
        json.dumps({"type": "response_item", "payload": {"type": "unknown_item"}}),
        json.dumps({"type": "event_msg", "payload": {"type": "token_count", "info": None}}),
        json.dumps({"type": "event_msg", "payload": None}),
        json.dumps([1, 2, 3]),  # non-dict record
    ]
    rollout.write_text("\n".join(lines) + "\n")
    parsed = parse_codex_rollout(rollout)
    assert parsed["blocks"] == []
    assert parsed["churn"] == []
    # Session id recovered from the filename.
    assert parsed["session_id"] == "0199cccc-0000-7000-8000-000000000abc"


def test_parser_empty_file(tmp_path):
    rollout = tmp_path / "rollout-2026-01-01T00-00-00-0199dddd-0000-7000-8000-000000000abc.jsonl"
    rollout.write_text("")
    parsed = parse_codex_rollout(rollout)
    assert parsed["churn"] == []
    assert parsed["turn_map"] == []


# ---------------------------------------------------------------------------
# Ingestion into SQLite
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "analyzer.db"


def test_ingest_codex_session(db_path):
    rec = ingest_codex_session(FIXTURE_SESSION_ID, codex_dir=FIXTURE_CODEX_DIR, db_path=db_path)
    assert rec is not None
    assert rec.agent == AGENT_CODEX
    assert rec.model == "gpt-5-codex"
    assert rec.project_path == "/home/dev/example-project"
    assert rec.total_api_calls == 3
    assert rec.total_turns == 2
    assert rec.total_input_tokens == 2000 + 1700 + 2700
    assert rec.total_output_tokens == 120 + 210 + 340
    assert rec.total_cache_read == 3000 + 4800 + 6400
    assert rec.total_cache_creation == 0
    assert rec.peak_context_tokens == 9100  # full prompt of the largest call
    assert rec.total_cost_usd == 0.0  # no fake pricing for Codex
    assert rec.started_at == "2026-01-05T10:00:00.000Z"

    factory = get_session_factory(get_engine(db_path))
    with factory() as db:
        assert db.query(ApiCallRecord).filter_by(session_id=FIXTURE_SESSION_ID).count() == 3
        assert db.query(BlockRecord).filter_by(session_id=FIXTURE_SESSION_ID).count() == rec.total_blocks
        turns = db.query(TurnRecord).filter_by(session_id=FIXTURE_SESSION_ID).order_by(TurnRecord.turn_number).all()
        assert len(turns) == 2
        assert turns[0].turn_number == 0
        assert turns[0].prompt_preview.startswith("Add a retry helper")


def test_ingest_codex_session_idempotent(db_path):
    rec1 = ingest_codex_session(FIXTURE_SESSION_ID, codex_dir=FIXTURE_CODEX_DIR, db_path=db_path)
    rec2 = ingest_codex_session(FIXTURE_SESSION_ID, codex_dir=FIXTURE_CODEX_DIR, db_path=db_path)
    assert rec1 is not None and rec2 is not None
    factory = get_session_factory(get_engine(db_path))
    with factory() as db:
        assert db.query(SessionRecord).count() == 1
        assert db.query(ApiCallRecord).count() == 3


def test_ingest_codex_session_missing(db_path, tmp_path):
    assert ingest_codex_session("0199ffff-0000-7000-8000-000000000000", codex_dir=tmp_path, db_path=db_path) is None


def test_agent_column_migration(tmp_path):
    """Databases created before the agent column exists get migrated in place."""
    db_file = tmp_path / "old.db"
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, model TEXT)")
    conn.execute("INSERT INTO sessions (session_id, model) VALUES ('old-session', 'claude-opus-4-6')")
    conn.commit()
    conn.close()

    engine = get_engine(db_file)
    with engine.connect() as c:
        from sqlalchemy import text

        row = c.execute(text("SELECT agent FROM sessions WHERE session_id='old-session'")).fetchone()
    assert row[0] == AGENT_CLAUDE_CODE


# ---------------------------------------------------------------------------
# Dashboard API
# ---------------------------------------------------------------------------


@pytest.fixture
def codex_client(tmp_path):
    app = create_app(
        trace_dir=tmp_path / "traces",
        transcript_dir=tmp_path / "transcripts",
        static_dir=tmp_path / "static",
        db_path=tmp_path / "analyzer.db",
        codex_dir=FIXTURE_CODEX_DIR,
    )
    return TestClient(app)


def test_api_sessions_lists_codex_session(codex_client):
    resp = codex_client.get("/api/sessions")
    assert resp.status_code == 200
    sessions = resp.json()
    assert len(sessions) == 1
    s = sessions[0]
    assert s["session_id"] == FIXTURE_SESSION_ID
    assert s["agent"] == AGENT_CODEX
    assert s["model"] == "gpt-5-codex"
    assert s["total_api_calls"] == 3
    assert s["first_prompt"].startswith("Add a retry helper")


def test_api_session_summary_codex(codex_client):
    resp = codex_client.get(f"/api/session/{FIXTURE_SESSION_ID}/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["agent"] == AGENT_CODEX
    assert data["total_turns"] == 2
    assert data["total_api_calls"] == 3
    assert data["peak_context_tokens"] == 9100
    # Codex has no hook events or subagents — empty, not faked.
    assert data["hook_event_counts"] == {}
    assert data["subagent_count"] == 0
    assert data["block_type_counts"]["tool_call"] == 2


def test_api_session_data_codex(codex_client):
    """Acceptance: a Codex fixture session renders via the session data API."""
    resp = codex_client.get(f"/api/session/{FIXTURE_SESSION_ID}/data")
    assert resp.status_code == 200
    data = resp.json()
    assert data["meta"]["session_id"] == FIXTURE_SESSION_ID
    assert data["meta"]["agent"] == AGENT_CODEX
    assert data["meta"]["model"] == "gpt-5-codex"
    assert len(data["churn"]) == 3
    assert len(data["turn_map"]) == 2
    assert len(data["blocks"]) > 0
    # Blocks carry everything the dashboard needs to render the tape.
    for block in data["blocks"]:
        for key in ("id", "type", "label", "tokens", "enter", "exit", "cached", "ref"):
            assert key in block


def test_api_sessions_trends_includes_codex(codex_client):
    resp = codex_client.get("/api/sessions/trends")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_count"] == 1
    assert data["sessions"][0]["agent"] == AGENT_CODEX
    assert data["total_api_calls"] == 3


def test_api_session_data_missing_still_404(codex_client):
    resp = codex_client.get("/api/session/0199ffff-0000-7000-8000-000000000000/data")
    assert resp.status_code == 404


def test_codex_disabled_by_default(tmp_path):
    """Without codex_dir, the app does not pick up Codex sessions."""
    app = create_app(
        trace_dir=tmp_path / "traces",
        transcript_dir=tmp_path / "transcripts",
        static_dir=tmp_path / "static",
        db_path=tmp_path / "analyzer.db",
    )
    client = TestClient(app)
    assert client.get("/api/sessions").json() == []
    assert client.get(f"/api/session/{FIXTURE_SESSION_ID}/data").status_code == 404
