"""Tests for multi-agent workflow run tracking.

Covers:
  (a) discovery/parse of workflow agents under subagents/workflows/wf_*/
  (b) ingest creating a WorkflowRunRecord + SubagentRecords with
      phase/label/workflow_id and api-call token rows
  (c) the /api/session/{id}/workflows endpoint grouped structure
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from context_tracker.ccscope.subagents import parse_workflows
from context_tracker.dashboard import create_app
from context_tracker.db import (
    SubagentApiCallRecord,
    SubagentRecord,
    WorkflowRunRecord,
    get_engine,
    get_session_factory,
)
from context_tracker.ingest import ingest_session

SESSION_ID = "wf-session-001"
WF_ID = "wf_test1234-abc"


# ---------------------------------------------------------------------------
# Fixture builders — mimic the real on-disk workflow layout
# ---------------------------------------------------------------------------


def _assistant_entry(usage: dict, agent_id: str, stop_reason: str = "end_turn") -> dict:
    return {
        "type": "assistant",
        "agentId": agent_id,
        "message": {
            "model": "claude-opus-4-8",
            "role": "assistant",
            "stop_reason": stop_reason,
            "content": [{"type": "text", "text": "ok"}],
            "usage": usage,
        },
        "uuid": "a1",
        "timestamp": "2026-06-09T23:50:00.000Z",
    }


def _write_agent(run_dir: Path, agent_id: str, usages: list[dict]) -> None:
    (run_dir / f"agent-{agent_id}.meta.json").write_text(json.dumps({"agentType": "workflow-subagent"}))
    jsonl = run_dir / f"agent-{agent_id}.jsonl"
    with jsonl.open("w") as f:
        for u in usages:
            f.write(json.dumps(_assistant_entry(u, agent_id)) + "\n")


def _usage(inp: int, out: int, cr: int = 0, cc: int = 0) -> dict:
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "cache_read_input_tokens": cr,
        "cache_creation_input_tokens": cc,
    }


def _build_workflow_layout(projects_dir: Path) -> Path:
    """Create projects_dir/<proj>/<session>/subagents/workflows/wf_*/ plus a
    minimal parent transcript so find_session_paths can discover the session.
    Returns the session's subagents/ dir.
    """
    session_dir = projects_dir / "proj" / SESSION_ID
    wf_dir = session_dir / "subagents" / "workflows" / WF_ID
    wf_dir.mkdir(parents=True)

    # Two agents share phase "k1" (parallel), one in phase "k2".
    _write_agent(wf_dir, "agentaaa", [_usage(1000, 50, cr=200), _usage(1500, 80, cr=300)])
    _write_agent(wf_dir, "agentbbb", [_usage(2000, 120, cc=400)])
    _write_agent(wf_dir, "agentccc", [_usage(800, 30)])

    journal = wf_dir / "journal.jsonl"
    lines = [
        {"type": "started", "key": "k1", "agentId": "agentaaa"},
        {"type": "started", "key": "k1", "agentId": "agentbbb"},
        {"type": "started", "key": "k2", "agentId": "agentccc"},
        {
            "type": "result",
            "key": "k1",
            "agentId": "agentaaa",
            "result": {"dimension": "Security review", "verdict": "ok"},
        },
        {
            "type": "result",
            "key": "k1",
            "agentId": "agentbbb",
            "result": {"dimension": "Performance review"},
        },
        # agentccc has no result line -> partial journal / still running
    ]
    with journal.open("w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    # Minimal parent transcript so ingest_session has something to reconcile.
    transcript = projects_dir / "proj" / f"{SESSION_ID}.jsonl"
    parent = [
        {
            "type": "user",
            "uuid": "u1",
            "timestamp": "2026-06-09T23:49:00.000Z",
            "message": {"content": "run the workflow"},
        },
        {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "u1",
            "timestamp": "2026-06-09T23:49:01.000Z",
            "message": {
                "model": "claude-opus-4-8",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "done"}],
                "usage": _usage(500, 20),
            },
        },
    ]
    with transcript.open("w") as f:
        for entry in parent:
            f.write(json.dumps(entry) + "\n")

    return session_dir / "subagents"


# ---------------------------------------------------------------------------
# (a) discovery / parse
# ---------------------------------------------------------------------------


def test_discovery_finds_workflow_agents(tmp_path):
    subagents_dir = _build_workflow_layout(tmp_path / "projects")

    runs = parse_workflows(subagents_dir)
    assert len(runs) == 1
    run = runs[0]
    assert run["wf_id"] == WF_ID
    assert len(run["agents"]) == 3

    by_id = {a["agent_id"]: a for a in run["agents"]}
    assert set(by_id) == {"agentaaa", "agentbbb", "agentccc"}

    # phase + label recovered from journal
    assert by_id["agentaaa"]["phase"] == "k1"
    assert by_id["agentaaa"]["label"] == "Security review"
    assert by_id["agentbbb"]["phase"] == "k1"
    assert by_id["agentccc"]["phase"] == "k2"
    # partial journal: no result line for agentccc
    assert by_id["agentccc"]["label"] is None

    # token reconstruction reused from subagent machinery
    assert by_id["agentaaa"]["api_calls"] == 2
    assert by_id["agentaaa"]["peak_resident"] == 1500 + 300  # input + cache_read


def test_parse_workflows_no_dir(tmp_path):
    # subagents dir without a workflows/ subdir -> empty
    d = tmp_path / "subagents"
    d.mkdir()
    assert parse_workflows(d) == []


# ---------------------------------------------------------------------------
# (b) ingest
# ---------------------------------------------------------------------------


def test_ingest_creates_workflow_records(tmp_path):
    projects_dir = tmp_path / "projects"
    _build_workflow_layout(projects_dir)
    db_path = tmp_path / "analyzer.db"

    rec = ingest_session(
        SESSION_ID,
        trace_dir=tmp_path / "traces",
        db_path=db_path,
        projects_dir=projects_dir,
    )
    assert rec is not None

    engine = get_engine(db_path)
    factory = get_session_factory(engine)
    with factory() as db:
        runs = db.query(WorkflowRunRecord).filter_by(session_id=SESSION_ID).all()
        assert len(runs) == 1
        run = runs[0]
        assert run.wf_id == WF_ID

        agents = db.query(SubagentRecord).filter_by(workflow_id=run.id).all()
        assert len(agents) == 3
        for a in agents:
            assert a.workflow_id == run.id
            assert a.phase in ("k1", "k2")

        by_id = {a.agent_id: a for a in agents}
        assert by_id["agentaaa"].label == "Security review"

        # per-call token rows persisted
        calls = (
            db.query(SubagentApiCallRecord)
            .filter_by(subagent_id=by_id["agentaaa"].id)
            .order_by(SubagentApiCallRecord.call_index)
            .all()
        )
        assert len(calls) == 2
        assert calls[0].input_tokens == 1000
        assert calls[1].cache_read == 300

        # plain (non-workflow) subagents stay null on workflow_id
        plain = db.query(SubagentRecord).filter_by(workflow_id=None).all()
        assert plain == []  # none in this fixture


def test_reingest_replaces_runs_no_duplicates(tmp_path):
    """Re-ingesting a session whose source changed must not duplicate runs
    (cascade delete via SessionRecord.workflow_runs)."""
    import os
    import time

    projects_dir = tmp_path / "projects"
    _build_workflow_layout(projects_dir)
    db_path = tmp_path / "analyzer.db"

    ingest_session(
        SESSION_ID,
        trace_dir=tmp_path / "traces",
        db_path=db_path,
        projects_dir=projects_dir,
    )

    # Bump the parent transcript mtime so the second ingest re-runs (delete + re-add).
    transcript = projects_dir / "proj" / f"{SESSION_ID}.jsonl"
    future = time.time() + 10
    os.utime(transcript, (future, future))

    ingest_session(
        SESSION_ID,
        trace_dir=tmp_path / "traces",
        db_path=db_path,
        projects_dir=projects_dir,
    )

    engine = get_engine(db_path)
    factory = get_session_factory(engine)
    with factory() as db:
        assert db.query(WorkflowRunRecord).filter_by(session_id=SESSION_ID).count() == 1
        assert db.query(SubagentRecord).filter_by(session_id=SESSION_ID).count() == 3


# ---------------------------------------------------------------------------
# (c) endpoint
# ---------------------------------------------------------------------------


def test_workflows_endpoint(tmp_path):
    projects_dir = tmp_path / "projects"
    _build_workflow_layout(projects_dir)
    db_path = tmp_path / "analyzer.db"

    app = create_app(
        trace_dir=tmp_path / "traces",
        transcript_dir=projects_dir,
        static_dir=tmp_path / "static",
        db_path=db_path,
    )
    client = TestClient(app)

    resp = client.get(f"/api/session/{SESSION_ID}/workflows")
    assert resp.status_code == 200
    data = resp.json()

    assert data["count"] == 1
    wf = data["workflows"][0]
    assert wf["wf_id"] == WF_ID
    assert wf["total_agents"] == 3
    assert wf["total_phases"] == 2
    assert wf["max_parallelism"] == 2  # two agents in phase k1

    # grouped run -> phase -> agents
    phases = {p["phase"]: p for p in wf["phases"]}
    assert set(phases) == {"k1", "k2"}
    assert len(phases["k1"]["agents"]) == 2
    assert len(phases["k2"]["agents"]) == 1
    assert phases["k1"]["label"] in ("Security review", "Performance review")

    # per-agent token totals present
    agent = phases["k1"]["agents"][0]
    assert "peak_resident" in agent
    assert "total_output_tokens" in agent
    assert "churn" in agent


def test_workflows_endpoint_empty_for_unknown_session(tmp_path):
    db_path = tmp_path / "analyzer.db"
    app = create_app(
        trace_dir=tmp_path / "traces",
        transcript_dir=tmp_path / "projects",
        static_dir=tmp_path / "static",
        db_path=db_path,
    )
    client = TestClient(app)
    resp = client.get("/api/session/no-such-session/workflows")
    assert resp.status_code == 200
    assert resp.json() == {"count": 0, "workflows": []}
