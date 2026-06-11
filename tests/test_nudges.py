"""Tests for the real-time session nudge engine."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from context_tracker.dashboard import create_app
from context_tracker.db import (
    HookEventRecord,
    SessionRecord,
    get_engine,
    get_session_factory,
)
from context_tracker.nudges import Nudge, evaluate_nudges

# ── Helpers ──────────────────────────────────────────────────────────


def _create_session(db_factory, session_id: str, **kwargs) -> None:  # type: ignore[no-untyped-def]
    """Insert a SessionRecord with the given overrides."""
    with db_factory() as db:
        rec = SessionRecord(session_id=session_id, **kwargs)
        db.add(rec)
        db.commit()


def _add_hook_events(db_factory, session_id: str, events: list[dict]) -> None:  # type: ignore[no-untyped-def]
    """Insert HookEventRecords."""
    with db_factory() as db:
        for evt in events:
            db.add(HookEventRecord(session_id=session_id, **evt))
        db.commit()


def _write_trace_events(trace_dir, session_id: str, events: list[dict]) -> None:  # type: ignore[no-untyped-def]
    """Write events to a hook trace JSONL file (simulating real-time hook output)."""
    trace_dir.mkdir(parents=True, exist_ok=True)
    filepath = trace_dir / f"{session_id}.jsonl"
    with open(filepath, "a", encoding="utf-8") as f:
        for evt in events:
            evt.setdefault("session_id", session_id)
            f.write(json.dumps(evt) + "\n")


# ── CONTEXT_THRESHOLD nudge ──────────────────────────────────────────


class TestContextThreshold:
    def test_fires_when_above_threshold(self, tmp_path):
        """CONTEXT_THRESHOLD fires when peak context > 600K tokens (60%)."""
        db_path = tmp_path / "test.db"
        engine = get_engine(db_path)
        factory = get_session_factory(engine)
        _create_session(factory, "sess-ct-high", peak_context_tokens=700_000)

        nudges = evaluate_nudges("sess-ct-high", db_path=db_path)

        ct_nudges = [n for n in nudges if n.code == "CONTEXT_THRESHOLD"]
        assert len(ct_nudges) == 1
        assert ct_nudges[0].severity == "warning"
        assert "70%" in ct_nudges[0].message
        engine.dispose()

    def test_critical_at_80_pct(self, tmp_path):
        """CONTEXT_THRESHOLD is critical at 80%+."""
        db_path = tmp_path / "test.db"
        engine = get_engine(db_path)
        factory = get_session_factory(engine)
        _create_session(factory, "sess-ct-crit", peak_context_tokens=850_000)

        nudges = evaluate_nudges("sess-ct-crit", db_path=db_path)

        ct_nudges = [n for n in nudges if n.code == "CONTEXT_THRESHOLD"]
        assert len(ct_nudges) == 1
        assert ct_nudges[0].severity == "critical"
        engine.dispose()

    def test_does_not_fire_below_threshold(self, tmp_path):
        """CONTEXT_THRESHOLD does not fire below 60%."""
        db_path = tmp_path / "test.db"
        engine = get_engine(db_path)
        factory = get_session_factory(engine)
        _create_session(factory, "sess-ct-low", peak_context_tokens=500_000)

        nudges = evaluate_nudges("sess-ct-low", db_path=db_path)

        ct_nudges = [n for n in nudges if n.code == "CONTEXT_THRESHOLD"]
        assert len(ct_nudges) == 0
        engine.dispose()

    def test_custom_threshold(self, tmp_path):
        """CONTEXT_THRESHOLD respects custom threshold config."""
        db_path = tmp_path / "test.db"
        engine = get_engine(db_path)
        factory = get_session_factory(engine)
        _create_session(factory, "sess-ct-custom", peak_context_tokens=400_000)

        # Default 60% threshold -> no nudge at 400K
        nudges = evaluate_nudges("sess-ct-custom", db_path=db_path)
        assert not any(n.code == "CONTEXT_THRESHOLD" for n in nudges)

        # Custom 30% threshold -> fires at 400K
        nudges = evaluate_nudges(
            "sess-ct-custom",
            db_path=db_path,
            config={"context_threshold_pct": 30},
        )
        assert any(n.code == "CONTEXT_THRESHOLD" for n in nudges)
        engine.dispose()


# ── COST_WARNING nudge ───────────────────────────────────────────────


class TestCostWarning:
    def test_fires_when_above_threshold(self, tmp_path):
        """COST_WARNING fires when total_cost_usd > $10."""
        db_path = tmp_path / "test.db"
        engine = get_engine(db_path)
        factory = get_session_factory(engine)
        _create_session(factory, "sess-cost-high", total_cost_usd=15.50)

        nudges = evaluate_nudges("sess-cost-high", db_path=db_path)

        cost_nudges = [n for n in nudges if n.code == "COST_WARNING"]
        assert len(cost_nudges) == 1
        assert "$15.50" in cost_nudges[0].message
        assert cost_nudges[0].severity == "warning"
        engine.dispose()

    def test_does_not_fire_below_threshold(self, tmp_path):
        """COST_WARNING does not fire below $10."""
        db_path = tmp_path / "test.db"
        engine = get_engine(db_path)
        factory = get_session_factory(engine)
        _create_session(factory, "sess-cost-low", total_cost_usd=5.00)

        nudges = evaluate_nudges("sess-cost-low", db_path=db_path)

        cost_nudges = [n for n in nudges if n.code == "COST_WARNING"]
        assert len(cost_nudges) == 0
        engine.dispose()


# ── REPEATED_READS nudge ─────────────────────────────────────────────


class TestRepeatedReads:
    def test_fires_when_file_read_3_times(self, tmp_path):
        """REPEATED_READS fires when the same tool_name appears 3+ times in trace."""
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()
        db_path = tmp_path / "test.db"
        engine = get_engine(db_path)
        factory = get_session_factory(engine)
        _create_session(factory, "sess-repeat")

        # Write events to JSONL trace (real-time source)
        _write_trace_events(
            trace_dir,
            "sess-repeat",
            [
                {
                    "event": "post_tool_use",
                    "tool_name": "src/main.py",
                    "input_payload_chars": 100,
                    "output_payload_chars": 200,
                },
                {
                    "event": "post_tool_use",
                    "tool_name": "src/main.py",
                    "input_payload_chars": 100,
                    "output_payload_chars": 200,
                },
                {
                    "event": "post_tool_use",
                    "tool_name": "src/main.py",
                    "input_payload_chars": 100,
                    "output_payload_chars": 200,
                },
                {
                    "event": "post_tool_use",
                    "tool_name": "src/other.py",
                    "input_payload_chars": 50,
                    "output_payload_chars": 50,
                },
            ],
        )

        nudges = evaluate_nudges("sess-repeat", db_path=db_path, trace_dir=trace_dir)

        rr_nudges = [n for n in nudges if n.code == "REPEATED_READS"]
        assert len(rr_nudges) == 1
        assert "src/main.py" in rr_nudges[0].message
        assert "3 times" in rr_nudges[0].message
        assert rr_nudges[0].severity == "info"
        engine.dispose()

    def test_does_not_fire_below_threshold(self, tmp_path):
        """REPEATED_READS does not fire when reads < 3."""
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()
        db_path = tmp_path / "test.db"
        engine = get_engine(db_path)
        factory = get_session_factory(engine)
        _create_session(factory, "sess-no-repeat")

        _write_trace_events(
            trace_dir,
            "sess-no-repeat",
            [
                {
                    "event": "post_tool_use",
                    "tool_name": "src/main.py",
                    "input_payload_chars": 100,
                    "output_payload_chars": 200,
                },
                {
                    "event": "post_tool_use",
                    "tool_name": "src/main.py",
                    "input_payload_chars": 100,
                    "output_payload_chars": 200,
                },
                {
                    "event": "post_tool_use",
                    "tool_name": "src/other.py",
                    "input_payload_chars": 50,
                    "output_payload_chars": 50,
                },
            ],
        )

        nudges = evaluate_nudges("sess-no-repeat", db_path=db_path, trace_dir=trace_dir)

        rr_nudges = [n for n in nudges if n.code == "REPEATED_READS"]
        assert len(rr_nudges) == 0
        engine.dispose()


# ── Healthy session ──────────────────────────────────────────────────


class TestHealthySession:
    def test_empty_nudges_for_healthy_session(self, tmp_path):
        """Healthy sessions return no nudges."""
        db_path = tmp_path / "test.db"
        engine = get_engine(db_path)
        factory = get_session_factory(engine)
        _create_session(
            factory,
            "sess-healthy",
            peak_context_tokens=200_000,
            total_cost_usd=2.50,
        )
        _add_hook_events(
            factory,
            "sess-healthy",
            [
                {"event_type": "post_tool_use", "tool_name": "src/a.py"},
                {"event_type": "post_tool_use", "tool_name": "src/b.py"},
            ],
        )

        nudges = evaluate_nudges("sess-healthy", db_path=db_path)

        assert nudges == []
        engine.dispose()

    def test_unknown_session_returns_empty(self, tmp_path):
        """Unknown session ID returns empty nudges (not an error)."""
        db_path = tmp_path / "test.db"
        get_engine(db_path)  # ensure DB exists
        nudges = evaluate_nudges("nonexistent", db_path=db_path)
        assert nudges == []


# ── Disabled config ──────────────────────────────────────────────────


class TestDisabledConfig:
    def test_disabled_returns_empty(self, tmp_path):
        """When enabled=False, no nudges are returned."""
        db_path = tmp_path / "test.db"
        engine = get_engine(db_path)
        factory = get_session_factory(engine)
        _create_session(factory, "sess-disabled", peak_context_tokens=900_000)

        nudges = evaluate_nudges(
            "sess-disabled",
            db_path=db_path,
            config={"enabled": False},
        )
        assert nudges == []
        engine.dispose()


# ── Integration: hook writes event then nudges evaluate ──────────────


class TestHookIntegration:
    def test_hook_writes_then_nudges_evaluate(self, tmp_path):
        """Integration: after a hook event is stored, nudges can evaluate it."""
        from context_tracker.hooks import process_hook_input
        from context_tracker.storage import append_event

        trace_dir = tmp_path / "traces"
        db_path = tmp_path / "test.db"
        engine = get_engine(db_path)
        factory = get_session_factory(engine)

        # Create session and add hook events to DB
        _create_session(
            factory,
            "sess-integ",
            peak_context_tokens=750_000,
            total_cost_usd=12.00,
        )
        # Write trace events for repeated reads detection
        _write_trace_events(
            trace_dir,
            "sess-integ",
            [
                {
                    "event": "post_tool_use",
                    "tool_name": "README.md",
                    "input_payload_chars": 100,
                    "output_payload_chars": 500,
                },
                {
                    "event": "post_tool_use",
                    "tool_name": "README.md",
                    "input_payload_chars": 100,
                    "output_payload_chars": 500,
                },
                {
                    "event": "post_tool_use",
                    "tool_name": "README.md",
                    "input_payload_chars": 100,
                    "output_payload_chars": 500,
                },
                {
                    "event": "post_tool_use",
                    "tool_name": "README.md",
                    "input_payload_chars": 100,
                    "output_payload_chars": 500,
                },
            ],
        )

        # Process a hook input (simulating what hooks.py does)
        hook_input = {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "sess-integ",
            "prompt": "fix the tests",
        }
        event = process_hook_input(json.dumps(hook_input))
        assert event is not None
        append_event(event, trace_dir=trace_dir)

        # Evaluate nudges (pass trace_dir for repeated reads detection)
        nudges = evaluate_nudges("sess-integ", db_path=db_path, trace_dir=trace_dir)

        codes = {n.code for n in nudges}
        assert "CONTEXT_THRESHOLD" in codes
        assert "COST_WARNING" in codes
        assert "REPEATED_READS" in codes
        engine.dispose()


# ── Dashboard API endpoint ───────────────────────────────────────────


class TestNudgesEndpoint:
    def test_nudges_endpoint_returns_nudges(self, tmp_path):
        """GET /api/session/{id}/nudges returns nudge list."""
        db_path = tmp_path / "test.db"
        engine = get_engine(db_path)
        factory = get_session_factory(engine)
        _create_session(factory, "sess-api", peak_context_tokens=800_000)

        app = create_app(
            trace_dir=tmp_path / "traces",
            transcript_dir=tmp_path / "transcripts",
            static_dir=tmp_path / "static",
            db_path=db_path,
        )
        client = TestClient(app)

        resp = client.get("/api/session/sess-api/nudges")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "sess-api"
        assert len(data["nudges"]) >= 1
        assert data["nudges"][0]["code"] == "CONTEXT_THRESHOLD"
        engine.dispose()

    def test_nudges_endpoint_empty_for_healthy(self, tmp_path):
        """GET /api/session/{id}/nudges returns empty for healthy session."""
        db_path = tmp_path / "test.db"
        engine = get_engine(db_path)
        factory = get_session_factory(engine)
        _create_session(factory, "sess-api-ok", peak_context_tokens=100_000)

        app = create_app(
            trace_dir=tmp_path / "traces",
            transcript_dir=tmp_path / "transcripts",
            static_dir=tmp_path / "static",
            db_path=db_path,
        )
        client = TestClient(app)

        resp = client.get("/api/session/sess-api-ok/nudges")
        assert resp.status_code == 200
        data = resp.json()
        assert data["nudges"] == []
        engine.dispose()

    def test_nudges_endpoint_unknown_session(self, tmp_path):
        """GET /api/session/{id}/nudges returns empty for unknown session."""
        db_path = tmp_path / "test.db"
        engine = get_engine(db_path)

        app = create_app(
            trace_dir=tmp_path / "traces",
            transcript_dir=tmp_path / "transcripts",
            static_dir=tmp_path / "static",
            db_path=db_path,
        )
        client = TestClient(app)

        resp = client.get("/api/session/nonexistent/nudges")
        assert resp.status_code == 200
        data = resp.json()
        assert data["nudges"] == []
        engine.dispose()


# ── Trace-only nudges (no DB, simulating live session) ──────────────


class TestTraceOnlyNudges:
    """Test nudges computed purely from hook trace JSONL, with no DB data."""

    def test_context_estimate_from_trace(self, tmp_path):
        """Context threshold fires from trace-estimated token count."""
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()
        db_path = tmp_path / "empty.db"
        get_engine(db_path)

        # Write enough tool I/O to exceed 600K tokens (~2.4M chars)
        events = []
        for i in range(100):
            events.append(
                {
                    "event": "post_tool_use",
                    "tool_name": f"file_{i}.py",
                    "input_payload_chars": 5000,
                    "output_payload_chars": 20000,
                }
            )
        _write_trace_events(trace_dir, "trace-ctx", events)

        nudges = evaluate_nudges("trace-ctx", db_path=db_path, trace_dir=trace_dir)
        ct = [n for n in nudges if n.code == "CONTEXT_THRESHOLD"]
        assert len(ct) == 1
        assert "%" in ct[0].message

    def test_cost_estimate_from_trace(self, tmp_path):
        """Cost warning fires from trace-estimated cost."""
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()
        db_path = tmp_path / "empty.db"
        get_engine(db_path)

        events = []
        for i in range(200):
            events.append(
                {
                    "event": "post_tool_use",
                    "tool_name": "Read",
                    "input_payload_chars": 2000,
                    "output_payload_chars": 10000,
                }
            )
            if i % 5 == 0:
                events.append({"event": "user_prompt", "prompt_length_chars": 100})
        _write_trace_events(trace_dir, "trace-cost", events)

        nudges = evaluate_nudges("trace-cost", db_path=db_path, trace_dir=trace_dir)
        cw = [n for n in nudges if n.code == "COST_WARNING"]
        assert len(cw) == 1
        assert "$" in cw[0].message

    def test_no_db_no_trace_returns_empty(self, tmp_path):
        """No DB session and no trace file returns empty nudges."""
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()
        db_path = tmp_path / "empty.db"
        get_engine(db_path)

        nudges = evaluate_nudges("nonexistent", db_path=db_path, trace_dir=trace_dir)
        assert nudges == []

    def test_compaction_resets_context_estimate(self, tmp_path):
        """Post-compact events reduce the estimated context size."""
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()
        db_path = tmp_path / "empty.db"
        get_engine(db_path)

        events = []
        for i in range(80):
            events.append(
                {
                    "event": "post_tool_use",
                    "tool_name": f"file_{i}.py",
                    "input_payload_chars": 5000,
                    "output_payload_chars": 20000,
                }
            )
        events.append({"event": "post_compact", "trigger": "auto", "compact_summary_length": 1000})
        _write_trace_events(trace_dir, "trace-compact", events)

        nudges = evaluate_nudges("trace-compact", db_path=db_path, trace_dir=trace_dir)
        ct = [n for n in nudges if n.code == "CONTEXT_THRESHOLD"]
        assert len(ct) == 0


# ── Nudge dataclass ──────────────────────────────────────────────────


class TestNudgeDataclass:
    def test_nudge_fields(self):
        nudge = Nudge(code="TEST", severity="info", message="test msg")
        assert nudge.code == "TEST"
        assert nudge.severity == "info"
        assert nudge.message == "test msg"

    def test_nudge_equality(self):
        a = Nudge(code="X", severity="warning", message="m")
        b = Nudge(code="X", severity="warning", message="m")
        assert a == b
