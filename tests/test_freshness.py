"""Tests for the context freshness / compact advisor analysis."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from context_tracker.analysis.freshness import (
    _compute_readiness,
    _detect_aged_out,
    _detect_failed_output,
    _detect_redundant,
    _detect_superseded,
    analyze_freshness,
)
from context_tracker.dashboard import create_app
from context_tracker.db import (
    BlockRecord,
    HookEventRecord,
    SessionRecord,
    get_engine,
    get_session_factory,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_block(
    session_id: str = "sess-1",
    block_id: str = "b1",
    block_type: str = "tool_result",
    label: str | None = None,
    tokens: int = 100,
    enter_turn: int = 0,
    exit_turn: int | None = None,
) -> BlockRecord:
    b = BlockRecord()
    b.session_id = session_id
    b.block_id = block_id
    b.block_type = block_type
    b.label = label
    b.tokens = tokens
    b.enter_turn = enter_turn
    b.exit_turn = exit_turn
    return b


def _make_hook(
    session_id: str = "sess-1",
    event_type: str = "post_tool_use_failure",
    tool_use_id: str | None = None,
) -> HookEventRecord:
    h = HookEventRecord()
    h.session_id = session_id
    h.event_type = event_type
    h.tool_use_id = tool_use_id
    return h


# ---------------------------------------------------------------------------
# Tests: _detect_aged_out
# ---------------------------------------------------------------------------


class TestDetectAgedOut:
    def test_exited_block_old_enough(self):
        blocks = [_make_block(enter_turn=0, exit_turn=5)]
        result = _detect_aged_out(blocks, latest_call=20)
        assert len(result) == 1
        assert result[0].category == "aged_out"

    def test_exited_block_too_recent(self):
        blocks = [_make_block(enter_turn=0, exit_turn=15)]
        result = _detect_aged_out(blocks, latest_call=20)
        assert len(result) == 0

    def test_present_block_very_old(self):
        blocks = [_make_block(enter_turn=0, exit_turn=None)]
        result = _detect_aged_out(blocks, latest_call=60)
        assert len(result) == 1

    def test_present_block_young(self):
        blocks = [_make_block(enter_turn=40, exit_turn=None)]
        result = _detect_aged_out(blocks, latest_call=60)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Tests: _detect_superseded
# ---------------------------------------------------------------------------


class TestDetectSuperseded:
    def test_same_label_newer_wins(self):
        blocks = [
            _make_block(block_id="old", label="file.py", enter_turn=0),
            _make_block(block_id="new", label="file.py", enter_turn=10),
        ]
        result = _detect_superseded(blocks)
        assert len(result) == 1
        assert result[0].block_id == "old"
        assert result[0].category == "superseded"

    def test_no_duplicate_labels(self):
        blocks = [
            _make_block(block_id="a", label="foo.py", enter_turn=0),
            _make_block(block_id="b", label="bar.py", enter_turn=5),
        ]
        result = _detect_superseded(blocks)
        assert len(result) == 0

    def test_three_reads_two_superseded(self):
        blocks = [
            _make_block(block_id="v1", label="f.py", enter_turn=0),
            _make_block(block_id="v2", label="f.py", enter_turn=5),
            _make_block(block_id="v3", label="f.py", enter_turn=10),
        ]
        result = _detect_superseded(blocks)
        assert len(result) == 2
        ids = {r.block_id for r in result}
        assert ids == {"v1", "v2"}

    def test_none_label_ignored(self):
        blocks = [
            _make_block(block_id="a", label=None, enter_turn=0),
            _make_block(block_id="b", label=None, enter_turn=5),
        ]
        result = _detect_superseded(blocks)
        assert len(result) == 0

    def test_exited_newer_does_not_supersede_present_older(self):
        """An exited newer block must NOT supersede a present older block."""
        blocks = [
            _make_block(block_id="old", label="file.py", enter_turn=0, exit_turn=None),
            _make_block(block_id="new", label="file.py", enter_turn=10, exit_turn=15),
        ]
        result = _detect_superseded(blocks)
        assert len(result) == 0

    def test_both_exited_newer_supersedes_older(self):
        """When both blocks are exited, the newer one supersedes the older."""
        blocks = [
            _make_block(block_id="old", label="file.py", enter_turn=0, exit_turn=5),
            _make_block(block_id="new", label="file.py", enter_turn=10, exit_turn=15),
        ]
        result = _detect_superseded(blocks)
        assert len(result) == 1
        assert result[0].block_id == "old"

    def test_present_newer_supersedes_present_older(self):
        """When both blocks are present, the newer one supersedes the older."""
        blocks = [
            _make_block(block_id="old", label="file.py", enter_turn=0, exit_turn=None),
            _make_block(block_id="new", label="file.py", enter_turn=10, exit_turn=None),
        ]
        result = _detect_superseded(blocks)
        assert len(result) == 1
        assert result[0].block_id == "old"

    def test_present_newer_supersedes_exited_older(self):
        """A present newer block supersedes an exited older block."""
        blocks = [
            _make_block(block_id="old", label="file.py", enter_turn=0, exit_turn=5),
            _make_block(block_id="new", label="file.py", enter_turn=10, exit_turn=None),
        ]
        result = _detect_superseded(blocks)
        assert len(result) == 1
        assert result[0].block_id == "old"


# ---------------------------------------------------------------------------
# Tests: _detect_failed_output
# ---------------------------------------------------------------------------


class TestDetectFailedOutput:
    def test_matching_failure_hook(self):
        blocks = [
            _make_block(
                block_id="t5-tool_result-toolu_ABC",
                block_type="tool_result",
            )
        ]
        hooks = [_make_hook(event_type="post_tool_use_failure", tool_use_id="toolu_ABC")]
        result = _detect_failed_output(blocks, hooks)
        assert len(result) == 1
        assert result[0].category == "failed_output"

    def test_no_matching_hook(self):
        blocks = [
            _make_block(
                block_id="t5-tool_result-toolu_XYZ",
                block_type="tool_result",
            )
        ]
        hooks = [_make_hook(event_type="post_tool_use_failure", tool_use_id="toolu_OTHER")]
        result = _detect_failed_output(blocks, hooks)
        assert len(result) == 0

    def test_non_tool_result_block(self):
        blocks = [
            _make_block(
                block_id="t5-assistant_text-abc",
                block_type="assistant_text",
            )
        ]
        hooks = [_make_hook(event_type="post_tool_use_failure", tool_use_id="abc")]
        result = _detect_failed_output(blocks, hooks)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Tests: _detect_redundant
# ---------------------------------------------------------------------------


class TestDetectRedundant:
    def test_three_same_label(self):
        blocks = [
            _make_block(block_id="r1", label="readme.md", enter_turn=0),
            _make_block(block_id="r2", label="readme.md", enter_turn=3),
            _make_block(block_id="r3", label="readme.md", enter_turn=6),
        ]
        result = _detect_redundant(blocks)
        # Only the oldest (beyond 2 newest) is marked redundant
        assert len(result) == 1
        assert result[0].block_id == "r1"

    def test_two_same_label_not_redundant(self):
        blocks = [
            _make_block(block_id="r1", label="readme.md", enter_turn=0),
            _make_block(block_id="r2", label="readme.md", enter_turn=3),
        ]
        result = _detect_redundant(blocks)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Tests: _compute_readiness
# ---------------------------------------------------------------------------


class TestComputeReadiness:
    def test_no_stale_no_pressure(self):
        assert _compute_readiness(0, 100) == 0

    def test_all_stale_low_pressure(self):
        score = _compute_readiness(1000, 1000)
        assert score == 60  # 0.6*1.0 + 0.4*(1000/1M) ≈ 0.6 => 60

    def test_high_pressure(self):
        # 500K total, 250K stale
        score = _compute_readiness(250_000, 500_000)
        # 0.6*0.5 + 0.4*0.5 = 0.3 + 0.2 = 0.5 => 50
        assert score == 50

    def test_max_caps_at_100(self):
        score = _compute_readiness(2_000_000, 2_000_000)
        assert score == 100

    def test_zero_total(self):
        assert _compute_readiness(0, 0) == 0


# ---------------------------------------------------------------------------
# Tests: analyze_freshness (integration with DB)
# ---------------------------------------------------------------------------


class TestAnalyzeFreshness:
    @pytest.fixture()
    def db_session(self, tmp_path):
        """Create a temp SQLite DB and return a session."""
        db_path = tmp_path / "test.db"
        engine = get_engine(db_path)
        factory = get_session_factory(engine)
        session = factory()
        yield session
        session.close()

    def test_empty_session(self, db_session):
        # Add session record but no blocks
        sr = SessionRecord()
        sr.session_id = "empty-session"
        sr.model = "claude-opus-4-6"
        db_session.add(sr)
        db_session.commit()

        report = analyze_freshness("empty-session", None, db_session)
        assert report.total_tokens == 0
        assert report.active_tokens == 0
        assert report.stale_tokens == 0
        assert report.compact_readiness_score == 0
        assert report.safe_to_drop == []

    def test_no_stale_blocks(self, db_session):
        sr = SessionRecord()
        sr.session_id = "fresh-session"
        sr.model = "claude-opus-4-6"
        db_session.add(sr)

        # Add a young block
        b = BlockRecord()
        b.session_id = "fresh-session"
        b.block_id = "b1"
        b.block_type = "user_prompt"
        b.label = "hello"
        b.tokens = 500
        b.enter_turn = 5
        b.exit_turn = None
        db_session.add(b)
        db_session.commit()

        report = analyze_freshness("fresh-session", 10, db_session)
        assert report.total_tokens == 500
        assert report.stale_tokens == 0
        assert report.active_tokens == 500
        assert report.compact_readiness_score == 0

    def test_all_stale_blocks(self, db_session):
        sr = SessionRecord()
        sr.session_id = "stale-session"
        sr.model = "claude-opus-4-6"
        db_session.add(sr)

        # Add a very old block
        b = BlockRecord()
        b.session_id = "stale-session"
        b.block_id = "old-block"
        b.block_type = "tool_result"
        b.label = "file.py"
        b.tokens = 10000
        b.enter_turn = 0
        b.exit_turn = None
        db_session.add(b)
        db_session.commit()

        report = analyze_freshness("stale-session", 100, db_session)
        assert report.stale_tokens == 10000
        assert report.active_tokens == 0
        assert report.compact_readiness_score > 0
        assert len(report.safe_to_drop) == 1
        assert report.safe_to_drop[0].category == "aged_out"

    def test_superseded_blocks_detected(self, db_session):
        sr = SessionRecord()
        sr.session_id = "dup-session"
        sr.model = "claude-opus-4-6"
        db_session.add(sr)

        # Two reads of the same file
        b1 = BlockRecord()
        b1.session_id = "dup-session"
        b1.block_id = "v1"
        b1.block_type = "tool_result"
        b1.label = "src/main.py"
        b1.tokens = 3000
        b1.enter_turn = 0
        b1.exit_turn = None
        db_session.add(b1)

        b2 = BlockRecord()
        b2.session_id = "dup-session"
        b2.block_id = "v2"
        b2.block_type = "tool_result"
        b2.label = "src/main.py"
        b2.tokens = 3200
        b2.enter_turn = 5
        b2.exit_turn = None
        db_session.add(b2)
        db_session.commit()

        report = analyze_freshness("dup-session", 10, db_session)
        assert report.stale_tokens == 3000
        superseded_blocks = [sb for sb in report.safe_to_drop if sb.category == "superseded"]
        assert len(superseded_blocks) == 1
        assert superseded_blocks[0].block_id == "v1"

    def test_turn_filter(self, db_session):
        sr = SessionRecord()
        sr.session_id = "turn-session"
        sr.model = "claude-opus-4-6"
        db_session.add(sr)

        # Block that enters at turn 10 should not appear at turn 5
        b = BlockRecord()
        b.session_id = "turn-session"
        b.block_id = "late-block"
        b.block_type = "user_prompt"
        b.label = "late"
        b.tokens = 500
        b.enter_turn = 10
        b.exit_turn = None
        db_session.add(b)
        db_session.commit()

        report = analyze_freshness("turn-session", 5, db_session)
        assert report.total_tokens == 0

    def test_estimated_savings_positive(self, db_session):
        sr = SessionRecord()
        sr.session_id = "cost-session"
        sr.model = "claude-opus-4-6"
        db_session.add(sr)

        b = BlockRecord()
        b.session_id = "cost-session"
        b.block_id = "old-b"
        b.block_type = "tool_result"
        b.label = "data.json"
        b.tokens = 50000
        b.enter_turn = 0
        b.exit_turn = None
        db_session.add(b)
        db_session.commit()

        report = analyze_freshness("cost-session", 100, db_session)
        assert report.estimated_savings_per_call > 0

    def test_safe_to_drop_excludes_exited_blocks(self, db_session):
        """safe_to_drop should only include present blocks, not exited ones."""
        sr = SessionRecord()
        sr.session_id = "drop-session"
        sr.model = "claude-opus-4-6"
        db_session.add(sr)

        # An exited block that is aged out -- should NOT be in safe_to_drop
        b1 = BlockRecord()
        b1.session_id = "drop-session"
        b1.block_id = "exited-old"
        b1.block_type = "tool_result"
        b1.label = "gone.py"
        b1.tokens = 5000
        b1.enter_turn = 0
        b1.exit_turn = 5
        db_session.add(b1)

        # A present block that is aged out -- should be in safe_to_drop
        b2 = BlockRecord()
        b2.session_id = "drop-session"
        b2.block_id = "present-old"
        b2.block_type = "tool_result"
        b2.label = "still-here.py"
        b2.tokens = 8000
        b2.enter_turn = 0
        b2.exit_turn = None
        db_session.add(b2)
        db_session.commit()

        report = analyze_freshness("drop-session", 100, db_session)
        drop_ids = {sb.block_id for sb in report.safe_to_drop}
        assert "present-old" in drop_ids
        assert "exited-old" not in drop_ids

    def test_exited_newer_does_not_supersede_present_older(self, db_session):
        """Integration test: exited newer block must not mark present older as stale."""
        sr = SessionRecord()
        sr.session_id = "supersede-session"
        sr.model = "claude-opus-4-6"
        db_session.add(sr)

        # The older block is still present
        b1 = BlockRecord()
        b1.session_id = "supersede-session"
        b1.block_id = "old-present"
        b1.block_type = "tool_result"
        b1.label = "important.py"
        b1.tokens = 3000
        b1.enter_turn = 5
        b1.exit_turn = None
        db_session.add(b1)

        # The newer block has exited -- it should NOT supersede the older one
        b2 = BlockRecord()
        b2.session_id = "supersede-session"
        b2.block_id = "new-exited"
        b2.block_type = "tool_result"
        b2.label = "important.py"
        b2.tokens = 3200
        b2.enter_turn = 10
        b2.exit_turn = 12
        db_session.add(b2)
        db_session.commit()

        report = analyze_freshness("supersede-session", 15, db_session)
        superseded = [sb for sb in report.safe_to_drop if sb.category == "superseded"]
        superseded_ids = {sb.block_id for sb in superseded}
        # The present older block should NOT be marked as superseded
        assert "old-present" not in superseded_ids


# ---------------------------------------------------------------------------
# Tests: API endpoint
# ---------------------------------------------------------------------------


class TestFreshnessEndpoint:
    @pytest.fixture()
    def client(self, tmp_path):
        """Create a test client with a transcript and DB."""
        transcript_dir = tmp_path / "transcripts"
        transcript_dir.mkdir(parents=True)
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir(parents=True)
        db_path = tmp_path / "db" / "analyzer.db"

        session_id = "test-fresh-123"
        messages = [
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello world"}],
                },
                "timestamp": "2026-06-01T10:00:00Z",
                "uuid": "u1",
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hi there."}],
                    "model": "claude-opus-4-6",
                    "usage": {
                        "input_tokens": 500,
                        "output_tokens": 30,
                        "cache_read_input_tokens": 200,
                        "cache_creation_input_tokens": 100,
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
            trace_dir=trace_dir,
            transcript_dir=transcript_dir,
            static_dir=tmp_path / "static",
            db_path=db_path,
        )
        return TestClient(app), session_id

    def test_freshness_returns_200(self, client):
        test_client, session_id = client
        resp = test_client.get(f"/api/session/{session_id}/freshness")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_tokens" in data
        assert "active_tokens" in data
        assert "stale_tokens" in data
        assert "stale_breakdown" in data
        assert "compact_readiness_score" in data
        assert "safe_to_drop" in data
        assert "estimated_savings_per_call" in data

    def test_freshness_with_turn_param(self, client):
        test_client, session_id = client
        resp = test_client.get(f"/api/session/{session_id}/freshness?turn=0")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["compact_readiness_score"], int)

    def test_freshness_not_found(self, client):
        test_client, _ = client
        resp = test_client.get("/api/session/nonexistent/freshness")
        # Should return 200 with zeros (session doesn't exist in DB but
        # the endpoint doesn't 404 for missing sessions -- it returns empty report)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_tokens"] == 0

    def test_freshness_invalid_session_id(self, client):
        test_client, _ = client
        resp = test_client.get("/api/session/invalid!id/freshness")
        assert resp.status_code == 400
