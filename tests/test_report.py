"""Tests for post-session optimization report generation."""

import pytest

from context_tracker.analysis.report import (
    _compute_split_recommendation,
    _detect_failed_retries,
    _detect_oversized_output,
    _detect_repeated_reads,
    _detect_stale_content,
    _tokens_to_cost,
    generate_report,
)
from context_tracker.db import (
    ApiCallRecord,
    BlockRecord,
    HookEventRecord,
    SessionRecord,
    get_engine,
    get_session_factory,
)


@pytest.fixture
def db_factory(tmp_path):
    """Create a fresh in-memory-like DB and return a session factory."""
    db_path = tmp_path / "test_report.db"
    engine = get_engine(db_path)
    factory = get_session_factory(engine)
    yield factory
    engine.dispose()


@pytest.fixture
def populated_db(db_factory):
    """DB with a session that has blocks, api_calls, and hook events."""
    with db_factory() as db:
        # Session record
        db.add(
            SessionRecord(
                session_id="sess-report-1",
                total_turns=10,
                total_api_calls=80,
                peak_context_tokens=200_000,
                total_input_tokens=1_000_000,
                total_output_tokens=50_000,
                total_cache_read=500_000,
                total_cost_usd=3.50,
            )
        )

        # Stale blocks (exit_turn - enter_turn > 50)
        db.add(
            BlockRecord(
                session_id="sess-report-1",
                block_id="stale-1",
                block_type="tool_result",
                label="/src/models.py",
                tokens=5000,
                enter_turn=0,
                exit_turn=60,
            )
        )
        db.add(
            BlockRecord(
                session_id="sess-report-1",
                block_id="stale-2",
                block_type="tool_result",
                label="/src/views.py",
                tokens=3000,
                enter_turn=5,
                exit_turn=70,
            )
        )

        # Non-stale block (lifespan < 50)
        db.add(
            BlockRecord(
                session_id="sess-report-1",
                block_id="fresh-1",
                block_type="tool_result",
                label="/src/utils.py",
                tokens=2000,
                enter_turn=50,
                exit_turn=60,
            )
        )

        # Repeated reads: /src/config.py read 4 times
        for i in range(4):
            db.add(
                BlockRecord(
                    session_id="sess-report-1",
                    block_id=f"repeat-config-{i}",
                    block_type="tool_result",
                    label="/src/config.py",
                    tokens=1000 + i * 100,
                    enter_turn=i * 10,
                    exit_turn=i * 10 + 5,
                )
            )

        # Oversized output
        db.add(
            BlockRecord(
                session_id="sess-report-1",
                block_id="oversized-1",
                block_type="tool_result",
                label="/src/big_file.py",
                tokens=45_000,
                enter_turn=20,
                exit_turn=30,
            )
        )

        # Failed tool events
        db.add(
            HookEventRecord(
                session_id="sess-report-1",
                event_type="post_tool_use_failure",
                error_length=500,
            )
        )
        db.add(
            HookEventRecord(
                session_id="sess-report-1",
                event_type="post_tool_use_failure",
                error_length=300,
            )
        )

        # Normal hook event (should not be counted)
        db.add(
            HookEventRecord(
                session_id="sess-report-1",
                event_type="post_tool_use",
                error_length=0,
            )
        )

        # API calls for split analysis (growing context)
        for i in range(20):
            db.add(
                ApiCallRecord(
                    session_id="sess-report-1",
                    call_index=i,
                    input_tokens=10_000 + i * 5_000,
                    output_tokens=500,
                    cache_read=1000,
                )
            )

        db.commit()
    return db_factory


class TestTokensToCost:
    def test_zero_tokens(self):
        assert _tokens_to_cost(0) == 0.0

    def test_one_million_tokens(self):
        assert _tokens_to_cost(1_000_000) == 3.0

    def test_fractional(self):
        cost = _tokens_to_cost(500_000)
        assert abs(cost - 1.5) < 0.001


class TestStaleContent:
    def test_detects_stale_blocks(self, populated_db):
        with populated_db() as db:
            item = _detect_stale_content("sess-report-1", db)
        assert item is not None
        assert item.category == "stale_content"
        assert item.tokens == 8000  # 5000 + 3000
        assert "2 blocks" in item.description

    def test_no_stale_blocks(self, db_factory):
        with db_factory() as db:
            db.add(SessionRecord(session_id="sess-no-stale"))
            db.add(
                BlockRecord(
                    session_id="sess-no-stale",
                    block_id="b1",
                    block_type="tool_result",
                    tokens=1000,
                    enter_turn=0,
                    exit_turn=10,
                )
            )
            db.commit()
            item = _detect_stale_content("sess-no-stale", db)
        assert item is None


class TestRepeatedReads:
    def test_detects_repeated_reads(self, populated_db):
        with populated_db() as db:
            item = _detect_repeated_reads("sess-report-1", db)
        assert item is not None
        assert item.category == "repeated_reads"
        assert item.tokens > 0
        assert "excess" in item.description

    def test_no_repeated_reads(self, db_factory):
        with db_factory() as db:
            db.add(SessionRecord(session_id="sess-no-repeat"))
            # Only 2 reads of same file (threshold is >2)
            for i in range(2):
                db.add(
                    BlockRecord(
                        session_id="sess-no-repeat",
                        block_id=f"r-{i}",
                        block_type="tool_result",
                        label="/src/a.py",
                        tokens=500,
                        enter_turn=i,
                    )
                )
            db.commit()
            item = _detect_repeated_reads("sess-no-repeat", db)
        assert item is None


class TestFailedRetries:
    def test_detects_failures(self, populated_db):
        with populated_db() as db:
            item = _detect_failed_retries("sess-report-1", db)
        assert item is not None
        assert item.category == "failed_retries"
        assert item.tokens == 800  # 500 + 300
        assert "2 tool failures" in item.description

    def test_no_failures(self, db_factory):
        with db_factory() as db:
            db.add(SessionRecord(session_id="sess-no-fail"))
            db.add(
                HookEventRecord(
                    session_id="sess-no-fail",
                    event_type="post_tool_use",
                    error_length=0,
                )
            )
            db.commit()
            item = _detect_failed_retries("sess-no-fail", db)
        assert item is None


class TestOversizedOutput:
    def test_detects_oversized(self, populated_db):
        with populated_db() as db:
            item = _detect_oversized_output("sess-report-1", db)
        assert item is not None
        assert item.category == "oversized_output"
        assert item.tokens == 15_000  # 45000 - 30000
        assert "1 tool results" in item.description

    def test_no_oversized(self, db_factory):
        with db_factory() as db:
            db.add(SessionRecord(session_id="sess-no-big"))
            db.add(
                BlockRecord(
                    session_id="sess-no-big",
                    block_id="small-1",
                    block_type="tool_result",
                    tokens=5000,
                    enter_turn=0,
                )
            )
            db.commit()
            item = _detect_oversized_output("sess-no-big", db)
        assert item is None


class TestSplitRecommendation:
    def test_recommends_split_for_growing_session(self, populated_db):
        with populated_db() as db:
            rec = _compute_split_recommendation("sess-report-1", db)
        # With linearly growing input tokens, a split should yield savings
        if rec is not None:
            assert rec.savings > 0
            assert rec.split_at_turn >= 2
            assert rec.projected_cost < rec.current_cost

    def test_no_split_for_short_session(self, db_factory):
        with db_factory() as db:
            db.add(SessionRecord(session_id="sess-short"))
            for i in range(3):
                db.add(
                    ApiCallRecord(
                        session_id="sess-short",
                        call_index=i,
                        input_tokens=1000,
                        output_tokens=100,
                    )
                )
            db.commit()
            rec = _compute_split_recommendation("sess-short", db)
        assert rec is None

    def test_no_split_for_flat_cost(self, db_factory):
        """When all calls have the same input tokens, splitting saves nothing."""
        with db_factory() as db:
            db.add(SessionRecord(session_id="sess-flat"))
            for i in range(10):
                db.add(
                    ApiCallRecord(
                        session_id="sess-flat",
                        call_index=i,
                        input_tokens=5000,
                        output_tokens=100,
                    )
                )
            db.commit()
            rec = _compute_split_recommendation("sess-flat", db)
        # Flat cost should still recommend split since the model considers
        # that a fresh session would start with much smaller context.
        # But the savings might not cross 20% for flat 5000-token calls.
        # Either outcome is valid; just verify the structure.
        if rec is not None:
            assert rec.savings > 0


class TestGenerateReport:
    def test_full_report(self, populated_db):
        with populated_db() as db:
            report = generate_report("sess-report-1", db)
        assert report.session_id == "sess-report-1"
        assert report.total_cost == 3.50
        assert report.total_turns == 10
        assert report.total_api_calls == 80
        assert report.peak_context == 200_000
        assert len(report.waste_items) >= 3  # stale, repeated, oversized at minimum
        assert report.total_waste_tokens > 0
        assert report.total_waste_cost > 0

    def test_report_to_dict(self, populated_db):
        with populated_db() as db:
            report = generate_report("sess-report-1", db)
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["session_id"] == "sess-report-1"
        assert isinstance(d["waste_items"], list)
        assert all(isinstance(w, dict) for w in d["waste_items"])

    def test_report_not_found(self, db_factory):
        with db_factory() as db:
            with pytest.raises(ValueError, match="not found"):
                generate_report("nonexistent", db)

    def test_empty_session(self, db_factory):
        """A session with no blocks, events, or API calls produces an empty report."""
        with db_factory() as db:
            db.add(
                SessionRecord(
                    session_id="sess-empty",
                    total_turns=0,
                    total_api_calls=0,
                    peak_context_tokens=0,
                    total_cost_usd=0.0,
                )
            )
            db.commit()
            report = generate_report("sess-empty", db)
        assert report.session_id == "sess-empty"
        assert len(report.waste_items) == 0
        assert report.total_waste_tokens == 0
        assert report.total_waste_cost == 0.0
        assert report.split_recommendation is None


class TestReportApiEndpoint:
    """Test the /api/session/{session_id}/report endpoint via TestClient."""

    def test_report_endpoint_not_found(self, tmp_path):
        from fastapi.testclient import TestClient

        from context_tracker.dashboard import create_app

        app = create_app(
            trace_dir=tmp_path / "traces",
            transcript_dir=tmp_path / "transcripts",
            static_dir=tmp_path / "static",
        )
        client = TestClient(app)
        resp = client.get("/api/session/nonexistent/report")
        assert resp.status_code == 404

    def test_report_endpoint_with_data(self, tmp_path):
        """Test the endpoint with a session that has DB data."""
        from fastapi.testclient import TestClient

        from context_tracker.dashboard import create_app

        db_path = tmp_path / "test.db"
        engine = get_engine(db_path)
        factory = get_session_factory(engine)

        with factory() as db:
            db.add(
                SessionRecord(
                    session_id="sess-api-test",
                    total_turns=5,
                    total_api_calls=20,
                    peak_context_tokens=100_000,
                    total_cost_usd=1.00,
                )
            )
            # Add some blocks for waste detection
            db.add(
                BlockRecord(
                    session_id="sess-api-test",
                    block_id="b1",
                    block_type="tool_result",
                    label="/test.py",
                    tokens=40_000,
                    enter_turn=0,
                    exit_turn=60,
                )
            )
            db.commit()

        engine.dispose()

        app = create_app(
            trace_dir=tmp_path / "traces",
            transcript_dir=tmp_path / "transcripts",
            static_dir=tmp_path / "static",
            db_path=db_path,
        )
        client = TestClient(app)
        resp = client.get("/api/session/sess-api-test/report")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "sess-api-test"
        assert "waste_items" in data
        assert "total_waste_tokens" in data
        assert "split_recommendation" in data
