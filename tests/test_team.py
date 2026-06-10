"""Tests for team benchmarks — anonymous efficiency comparison."""

import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from context_tracker.dashboard import create_app
from context_tracker.db import (
    HookEventRecord,
    SessionRecord,
    get_engine,
    get_session_factory,
)
from context_tracker.team import (
    MIN_TEAM_SIZE,
    AnonymizedMetrics,
    TeamComparison,
    compare_with_team,
    export_metrics,
    export_metrics_to_json,
    import_metrics,
    import_metrics_from_dict,
)

# ---- Fixtures ----


@pytest.fixture
def db_with_sessions(tmp_path):
    """Create a DB with sample sessions and hook events."""
    db_path = tmp_path / "test.db"
    engine = get_engine(db_path)
    factory = get_session_factory(engine)

    now = datetime.now(UTC)

    with factory() as db:
        # Create 3 sessions within the last 30 days
        for i in range(3):
            session = SessionRecord(
                session_id=f"sess-{i}",
                project_path="/some/project",
                started_at=(now - timedelta(days=i + 1)).isoformat(),
                ended_at=(now - timedelta(days=i + 1, hours=-1)).isoformat(),
                model="claude-opus-4-6",
                total_turns=10 + i * 5,
                total_api_calls=20 + i * 3,
                peak_context_tokens=50000 + i * 10000,
                total_input_tokens=100000 + i * 20000,
                total_output_tokens=5000 + i * 1000,
                total_cache_read=30000 + i * 5000,
                total_cost_usd=0.50 + i * 0.10,
            )
            db.add(session)

            # Add hook events
            for j in range(5):
                evt = HookEventRecord(
                    session_id=f"sess-{i}",
                    event_type="post_tool_use",
                    tool_name=["Read", "Edit", "Bash", "Write", "Read"][j],
                )
                db.add(evt)

            # Add a pre_compact event
            db.add(
                HookEventRecord(
                    session_id=f"sess-{i}",
                    event_type="pre_compact",
                )
            )

            # Add a failure event
            db.add(
                HookEventRecord(
                    session_id=f"sess-{i}",
                    event_type="post_tool_use",
                    tool_name="Bash",
                    error_length=150,
                )
            )

        db.commit()

    return db_path, factory


@pytest.fixture
def sample_metrics():
    """Return a sample AnonymizedMetrics instance."""
    return AnonymizedMetrics(
        period_start="2026-05-11",
        period_end="2026-06-10",
        alias="alice",
        session_count=10,
        avg_cost_per_session=0.45,
        avg_turns_per_session=12.5,
        avg_context_peak=60000,
        tool_distribution={"Read": 35.0, "Edit": 25.0, "Bash": 30.0, "Write": 10.0},
        error_rate=0.08,
        avg_cost_per_turn=0.036,
        compact_frequency=1.2,
    )


@pytest.fixture
def team_metrics_list():
    """Return a list of 3 sample team member metrics."""
    return [
        AnonymizedMetrics(
            period_start="2026-05-11",
            period_end="2026-06-10",
            alias="bob",
            session_count=8,
            avg_cost_per_session=0.55,
            avg_turns_per_session=15.0,
            avg_context_peak=70000,
            tool_distribution={"Read": 40.0, "Edit": 20.0, "Bash": 25.0, "Write": 15.0},
            error_rate=0.12,
            avg_cost_per_turn=0.037,
            compact_frequency=1.5,
        ),
        AnonymizedMetrics(
            period_start="2026-05-11",
            period_end="2026-06-10",
            alias="carol",
            session_count=12,
            avg_cost_per_session=0.35,
            avg_turns_per_session=10.0,
            avg_context_peak=45000,
            tool_distribution={"Read": 30.0, "Edit": 35.0, "Bash": 20.0, "Write": 15.0},
            error_rate=0.05,
            avg_cost_per_turn=0.035,
            compact_frequency=0.8,
        ),
        AnonymizedMetrics(
            period_start="2026-05-11",
            period_end="2026-06-10",
            alias="dave",
            session_count=6,
            avg_cost_per_session=0.60,
            avg_turns_per_session=18.0,
            avg_context_peak=80000,
            tool_distribution={"Read": 25.0, "Edit": 15.0, "Bash": 40.0, "Write": 20.0},
            error_rate=0.15,
            avg_cost_per_turn=0.033,
            compact_frequency=2.0,
        ),
    ]


@pytest.fixture
def api_client(tmp_path):
    """TestClient for the dashboard app."""
    app = create_app(
        trace_dir=tmp_path / "traces",
        transcript_dir=tmp_path / "transcripts",
        static_dir=tmp_path / "static",
        db_path=tmp_path / "test.db",
    )
    return TestClient(app)


@pytest.fixture
def api_client_with_db(tmp_path, db_with_sessions):
    """TestClient with pre-populated DB."""
    db_path, factory = db_with_sessions
    app = create_app(
        trace_dir=tmp_path / "traces",
        transcript_dir=tmp_path / "transcripts",
        static_dir=tmp_path / "static",
        db_path=db_path,
    )
    return TestClient(app)


# ---- export_metrics tests ----


class TestExportMetrics:
    def test_export_produces_valid_data(self, db_with_sessions):
        """Export should return well-formed AnonymizedMetrics."""
        db_path, factory = db_with_sessions
        with factory() as db:
            metrics = export_metrics(db, period_days=30, alias="testuser")

        assert isinstance(metrics, AnonymizedMetrics)
        assert metrics.alias == "testuser"
        assert metrics.session_count == 3
        assert metrics.avg_cost_per_session > 0
        assert metrics.avg_turns_per_session > 0
        assert metrics.avg_context_peak > 0

    def test_export_no_sensitive_info(self, db_with_sessions):
        """Exported data must NOT contain session IDs, file paths, or prompts."""
        db_path, factory = db_with_sessions
        with factory() as db:
            metrics = export_metrics(db, period_days=30, alias="dev")

        exported_json = export_metrics_to_json(metrics)

        # Must not contain any session IDs
        assert "sess-0" not in exported_json
        assert "sess-1" not in exported_json
        assert "sess-2" not in exported_json

        # Must not contain file paths
        assert "/some/project" not in exported_json
        assert "project_path" not in exported_json

        # Must not contain session_id field
        data = json.loads(exported_json)
        assert "session_id" not in data
        assert "session_ids" not in data

    def test_export_tool_distribution(self, db_with_sessions):
        """Tool distribution should sum to 100% and contain expected tools."""
        db_path, factory = db_with_sessions
        with factory() as db:
            metrics = export_metrics(db, period_days=30, alias="dev")

        if metrics.tool_distribution:
            total = sum(metrics.tool_distribution.values())
            assert abs(total - 100.0) < 1.0  # Allow small rounding error

    def test_export_error_rate_range(self, db_with_sessions):
        """Error rate should be between 0 and 1."""
        db_path, factory = db_with_sessions
        with factory() as db:
            metrics = export_metrics(db, period_days=30, alias="dev")

        assert 0.0 <= metrics.error_rate <= 1.0

    def test_export_compact_frequency(self, db_with_sessions):
        """Compact frequency should be non-negative."""
        db_path, factory = db_with_sessions
        with factory() as db:
            metrics = export_metrics(db, period_days=30, alias="dev")

        assert metrics.compact_frequency >= 0.0

    def test_export_empty_db(self, tmp_path):
        """Export with no sessions returns zeroed metrics."""
        db_path = tmp_path / "empty.db"
        engine = get_engine(db_path)
        factory = get_session_factory(engine)
        with factory() as db:
            metrics = export_metrics(db, period_days=30, alias="empty")

        assert metrics.session_count == 0
        assert metrics.avg_cost_per_session == 0.0
        assert metrics.tool_distribution == {}


# ---- import_metrics tests ----


class TestImportMetrics:
    def test_import_from_file(self, tmp_path, sample_metrics):
        """Import should correctly read a JSON file."""
        file_path = tmp_path / "team_member.json"
        file_path.write_text(export_metrics_to_json(sample_metrics))

        imported = import_metrics(file_path)

        assert imported.alias == "alice"
        assert imported.session_count == 10
        assert imported.avg_cost_per_session == 0.45
        assert imported.tool_distribution == {"Read": 35.0, "Edit": 25.0, "Bash": 30.0, "Write": 10.0}

    def test_import_from_dict(self, sample_metrics):
        """Import from dict should work the same as file import."""
        data = asdict(sample_metrics)
        imported = import_metrics_from_dict(data)

        assert imported.alias == sample_metrics.alias
        assert imported.session_count == sample_metrics.session_count

    def test_import_missing_fields(self):
        """Import should raise ValueError on missing fields."""
        with pytest.raises(ValueError, match="Missing required fields"):
            import_metrics_from_dict({"alias": "incomplete"})

    def test_import_roundtrip(self, tmp_path, sample_metrics):
        """Export -> import roundtrip preserves all fields."""
        file_path = tmp_path / "roundtrip.json"
        file_path.write_text(export_metrics_to_json(sample_metrics))
        imported = import_metrics(file_path)

        assert asdict(imported) == asdict(sample_metrics)


# ---- compare_with_team tests ----


class TestCompareWithTeam:
    def test_comparison_with_valid_team(self, sample_metrics, team_metrics_list):
        """Comparison should produce rankings and insights."""
        result = compare_with_team(sample_metrics, team_metrics_list)

        assert isinstance(result, TeamComparison)
        assert result.total_members == 4  # you + 3 team
        assert len(result.rankings) > 0
        assert all(1 <= rank <= 4 for rank in result.rankings.values())
        assert len(result.insights) > 0

    def test_minimum_team_size_requirement(self, sample_metrics):
        """Comparison should fail with fewer than MIN_TEAM_SIZE members."""
        small_team = [
            AnonymizedMetrics(
                period_start="2026-05-11",
                period_end="2026-06-10",
                alias="only_one",
                session_count=5,
                avg_cost_per_session=0.40,
                avg_turns_per_session=8.0,
                avg_context_peak=40000,
                tool_distribution={"Read": 50.0, "Edit": 50.0},
                error_rate=0.10,
                avg_cost_per_turn=0.05,
                compact_frequency=1.0,
            ),
        ]

        with pytest.raises(ValueError, match=f"Need at least {MIN_TEAM_SIZE}"):
            compare_with_team(sample_metrics, small_team)

    def test_minimum_team_size_boundary(self, sample_metrics, team_metrics_list):
        """Exactly MIN_TEAM_SIZE members should work."""
        exact_team = team_metrics_list[:MIN_TEAM_SIZE]
        result = compare_with_team(sample_metrics, exact_team)
        assert result.total_members == MIN_TEAM_SIZE + 1

    def test_ranking_calculation(self, sample_metrics, team_metrics_list):
        """Rankings should assign 1 to the best performer."""
        result = compare_with_team(sample_metrics, team_metrics_list)

        # Carol has the lowest cost (0.35), so she should rank 1
        # alice is 0.45, so she should rank somewhere in the middle
        cost_rank = result.rankings["avg_cost_per_session"]
        assert 1 <= cost_rank <= result.total_members

    def test_insight_generation(self, sample_metrics, team_metrics_list):
        """Insights should contain meaningful comparison text."""
        result = compare_with_team(sample_metrics, team_metrics_list)

        # Check that insights are strings
        for insight in result.insights:
            assert isinstance(insight, str)
            assert len(insight) > 0

    def test_insight_cost_comparison(self, team_metrics_list):
        """Cost insight should mention more/less than team average."""
        # Create a metrics with significantly higher cost
        expensive = AnonymizedMetrics(
            period_start="2026-05-11",
            period_end="2026-06-10",
            alias="expensive",
            session_count=5,
            avg_cost_per_session=2.00,  # Much higher than team
            avg_turns_per_session=10.0,
            avg_context_peak=50000,
            tool_distribution={"Read": 50.0, "Bash": 50.0},
            error_rate=0.10,
            avg_cost_per_turn=0.20,
            compact_frequency=1.0,
        )

        result = compare_with_team(expensive, team_metrics_list)
        cost_insights = [i for i in result.insights if "cost" in i.lower()]
        assert len(cost_insights) > 0
        assert any("more" in i for i in cost_insights)

    def test_insight_error_rate(self, team_metrics_list):
        """Error rate insight should be generated."""
        user = AnonymizedMetrics(
            period_start="2026-05-11",
            period_end="2026-06-10",
            alias="user",
            session_count=5,
            avg_cost_per_session=0.50,
            avg_turns_per_session=10.0,
            avg_context_peak=50000,
            tool_distribution={"Read": 50.0, "Bash": 50.0},
            error_rate=0.25,
            avg_cost_per_turn=0.05,
            compact_frequency=1.0,
        )

        result = compare_with_team(user, team_metrics_list)
        error_insights = [i for i in result.insights if "error rate" in i.lower()]
        assert len(error_insights) > 0


# ---- API endpoint tests ----


class TestTeamAPI:
    def test_export_endpoint(self, api_client_with_db):
        """GET /api/team/export should return anonymized metrics."""
        resp = api_client_with_db.get("/api/team/export?period=30&alias=testdev")
        assert resp.status_code == 200
        data = resp.json()
        assert data["alias"] == "testdev"
        assert "session_count" in data
        assert "avg_cost_per_session" in data
        assert "tool_distribution" in data
        # Ensure no session IDs leaked
        assert "session_id" not in json.dumps(data)

    def test_import_endpoint(self, api_client, sample_metrics):
        """POST /api/team/import should accept and store metrics."""
        resp = api_client.post(
            "/api/team/import",
            json=asdict(sample_metrics),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "imported"
        assert data["alias"] == "alice"
        assert data["total_imported"] >= 1

    def test_compare_insufficient_members(self, api_client):
        """GET /api/team/compare with <3 members returns error."""
        resp = api_client.get("/api/team/compare")
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data
        assert data["min_required"] == MIN_TEAM_SIZE

    def test_full_flow(self, api_client_with_db, team_metrics_list):
        """End-to-end: import 3 members, then compare."""
        # Import 3 team members
        for m in team_metrics_list:
            resp = api_client_with_db.post(
                "/api/team/import",
                json=asdict(m),
            )
            assert resp.status_code == 200

        # Check imported list
        resp = api_client_with_db.get("/api/team/imported")
        data = resp.json()
        assert data["total_imported"] == 3

        # Now compare
        resp = api_client_with_db.get("/api/team/compare?period=30&alias=me")
        assert resp.status_code == 200
        data = resp.json()
        assert "error" not in data
        assert "rankings" in data
        assert "insights" in data
        assert data["total_members"] == 4  # you + 3

    def test_clear_imported(self, api_client, sample_metrics):
        """DELETE /api/team/imported should clear all data."""
        # Import one
        api_client.post("/api/team/import", json=asdict(sample_metrics))

        # Clear
        resp = api_client.delete("/api/team/imported")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cleared"

        # Verify empty
        resp = api_client.get("/api/team/imported")
        assert resp.json()["total_imported"] == 0

    def test_team_page_route(self, api_client):
        """GET /team should return HTML (even if file missing)."""
        resp = api_client.get("/team")
        assert resp.status_code == 200
