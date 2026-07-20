"""Tests for cross-session pattern detection and trend analysis."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from context_tracker.analysis.patterns import (
    _classify_direction,
    _linear_regression,
    _rolling_average,
    analyze_patterns,
    analyze_trends,
)
from context_tracker.db import Base, HookEventRecord, SessionRecord


def _day(d: int, hour: int = 10) -> str:
    """ISO timestamp for 'day d' of a recent 8-day window (d=1 oldest).

    Dates must be relative to now: analyze_trends filters by period_days,
    so hardcoded dates silently age out of the window and flip test behavior.
    """
    ts = datetime.now(UTC) - timedelta(days=9 - d)
    return ts.strftime(f"%Y-%m-%dT{hour:02d}:00:00Z")


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database with schema for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    session = factory()
    yield session
    session.close()


def _make_session(
    session_id: str,
    started_at: str,
    total_turns: int = 10,
    total_api_calls: int = 20,
    total_cost_usd: float = 1.0,
    peak_context_tokens: int = 100000,
    total_input_tokens: int = 50000,
    total_output_tokens: int = 5000,
    total_cache_read: int = 30000,
) -> SessionRecord:
    return SessionRecord(
        session_id=session_id,
        started_at=started_at,
        total_turns=total_turns,
        total_api_calls=total_api_calls,
        total_cost_usd=total_cost_usd,
        peak_context_tokens=peak_context_tokens,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_cache_read=total_cache_read,
    )


def _make_hook_event(
    session_id: str,
    event_type: str,
    tool_name: str | None = None,
) -> HookEventRecord:
    return HookEventRecord(
        session_id=session_id,
        event_type=event_type,
        tool_name=tool_name,
    )


# -------------------------------------------------------------------
# Unit tests for helper functions
# -------------------------------------------------------------------


class TestLinearRegression:
    def test_flat_line(self):
        slope, intercept = _linear_regression([5.0, 5.0, 5.0, 5.0])
        assert slope == 0.0
        assert intercept == 5.0

    def test_upward_slope(self):
        slope, _ = _linear_regression([1.0, 2.0, 3.0, 4.0, 5.0])
        assert slope == pytest.approx(1.0)

    def test_downward_slope(self):
        slope, _ = _linear_regression([5.0, 4.0, 3.0, 2.0, 1.0])
        assert slope == pytest.approx(-1.0)

    def test_single_value(self):
        slope, intercept = _linear_regression([42.0])
        assert slope == 0.0
        assert intercept == 42.0

    def test_empty(self):
        slope, intercept = _linear_regression([])
        assert slope == 0.0
        assert intercept == 0.0


class TestRollingAverage:
    def test_basic(self):
        result = _rolling_average([1.0, 2.0, 3.0, 4.0, 5.0], window=3)
        assert len(result) == 5
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(1.5)
        assert result[2] == pytest.approx(2.0)
        assert result[3] == pytest.approx(3.0)
        assert result[4] == pytest.approx(4.0)

    def test_window_larger_than_data(self):
        result = _rolling_average([10.0, 20.0], window=5)
        assert len(result) == 2
        assert result[0] == pytest.approx(10.0)
        assert result[1] == pytest.approx(15.0)

    def test_empty(self):
        assert _rolling_average([]) == []


class TestClassifyDirection:
    def test_improving_lower_is_better(self):
        direction, magnitude = _classify_direction([10.0, 8.0, 6.0, 4.0, 2.0], lower_is_better=True)
        assert direction == "improving"
        assert magnitude > 5.0

    def test_degrading_lower_is_better(self):
        direction, magnitude = _classify_direction([2.0, 4.0, 6.0, 8.0, 10.0], lower_is_better=True)
        assert direction == "degrading"
        assert magnitude > 5.0

    def test_stable(self):
        direction, magnitude = _classify_direction([5.0, 5.01, 4.99, 5.0, 5.0], lower_is_better=True)
        assert direction == "stable"

    def test_improving_higher_is_better(self):
        direction, _ = _classify_direction([2.0, 4.0, 6.0, 8.0, 10.0], lower_is_better=False)
        assert direction == "improving"

    def test_single_value(self):
        direction, magnitude = _classify_direction([5.0])
        assert direction == "stable"
        assert magnitude == 0.0


# -------------------------------------------------------------------
# Pattern detection tests
# -------------------------------------------------------------------


class TestMinimumSessionThreshold:
    def test_returns_empty_with_fewer_than_min_sessions(self, db_session):
        """With fewer than 5 sessions, analyze_patterns should return empty."""
        for i in range(4):
            db_session.add(
                _make_session(
                    f"s{i}",
                    _day(i + 1),
                    total_turns=10,
                    total_cost_usd=1.0,
                )
            )
        db_session.commit()

        patterns = analyze_patterns(db_session, min_sessions=5)
        assert patterns == []

    def test_returns_patterns_with_enough_sessions(self, db_session):
        """With 5+ sessions (varying data), patterns can be detected."""
        # Create sessions with varying turn counts to trigger sweet spot
        configs = [
            ("s0", _day(1), 5, 0.50),
            ("s1", _day(2), 5, 0.45),
            ("s2", _day(3), 15, 2.0),
            ("s3", _day(4), 15, 1.8),
            ("s4", _day(5), 35, 8.0),
            ("s5", _day(6), 35, 9.0),
        ]
        for sid, ts, turns, cost in configs:
            db_session.add(_make_session(sid, ts, total_turns=turns, total_cost_usd=cost))
        db_session.commit()

        patterns = analyze_patterns(db_session, min_sessions=5)
        # Should find at least the session_length_sweet_spot pattern
        assert len(patterns) >= 1
        names = [p.name for p in patterns]
        assert "session_length_sweet_spot" in names


class TestSessionLengthSweetSpot:
    def test_detects_sweet_spot(self, db_session):
        """Sessions with varying turn counts should reveal a sweet spot."""
        # Short sessions: cheap per turn
        for i in range(3):
            db_session.add(
                _make_session(
                    f"short-{i}",
                    _day(i + 1),
                    total_turns=5,
                    total_cost_usd=0.5,
                )
            )
        # Long sessions: expensive per turn
        for i in range(3):
            db_session.add(
                _make_session(
                    f"long-{i}",
                    _day(i + 4),
                    total_turns=35,
                    total_cost_usd=10.0,
                )
            )
        db_session.commit()

        patterns = analyze_patterns(db_session, min_sessions=5)
        sweet_spot = [p for p in patterns if p.name == "session_length_sweet_spot"]
        assert len(sweet_spot) == 1
        pattern = sweet_spot[0]
        assert "1-10" in pattern.description  # short sessions are most efficient
        assert pattern.confidence > 0
        assert pattern.confidence <= 1.0


class TestTimeOfDayPatterns:
    def test_detects_time_difference(self, db_session):
        """Morning sessions cheaper than evening -> should detect pattern."""
        # Morning sessions - cheap
        for i in range(3):
            db_session.add(
                _make_session(
                    f"morning-{i}",
                    _day(i + 1, hour=9),
                    total_cost_usd=1.0,
                )
            )
        # Evening sessions - expensive
        for i in range(3):
            db_session.add(
                _make_session(
                    f"evening-{i}",
                    _day(i + 4, hour=20),
                    total_cost_usd=5.0,
                )
            )
        db_session.commit()

        patterns = analyze_patterns(db_session, min_sessions=5)
        tod_patterns = [p for p in patterns if p.name == "time_of_day"]
        assert len(tod_patterns) == 1
        assert "morning" in tod_patterns[0].actionable.lower()

    def test_no_pattern_when_similar(self, db_session):
        """No time-of-day pattern when costs are similar across periods."""
        for i in range(6):
            hour = 9 if i < 3 else 20
            db_session.add(
                _make_session(
                    f"session-{i}",
                    _day(i + 1, hour=hour),
                    total_cost_usd=2.0,
                )
            )
        db_session.commit()

        patterns = analyze_patterns(db_session, min_sessions=5)
        tod_patterns = [p for p in patterns if p.name == "time_of_day"]
        assert len(tod_patterns) == 0


class TestCostTrajectory:
    def test_increasing_costs(self, db_session):
        """Sessions with increasing costs should be detected."""
        for i in range(6):
            db_session.add(
                _make_session(
                    f"s{i}",
                    _day(i + 1),
                    total_cost_usd=1.0 + i * 2.0,
                )
            )
        db_session.commit()

        patterns = analyze_patterns(db_session, min_sessions=5)
        cost_patterns = [p for p in patterns if p.name == "cost_trajectory"]
        assert len(cost_patterns) == 1
        assert "increasing" in cost_patterns[0].description

    def test_decreasing_costs(self, db_session):
        """Sessions with decreasing costs should be detected."""
        for i in range(6):
            db_session.add(
                _make_session(
                    f"s{i}",
                    _day(i + 1),
                    total_cost_usd=10.0 - i * 1.5,
                )
            )
        db_session.commit()

        patterns = analyze_patterns(db_session, min_sessions=5)
        cost_patterns = [p for p in patterns if p.name == "cost_trajectory"]
        assert len(cost_patterns) == 1
        assert "decreasing" in cost_patterns[0].description

    def test_stable_costs(self, db_session):
        """Sessions with stable costs should not trigger the pattern."""
        for i in range(6):
            db_session.add(
                _make_session(
                    f"s{i}",
                    _day(i + 1),
                    total_cost_usd=5.0,
                )
            )
        db_session.commit()

        patterns = analyze_patterns(db_session, min_sessions=5)
        cost_patterns = [p for p in patterns if p.name == "cost_trajectory"]
        assert len(cost_patterns) == 0


class TestErrorRateTrend:
    def test_detects_error_rate_change(self, db_session):
        """Sessions with increasing error rates should be detected."""
        for i in range(6):
            sid = f"s{i}"
            db_session.add(_make_session(sid, _day(i + 1)))
            # Add tool events: increasing failure rate
            success_count = 10 - i
            failure_count = i * 2
            for _j in range(success_count):
                db_session.add(_make_hook_event(sid, "post_tool_use", tool_name="Bash"))
            for _j in range(failure_count):
                db_session.add(_make_hook_event(sid, "post_tool_use_failure", tool_name="Bash"))
        db_session.commit()

        patterns = analyze_patterns(db_session, min_sessions=5)
        err_patterns = [p for p in patterns if p.name == "error_rate_trend"]
        assert len(err_patterns) == 1
        assert "degrading" in err_patterns[0].description


class TestToolPreferenceShifts:
    def test_detects_shift(self, db_session):
        """Tool usage change across old vs new sessions should be detected."""
        # Old sessions: heavy Bash usage
        for i in range(3):
            sid = f"old-{i}"
            db_session.add(_make_session(sid, _day(i + 1)))
            for _ in range(10):
                db_session.add(_make_hook_event(sid, "post_tool_use", tool_name="Bash"))
            for _ in range(2):
                db_session.add(_make_hook_event(sid, "post_tool_use", tool_name="Read"))

        # New sessions: heavy Read usage
        for i in range(3):
            sid = f"new-{i}"
            db_session.add(_make_session(sid, _day(i + 4)))
            for _ in range(2):
                db_session.add(_make_hook_event(sid, "post_tool_use", tool_name="Bash"))
            for _ in range(10):
                db_session.add(_make_hook_event(sid, "post_tool_use", tool_name="Read"))
        db_session.commit()

        patterns = analyze_patterns(db_session, min_sessions=5)
        shift_patterns = [p for p in patterns if p.name == "tool_preference_shift"]
        assert len(shift_patterns) == 1
        assert "shifted" in shift_patterns[0].description


# -------------------------------------------------------------------
# Trend analysis tests
# -------------------------------------------------------------------


class TestAnalyzeTrends:
    def test_returns_trends(self, db_session):
        """Should return trends for each metric with enough data."""
        for i in range(6):
            sid = f"s{i}"
            db_session.add(
                _make_session(
                    sid,
                    _day(i + 1),
                    total_turns=10 + i,
                    total_cost_usd=1.0 + i * 0.5,
                )
            )
            # Add some tool events for error_rate metric
            for _ in range(5):
                db_session.add(_make_hook_event(sid, "post_tool_use", tool_name="Bash"))
            db_session.add(_make_hook_event(sid, "post_tool_use_failure", tool_name="Bash"))
        db_session.commit()

        trends = analyze_trends(db_session, period_days=30)
        assert len(trends) >= 3

        metrics = {t.metric for t in trends}
        assert "cost" in metrics
        assert "turns" in metrics
        assert "efficiency" in metrics

        for t in trends:
            assert t.direction in ("improving", "stable", "degrading")
            assert t.data_points > 0
            assert len(t.values) > 0
            assert t.period == "last 30 days"

    def test_too_few_sessions(self, db_session):
        """With only 1 session, trends should be empty."""
        db_session.add(_make_session("s0", _day(1)))
        db_session.commit()

        trends = analyze_trends(db_session, period_days=30)
        assert trends == []

    def test_improving_cost_trend(self, db_session):
        """Decreasing costs should be classified as improving."""
        for i in range(6):
            db_session.add(
                _make_session(
                    f"s{i}",
                    _day(i + 1),
                    total_cost_usd=10.0 - i * 1.5,
                    total_turns=10,
                )
            )
        db_session.commit()

        trends = analyze_trends(db_session, period_days=30)
        cost_trend = [t for t in trends if t.metric == "cost"]
        assert len(cost_trend) == 1
        assert cost_trend[0].direction == "improving"

    def test_degrading_cost_trend(self, db_session):
        """Increasing costs should be classified as degrading."""
        for i in range(6):
            db_session.add(
                _make_session(
                    f"s{i}",
                    _day(i + 1),
                    total_cost_usd=1.0 + i * 3.0,
                    total_turns=10,
                )
            )
        db_session.commit()

        trends = analyze_trends(db_session, period_days=30)
        cost_trend = [t for t in trends if t.metric == "cost"]
        assert len(cost_trend) == 1
        assert cost_trend[0].direction == "degrading"

    def test_fallback_period_label(self, db_session):
        """When fewer than 3 sessions in the period, label should say 'all time'."""
        # Sessions well outside the 7-day window
        for i in range(4):
            db_session.add(
                _make_session(
                    f"s{i}",
                    f"2025-01-0{i + 1}T10:00:00Z",
                    total_cost_usd=1.0 + i * 0.5,
                    total_turns=10,
                )
            )
        db_session.commit()

        trends = analyze_trends(db_session, period_days=7)
        assert len(trends) >= 1
        for t in trends:
            assert "all time" in t.period
            assert "4 sessions" in t.period

    def test_sparkline_values(self, db_session):
        """Trend values should contain rolling averages for sparkline rendering."""
        for i in range(8):
            db_session.add(
                _make_session(
                    f"s{i}",
                    _day(i + 1),
                    total_cost_usd=float(i + 1),
                    total_turns=10,
                )
            )
        db_session.commit()

        trends = analyze_trends(db_session, period_days=30)
        cost_trend = next(t for t in trends if t.metric == "cost")
        assert len(cost_trend.values) == 8
        # Rolling averages should be smoother than raw values
        # First value should equal the first raw value
        assert cost_trend.values[0] == pytest.approx(1.0, abs=0.01)


# -------------------------------------------------------------------
# Dashboard API endpoint tests
# -------------------------------------------------------------------


class TestDashboardEndpoints:
    @pytest.fixture
    def client(self, tmp_path):
        from fastapi.testclient import TestClient

        from context_tracker.dashboard import create_app

        db_path = tmp_path / "test.db"
        app = create_app(
            trace_dir=tmp_path / "traces",
            transcript_dir=tmp_path / "transcripts",
            static_dir=tmp_path / "static",
            db_path=db_path,
        )
        return TestClient(app), db_path

    def test_patterns_endpoint_empty(self, client):
        test_client, _ = client
        resp = test_client.get("/api/sessions/patterns")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_trends_detailed_endpoint_empty(self, client):
        test_client, _ = client
        resp = test_client.get("/api/sessions/trends/detailed")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_patterns_endpoint_with_data(self, client):
        test_client, db_path = client
        # Seed database directly
        from context_tracker.db import get_engine, get_session_factory

        engine = get_engine(db_path)
        factory = get_session_factory(engine)
        with factory() as db:
            configs = [
                ("s0", _day(1), 5, 0.50),
                ("s1", _day(2), 5, 0.45),
                ("s2", _day(3), 15, 2.0),
                ("s3", _day(4), 15, 1.8),
                ("s4", _day(5), 35, 8.0),
                ("s5", _day(6), 35, 9.0),
            ]
            for sid, ts, turns, cost in configs:
                db.add(
                    SessionRecord(
                        session_id=sid,
                        started_at=ts,
                        total_turns=turns,
                        total_cost_usd=cost,
                    )
                )
            db.commit()

        resp = test_client.get("/api/sessions/patterns")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        pattern = data[0]
        assert "name" in pattern
        assert "description" in pattern
        assert "confidence" in pattern
        assert "actionable" in pattern

    def test_trends_detailed_endpoint_with_data(self, client):
        test_client, db_path = client
        from context_tracker.db import get_engine, get_session_factory

        engine = get_engine(db_path)
        factory = get_session_factory(engine)
        with factory() as db:
            for i in range(6):
                db.add(
                    SessionRecord(
                        session_id=f"s{i}",
                        started_at=_day(i + 1),
                        total_turns=10,
                        total_cost_usd=1.0 + i * 0.5,
                    )
                )
            db.commit()

        resp = test_client.get("/api/sessions/trends/detailed?period=30")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 2
        for trend in data:
            assert "metric" in trend
            assert "direction" in trend
            assert "values" in trend
            assert trend["direction"] in ("improving", "stable", "degrading")
