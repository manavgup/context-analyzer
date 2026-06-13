"""Tests for prompt pattern detection and specificity scoring."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from context_tracker.analysis.prompts import (
    PromptAnalysis,
    analyze_session_prompts,
    classify_prompt,
    compute_aggregate_stats,
)
from context_tracker.db import (
    ApiCallRecord,
    Base,
    SessionRecord,
    TurnRecord,
    get_engine,
    get_session_factory,
)

# ---------------------------------------------------------------------------
# classify_prompt unit tests
# ---------------------------------------------------------------------------


class TestClassifyPrompt:
    """Tests for the classify_prompt heuristic scorer."""

    def test_empty_prompt_scores_zero(self):
        score, signals = classify_prompt("")
        assert score == 0.0
        assert signals == []

    def test_whitespace_prompt_scores_zero(self):
        score, signals = classify_prompt("   \n\t  ")
        assert score == 0.0
        assert signals == []

    def test_specific_prompt_scores_high(self):
        """A prompt with file path, line number, function name, and constraints should score > 0.6."""
        prompt = (
            "Fix the bug in src/context_tracker/dashboard.py:123 where "
            "render_chart() throws 'TypeError: undefined is not a function'. "
            "Don't change the existing API interface."
        )
        score, signals = classify_prompt(prompt)
        assert score > 0.6
        assert "file_path" in signals
        assert "constraint_language" in signals

    def test_vague_prompt_scores_low(self):
        """Short vague prompts should score < 0.3."""
        score, signals = classify_prompt("Fix this")
        assert score < 0.3

    def test_vague_prompt_make_it_work(self):
        score, signals = classify_prompt("Make it work")
        assert score < 0.3

    def test_vague_prompt_looks_wrong(self):
        score, signals = classify_prompt("It looks wrong")
        assert score < 0.3

    def test_file_path_detection(self):
        """File paths like foo/bar.py should trigger the file_path signal."""
        score, signals = classify_prompt("Look at src/module.py")
        assert "file_path" in signals

    def test_file_path_with_line(self):
        score, signals = classify_prompt("Check src/module.py:42")
        assert "file_path" in signals
        assert "line_number" in signals

    def test_function_name_detection(self):
        """Function references like snake_case_func() should trigger."""
        score, signals = classify_prompt("The issue is in process_data() when it receives None")
        assert "function_name" in signals

    def test_class_name_detection(self):
        """PascalCase class names should trigger function_name signal."""
        score, signals = classify_prompt("ContextBlock doesn't have the right attribute")
        assert "function_name" in signals

    def test_ordinary_capitalized_word_no_function_signal(self):
        """Ordinary capitalized words like 'Please' should NOT trigger function_name."""
        score, signals = classify_prompt("Please fix this issue")
        assert "function_name" not in signals

    def test_sentence_start_capitals_no_function_signal(self):
        """Sentence-starting capitals should NOT trigger function_name."""
        score, signals = classify_prompt("Update the header component")
        assert "function_name" not in signals

    def test_backticked_identifier_detection(self):
        """Backticked identifiers like `functionName` should trigger function_name signal."""
        score, signals = classify_prompt("The issue is in `process_data` when it receives None")
        assert "function_name" in signals

    def test_qualified_access_detection(self):
        """Qualified access like ClassName.method should trigger function_name signal."""
        score, signals = classify_prompt("Check Response.status_code for the error")
        assert "function_name" in signals

    def test_error_message_detection(self):
        """Quoted error messages should trigger error_message signal."""
        score, signals = classify_prompt('I see "TypeError: cannot read property" in the console')
        assert "error_message" in signals

    def test_error_message_with_backticks(self):
        score, signals = classify_prompt("Getting `FileNotFoundError: No such file` when running tests")
        assert "error_message" in signals

    def test_constraint_language_dont(self):
        """Constraint language like 'don't' should trigger."""
        score, signals = classify_prompt("Update the header but don't change the footer")
        assert "constraint_language" in signals

    def test_constraint_language_without_changing(self):
        score, signals = classify_prompt("Refactor this without changing the public API")
        assert "constraint_language" in signals

    def test_constraint_language_only_modify(self):
        score, signals = classify_prompt("Only modify the test file, leave production code alone")
        assert "constraint_language" in signals

    def test_length_signal_short(self):
        """Prompts under 100 chars should not get length signal."""
        score, signals = classify_prompt("Fix bug")
        assert "length_signal" not in signals

    def test_length_signal_long(self):
        """Prompts over 100 chars should get length signal."""
        prompt = "a " * 60  # 120 chars
        score, signals = classify_prompt(prompt)
        assert "length_signal" in signals

    def test_score_capped_at_one(self):
        """Even with all signals present, score should not exceed 1.0."""
        prompt = (
            "Fix the bug in src/context_tracker/analysis/prompts.py:42 where "
            "classify_prompt() throws 'ValueError: invalid signal weight'. "
            "Don't change the test file. Only modify the scoring logic. "
            "The error appears on line 42 when processing L42. "
            "Keep the existing API interface intact."
        )
        score, signals = classify_prompt(prompt)
        assert score <= 1.0
        assert score > 0.6  # should be highly specific

    def test_multiple_signals_accumulate(self):
        """Multiple signals should add up."""
        # Just file_path
        score1, _ = classify_prompt("Check src/foo.py")
        # File_path + constraint
        score2, _ = classify_prompt("Check src/foo.py but don't change the tests")
        assert score2 > score1


# ---------------------------------------------------------------------------
# analyze_session_prompts with mock DB data
# ---------------------------------------------------------------------------


class TestAnalyzeSessionPrompts:
    """Tests for session-level prompt analysis with resolution tracking."""

    @pytest.fixture()
    def db_session(self, tmp_path):
        """Create an in-memory DB session with test data."""
        db_path = tmp_path / "test.db"
        engine = get_engine(db_path)
        Base.metadata.create_all(engine)
        factory = get_session_factory(engine)

        session = factory()
        # Insert a session
        session.add(
            SessionRecord(
                session_id="test-sess",
                model="claude-opus-4-6",
                total_turns=3,
                total_api_calls=5,
            )
        )
        # Insert turns
        session.add(
            TurnRecord(
                session_id="test-sess",
                turn_number=0,
                first_api_call=0,
                last_api_call=1,
                prompt_preview="Fix the bug in src/dashboard.py:42 where render() fails. Don't change tests.",
            )
        )
        session.add(
            TurnRecord(
                session_id="test-sess",
                turn_number=1,
                first_api_call=2,
                last_api_call=2,
                prompt_preview="Make it work",
            )
        )
        session.add(
            TurnRecord(
                session_id="test-sess",
                turn_number=2,
                first_api_call=3,
                last_api_call=4,
                prompt_preview="Refactor process_data() in src/utils.py without changing the API",
            )
        )
        # Insert API calls
        for i in range(5):
            session.add(
                ApiCallRecord(
                    session_id="test-sess",
                    call_index=i,
                    input_tokens=1000 * (i + 1),
                    output_tokens=200 * (i + 1),
                    cache_read=500,
                    cache_creation=100,
                )
            )
        session.commit()
        yield session
        session.close()

    def test_returns_correct_number_of_analyses(self, db_session):
        results = analyze_session_prompts("test-sess", db_session, "claude-opus-4-6")
        assert len(results) == 3

    def test_specific_prompt_scores_higher(self, db_session):
        results = analyze_session_prompts("test-sess", db_session, "claude-opus-4-6")
        # Turn 0 is specific, turn 1 is vague
        assert results[0].specificity_score > results[1].specificity_score

    def test_resolution_turns_computed(self, db_session):
        results = analyze_session_prompts("test-sess", db_session, "claude-opus-4-6")
        # Turn 0: calls 0-1 = 2 resolution turns
        assert results[0].resolution_turns == 2
        # Turn 1: call 2 = 1 resolution turn
        assert results[1].resolution_turns == 1
        # Turn 2: calls 3-4 = 2 resolution turns
        assert results[2].resolution_turns == 2

    def test_resolution_cost_positive(self, db_session):
        results = analyze_session_prompts("test-sess", db_session, "claude-opus-4-6")
        for r in results:
            assert r.resolution_cost >= 0.0

    def test_signals_populated(self, db_session):
        results = analyze_session_prompts("test-sess", db_session, "claude-opus-4-6")
        # Turn 0 should have file_path signal
        assert "file_path" in results[0].signals
        # Turn 1 should have no signals (or minimal)
        assert results[1].specificity_score < 0.3

    def test_tool_failures_always_zero_per_prompt(self, db_session):
        """Tool failures should be 0 for every prompt (session-level metric only)."""
        results = analyze_session_prompts("test-sess", db_session, "claude-opus-4-6")
        for r in results:
            assert r.tool_failures == 0

    def test_no_api_range_gives_zero_resolution_turns(self, tmp_path):
        """A turn with no first/last API call should have resolution_turns=0, not 1."""
        db_path = tmp_path / "no_range.db"
        engine = get_engine(db_path)
        Base.metadata.create_all(engine)
        factory = get_session_factory(engine)
        session = factory()
        session.add(SessionRecord(session_id="no-range-sess", total_turns=1))
        session.add(
            TurnRecord(
                session_id="no-range-sess",
                turn_number=0,
                first_api_call=None,
                last_api_call=None,
                prompt_preview="Do something",
            )
        )
        session.commit()

        results = analyze_session_prompts("no-range-sess", session, "_default")
        assert len(results) == 1
        assert results[0].resolution_turns == 0
        assert results[0].resolution_cost == 0.0
        session.close()

    def test_empty_session(self, tmp_path):
        """Analyzing a session with no turns returns empty list."""
        db_path = tmp_path / "empty.db"
        engine = get_engine(db_path)
        Base.metadata.create_all(engine)
        factory = get_session_factory(engine)
        session = factory()
        session.add(SessionRecord(session_id="empty-sess"))
        session.commit()

        results = analyze_session_prompts("empty-sess", session, "_default")
        assert results == []
        session.close()


# ---------------------------------------------------------------------------
# compute_aggregate_stats tests
# ---------------------------------------------------------------------------


class TestComputeAggregateStats:
    """Tests for aggregate stats computation."""

    def test_empty_list(self):
        stats = compute_aggregate_stats([])
        assert stats["total_prompts"] == 0
        assert stats["avg_specificity"] == 0.0

    def test_basic_aggregation(self):
        analyses = [
            PromptAnalysis(
                turn_number=0,
                prompt_preview="specific prompt",
                specificity_score=0.8,
                signals=["file_path", "length_signal"],
                resolution_cost=0.05,
                resolution_turns=2,
            ),
            PromptAnalysis(
                turn_number=1,
                prompt_preview="vague",
                specificity_score=0.1,
                signals=[],
                resolution_cost=0.15,
                resolution_turns=5,
            ),
        ]
        stats = compute_aggregate_stats(analyses)
        assert stats["total_prompts"] == 2
        assert stats["specific_prompts"]["count"] == 1
        assert stats["vague_prompts"]["count"] == 1
        assert stats["specific_prompts"]["avg_cost"] < stats["vague_prompts"]["avg_cost"]

    def test_moderate_bucket(self):
        analyses = [
            PromptAnalysis(
                turn_number=0,
                prompt_preview="moderate",
                specificity_score=0.45,
                signals=["length_signal"],
                resolution_cost=0.10,
            ),
        ]
        stats = compute_aggregate_stats(analyses)
        assert stats["moderate_prompts"]["count"] == 1
        assert stats["specific_prompts"]["count"] == 0
        assert stats["vague_prompts"]["count"] == 0


# ---------------------------------------------------------------------------
# API endpoint test
# ---------------------------------------------------------------------------


class TestPromptEndpoint:
    """Test the /api/session/{id}/prompts endpoint."""

    @pytest.fixture()
    def client_with_db(self, tmp_path):
        """Create a test client with a pre-populated DB."""
        db_path = tmp_path / "api_test.db"
        engine = get_engine(db_path)
        Base.metadata.create_all(engine)
        factory = get_session_factory(engine)

        with factory() as sess:
            sess.add(
                SessionRecord(
                    session_id="api-test-sess",
                    model="claude-opus-4-6",
                    total_turns=2,
                    total_api_calls=3,
                )
            )
            sess.add(
                TurnRecord(
                    session_id="api-test-sess",
                    turn_number=0,
                    first_api_call=0,
                    last_api_call=0,
                    prompt_preview="Fix src/main.py:10 don't break tests",
                )
            )
            sess.add(
                TurnRecord(
                    session_id="api-test-sess",
                    turn_number=1,
                    first_api_call=1,
                    last_api_call=2,
                    prompt_preview="Do something",
                )
            )
            for i in range(3):
                sess.add(
                    ApiCallRecord(
                        session_id="api-test-sess",
                        call_index=i,
                        input_tokens=5000,
                        output_tokens=1000,
                    )
                )
            sess.commit()

        from context_tracker.dashboard import create_app

        app = create_app(
            trace_dir=tmp_path / "traces",
            transcript_dir=tmp_path / "transcripts",
            static_dir=tmp_path / "static",
            db_path=db_path,
        )
        return TestClient(app)

    def test_prompts_endpoint_returns_data(self, client_with_db):
        resp = client_with_db.get("/api/session/api-test-sess/prompts")
        assert resp.status_code == 200
        data = resp.json()
        assert "prompts" in data
        assert "aggregate" in data
        assert len(data["prompts"]) == 2

    def test_prompts_endpoint_specificity_scores(self, client_with_db):
        resp = client_with_db.get("/api/session/api-test-sess/prompts")
        data = resp.json()
        prompts = data["prompts"]
        # First prompt (with file path + constraint) should score higher
        assert prompts[0]["specificity_score"] > prompts[1]["specificity_score"]

    def test_prompts_endpoint_aggregate(self, client_with_db):
        resp = client_with_db.get("/api/session/api-test-sess/prompts")
        data = resp.json()
        agg = data["aggregate"]
        assert agg["total_prompts"] == 2
        assert "specific_prompts" in agg
        assert "vague_prompts" in agg

    def test_prompts_endpoint_not_found(self, client_with_db):
        resp = client_with_db.get("/api/session/nonexistent/prompts")
        assert resp.status_code == 404

    def test_prompts_endpoint_invalid_id(self, client_with_db):
        resp = client_with_db.get("/api/session/bad%20id!@/prompts")
        assert resp.status_code == 400
