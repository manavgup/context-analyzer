"""Tests for the personal stats card (`context-tracker stats`)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import sessionmaker

from context_tracker.db import Base, BlockRecord, SessionRecord, TurnRecord
from context_tracker.stats import compute_stats, render_card, render_share_markdown

# Sensitive fixture values that must NEVER appear in --share output.
SESSION_A = "aaaa1111-2222-3333-4444-555566667777"
SESSION_B = "bbbb8888-9999-0000-1111-222233334444"
PROJECT_A = "/Users/testuser/projects/secret-repo"
PROJECT_B = "/Users/testuser/projects/other-repo"
PROMPT_TEXT = "please fix the login bug in auth.py"
CONTENT_PREVIEW = "def login(): raise SecretInternalError"


@pytest.fixture
def db_session() -> Iterator[DbSession]:
    """In-memory SQLite DB with the full schema."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    session = factory()
    yield session
    session.close()


@pytest.fixture
def fixture_db(db_session: DbSession) -> DbSession:
    """Two sessions with hand-checkable numbers.

    Session A: cost $10, 10 API calls (last call index 9), input 1000, cache 9000.
      Blocks (tokens, enter, exit, ref, cached) -> residency, token-calls:
        b1 (100, 0, None, 1, 0) -> 9 -> 900   live
        b2 (200, 1, 6,    0, 0) -> 5 -> 1000  DEAD
        b3 (300, 0, None, 0, 1) -> 9 -> 2700  cached, not dead
      waste_A = 10 * 1000/4600 = 2.173913...

    Session B: cost $2, 4 API calls (last call index 3), input 2000, cache 2000.
        b4 (100, 0, None, 1, 0) -> 3 -> 300   live
        b5 (50,  1, 3,    0, 0) -> 2 -> 100   DEAD
      waste_B = 2 * 100/400 = 0.5

    Totals: sessions=2, api_calls=14, spend=$12, waste=2.673913...
    Cache efficiency = 11000 / (11000 + 3000) = 78.571...%
    Most expensive: session A ($10, peak 50,000 tokens, 2026-07-01).
    """
    db_session.add(
        SessionRecord(
            session_id=SESSION_A,
            project_path=PROJECT_A,
            started_at="2026-07-01T10:00:00Z",
            total_api_calls=10,
            total_input_tokens=1000,
            total_cache_read=9000,
            total_cost_usd=10.0,
            peak_context_tokens=50000,
        )
    )
    db_session.add(
        SessionRecord(
            session_id=SESSION_B,
            project_path=PROJECT_B,
            started_at="2026-07-05T09:00:00Z",
            total_api_calls=4,
            total_input_tokens=2000,
            total_cache_read=2000,
            total_cost_usd=2.0,
            peak_context_tokens=20000,
        )
    )
    db_session.add(TurnRecord(session_id=SESSION_A, turn_number=1, prompt_preview=PROMPT_TEXT))
    blocks = [
        (SESSION_A, "b1", 100, 0, None, 1, 0),
        (SESSION_A, "b2", 200, 1, 6, 0, 0),
        (SESSION_A, "b3", 300, 0, None, 0, 1),
        (SESSION_B, "b4", 100, 0, None, 1, 0),
        (SESSION_B, "b5", 50, 1, 3, 0, 0),
    ]
    for sid, bid, tokens, enter, exit_, ref, cached in blocks:
        db_session.add(
            BlockRecord(
                session_id=sid,
                block_id=bid,
                block_type="tool_result",
                label=f"Read {PROJECT_A}/auth.py",
                tokens=tokens,
                enter_turn=enter,
                exit_turn=exit_,
                ref=ref,
                cached=cached,
                content_preview=CONTENT_PREVIEW,
            )
        )
    db_session.commit()
    return db_session


class TestComputeStats:
    def test_card_math(self, fixture_db: DbSession) -> None:
        card = compute_stats(fixture_db)

        assert card.total_sessions == 2
        assert card.total_api_calls == 14
        assert card.total_spend_usd == pytest.approx(12.0)
        # waste = 10 * (1000/4600) + 2 * (100/400)
        assert card.wasted_spend_usd == pytest.approx(10 * 1000 / 4600 + 0.5)
        assert card.cache_read_tokens == 11000
        assert card.input_tokens == 3000
        assert card.cache_efficiency == pytest.approx(11000 / 14000)
        assert card.top_session_cost_usd == pytest.approx(10.0)
        assert card.top_session_peak_context == 50000
        assert card.top_session_date == "2026-07-01"

    def test_empty_db(self, db_session: DbSession) -> None:
        card = compute_stats(db_session)
        assert card.total_sessions == 0
        assert card.total_spend_usd == 0.0
        assert card.wasted_spend_usd == 0.0
        assert card.cache_efficiency == 0.0
        assert card.top_session_date == "unknown date"


class TestRendering:
    def test_terminal_card_contains_numbers(self, fixture_db: DbSession) -> None:
        out = render_card(compute_stats(fixture_db))
        assert "$12.00" in out
        assert "$2.67" in out  # wasted spend, rounded
        assert "78.6%" in out
        assert "50,000" in out
        assert "2026-07-01" in out

    def test_share_output_contains_numbers(self, fixture_db: DbSession) -> None:
        out = render_share_markdown(compute_stats(fixture_db))
        assert "**Sessions analyzed:** 2" in out
        assert "$12.00" in out
        assert "$2.67" in out
        assert "78.6%" in out
        assert "2026-07-01" in out

    def test_share_output_has_no_leakage(self, fixture_db: DbSession) -> None:
        """The --share artifact must contain no sensitive fixture content."""
        out = render_share_markdown(compute_stats(fixture_db))

        # No session ids (full or fragments)
        assert SESSION_A not in out
        assert SESSION_B not in out
        assert "aaaa1111" not in out
        assert "bbbb8888" not in out
        # No project names or file paths
        assert "secret-repo" not in out
        assert "other-repo" not in out
        assert PROJECT_A not in out
        assert "/Users/" not in out
        assert "testuser" not in out
        assert "auth.py" not in out
        # No prompt text or block content
        assert PROMPT_TEXT not in out
        assert "login bug" not in out
        assert CONTENT_PREVIEW not in out
        assert "SecretInternalError" not in out

    def test_terminal_card_has_no_leakage(self, fixture_db: DbSession) -> None:
        """The terminal card identifies sessions by date only, too."""
        out = render_card(compute_stats(fixture_db))
        assert SESSION_A not in out
        assert "secret-repo" not in out
        assert "/Users/" not in out
        assert PROMPT_TEXT not in out
