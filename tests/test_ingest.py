"""Tests for the ingest pipeline (ingest.py)."""

from __future__ import annotations

import json
import os
import time

from sqlalchemy import select

from context_tracker.db import ApiCallRecord, SessionRecord, get_engine, get_session_factory
from context_tracker.ingest import get_or_ingest, ingest_all, ingest_session

# ---------------------------------------------------------------------------
# Helpers — minimal transcript builders (same pattern as test_parse_transcript_cs.py)
# ---------------------------------------------------------------------------


def _user_entry(content, uuid="u1"):
    """Build a user entry dict."""
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": None,
        "timestamp": "2026-01-01T00:00:00.000Z",
        "message": {"content": content},
    }


def _assistant_entry(content_blocks, usage, uuid="a1", parent="u1"):
    """Build a completed assistant entry dict."""
    return {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": parent,
        "timestamp": "2026-01-01T00:00:01.000Z",
        "message": {
            "model": "claude-opus-4-6",
            "stop_reason": "end_turn",
            "content": content_blocks,
            "usage": usage,
        },
    }


def _write_jsonl(entries, path):
    """Write entries as JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def _make_transcript(projects_dir, session_id, entries):
    """Create a transcript JSONL in a projects_dir structure that find_session_paths can discover.

    find_session_paths uses projects_dir.rglob("{session_id}.jsonl") to find transcripts.
    """
    # Put transcript inside a project subdirectory (mimics real layout)
    transcript_path = projects_dir / "test-project" / f"{session_id}.jsonl"
    _write_jsonl(entries, transcript_path)
    return transcript_path


def _simple_entries():
    """Minimal one-turn transcript: user + assistant with known token counts."""
    return [
        _user_entry("Hello, help me with code"),
        _assistant_entry(
            content_blocks=[{"type": "text", "text": "Sure, I can help!"}],
            usage={
                "input_tokens": 5,
                "cache_creation_input_tokens": 10000,
                "cache_read_input_tokens": 0,
                "output_tokens": 20,
            },
        ),
    ]


def _two_turn_entries():
    """Two-turn transcript for testing totals across multiple API calls."""
    return [
        _user_entry("First question", uuid="u1"),
        _assistant_entry(
            content_blocks=[{"type": "text", "text": "First answer"}],
            usage={
                "input_tokens": 5,
                "cache_creation_input_tokens": 10000,
                "cache_read_input_tokens": 0,
                "output_tokens": 20,
            },
            uuid="a1",
            parent="u1",
        ),
        _user_entry("Second question", uuid="u2"),
        _assistant_entry(
            content_blocks=[{"type": "text", "text": "Second answer"}],
            usage={
                "input_tokens": 10,
                "cache_creation_input_tokens": 500,
                "cache_read_input_tokens": 10000,
                "output_tokens": 30,
            },
            uuid="a2",
            parent="u2",
        ),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIngestSessionBasic:
    def test_ingest_creates_session_record(self, tmp_path):
        projects_dir = tmp_path / "projects"
        db_path = tmp_path / "test.db"
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()

        _make_transcript(projects_dir, "sess-100", _simple_entries())

        result = ingest_session(
            "sess-100",
            trace_dir=trace_dir,
            db_path=db_path,
            projects_dir=projects_dir,
        )
        assert result is not None
        assert result.session_id == "sess-100"

    def test_ingest_correct_totals(self, tmp_path):
        projects_dir = tmp_path / "projects"
        db_path = tmp_path / "test.db"
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()

        _make_transcript(projects_dir, "sess-101", _simple_entries())

        result = ingest_session(
            "sess-101",
            trace_dir=trace_dir,
            db_path=db_path,
            projects_dir=projects_dir,
        )
        assert result is not None
        assert result.total_api_calls == 1
        assert result.total_input_tokens == 5
        assert result.total_output_tokens == 20
        assert result.total_cache_read == 0
        assert result.total_cache_creation == 10000
        # Peak context = cache_read + cache_creation + input = 0 + 10000 + 5
        assert result.peak_context_tokens == 10005

    def test_ingest_two_turns_correct_totals(self, tmp_path):
        projects_dir = tmp_path / "projects"
        db_path = tmp_path / "test.db"
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()

        _make_transcript(projects_dir, "sess-102", _two_turn_entries())

        result = ingest_session(
            "sess-102",
            trace_dir=trace_dir,
            db_path=db_path,
            projects_dir=projects_dir,
        )
        assert result is not None
        assert result.total_api_calls == 2
        assert result.total_input_tokens == 15  # 5 + 10
        assert result.total_output_tokens == 50  # 20 + 30
        assert result.total_cache_read == 10000  # 0 + 10000
        assert result.total_cache_creation == 10500  # 10000 + 500
        # Peak context: max(10005, 10510) = 10510
        assert result.peak_context_tokens == 10510

    def test_ingest_creates_api_call_records(self, tmp_path):
        projects_dir = tmp_path / "projects"
        db_path = tmp_path / "test.db"
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()

        _make_transcript(projects_dir, "sess-103", _two_turn_entries())

        ingest_session(
            "sess-103",
            trace_dir=trace_dir,
            db_path=db_path,
            projects_dir=projects_dir,
        )

        engine = get_engine(db_path)
        factory = get_session_factory(engine)
        with factory() as db:
            calls = (
                db.execute(
                    select(ApiCallRecord)
                    .where(ApiCallRecord.session_id == "sess-103")
                    .order_by(ApiCallRecord.call_index)
                )
                .scalars()
                .all()
            )
            assert len(calls) == 2
            assert calls[0].input_tokens == 5
            assert calls[0].cache_creation == 10000
            assert calls[1].input_tokens == 10
            assert calls[1].cache_read == 10000
        engine.dispose()

    def test_ingest_returns_none_for_missing_transcript(self, tmp_path):
        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()
        db_path = tmp_path / "test.db"
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()

        result = ingest_session(
            "nonexistent-session",
            trace_dir=trace_dir,
            db_path=db_path,
            projects_dir=projects_dir,
        )
        assert result is None


class TestIngestIdempotent:
    def test_double_ingest_produces_one_record(self, tmp_path):
        projects_dir = tmp_path / "projects"
        db_path = tmp_path / "test.db"
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()

        _make_transcript(projects_dir, "sess-200", _simple_entries())

        result1 = ingest_session(
            "sess-200",
            trace_dir=trace_dir,
            db_path=db_path,
            projects_dir=projects_dir,
        )
        result2 = ingest_session(
            "sess-200",
            trace_dir=trace_dir,
            db_path=db_path,
            projects_dir=projects_dir,
        )
        assert result1 is not None
        assert result2 is not None

        # Verify only one session record exists
        engine = get_engine(db_path)
        factory = get_session_factory(engine)
        with factory() as db:
            all_sessions = db.execute(select(SessionRecord)).scalars().all()
            assert len(all_sessions) == 1
        engine.dispose()


class TestIngestReIngestsOnNewerSource:
    def test_re_ingests_when_source_is_newer(self, tmp_path):
        projects_dir = tmp_path / "projects"
        db_path = tmp_path / "test.db"
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()

        transcript_path = _make_transcript(projects_dir, "sess-300", _simple_entries())

        # First ingest
        result1 = ingest_session(
            "sess-300",
            trace_dir=trace_dir,
            db_path=db_path,
            projects_dir=projects_dir,
        )
        assert result1 is not None
        original_mtime = result1.source_mtime

        # Touch the source file to make it newer
        time.sleep(0.05)
        os.utime(transcript_path, None)

        # Re-ingest (force=False) -- should re-ingest because source is newer
        result2 = ingest_session(
            "sess-300",
            trace_dir=trace_dir,
            db_path=db_path,
            projects_dir=projects_dir,
        )
        assert result2 is not None
        assert result2.source_mtime > original_mtime

        # Still only one session record
        engine = get_engine(db_path)
        factory = get_session_factory(engine)
        with factory() as db:
            all_sessions = db.execute(select(SessionRecord)).scalars().all()
            assert len(all_sessions) == 1
        engine.dispose()

    def test_skips_when_source_not_newer(self, tmp_path):
        projects_dir = tmp_path / "projects"
        db_path = tmp_path / "test.db"
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()

        _make_transcript(projects_dir, "sess-301", _simple_entries())

        result1 = ingest_session(
            "sess-301",
            trace_dir=trace_dir,
            db_path=db_path,
            projects_dir=projects_dir,
        )
        mtime1 = result1.source_mtime

        # Don't touch the file -- same ingest again
        result2 = ingest_session(
            "sess-301",
            trace_dir=trace_dir,
            db_path=db_path,
            projects_dir=projects_dir,
        )
        assert result2 is not None
        # mtime should be unchanged (returned the existing record)
        assert result2.source_mtime == mtime1


class TestIngestAll:
    def test_ingests_multiple_sessions(self, tmp_path):
        projects_dir = tmp_path / "projects"
        db_path = tmp_path / "test.db"
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()

        # list_sessions looks for *.jsonl files in trace_dir
        # But find_session_paths looks for transcripts in projects_dir
        # So we need JSONL files in trace_dir (for list_sessions) AND in projects_dir (for find_session_paths)
        for sid in ["sess-400", "sess-401", "sess-402"]:
            _make_transcript(projects_dir, sid, _simple_entries())
            # list_sessions needs .jsonl files in trace_dir
            (trace_dir / f"{sid}.jsonl").write_text("{}\n")

        ingested = ingest_all(
            trace_dir=trace_dir,
            db_path=db_path,
            projects_dir=projects_dir,
        )
        assert len(ingested) == 3
        assert set(ingested) == {"sess-400", "sess-401", "sess-402"}


class TestGetOrIngest:
    def test_returns_existing_record(self, tmp_path):
        projects_dir = tmp_path / "projects"
        db_path = tmp_path / "test.db"
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()

        _make_transcript(projects_dir, "sess-500", _simple_entries())

        # First ingest
        ingest_session(
            "sess-500",
            trace_dir=trace_dir,
            db_path=db_path,
            projects_dir=projects_dir,
        )

        # get_or_ingest should return existing
        result = get_or_ingest(
            "sess-500",
            trace_dir=trace_dir,
            db_path=db_path,
            projects_dir=projects_dir,
        )
        assert result is not None
        assert result.session_id == "sess-500"

    def test_ingests_if_not_in_db(self, tmp_path):
        projects_dir = tmp_path / "projects"
        db_path = tmp_path / "test.db"
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()

        _make_transcript(projects_dir, "sess-501", _simple_entries())

        # get_or_ingest should ingest on first call
        result = get_or_ingest(
            "sess-501",
            trace_dir=trace_dir,
            db_path=db_path,
            projects_dir=projects_dir,
        )
        assert result is not None
        assert result.session_id == "sess-501"
        assert result.total_api_calls == 1

    def test_returns_none_for_missing(self, tmp_path):
        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()
        db_path = tmp_path / "test.db"
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()

        result = get_or_ingest(
            "nonexistent",
            trace_dir=trace_dir,
            db_path=db_path,
            projects_dir=projects_dir,
        )
        assert result is None
