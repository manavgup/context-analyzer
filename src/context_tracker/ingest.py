"""Ingest session data from JSONL traces and transcripts into SQLite."""

from __future__ import annotations

import logging
from pathlib import Path

from context_tracker.ccscope.parse_transcript import parse_transcript_to_blocks
from context_tracker.ccscope.reconcile import find_session_paths
from context_tracker.db import (
    DEFAULT_DB_PATH,
    ApiCallRecord,
    SessionRecord,
    get_engine,
    get_session_factory,
)
from context_tracker.storage import DEFAULT_TRACE_DIR, list_sessions

logger = logging.getLogger(__name__)


def ingest_session(
    session_id: str,
    trace_dir: Path = DEFAULT_TRACE_DIR,
    db_path: Path = DEFAULT_DB_PATH,
    force: bool = False,
    projects_dir: Path | None = None,
) -> SessionRecord | None:
    """Ingest a single session into SQLite.

    Returns the SessionRecord if ingested, None if skipped (no transcript found).
    Idempotent: re-ingests if source file is newer than stored mtime.
    """
    engine = get_engine(db_path)
    session_factory = get_session_factory(engine)

    # Find transcript path
    paths = find_session_paths(session_id, projects_dir=projects_dir, trace_dir=trace_dir)
    transcript_path = paths.get("transcript")
    if not transcript_path or not Path(transcript_path).exists():
        logger.warning("No transcript found for session %s", session_id)
        return None

    transcript_path = Path(transcript_path)
    source_mtime = transcript_path.stat().st_mtime

    with session_factory() as db:
        # Check if already ingested and up-to-date
        existing = db.get(SessionRecord, session_id)
        if existing and not force:
            if existing.source_mtime >= source_mtime:
                return existing  # Already up-to-date
            # Source is newer -- delete and re-ingest
            db.delete(existing)
            db.flush()

        # Parse transcript to get blocks + churn
        try:
            _blocks, churn = parse_transcript_to_blocks(transcript_path)
        except Exception:
            logger.exception("Failed to parse transcript for %s", session_id)
            return None

        # Build session record from churn data
        total_input = sum(c.get("input", 0) for c in churn)
        total_output = sum(c.get("output", 0) for c in churn)
        total_cache_read = sum(c.get("cache_read", 0) for c in churn)
        total_cache_creation = sum(c.get("cache_creation", 0) for c in churn)

        # Peak context = max resident tokens across all API calls
        peak_context = 0
        for c in churn:
            resident = c.get("cache_read", 0) + c.get("cache_creation", 0) + c.get("input", 0)
            peak_context = max(peak_context, resident)

        # Estimate cost (using Opus 4.6 1M pricing)
        cost = (
            total_input * 15.0 / 1e6
            + total_output * 75.0 / 1e6
            + total_cache_read * 1.875 / 1e6
            + total_cache_creation * 18.75 / 1e6
        )

        # Detect model from first churn entry if available
        model = None
        if churn and "model" in churn[0]:
            model = churn[0]["model"]

        session_rec = SessionRecord(
            session_id=session_id,
            model=model,
            total_api_calls=len(churn),
            peak_context_tokens=peak_context,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_cache_read=total_cache_read,
            total_cache_creation=total_cache_creation,
            total_cost_usd=round(cost, 4),
            source_mtime=source_mtime,
        )
        db.add(session_rec)

        # Add API call records
        for i, c in enumerate(churn):
            call_rec = ApiCallRecord(
                session_id=session_id,
                call_index=i,
                input_tokens=c.get("input", 0),
                output_tokens=c.get("output", 0),
                cache_read=c.get("cache_read", 0),
                cache_creation=c.get("cache_creation", 0),
                system_tokens=c.get("system_tokens", 0),
                working_tokens=c.get("working_tokens", 0),
            )
            db.add(call_rec)

        db.commit()
        db.refresh(session_rec)
        return session_rec


def ingest_all(
    trace_dir: Path = DEFAULT_TRACE_DIR,
    db_path: Path = DEFAULT_DB_PATH,
    force: bool = False,
    projects_dir: Path | None = None,
) -> list[str]:
    """Ingest all sessions. Returns list of session IDs that were ingested."""
    sessions = list_sessions(trace_dir)
    ingested = []
    for sid in sessions:
        result = ingest_session(
            sid, trace_dir=trace_dir, db_path=db_path, force=force, projects_dir=projects_dir,
        )
        if result is not None:
            ingested.append(sid)
    return ingested


def get_or_ingest(
    session_id: str,
    trace_dir: Path = DEFAULT_TRACE_DIR,
    db_path: Path = DEFAULT_DB_PATH,
    projects_dir: Path | None = None,
) -> SessionRecord | None:
    """Get a session from SQLite, ingesting first if needed."""
    engine = get_engine(db_path)
    session_factory = get_session_factory(engine)

    with session_factory() as db:
        existing = db.get(SessionRecord, session_id)
        if existing:
            return existing

    # Not in DB -- try to ingest
    return ingest_session(
        session_id, trace_dir=trace_dir, db_path=db_path, projects_dir=projects_dir,
    )
