"""Ingest session data from JSONL traces and transcripts into SQLite."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import cast

from sqlalchemy.orm import Session

from context_tracker.ccscope.reconcile import find_session_paths, reconcile
from context_tracker.ccscope.subagents import parse_workflows
from context_tracker.db import (
    DEFAULT_DB_PATH,
    ApiCallRecord,
    BlockRecord,
    HookEventRecord,
    SessionRecord,
    SubagentApiCallRecord,
    SubagentRecord,
    ToolResultOffloadRecord,
    TurnRecord,
    WorkflowRunRecord,
    get_engine,
    get_session_factory,
)
from context_tracker.storage import DEFAULT_TRACE_DIR, list_sessions

logger = logging.getLogger(__name__)


def _parse_hook_events(hook_path: Path) -> list[dict]:
    """Parse hook events JSONL into a list of dicts."""
    events = []
    with open(hook_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _build_turn_map(churn: list[dict], blocks: list[dict]) -> list[dict]:
    """Build conversation turn map from blocks and churn.

    Groups API calls into conversation turns by finding user blocks
    that mark the start of each turn.
    """
    turns: list[dict] = []
    # Find user blocks and their enter turn (API call index)
    user_entries = []
    for b in blocks:
        if b.get("type") == "user" and b.get("enter") is not None:
            user_entries.append(b)

    if not user_entries:
        return turns

    # Sort by enter turn
    user_entries.sort(key=lambda b: b["enter"])

    # Deduplicate — multiple user blocks can share the same enter turn
    seen_turns = set()
    unique_entries = []
    for b in user_entries:
        t = b["enter"]
        if t not in seen_turns:
            seen_turns.add(t)
            unique_entries.append(b)

    max_call = len(churn) - 1

    for i, ub in enumerate(unique_entries):
        first_call = ub["enter"]
        if i + 1 < len(unique_entries):
            last_call = unique_entries[i + 1]["enter"] - 1
        else:
            last_call = max_call

        prompt_preview = ub.get("content", "")[:200] if ub.get("content") else ""
        turns.append(
            {
                "turn_number": i,
                "first_api_call": first_call,
                "last_api_call": last_call,
                "prompt_preview": prompt_preview,
            }
        )

    return turns


def _add_subagent(
    db: Session,
    session_id: str,
    sa: dict,
    workflow_id: int | None = None,
    phase: str | None = None,
    label: str | None = None,
) -> SubagentRecord:
    """Persist a SubagentRecord + its per-call churn. Shared by plain subagents
    and workflow agents (the latter set workflow_id/phase/label)."""
    sa_rec = SubagentRecord(
        session_id=session_id,
        agent_id=sa.get("agent_id", ""),
        agent_type=sa.get("agent_type"),
        description=sa.get("description"),
        peak_resident=sa.get("peak_resident", 0),
        total_cache_read=sa.get("total_cache_read", 0),
        total_api_calls=sa.get("api_calls", 0),
        total_output_tokens=sa.get("total_output", 0),
        workflow_id=workflow_id,
        phase=phase,
        label=label,
    )
    db.add(sa_rec)
    db.flush()  # get sa_rec.id for FK

    for sc in sa.get("churn", []):
        db.add(
            SubagentApiCallRecord(
                subagent_id=sa_rec.id,
                session_id=session_id,
                call_index=sc.get("turn", 0),
                input_tokens=sc.get("input", 0),
                output_tokens=sc.get("output", 0),
                cache_read=sc.get("cache_read", 0),
                cache_creation=sc.get("cache_creation", 0),
            )
        )
    return sa_rec


def ingest_session(
    session_id: str,
    trace_dir: Path = DEFAULT_TRACE_DIR,
    db_path: Path = DEFAULT_DB_PATH,
    force: bool = False,
    projects_dir: Path | None = None,
) -> SessionRecord | None:
    """Ingest a single session into SQLite — all tables.

    Returns the SessionRecord if ingested, None if skipped (no transcript found).
    Idempotent: re-ingests if source file is newer than stored mtime.
    """
    engine = get_engine(db_path)
    session_factory = get_session_factory(engine)

    # Find all data paths
    paths = find_session_paths(session_id, projects_dir=projects_dir, trace_dir=trace_dir)
    transcript_path = paths.get("transcript")
    if not transcript_path or not Path(transcript_path).exists():
        logger.warning("No transcript found for session %s", session_id)
        return None

    transcript_path = Path(transcript_path)
    source_mtime = transcript_path.stat().st_mtime

    with session_factory() as db:
        # Check if already ingested and up-to-date
        existing: SessionRecord | None = db.get(SessionRecord, session_id)
        if existing and not force:
            if existing.source_mtime >= source_mtime:
                return existing  # Already up-to-date
            # Source is newer -- delete and re-ingest
            db.delete(existing)
            db.flush()

        # --- Parse all data sources via reconcile ---
        try:
            blocks, churn, subagent_summaries = reconcile(
                session_id,
                projects_dir=projects_dir,
                trace_dir=trace_dir,
            )
        except Exception:
            logger.exception("Failed to reconcile data for %s", session_id)
            return None

        # --- Build session summary from churn ---
        total_input = sum(c.get("input", 0) for c in churn)
        total_output = sum(c.get("output", 0) for c in churn)
        total_cache_read = sum(c.get("cache_read", 0) for c in churn)
        total_cache_creation = sum(c.get("cache_creation", 0) for c in churn)

        peak_context = 0
        for c in churn:
            resident = c.get("cache_read", 0) + c.get("cache_creation", 0) + c.get("input", 0)
            peak_context = max(peak_context, resident)

        cost = (
            total_input * 15.0 / 1e6
            + total_output * 75.0 / 1e6
            + total_cache_read * 1.875 / 1e6
            + total_cache_creation * 18.75 / 1e6
        )

        model = None
        if churn and "model" in churn[0]:
            model = churn[0]["model"]

        # Build turn map
        turn_map = _build_turn_map(churn, blocks)

        session_rec = SessionRecord(
            session_id=session_id,
            model=model,
            total_turns=len(turn_map),
            total_api_calls=len(churn),
            total_blocks=len(blocks),
            peak_context_tokens=peak_context,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_cache_read=total_cache_read,
            total_cache_creation=total_cache_creation,
            total_cost_usd=round(cost, 4),
            source_mtime=source_mtime,
        )
        db.add(session_rec)

        # --- API call records ---
        for i, c in enumerate(churn):
            db.add(
                ApiCallRecord(
                    session_id=session_id,
                    call_index=i,
                    input_tokens=c.get("input", 0),
                    output_tokens=c.get("output", 0),
                    cache_read=c.get("cache_read", 0),
                    cache_creation=c.get("cache_creation", 0),
                    system_tokens=c.get("system_tokens", 0),
                    working_tokens=c.get("working_tokens", 0),
                )
            )

        # --- Block records ---
        for b in blocks:
            db.add(
                BlockRecord(
                    session_id=session_id,
                    block_id=b.get("id", ""),
                    block_type=b.get("type", ""),
                    label=b.get("label"),
                    tokens=b.get("tokens", 0),
                    enter_turn=b.get("enter"),
                    exit_turn=b.get("exit"),
                    cached=1 if b.get("cached") else 0,
                    ref=1 if b.get("ref") else 0,
                    content_preview=b.get("content", "")[:500] if b.get("content") else None,
                )
            )

        # --- Turn records ---
        for t in turn_map:
            db.add(
                TurnRecord(
                    session_id=session_id,
                    turn_number=t["turn_number"],
                    first_api_call=t["first_api_call"],
                    last_api_call=t["last_api_call"],
                    prompt_preview=t.get("prompt_preview"),
                )
            )

        # --- Hook event records ---
        hook_path = paths.get("hook_events")
        if hook_path and Path(hook_path).exists():
            for evt in _parse_hook_events(Path(hook_path)):
                event_type = evt.get("event", "")
                tool_name = evt.get("tool_name")
                tool_use_id = evt.get("tool_use_id")
                payload_chars = evt.get("input_payload_chars", 0) + evt.get("output_payload_chars", 0)
                error_length = evt.get("error_length", 0)

                # Store event-specific fields as JSON
                skip_keys = {
                    "event",
                    "session_id",
                    "timestamp",
                    "tool_name",
                    "tool_use_id",
                    "input_payload_chars",
                    "output_payload_chars",
                    "error_length",
                }
                extra = {k: v for k, v in evt.items() if k not in skip_keys}
                metadata_json = json.dumps(extra) if extra else None

                db.add(
                    HookEventRecord(
                        session_id=session_id,
                        event_type=event_type,
                        timestamp=evt.get("timestamp"),
                        tool_name=tool_name,
                        tool_use_id=tool_use_id,
                        payload_chars=payload_chars,
                        error_length=error_length,
                        metadata_json=metadata_json,
                    )
                )

        # --- Subagent records + their per-call churn (plain Task subagents) ---
        for sa in subagent_summaries:
            _add_subagent(db, session_id, sa)

        # --- Multi-agent workflow runs + their subagents ---
        sa_dir = paths.get("subagents")
        if sa_dir and Path(sa_dir).exists():
            for run in parse_workflows(Path(sa_dir)):
                agents = run.get("agents", [])
                starts = [a.get("started_at") for a in agents if a.get("started_at")]
                ends = [a.get("ended_at") for a in agents if a.get("ended_at")]
                wf_rec = WorkflowRunRecord(
                    wf_id=run.get("wf_id", ""),
                    session_id=session_id,
                    name=run.get("name"),
                    started_at=min(starts) if starts else None,
                    ended_at=max(ends) if ends else None,
                )
                db.add(wf_rec)
                db.flush()  # get wf_rec.id for FK
                for sa in agents:
                    _add_subagent(
                        db,
                        session_id,
                        sa,
                        workflow_id=cast(int, wf_rec.id),
                        phase=sa.get("phase"),
                        label=sa.get("label"),
                    )

        # --- Tool result offloads ---
        tr_path = paths.get("tool_results")
        if tr_path and Path(tr_path).exists():
            for f in Path(tr_path).iterdir():
                if not f.is_file():
                    continue
                size_bytes = f.stat().st_size
                # Read first 500 chars for preview
                try:
                    preview = f.read_text(encoding="utf-8", errors="replace")[:500]
                except OSError:
                    preview = None
                db.add(
                    ToolResultOffloadRecord(
                        session_id=session_id,
                        filename=f.name,
                        size_bytes=size_bytes,
                        content_preview=preview,
                    )
                )

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
    sessions = list_sessions(trace_dir, projects_dir=projects_dir)
    ingested = []
    for sid in sessions:
        result = ingest_session(
            sid,
            trace_dir=trace_dir,
            db_path=db_path,
            force=force,
            projects_dir=projects_dir,
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
        existing: SessionRecord | None = db.get(SessionRecord, session_id)
        if existing:
            return existing

    # Not in DB -- try to ingest
    return ingest_session(
        session_id,
        trace_dir=trace_dir,
        db_path=db_path,
        projects_dir=projects_dir,
    )
