"""JSONL storage with atomic append for context trace events."""

from __future__ import annotations

import os
import re
from pathlib import Path

from context_tracker.models import BaseEvent, TrackerEvent, parse_event

DEFAULT_TRACE_DIR = Path.home() / ".claude" / "context-trace"

_SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def _validate_session_id(session_id: str) -> None:
    if not _SESSION_ID_RE.match(session_id):
        raise ValueError(f"Invalid session_id: {session_id!r}")


def _session_path(session_id: str, trace_dir: Path) -> Path:
    return trace_dir / f"{session_id}.jsonl"


def append_event(event: BaseEvent, trace_dir: Path = DEFAULT_TRACE_DIR) -> None:
    """Append a single event as a JSONL line. Uses O_APPEND for atomic writes."""
    _validate_session_id(event.session_id)
    trace_dir.mkdir(parents=True, exist_ok=True)
    filepath = _session_path(event.session_id, trace_dir)
    line = event.to_jsonl() + "\n"
    fd = os.open(str(filepath), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)


def read_events(session_id: str, trace_dir: Path = DEFAULT_TRACE_DIR) -> list[TrackerEvent]:
    """Read all events for a session. Skips malformed lines."""
    _validate_session_id(session_id)
    filepath = _session_path(session_id, trace_dir)
    if not filepath.exists():
        return []

    events: list[TrackerEvent] = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parsed = parse_event(line)
            if parsed is not None:
                events.append(parsed)
    return events


DEFAULT_PROJECTS_DIR = Path.home() / ".claude" / "projects"

# UUID pattern for session IDs (Claude Code transcript filenames)
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def list_sessions(
    trace_dir: Path = DEFAULT_TRACE_DIR,
    projects_dir: Path | None = None,
) -> list[str]:
    """List all session IDs from hook traces and transcripts.

    Merges both sources, deduplicates, and sorts by modification time
    (newest first). Transcript files in subagents/ directories are excluded.
    """
    # session_id -> newest mtime across sources
    sessions: dict[str, float] = {}

    # Source 1: Hook trace files (existing behavior)
    if trace_dir.exists():
        for f in trace_dir.glob("*.jsonl"):
            sessions[f.stem] = f.stat().st_mtime

    # Source 2: Transcript files from projects directory
    if projects_dir is None:
        projects_dir = DEFAULT_PROJECTS_DIR
    if projects_dir.exists():
        for f in projects_dir.rglob("*.jsonl"):
            # Skip subagent transcripts (agent-*.jsonl inside subagents/ dirs)
            if "subagents" in f.parts:
                continue
            sid = f.stem
            # Only include UUID-formatted session IDs
            if not _UUID_RE.match(sid):
                continue
            mtime = f.stat().st_mtime
            if sid not in sessions or mtime > sessions[sid]:
                sessions[sid] = mtime

    # Sort by mtime descending (newest first)
    return sorted(sessions, key=lambda s: sessions[s], reverse=True)
