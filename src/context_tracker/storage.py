"""JSONL storage with atomic append for context trace events."""

from __future__ import annotations

import os
import re
from pathlib import Path

from context_tracker.models import BaseEvent, TrackerEvent, parse_event

DEFAULT_TRACE_DIR = Path.home() / ".claude" / "context-trace"

_SESSION_ID_RE = re.compile(r'^[a-zA-Z0-9_-]+$')


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


def read_events(
    session_id: str, trace_dir: Path = DEFAULT_TRACE_DIR
) -> list[TrackerEvent]:
    """Read all events for a session. Skips malformed lines."""
    _validate_session_id(session_id)
    filepath = _session_path(session_id, trace_dir)
    if not filepath.exists():
        return []

    events: list[TrackerEvent] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parsed = parse_event(line)
            if parsed is not None:
                events.append(parsed)
    return events


def list_sessions(trace_dir: Path = DEFAULT_TRACE_DIR) -> list[str]:
    """List all session IDs with trace files, sorted by modification time (newest first)."""
    if not trace_dir.exists():
        return []
    files = sorted(trace_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [f.stem for f in files]
