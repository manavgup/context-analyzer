"""FastMCP server exposing context usage query tools."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from time import time

from fastmcp import FastMCP

from context_tracker.models import (
    ApiTurnEvent,
    PostCompactEvent,
    PostToolUseEvent,
    SessionStartEvent,
)
from context_tracker.storage import DEFAULT_TRACE_DIR, list_sessions, read_events
from context_tracker.transcript import parse_transcript

mcp = FastMCP(name="context-tracker", version="0.1.0")

DEFAULT_TRANSCRIPT_DIR = Path.home() / ".claude" / "projects"

# Cache read_events for 2 seconds (covers multiple tool calls in quick succession)
_cache: dict = {}
_cache_ttl = 2.0


def _cached_read_events(session_id: str, trace_dir: Path = DEFAULT_TRACE_DIR):
    key = (session_id, str(trace_dir))
    now = time()
    if key in _cache and now - _cache[key][0] < _cache_ttl:
        return _cache[key][1]
    events = read_events(session_id, trace_dir=trace_dir)
    _cache[key] = (now, events)
    return events


def _find_transcript(
    session_id: str, transcript_dir: Path = DEFAULT_TRANSCRIPT_DIR
) -> Path | None:
    """Find a transcript JSONL file for a session ID across all project dirs."""
    # Direct match
    direct = transcript_dir / f"{session_id}.jsonl"
    if direct.exists():
        return direct

    # Search in subdirectories (Claude Code stores transcripts under project dirs)
    for jsonl_file in transcript_dir.rglob(f"{session_id}.jsonl"):
        return jsonl_file

    return None


def get_session_summary(
    session_id: str,
    trace_dir: Path = DEFAULT_TRACE_DIR,
    transcript_dir: Path = DEFAULT_TRANSCRIPT_DIR,
) -> dict:
    """Compute summary statistics for a session."""
    events = _cached_read_events(session_id, trace_dir=trace_dir)

    tool_calls = [e for e in events if isinstance(e, PostToolUseEvent)]
    post_compactions = [e for e in events if isinstance(e, PostCompactEvent)]
    starts = [e for e in events if isinstance(e, SessionStartEvent)]

    # Parse transcript for exact token counts
    transcript_path = _find_transcript(session_id, transcript_dir)
    api_turns: list[ApiTurnEvent] = []
    if transcript_path:
        api_turns = parse_transcript(transcript_path)

    total_input = sum(t.input_tokens for t in api_turns)
    total_output = sum(t.output_tokens for t in api_turns)
    total_cache_read = sum(t.cache_read_input_tokens for t in api_turns)
    total_cache_create = sum(t.cache_creation_input_tokens for t in api_turns)

    cache_hit_rate = 0.0
    cache_total = total_cache_read + total_cache_create + total_input
    if cache_total > 0:
        cache_hit_rate = round(total_cache_read / cache_total, 3)

    model = starts[0].model if starts else "unknown"

    # Duration from first to last event timestamp
    timestamps = [e.timestamp for e in events if e.timestamp]
    duration_seconds = None
    if len(timestamps) >= 2:
        try:
            first = datetime.fromisoformat(timestamps[0])
            last = datetime.fromisoformat(timestamps[-1])
            duration_seconds = round((last - first).total_seconds())
        except ValueError:
            pass

    return {
        "session_id": session_id,
        "model": model,
        "tool_calls": len(tool_calls),
        "compactions": len(post_compactions),
        "total_events": len(events),
        "api_turns": len(api_turns),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_cache_read_tokens": total_cache_read,
        "total_cache_creation_tokens": total_cache_create,
        "cache_hit_rate": cache_hit_rate,
        "duration_seconds": duration_seconds,
    }


def get_tool_breakdown(
    session_id: str,
    trace_dir: Path = DEFAULT_TRACE_DIR,
) -> list[dict]:
    """Rank tools by total output payload size."""
    events = _cached_read_events(session_id, trace_dir=trace_dir)
    tool_calls = [e for e in events if isinstance(e, PostToolUseEvent)]

    by_tool: dict[str, dict] = defaultdict(lambda: {
        "call_count": 0,
        "total_input_payload_chars": 0,
        "total_output_payload_chars": 0,
    })

    for tc in tool_calls:
        entry = by_tool[tc.tool_name]
        entry["call_count"] += 1
        entry["total_input_payload_chars"] += tc.input_payload_chars
        entry["total_output_payload_chars"] += tc.output_payload_chars

    result = [
        {"tool_name": name, **stats}
        for name, stats in by_tool.items()
    ]
    result.sort(key=lambda x: x["total_output_payload_chars"], reverse=True)
    return result


def get_compaction_history(
    session_id: str,
    trace_dir: Path = DEFAULT_TRACE_DIR,
) -> list[dict]:
    """Timeline of compaction events."""
    events = _cached_read_events(session_id, trace_dir=trace_dir)

    result = []
    for e in events:
        if isinstance(e, PostCompactEvent):
            result.append({
                "timestamp": e.timestamp,
                "trigger": e.trigger,
                "summary_length": e.compact_summary_length,
            })
    return result


def get_context_hogs(
    session_id: str,
    top_n: int = 10,
    trace_dir: Path = DEFAULT_TRACE_DIR,
) -> list[dict]:
    """Top N tool calls by output payload size."""
    events = _cached_read_events(session_id, trace_dir=trace_dir)
    tool_calls = [e for e in events if isinstance(e, PostToolUseEvent)]
    tool_calls.sort(key=lambda e: e.output_payload_chars, reverse=True)

    return [
        {
            "tool_name": tc.tool_name,
            "output_payload_chars": tc.output_payload_chars,
            "input_payload_chars": tc.input_payload_chars,
            "tool_use_id": tc.tool_use_id,
            "timestamp": tc.timestamp,
        }
        for tc in tool_calls[:top_n]
    ]


def get_session_history(
    trace_dir: Path = DEFAULT_TRACE_DIR,
    transcript_dir: Path = DEFAULT_TRANSCRIPT_DIR,
) -> list[dict]:
    """List all sessions with summary stats."""
    session_ids = list_sessions(trace_dir=trace_dir)
    return [
        get_session_summary(sid, trace_dir=trace_dir, transcript_dir=transcript_dir)
        for sid in session_ids
    ]


def get_bloat_events(
    session_id: str,
    threshold: int = 5000,
    trace_dir: Path = DEFAULT_TRACE_DIR,
    transcript_dir: Path = DEFAULT_TRANSCRIPT_DIR,
) -> list[dict]:
    """Find turns where context grew significantly."""
    events = _cached_read_events(session_id, trace_dir=trace_dir)
    tool_calls = [e for e in events if isinstance(e, PostToolUseEvent) and e.output_payload_chars > threshold]
    tool_calls.sort(key=lambda e: e.output_payload_chars, reverse=True)
    return [
        {
            "tool_name": tc.tool_name,
            "output_payload_chars": tc.output_payload_chars,
            "tool_use_id": tc.tool_use_id,
            "timestamp": tc.timestamp,
        }
        for tc in tool_calls
    ]


def should_clear(
    session_id: str,
    trace_dir: Path = DEFAULT_TRACE_DIR,
    transcript_dir: Path = DEFAULT_TRANSCRIPT_DIR,
) -> dict:
    """Recommend whether to clear/start a new session."""
    summary = get_session_summary(session_id, trace_dir=trace_dir, transcript_dir=transcript_dir)

    total_input = summary["total_input_tokens"] + summary["total_cache_read_tokens"] + summary["total_cache_creation_tokens"]
    context_pct = (total_input / 1_000_000 * 100) if total_input > 0 else 0
    cache_hit = summary["cache_hit_rate"] * 100

    reasons = []
    if context_pct > 70:
        reasons.append("context above 70%")
    elif context_pct > 50:
        reasons.append("context above 50%")
    if cache_hit < 50:
        reasons.append("cache hit rate below 50%")
    if summary["compactions"] >= 2:
        reasons.append(f"{summary['compactions']} compactions occurred")

    if any("70%" in r for r in reasons) or len(reasons) >= 2:
        rec = "urgent_clear"
    elif reasons:
        rec = "clear"
    else:
        rec = "continue"

    return {
        "recommendation": rec,
        "context_pct": round(context_pct, 1),
        "cache_hit_rate": round(cache_hit, 1),
        "compactions": summary["compactions"],
        "reasons": reasons,
    }


# --- MCP Tool Registrations ---


@mcp.tool(description="Get summary statistics for a Claude Code session including tool calls, compactions, exact API token counts, and cache efficiency.")
def mcp_get_session_summary(session_id: str = "") -> str:
    """If session_id is empty, uses the most recent session."""
    if not session_id:
        sessions = list_sessions()
        if not sessions:
            return json.dumps({"error": "No sessions found"})
        session_id = sessions[0]
    return json.dumps(get_session_summary(session_id), indent=2)


@mcp.tool(description="Get ranked breakdown of tools by payload size and call frequency for a session.")
def mcp_get_tool_breakdown(session_id: str = "") -> str:
    if not session_id:
        sessions = list_sessions()
        if not sessions:
            return json.dumps({"error": "No sessions found"})
        session_id = sessions[0]
    return json.dumps(get_tool_breakdown(session_id), indent=2)


@mcp.tool(description="Get timeline of compaction events for a session.")
def mcp_get_compaction_history(session_id: str = "") -> str:
    if not session_id:
        sessions = list_sessions()
        if not sessions:
            return json.dumps({"error": "No sessions found"})
        session_id = sessions[0]
    return json.dumps(get_compaction_history(session_id), indent=2)


@mcp.tool(description="Get the top N tool calls by output payload size for a session.")
def mcp_get_context_hogs(session_id: str = "", top_n: int = 10) -> str:
    if not session_id:
        sessions = list_sessions()
        if not sessions:
            return json.dumps({"error": "No sessions found"})
        session_id = sessions[0]
    return json.dumps(get_context_hogs(session_id, top_n=top_n), indent=2)


@mcp.tool(description="List all tracked sessions with summary stats.")
def mcp_get_session_history() -> str:
    return json.dumps(get_session_history(), indent=2)


@mcp.tool(description="Find tool calls where context grew by more than a threshold (default 5000 chars). Returns events sorted by output size descending.")
def mcp_get_bloat_events(session_id: str = "", threshold: int = 5000) -> str:
    if not session_id:
        sessions = list_sessions()
        if not sessions:
            return json.dumps({"error": "No sessions found"})
        session_id = sessions[0]
    return json.dumps(get_bloat_events(session_id, threshold=threshold), indent=2)


@mcp.tool(description="Get a recommendation on whether to start a new Claude Code session based on context usage, cache efficiency, and compaction history.")
def mcp_should_clear(session_id: str = "") -> str:
    if not session_id:
        sessions = list_sessions()
        if not sessions:
            return json.dumps({"error": "No sessions found"})
        session_id = sessions[0]
    return json.dumps(should_clear(session_id), indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Context Tracker MCP Server")
    subparsers = parser.add_subparsers(dest="command")

    # Default: MCP server
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "http"],
        default="stdio",
        help="Transport mode (default: stdio)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9200)

    # Dashboard subcommand
    dash_parser = subparsers.add_parser("dashboard", help="Launch web dashboard")
    dash_parser.add_argument("--host", default="127.0.0.1", dest="dash_host")
    dash_parser.add_argument("--port", type=int, default=9201, dest="dash_port")

    args = parser.parse_args()

    if args.command == "dashboard":
        from context_tracker.dashboard import create_app
        import uvicorn
        app = create_app()
        uvicorn.run(app, host=args.dash_host, port=args.dash_port)
    else:
        if args.transport == "stdio":
            mcp.run()
        else:
            mcp.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
