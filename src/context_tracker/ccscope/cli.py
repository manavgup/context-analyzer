"""ccscope CLI — Context Scope data pipeline.

Usage:
    ccscope list                     List available sessions
    ccscope build [SESSION]          Build blocks.json + churn.json
    ccscope open [SESSION]           Build + serve + open browser
"""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from pathlib import Path

from context_tracker.ccscope.parse_transcript import build_turn_map
from context_tracker.ccscope.reconcile import find_session_paths, reconcile, write_output

DEFAULT_PROJECTS_DIR = Path.home() / ".claude" / "projects"


def default_output_dir() -> Path:
    """User-writable default location for build artifacts.

    The packaged dashboard HTML may live in a read-only site-packages (wheel
    installs into system-managed environments), so generated data
    (blocks.json, churn.json, meta.json, turn_map.json) must not be written
    next to it. Default to the user cache directory instead:
    ``$XDG_CACHE_HOME/context-tracker/ccscope`` when set, otherwise
    ``~/.cache/context-tracker/ccscope``. The dashboard is pointed at this
    directory via ``create_app(data_dir=...)``.
    """
    cache_home = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(cache_home) if cache_home else Path.home() / ".cache"
    return base / "context-tracker" / "ccscope"


def cmd_list(args: argparse.Namespace) -> int:
    """List available sessions."""
    projects_dir = Path(args.projects_dir) if args.projects_dir else DEFAULT_PROJECTS_DIR

    if not projects_dir.exists():
        print(f"Projects directory not found: {projects_dir}", file=sys.stderr)
        return 1

    sessions = []
    for jsonl in sorted(projects_dir.rglob("*.jsonl")):
        # Skip subagent transcripts
        if "subagents" in str(jsonl):
            continue
        session_id = jsonl.stem
        # Skip non-UUID-like names
        if len(session_id) < 8:
            continue
        size_mb = jsonl.stat().st_size / (1024 * 1024)
        from datetime import datetime

        mtime = datetime.fromtimestamp(jsonl.stat().st_mtime)

        # Check if session folder exists (has subagents/tool-results)
        session_dir = jsonl.parent / session_id
        has_subagents = (session_dir / "subagents").exists() if session_dir.exists() else False
        has_offloads = (session_dir / "tool-results").exists() if session_dir.exists() else False

        extras = []
        if has_subagents:
            extras.append("subagents")
        if has_offloads:
            extras.append("offloads")
        extra_str = f" [{', '.join(extras)}]" if extras else ""

        sessions.append((mtime, session_id, size_mb, extra_str, str(jsonl.parent.name)))

    if not sessions:
        print("No sessions found.")
        return 0

    sessions.sort(reverse=True)  # newest first
    print(f"{'SESSION ID':<44} {'SIZE':>7} {'DATE':<20} {'PROJECT':<40} SOURCES")
    print("-" * 130)
    for mtime, sid, size, extra_str, project in sessions:
        print(f"{sid:<44} {size:6.1f}M {mtime.strftime('%Y-%m-%d %H:%M'):<20} {project[:40]:<40}{extra_str}")

    return 0


def cmd_build(args: argparse.Namespace) -> int:
    """Build blocks.json + churn.json for a session."""
    session_id = args.session
    if not session_id:
        print("Usage: ccscope build <SESSION_ID>", file=sys.stderr)
        return 1

    output_dir = Path(args.output) if args.output else default_output_dir()

    print(f"Building Context Scope data for session {session_id}...")

    try:
        blocks, churn, subagents = reconcile(session_id)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    blocks_path, churn_path = write_output(blocks, churn, output_dir)

    # Write meta.json for the dashboard to know the session ID
    import json as _json

    meta_path = output_dir / "meta.json"
    meta_path.write_text(_json.dumps({"session_id": session_id}), encoding="utf-8")

    # Write turn_map.json for conversation turn grouping
    paths = find_session_paths(session_id)
    if paths["transcript"]:
        turn_map = build_turn_map(paths["transcript"])
        turn_map_path = output_dir / "turn_map.json"
        turn_map_path.write_text(_json.dumps(turn_map), encoding="utf-8")
    else:
        turn_map = []

    # Summary
    total_cr = sum(c["cache_read"] for c in churn)
    total_in = sum(c["input"] for c in churn)
    pinned = [b for b in blocks if b.get("cached")]
    spilled = [b for b in blocks if b.get("spilled_tokens")]

    print("\nDone!")
    print(f"  Blocks:     {len(blocks)} ({len(pinned)} pinned, {len(spilled)} offloaded)")
    print(f"  Churn:      {len(churn)} API calls")
    print(f"  Conv turns: {len(turn_map)}")
    print(f"  Cache read: {total_cr:,} tokens")
    print(f"  New input:  {total_in:,} tokens")
    print(f"  Ratio:      {total_cr // max(total_in, 1):,}x")
    if subagents:
        sub_churn = sum(s["total_cache_read"] for s in subagents)
        print(f"  Subagents:  {len(subagents)} ({sub_churn:,} cache_read)")
    print(f"\n  Written to: {blocks_path}")
    print(f"              {churn_path}")

    return 0


def cmd_open(args: argparse.Namespace) -> int:
    """Build + serve + open browser."""
    session_id = args.session
    if not session_id:
        print("Usage: ccscope open <SESSION_ID>", file=sys.stderr)
        return 1

    # Build first (into a user-writable dir; site-packages may be read-only).
    output_dir = Path(args.output) if getattr(args, "output", None) else default_output_dir()
    args.output = str(output_dir)
    result = cmd_build(args)
    if result != 0:
        return result

    # Serve
    host = args.host if hasattr(args, "host") else "127.0.0.1"
    port = args.port if hasattr(args, "port") else 9201

    print(f"\nServing at http://{host}:{port}")
    webbrowser.open(f"http://{host}:{port}")

    import uvicorn

    from context_tracker.dashboard import create_app

    app = create_app(data_dir=output_dir)
    uvicorn.run(app, host=host, port=port)

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ccscope",
        description="Context Scope — context window forensics for Claude Code sessions",
    )
    subparsers = parser.add_subparsers(dest="command")

    # list
    list_parser = subparsers.add_parser("list", help="List available sessions")
    list_parser.add_argument("--projects-dir", default=None, help="Override projects directory")

    # build
    build_parser = subparsers.add_parser("build", help="Build blocks.json + churn.json")
    build_parser.add_argument("session", nargs="?", help="Session ID")
    build_parser.add_argument("--output", "-o", default=None, help="Output directory")
    build_parser.add_argument("--projects-dir", default=None)

    # open
    open_parser = subparsers.add_parser("open", help="Build + serve + open browser")
    open_parser.add_argument("session", nargs="?", help="Session ID")
    open_parser.add_argument("--output", "-o", default=None, help="Output directory")
    open_parser.add_argument("--host", default="127.0.0.1")
    open_parser.add_argument("--port", type=int, default=9201)
    open_parser.add_argument("--projects-dir", default=None)

    args = parser.parse_args()

    if args.command == "list":
        sys.exit(cmd_list(args))
    elif args.command == "build":
        sys.exit(cmd_build(args))
    elif args.command == "open":
        sys.exit(cmd_open(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
