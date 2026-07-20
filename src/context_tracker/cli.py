"""Top-level console entry point for the ``context-tracker`` command.

Quickstart subcommands (issue #77):

- ``context-tracker up``    — install Claude Code hooks (skippable with
  ``--no-hooks``), ingest existing transcripts from ``~/.claude/projects/``,
  and start the web dashboard. The dashboard shows data from existing
  transcripts even when hooks are skipped, so there is value with zero setup.
- ``context-tracker down``  — remove only this tool's hook entries from
  ``~/.claude/settings.json`` (other hooks are left untouched). A backup of
  settings.json is written as ``settings.json.bak`` on every install.
- ``context-tracker install`` / ``uninstall`` — hooks only, no dashboard.

Anything else (``dashboard``, the default MCP server mode, ``--transport``
flags) is delegated to :func:`context_tracker.server.main` so existing MCP
configurations keep working unchanged.
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI

from context_tracker.db import DEFAULT_DB_PATH
from context_tracker.installer import install_hooks, uninstall_hooks
from context_tracker.profiles import DEFAULT_SETTINGS_PATH
from context_tracker.storage import DEFAULT_PROJECTS_DIR, DEFAULT_TRACE_DIR

ServeFn = Callable[[FastAPI, str, int], None]


def _default_serve(app: FastAPI, host: str, port: int) -> None:
    """Run the dashboard app with uvicorn (blocks until interrupted)."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


def up(
    *,
    install: bool = True,
    host: str = "127.0.0.1",
    port: int = 8080,
    open_browser: bool = False,
    settings_path: Path | None = None,
    trace_dir: Path = DEFAULT_TRACE_DIR,
    projects_dir: Path | None = None,
    db_path: Path = DEFAULT_DB_PATH,
    serve: ServeFn | None = None,
) -> FastAPI:
    """One-command quickstart: hooks (optional) + ingest + dashboard.

    Ingestion runs regardless of whether hooks are installed, so the dashboard
    shows real sessions from existing transcripts in ``~/.claude/projects/``
    immediately — hooks only add richer live data for future sessions.

    Returns the FastAPI app (useful for tests; ``serve`` normally blocks).
    """
    from context_tracker.dashboard import create_app
    from context_tracker.ingest import ingest_all

    # 1. Hooks (reversible; settings.json is backed up to settings.json.bak).
    if install:
        install_hooks(settings_path=settings_path)
    else:
        print("Skipping hook install (--no-hooks). Dashboard will use existing transcripts only.")

    # 2. Ingest existing transcripts so the dashboard has data on first load.
    print("Ingesting existing sessions from transcripts...")
    ingested = ingest_all(trace_dir=trace_dir, db_path=db_path, projects_dir=projects_dir)
    print(f"Ingested {len(ingested)} session(s).")

    # 3. Dashboard.
    app = create_app(
        trace_dir=trace_dir,
        transcript_dir=projects_dir if projects_dir is not None else DEFAULT_PROJECTS_DIR,
        db_path=db_path,
    )
    url = f"http://{host}:{port}"
    print(f"Dashboard: {url}  (Ctrl+C to stop; 'context-tracker down' to remove hooks)")
    if open_browser:
        # Delay so the server is listening before the browser hits it.
        threading.Timer(1.0, webbrowser.open, [url]).start()

    (serve or _default_serve)(app, host, port)
    return app


def down(*, settings_path: Path | None = None) -> None:
    """Reversible uninstall: remove only context-tracker's hook entries.

    Other hooks in settings.json are left untouched; a pre-install backup
    exists at ``settings.json.bak`` if a full restore is ever needed.
    """
    uninstall_hooks(settings_path=settings_path)
    resolved = settings_path if settings_path is not None else DEFAULT_SETTINGS_PATH
    backup = resolved.with_suffix(".json.bak")
    if backup.exists():
        print(f"Backup of your pre-install settings remains at {backup}")


def _up_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="context-tracker up",
        description="Install hooks, ingest existing transcripts, and start the dashboard.",
    )
    parser.add_argument(
        "--no-hooks",
        action="store_true",
        help="Skip installing Claude Code hooks (dashboard still works from existing transcripts)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Dashboard host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Dashboard port (default: 8080)")
    parser.add_argument("--open", action="store_true", help="Open the dashboard in a browser once started")
    return parser


def _no_args_parser(name: str, description: str) -> argparse.ArgumentParser:
    """Parser for subcommands that take no arguments.

    Parsing before dispatch guarantees ``--help`` prints help (exit 0) and any
    unexpected argument is a usage error (exit 2) — in both cases *before* the
    side-effectful installer/uninstaller ever runs, so ``settings.json`` is
    never mutated by e.g. ``context-tracker down --help``.
    """
    return argparse.ArgumentParser(prog=f"context-tracker {name}", description=description)


def main() -> None:
    """Dispatch up/down/install/uninstall; delegate everything else to the MCP server CLI."""
    argv = sys.argv[1:]
    command = argv[0] if argv else None

    if command == "up":
        args = _up_parser().parse_args(argv[1:])
        up(
            install=not args.no_hooks,
            host=args.host,
            port=args.port,
            open_browser=args.open,
        )
    elif command == "down":
        _no_args_parser(
            "down",
            "Remove context-tracker's hook entries from ~/.claude/settings.json (other hooks are left untouched).",
        ).parse_args(argv[1:])
        down()
    elif command == "install":
        _no_args_parser(
            "install",
            "Install Claude Code hooks into ~/.claude/settings.json (no dashboard).",
        ).parse_args(argv[1:])
        install_hooks()
    elif command == "uninstall":
        _no_args_parser(
            "uninstall",
            "Remove context-tracker's hook entries from ~/.claude/settings.json (no dashboard).",
        ).parse_args(argv[1:])
        uninstall_hooks()
    else:
        # Backward compatibility: `context-tracker` (MCP server over stdio),
        # `context-tracker dashboard`, `--transport`, etc.
        from context_tracker.server import main as server_main

        server_main()


if __name__ == "__main__":
    main()
