"""FastAPI dashboard server for context analysis."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from context_tracker.storage import DEFAULT_TRACE_DIR, list_sessions, read_events
from context_tracker.transcript_parser import parse_raw_transcript
from context_tracker.analysis.reconstruction import reconstruct_session

DEFAULT_TRANSCRIPT_DIR = Path.home() / ".claude" / "projects"
DEFAULT_STATIC_DIR = Path(__file__).parent.parent.parent / "static"


def _find_transcript(session_id: str, transcript_dir: Path) -> Path | None:
    direct = transcript_dir / f"{session_id}.jsonl"
    if direct.exists():
        return direct
    for jsonl_file in transcript_dir.rglob(f"{session_id}.jsonl"):
        return jsonl_file
    return None


def create_app(
    trace_dir: Path = DEFAULT_TRACE_DIR,
    transcript_dir: Path = DEFAULT_TRANSCRIPT_DIR,
    static_dir: Path = DEFAULT_STATIC_DIR,
) -> FastAPI:
    app = FastAPI(title="Context Analyzer", version="0.2.0")

    @app.get("/api/health")
    def health_check():
        return {"status": "ok"}

    @app.get("/api/sessions")
    def get_sessions():
        sessions = list_sessions(trace_dir=trace_dir)
        return sessions

    @app.get("/api/session/{session_id}/summary")
    def get_session_summary(session_id: str):
        transcript_path = _find_transcript(session_id, transcript_dir)
        if transcript_path is None:
            events = read_events(session_id, trace_dir=trace_dir)
            if not events:
                raise HTTPException(status_code=404, detail="Session not found")

        return {"session_id": session_id, "status": "ok"}

    @app.get("/api/session/{session_id}/turns")
    def get_session_turns(session_id: str):
        transcript_path = _find_transcript(session_id, transcript_dir)
        if transcript_path is None:
            raise HTTPException(status_code=404, detail="Transcript not found")

        messages, warnings = parse_raw_transcript(transcript_path)
        hook_events = read_events(session_id, trace_dir=trace_dir)
        turns, snapshots, content_store, epochs, recon_warnings = reconstruct_session(
            messages, hook_events
        )

        return {
            "turn_count": len(turns),
            "snapshot_count": len(snapshots),
            "block_count": len(content_store),
            "epoch_count": len(epochs),
            "warnings": len(warnings) + len(recon_warnings),
        }

    # Serve static files if directory exists
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/")
        def serve_dashboard():
            index = static_dir / "dashboard.html"
            if index.exists():
                return FileResponse(str(index))
            return HTMLResponse("<h1>Context Analyzer</h1><p>Dashboard not built yet.</p>")

    return app


def main() -> None:
    """Entry point: context-tracker dashboard."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Context Analyzer Dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9201)
    args = parser.parse_args()

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
