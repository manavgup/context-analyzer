"""FastAPI dashboard server for context analysis."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from context_tracker.analysis.config import StalenessConfig
from context_tracker.analysis.staleness import compute_staleness, detect_superseded
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
    app = FastAPI(title="Context Analyzer", version="0.3.0")

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
        """Per-turn summary data for sediment chart and scorecards."""
        transcript_path = _find_transcript(session_id, transcript_dir)
        if transcript_path is None:
            raise HTTPException(status_code=404, detail="Transcript not found")

        messages, warnings = parse_raw_transcript(transcript_path)
        hook_events = read_events(session_id, trace_dir=trace_dir)
        turns, snapshots, content_store, epochs, recon_warnings, block_registry = (
            reconstruct_session(messages, hook_events)
        )

        config = StalenessConfig()
        superseded = detect_superseded(list(block_registry.values()))

        # Build resource_last_used map incrementally per turn
        resource_last_used: dict[str, int] = {}

        turn_data = []
        for snap in snapshots:
            # Update resource_last_used for blocks entering this turn
            for bid in snap.blocks_entered_ids:
                block = block_registry.get(bid)
                if block and block.resource:
                    resource_last_used[block.resource] = snap.turn_number

            # Score all blocks at this turn
            system_tokens = 0
            active_tokens = 0
            stale_tokens = 0
            dead_weight_tokens = 0
            block_count = len(snap.block_ids)
            stale_count = 0
            dead_count = 0

            for bid in snap.block_ids:
                block = block_registry.get(bid)
                if not block:
                    continue

                score_val, label = compute_staleness(
                    block=block,
                    current_turn=snap.turn_number,
                    config=config,
                    resource_last_used=resource_last_used,
                    messages_since_block=[],
                    active_resources=set(resource_last_used.keys()),
                    task_boundaries=[],
                    superseded_map=superseded,
                )

                tokens = block.size_tokens_est
                if block.is_pinned or label == "pinned":
                    system_tokens += tokens
                elif label in ("active", "warm"):
                    active_tokens += tokens
                elif label == "stale":
                    stale_tokens += tokens
                    stale_count += 1
                else:  # dead_weight
                    dead_weight_tokens += tokens
                    dead_count += 1

            total = system_tokens + active_tokens + stale_tokens + dead_weight_tokens

            turn_data.append({
                "turn": snap.turn_number,
                "system_tokens": system_tokens,
                "active_tokens": active_tokens,
                "stale_tokens": stale_tokens + dead_weight_tokens,
                "total_tokens": total,
                "block_count": block_count,
                "stale_block_count": stale_count + dead_count,
                "input_tokens": snap.input_tokens,
                "output_tokens": snap.output_tokens,
                "cache_read_tokens": snap.cache_read_tokens,
                "cache_creation_tokens": snap.cache_creation_tokens,
                "api_call_count": snap.api_call_count,
            })

        # Detect model from messages
        model = "unknown"
        for msg in messages:
            if msg.model:
                model = msg.model
                break

        return {
            "session_id": session_id,
            "model": model,
            "turn_count": len(turns),
            "block_count": len(block_registry),
            "epoch_count": len(epochs),
            "turns": turn_data,
        }

    @app.get("/api/session/{session_id}/blocks")
    def get_session_blocks(session_id: str):
        """Block metadata for context tape. No content (lazy loaded)."""
        transcript_path = _find_transcript(session_id, transcript_dir)
        if transcript_path is None:
            raise HTTPException(status_code=404, detail="Transcript not found")

        messages, _ = parse_raw_transcript(transcript_path)
        hook_events = read_events(session_id, trace_dir=trace_dir)
        turns, snapshots, content_store, epochs, _, block_registry = (
            reconstruct_session(messages, hook_events)
        )

        config = StalenessConfig()
        superseded = detect_superseded(list(block_registry.values()))

        # Build resource_last_used from all turns
        resource_last_used: dict[str, int] = {}
        for snap in snapshots:
            for bid in snap.blocks_entered_ids:
                block = block_registry.get(bid)
                if block and block.resource:
                    resource_last_used[block.resource] = snap.turn_number

        last_turn = snapshots[-1].turn_number if snapshots else 0

        blocks_out = []
        for bid in (snapshots[-1].block_ids if snapshots else []):
            block = block_registry.get(bid)
            if not block:
                continue
            score_val, label = compute_staleness(
                block=block,
                current_turn=last_turn,
                config=config,
                resource_last_used=resource_last_used,
                messages_since_block=[],
                active_resources=set(resource_last_used.keys()),
                task_boundaries=[],
                superseded_map=superseded,
            )
            blocks_out.append({
                "block_id": block.block_id,
                "turn_entered": block.turn_entered,
                "block_type": block.block_type.value,
                "resource": block.resource,
                "tool_name": block.tool_name,
                "size_chars": block.size_chars,
                "size_tokens_est": block.size_tokens_est,
                "is_pinned": block.is_pinned,
                "staleness_score": round(score_val, 3),
                "staleness_label": label,
            })

        return {"session_id": session_id, "blocks": blocks_out}

    @app.get("/api/session/{session_id}/turn/{turn_number}/messages")
    def get_turn_messages(session_id: str, turn_number: int):
        """Full message content for a specific turn (drilldown)."""
        transcript_path = _find_transcript(session_id, transcript_dir)
        if transcript_path is None:
            raise HTTPException(status_code=404, detail="Transcript not found")

        messages, _ = parse_raw_transcript(transcript_path)
        hook_events = read_events(session_id, trace_dir=trace_dir)
        turns, snapshots, content_store, epochs, _, block_registry = (
            reconstruct_session(messages, hook_events)
        )

        if turn_number < 1 or turn_number > len(turns):
            raise HTTPException(status_code=404, detail="Turn not found")

        # Find blocks that entered in this turn
        snap = snapshots[turn_number - 1] if turn_number <= len(snapshots) else None
        if not snap:
            return {"turn": turn_number, "messages": []}

        msgs_out = []
        for bid in snap.blocks_entered_ids:
            block = block_registry.get(bid)
            if not block:
                continue
            content = content_store.get_content(bid)
            msgs_out.append({
                "block_id": bid,
                "block_type": block.block_type.value,
                "tool_name": block.tool_name,
                "resource": block.resource,
                "size_chars": block.size_chars,
                "size_tokens_est": block.size_tokens_est,
                "content": content[:5000],
                "is_truncated": len(content) > 5000,
            })

        return {"turn": turn_number, "messages": msgs_out}

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
