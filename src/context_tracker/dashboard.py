"""FastAPI dashboard server for context analysis."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from context_tracker.analysis.config import StalenessConfig
from context_tracker.analysis.models import BlockType, ContextBlock
from context_tracker.analysis.staleness import (
    compute_staleness,
    detect_superseded,
    detect_task_boundaries,
)
from context_tracker.storage import DEFAULT_TRACE_DIR, list_sessions, read_events
from context_tracker.transcript_parser import parse_raw_transcript
from context_tracker.analysis.reconstruction import reconstruct_session

# Reuse the same session ID pattern as storage.py
_SESSION_ID_RE = re.compile(r'^[a-zA-Z0-9_-]+$')


def _validate_session_id(session_id: str) -> None:
    """Validate session ID format. Raises HTTPException(400) for invalid IDs."""
    if not _SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=400, detail="Invalid session ID")

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
        _validate_session_id(session_id)
        transcript_path = _find_transcript(session_id, transcript_dir)
        if transcript_path is None:
            events = read_events(session_id, trace_dir=trace_dir)
            if not events:
                raise HTTPException(status_code=404, detail="Session not found")

        return {"session_id": session_id, "status": "ok"}

    @app.get("/api/session/{session_id}/turns")
    def get_session_turns(session_id: str):
        """Per-turn summary data for sediment chart and scorecards."""
        _validate_session_id(session_id)
        transcript_path = _find_transcript(session_id, transcript_dir)
        if transcript_path is None:
            raise HTTPException(status_code=404, detail="Transcript not found")

        messages, warnings = parse_raw_transcript(transcript_path)
        hook_events = read_events(session_id, trace_dir=trace_dir)
        turns, snapshots, content_store, epochs, recon_warnings, block_registry = (
            reconstruct_session(messages, hook_events)
        )

        config = StalenessConfig()

        # P2-4: Detect task boundaries from turns
        task_boundaries = detect_task_boundaries(turns, config)

        # P2-3: Build assistant text indexed by turn number for reference scanning
        assistant_texts_by_turn: dict[int, list[str]] = {}
        for bid, block in block_registry.items():
            if block.block_type == BlockType.ASSISTANT_TEXT:
                tn = block.turn_entered
                if tn not in assistant_texts_by_turn:
                    assistant_texts_by_turn[tn] = []
                text = content_store.get_content(bid)
                if text:
                    assistant_texts_by_turn[tn].append(text)

        # Build resource_last_used map incrementally per turn
        resource_last_used: dict[str, int] = {}
        # P2-2: Track blocks seen so far for incremental superseded computation
        blocks_seen_so_far: list[ContextBlock] = []

        turn_data = []
        for snap in snapshots:
            # Update resource_last_used for blocks entering this turn
            for bid in snap.blocks_entered_ids:
                block = block_registry.get(bid)
                if block and block.resource:
                    resource_last_used[block.resource] = snap.turn_number
                # P2-2: Accumulate blocks for incremental superseded map
                if block:
                    blocks_seen_so_far.append(block)

            # P2-2: Compute superseded only from blocks seen up to this turn
            superseded = detect_superseded(blocks_seen_so_far)

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

                # P2-3: Gather assistant messages from turns after block entered
                messages_since: list[str] = []
                for t in range(block.turn_entered + 1, snap.turn_number + 1):
                    messages_since.extend(assistant_texts_by_turn.get(t, []))

                score_val, label = compute_staleness(
                    block=block,
                    current_turn=snap.turn_number,
                    config=config,
                    resource_last_used=resource_last_used,
                    messages_since_block=messages_since,
                    active_resources=set(resource_last_used.keys()),
                    task_boundaries=task_boundaries,
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
                "actual_context_tokens": snap.actual_context_tokens,
                "block_count": block_count,
                "stale_block_count": stale_count + dead_count,
                "input_tokens": snap.input_tokens,
                "output_tokens": snap.output_tokens,
                "cache_read_tokens": snap.cache_read_tokens,
                "cache_creation_tokens": snap.cache_creation_tokens,
                "compaction_detected": snap.compaction_detected,
                "epoch": snap.epoch,
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
        _validate_session_id(session_id)
        transcript_path = _find_transcript(session_id, transcript_dir)
        if transcript_path is None:
            raise HTTPException(status_code=404, detail="Transcript not found")

        messages, _ = parse_raw_transcript(transcript_path)
        hook_events = read_events(session_id, trace_dir=trace_dir)
        turns, snapshots, content_store, epochs, _, block_registry = (
            reconstruct_session(messages, hook_events)
        )

        config = StalenessConfig()
        task_boundaries = detect_task_boundaries(turns, config)

        # Build resource_last_used from all turns (final state for /blocks)
        resource_last_used: dict[str, int] = {}
        for snap in snapshots:
            for bid in snap.blocks_entered_ids:
                block = block_registry.get(bid)
                if block and block.resource:
                    resource_last_used[block.resource] = snap.turn_number

        # Superseded at final turn uses all blocks (correct for final snapshot)
        superseded = detect_superseded(list(block_registry.values()))

        # Build assistant text by turn for reference scanning
        assistant_texts_by_turn: dict[int, list[str]] = {}
        for bid, blk in block_registry.items():
            if blk.block_type == BlockType.ASSISTANT_TEXT:
                tn = blk.turn_entered
                if tn not in assistant_texts_by_turn:
                    assistant_texts_by_turn[tn] = []
                text = content_store.get_content(bid)
                if text:
                    assistant_texts_by_turn[tn].append(text)

        last_turn = snapshots[-1].turn_number if snapshots else 0

        blocks_out = []
        for bid in (snapshots[-1].block_ids if snapshots else []):
            block = block_registry.get(bid)
            if not block:
                continue

            # Gather assistant messages from turns after block entered
            messages_since: list[str] = []
            for t in range(block.turn_entered + 1, last_turn + 1):
                messages_since.extend(assistant_texts_by_turn.get(t, []))

            score_val, label = compute_staleness(
                block=block,
                current_turn=last_turn,
                config=config,
                resource_last_used=resource_last_used,
                messages_since_block=messages_since,
                active_resources=set(resource_last_used.keys()),
                task_boundaries=task_boundaries,
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
        _validate_session_id(session_id)
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

    @app.get("/")
    def serve_dashboard():
        # Prefer v3 dashboard, fall back to v2
        v3 = static_dir / "dashboard-v3.html"
        if v3.exists():
            return FileResponse(str(v3))
        v2 = static_dir / "context-scope.html"
        if v2.exists():
            return FileResponse(str(v2))
        return HTMLResponse("<h1>Context Analyzer</h1><p>Run ccscope build first.</p>")

    @app.get("/blocks.json")
    def get_blocks_json():
        blocks_path = static_dir / "blocks.json"
        if blocks_path.exists():
            return FileResponse(str(blocks_path), media_type="application/json")
        raise HTTPException(status_code=404, detail="Run ccscope build first")

    @app.get("/churn.json")
    def get_churn_json():
        churn_path = static_dir / "churn.json"
        if churn_path.exists():
            return FileResponse(str(churn_path), media_type="application/json")
        raise HTTPException(status_code=404, detail="Run ccscope build first")

    # Serve static files if directory exists
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app


def main() -> None:
    """Entry point: context-tracker dashboard."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Context Analyzer Dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9201)
    args = parser.parse_args()

    if args.host != "127.0.0.1":
        import sys
        print(
            f"WARNING: Binding to {args.host} exposes session data on the network. "
            "Use 127.0.0.1 (default) to restrict access to localhost.",
            file=sys.stderr,
        )

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
