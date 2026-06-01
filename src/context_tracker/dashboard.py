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

    @app.get("/api/session/{session_id}/call/{call_index}/content")
    def get_call_content(session_id: str, call_index: int):
        """Full message content for a specific API call (by call index 0-based).

        Uses the ccscope transcript parser which indexes by API call,
        matching the v3 dashboard's turn numbering.
        """
        _validate_session_id(session_id)
        transcript_path = _find_transcript(session_id, transcript_dir)
        if transcript_path is None:
            raise HTTPException(status_code=404, detail="Transcript not found")

        import json

        # Parse transcript entries directly — we need the raw content
        # for the specific API call
        if not transcript_path.exists():
            raise HTTPException(status_code=404, detail="Transcript not found")

        entries = []
        api_call_idx = -1
        target_entries = []
        pending_user_entries = []

        with open(transcript_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                entry_type = entry.get("type", "")
                if entry_type in ("file-history-snapshot", "last-prompt", "pr-link", "queue-operation"):
                    continue

                if entry_type == "user":
                    # Buffer user messages — they belong to the next API call
                    pending_user_entries.append(entry)
                    continue

                if entry_type == "assistant":
                    message = entry.get("message", {})
                    usage = message.get("usage", {})
                    stop_reason = message.get("stop_reason")
                    output_tokens = usage.get("output_tokens", 0)
                    model = message.get("model", "")

                    if stop_reason is None or output_tokens == 0 or model == "synthetic":
                        continue

                    api_call_idx += 1

                    if api_call_idx == call_index:
                        # This is the API call we want
                        target_entries = list(pending_user_entries)
                        target_entries.append(entry)
                        break

                    # Clear pending for next call
                    pending_user_entries = []

        if not target_entries:
            raise HTTPException(status_code=404, detail=f"API call {call_index} not found")

        # Extract content blocks from the target entries
        messages_out = []
        for entry in target_entries:
            entry_type = entry.get("type", "")
            message = entry.get("message", {})
            content = message.get("content", "")
            timestamp = entry.get("timestamp", "")

            if isinstance(content, str) and content:
                messages_out.append({
                    "type": "user" if entry_type == "user" else "assistant",
                    "role": "user" if entry_type == "user" else "assistant",
                    "content": content[:8000],
                    "size_chars": len(content),
                    "is_truncated": len(content) > 8000,
                    "timestamp": timestamp,
                })
            elif isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    block_type = item.get("type", "")

                    if block_type == "text":
                        text = item.get("text", "")
                        messages_out.append({
                            "type": "assistant_text" if entry_type == "assistant" else "user_text",
                            "role": entry_type,
                            "content": text[:8000],
                            "size_chars": len(text),
                            "is_truncated": len(text) > 8000,
                            "timestamp": timestamp,
                        })

                    elif block_type == "thinking":
                        text = item.get("thinking", "")
                        messages_out.append({
                            "type": "thinking",
                            "role": "assistant",
                            "content": text[:8000],
                            "size_chars": len(text),
                            "is_truncated": len(text) > 8000,
                            "timestamp": timestamp,
                        })

                    elif block_type == "tool_use":
                        tool_input = item.get("input", {})
                        input_str = json.dumps(tool_input, indent=2) if isinstance(tool_input, dict) else str(tool_input)
                        tool_name = item.get("name", "unknown")
                        resource = ""
                        if tool_name in ("Read", "Edit", "Write"):
                            resource = tool_input.get("file_path", "") if isinstance(tool_input, dict) else ""
                        elif tool_name == "Bash":
                            resource = (tool_input.get("command", "")[:100] if isinstance(tool_input, dict) else "")
                        messages_out.append({
                            "type": "tool_use",
                            "role": "assistant",
                            "tool_name": tool_name,
                            "resource": resource,
                            "content": input_str[:8000],
                            "size_chars": len(input_str),
                            "is_truncated": len(input_str) > 8000,
                            "tool_use_id": item.get("id", ""),
                            "timestamp": timestamp,
                        })

                    elif block_type == "tool_result":
                        result_content = item.get("content", "")
                        if isinstance(result_content, list):
                            result_content = "\n".join(
                                b.get("text", "") if isinstance(b, dict) else str(b)
                                for b in result_content
                            )
                        elif not isinstance(result_content, str):
                            result_content = str(result_content)
                        messages_out.append({
                            "type": "tool_result",
                            "role": "tool",
                            "content": result_content[:8000],
                            "size_chars": len(result_content),
                            "is_truncated": len(result_content) > 8000,
                            "is_error": bool(item.get("is_error", False)),
                            "tool_use_id": item.get("tool_use_id", ""),
                            "timestamp": timestamp,
                        })

        # Add usage info from the assistant entry
        for entry in target_entries:
            if entry.get("type") == "assistant":
                usage = entry.get("message", {}).get("usage", {})
                return {
                    "call_index": call_index,
                    "messages": messages_out,
                    "usage": {
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                        "cache_read": usage.get("cache_read_input_tokens", 0),
                        "cache_creation": usage.get("cache_creation_input_tokens", 0),
                    },
                }

        return {"call_index": call_index, "messages": messages_out, "usage": {}}

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

    @app.get("/meta.json")
    def get_meta_json():
        meta_path = static_dir / "meta.json"
        if meta_path.exists():
            return FileResponse(str(meta_path), media_type="application/json")
        raise HTTPException(status_code=404, detail="No meta.json")

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
