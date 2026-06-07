"""FastAPI dashboard server for context analysis."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from context_tracker.analysis.config import StalenessConfig
from context_tracker.analysis.health import (
    build_health_signals,
    classify_recommendation,
    compute_urgency,
    generate_recommendations,
)
from context_tracker.analysis.models import BlockType, ContextBlock
from context_tracker.analysis.reconstruction import reconstruct_session
from context_tracker.analysis.staleness import (
    compute_staleness,
    detect_superseded,
    detect_task_boundaries,
)
from context_tracker.ccscope.tokens import image_dimensions, image_tokens
from context_tracker.db import (
    DEFAULT_DB_PATH,
    BlockRecord,
    HookEventRecord,
    SessionRecord,
    SubagentRecord,
    TurnRecord,
    get_engine,
    get_session_factory,
)
from context_tracker.ingest import ingest_session
from context_tracker.storage import DEFAULT_TRACE_DIR, list_sessions, read_events
from context_tracker.transcript_parser import parse_raw_transcript

logger = logging.getLogger(__name__)

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


def _ensure_ingested(
    session_id: str,
    trace_dir: Path,
    db_path: Path,
    projects_dir: Path | None = None,
) -> SessionRecord | None:
    """Get session from SQLite, auto-ingesting if missing."""
    engine = get_engine(db_path)
    factory = get_session_factory(engine)
    with factory() as db:
        existing = db.get(SessionRecord, session_id)
        if existing:
            return existing
    # Not in DB — try to ingest
    return ingest_session(
        session_id, trace_dir=trace_dir, db_path=db_path, projects_dir=projects_dir,
    )


def _extract_image_metadata(source: dict, media_type: str) -> dict:
    """Extract width/height/tokens from an image source block."""
    source_type = source.get("type", "base64")
    w, h = 1024, 1024  # fallback
    mt = media_type

    if source_type == "base64":
        mt = source.get("media_type", media_type)
        b64_data = source.get("data", "")
        w, h = image_dimensions(b64_data, mt)
    elif source_type == "url":
        url = source.get("url", "")
        if url.startswith("data:"):
            # Parse data URI: data:image/png;base64,<data>
            try:
                header, b64_data = url.split(",", 1)
                mt = header.split(":")[1].split(";")[0] if ":" in header else media_type
                w, h = image_dimensions(b64_data, mt)
            except (ValueError, IndexError):
                pass
        # For http(s) URLs, we can't fetch dimensions without network — use fallback

    return {
        "media_type": mt,
        "width": w,
        "height": h,
        "tokens": image_tokens(w, h),
    }


def _extract_content_blocks_with_images(
    content: str | list,
    entry_type: str,
    timestamp: str,
    item: dict | None = None,
) -> list[dict]:
    """Extract message dicts from raw content, including image metadata.

    This shared helper is used by both get_call_content and
    get_conv_turn_content to ensure consistent flattening order.

    When item is provided, it's the top-level content block dict (e.g.,
    a tool_result or image block). When content is a string or list,
    item can be None and we process the content directly.

    Returns a list of message dicts ready for the API response.
    """
    messages: list[dict] = []

    if isinstance(content, str) and content:
        messages.append({
            "type": "user" if entry_type == "user" else "assistant",
            "role": "user" if entry_type == "user" else "assistant",
            "content": content[:8000],
            "size_chars": len(content),
            "is_truncated": len(content) > 8000,
            "timestamp": timestamp,
        })
        return messages

    if not isinstance(content, list):
        return messages

    for block_item in content:
        if not isinstance(block_item, dict):
            continue
        block_type = block_item.get("type", "")

        if block_type == "text":
            text = block_item.get("text", "")
            messages.append({
                "type": "assistant_text" if entry_type == "assistant" else "user_text",
                "role": entry_type,
                "content": text[:8000],
                "size_chars": len(text),
                "is_truncated": len(text) > 8000,
                "timestamp": timestamp,
            })

        elif block_type == "thinking":
            text = block_item.get("thinking", "")
            messages.append({
                "type": "thinking",
                "role": "assistant",
                "content": text[:8000],
                "size_chars": len(text),
                "is_truncated": len(text) > 8000,
                "timestamp": timestamp,
            })

        elif block_type == "tool_use":
            tool_input = block_item.get("input", {})
            input_str = (
                json.dumps(tool_input, indent=2)
                if isinstance(tool_input, dict)
                else str(tool_input)
            )
            tool_name = block_item.get("name", "unknown")
            resource = ""
            if tool_name in ("Read", "Edit", "Write"):
                resource = (
                    tool_input.get("file_path", "")
                    if isinstance(tool_input, dict)
                    else ""
                )
            elif tool_name == "Bash":
                resource = (
                    tool_input.get("command", "")[:100]
                    if isinstance(tool_input, dict)
                    else ""
                )
            messages.append({
                "type": "tool_use",
                "role": "assistant",
                "tool_name": tool_name,
                "resource": resource,
                "content": input_str[:8000],
                "size_chars": len(input_str),
                "is_truncated": len(input_str) > 8000,
                "tool_use_id": block_item.get("id", ""),
                "timestamp": timestamp,
            })

        elif block_type == "tool_result":
            result_content = block_item.get("content", "")
            images_meta: list[dict] = []
            if isinstance(result_content, list):
                text_parts = []
                img_idx = 0
                for sub in result_content:
                    if not isinstance(sub, dict):
                        text_parts.append(str(sub))
                        continue
                    if sub.get("type") == "image":
                        source = sub.get("source", {})
                        meta = _extract_image_metadata(
                            source,
                            source.get("media_type", "image/png"),
                        )
                        meta["index"] = img_idx
                        images_meta.append(meta)
                        img_idx += 1
                    else:
                        text_parts.append(sub.get("text", ""))
                result_content = "\n".join(text_parts)
            elif not isinstance(result_content, str):
                result_content = str(result_content)

            msg_dict: dict = {
                "type": "tool_result",
                "role": "tool",
                "content": result_content[:8000],
                "size_chars": len(result_content),
                "is_truncated": len(result_content) > 8000,
                "is_error": bool(block_item.get("is_error", False)),
                "tool_use_id": block_item.get("tool_use_id", ""),
                "timestamp": timestamp,
            }
            if images_meta:
                msg_dict["images"] = images_meta
            messages.append(msg_dict)

        elif block_type == "image":
            # Top-level image block (user-pasted screenshot)
            source = block_item.get("source", {})
            meta = _extract_image_metadata(
                source,
                source.get("media_type", "image/png"),
            )
            meta["index"] = 0
            messages.append({
                "type": "image",
                "role": entry_type,
                "content": f"[image: {meta['media_type']} {meta['width']}x{meta['height']}]",
                "size_chars": 0,
                "is_truncated": False,
                "timestamp": timestamp,
                "images": [meta],
            })

    return messages


def _walk_transcript_for_range(
    transcript_path: Path,
    first_call: int,
    last_call: int,
) -> list[dict]:
    """Walk a transcript JSONL and return raw entries for a range of API calls.

    Shared between get_conv_turn_content and the image endpoint to ensure
    consistent flattening order (msg_index alignment).
    """
    entries_raw: list[dict] = []
    api_call_idx = -1
    pending_user_entries: list[dict] = []

    with open(transcript_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_type = entry.get("type", "")
            if entry_type in (
                "file-history-snapshot", "last-prompt", "pr-link", "queue-operation",
            ):
                continue

            if entry_type == "user":
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

                if first_call <= api_call_idx <= last_call:
                    entries_raw.extend(pending_user_entries)
                    entries_raw.append(entry)
                elif api_call_idx > last_call:
                    break

                pending_user_entries = []

    return entries_raw


def _flatten_entries_to_messages(entries_raw: list[dict]) -> list[dict]:
    """Flatten raw transcript entries into message dicts using shared helper.

    Returns list of message dicts with images[] metadata where applicable.
    """
    messages_out: list[dict] = []
    for entry in entries_raw:
        entry_type = entry.get("type", "")
        message = entry.get("message", {})
        content = message.get("content", "")
        timestamp = entry.get("timestamp", "")

        msgs = _extract_content_blocks_with_images(
            content, entry_type, timestamp,
        )
        messages_out.extend(msgs)
    return messages_out


def _serve_image_source(source: dict) -> dict:
    """Return the appropriate response for an image source block.

    Branches on source.type:
    - base64: returns {"data_uri": "data:{media_type};base64,{data}"}
    - url with data: prefix: returns {"data_uri": url}
    - url with http(s): returns {"url": url}
    """
    source_type = source.get("type", "base64")

    if source_type == "base64":
        media_type = source.get("media_type", "image/png")
        data = source.get("data", "")
        return {"data_uri": f"data:{media_type};base64,{data}"}

    if source_type == "url":
        url = source.get("url", "")
        if url.startswith("data:"):
            return {"data_uri": url}
        return {"url": url}

    # Unknown source type — try base64 as fallback
    media_type = source.get("media_type", "image/png")
    data = source.get("data", "")
    if data:
        return {"data_uri": f"data:{media_type};base64,{data}"}
    return {"error": "Unknown image source type"}


def create_app(
    trace_dir: Path = DEFAULT_TRACE_DIR,
    transcript_dir: Path = DEFAULT_TRANSCRIPT_DIR,
    static_dir: Path = DEFAULT_STATIC_DIR,
    db_path: Path = DEFAULT_DB_PATH,
) -> FastAPI:
    app = FastAPI(title="Context Analyzer", version="0.4.0")

    @app.get("/api/health")
    def health_check():
        return {"status": "ok"}

    @app.get("/api/sessions")
    def get_sessions():
        """List all sessions with summary stats from SQLite."""
        session_ids = list_sessions(trace_dir=trace_dir)
        results = []
        engine = get_engine(db_path)
        factory = get_session_factory(engine)
        with factory() as db:
            for sid in session_ids:
                rec = db.get(SessionRecord, sid)
                if not rec:
                    # Auto-ingest on first access
                    rec = _ensure_ingested(sid, trace_dir, db_path, transcript_dir)
                    if not rec:
                        results.append({"session_id": sid})
                        continue
                    # Re-fetch inside this session
                    rec = db.get(SessionRecord, sid)
                    if not rec:
                        results.append({"session_id": sid})
                        continue
                # Get first turn prompt for session label
                first_turn = (
                    db.query(TurnRecord)
                    .filter_by(session_id=sid, turn_number=0)
                    .first()
                )
                first_prompt = ""
                if first_turn and first_turn.prompt_preview:
                    first_prompt = first_turn.prompt_preview[:80]

                results.append({
                    "session_id": rec.session_id,
                    "model": rec.model,
                    "total_turns": rec.total_turns,
                    "total_api_calls": rec.total_api_calls,
                    "total_blocks": rec.total_blocks,
                    "peak_context_tokens": rec.peak_context_tokens,
                    "total_input_tokens": rec.total_input_tokens,
                    "total_output_tokens": rec.total_output_tokens,
                    "total_cache_read": rec.total_cache_read,
                    "total_cost_usd": rec.total_cost_usd,
                    "source_mtime": rec.source_mtime,
                    "first_prompt": first_prompt,
                })
        return results

    @app.get("/api/session/{session_id}/summary")
    def get_session_summary(session_id: str):
        """Full session summary from SQLite."""
        _validate_session_id(session_id)
        rec = _ensure_ingested(session_id, trace_dir, db_path, transcript_dir)
        if not rec:
            raise HTTPException(status_code=404, detail="Session not found")

        engine = get_engine(db_path)
        factory = get_session_factory(engine)
        with factory() as db:
            rec = db.get(SessionRecord, session_id)
            if not rec:
                raise HTTPException(status_code=404, detail="Session not found")

            block_types = {}
            for b in db.query(BlockRecord).filter_by(session_id=session_id):
                block_types[b.block_type] = block_types.get(b.block_type, 0) + 1

            hook_types = {}
            for h in db.query(HookEventRecord).filter_by(session_id=session_id):
                hook_types[h.event_type] = hook_types.get(h.event_type, 0) + 1

            subagent_count = db.query(SubagentRecord).filter_by(session_id=session_id).count()

            return {
                "session_id": rec.session_id,
                "model": rec.model,
                "total_turns": rec.total_turns,
                "total_api_calls": rec.total_api_calls,
                "total_blocks": rec.total_blocks,
                "peak_context_tokens": rec.peak_context_tokens,
                "total_input_tokens": rec.total_input_tokens,
                "total_output_tokens": rec.total_output_tokens,
                "total_cache_read": rec.total_cache_read,
                "total_cache_creation": rec.total_cache_creation,
                "total_cost_usd": rec.total_cost_usd,
                "health_score": rec.health_score,
                "block_type_counts": block_types,
                "hook_event_counts": hook_types,
                "subagent_count": subagent_count,
            }

    @app.get("/api/sessions/trends")
    def get_session_trends():
        """Cross-session aggregation for trend analysis."""
        session_ids = list_sessions(trace_dir=trace_dir)
        engine = get_engine(db_path)
        factory = get_session_factory(engine)

        trends = []
        with factory() as db:
            for sid in session_ids:
                rec = db.get(SessionRecord, sid)
                if not rec:
                    rec_obj = _ensure_ingested(sid, trace_dir, db_path, transcript_dir)
                    if not rec_obj:
                        continue
                    rec = db.get(SessionRecord, sid)
                    if not rec:
                        continue
                trends.append({
                    "session_id": rec.session_id,
                    "model": rec.model,
                    "total_turns": rec.total_turns,
                    "total_api_calls": rec.total_api_calls,
                    "peak_context_tokens": rec.peak_context_tokens,
                    "total_cache_read": rec.total_cache_read,
                    "total_output_tokens": rec.total_output_tokens,
                    "total_cost_usd": rec.total_cost_usd,
                    "source_mtime": rec.source_mtime,
                })

        return {
            "session_count": len(trends),
            "total_cost": round(sum(t["total_cost_usd"] for t in trends), 2),
            "total_cache_read": sum(t["total_cache_read"] for t in trends),
            "total_api_calls": sum(t["total_api_calls"] for t in trends),
            "sessions": trends,
        }

    @app.get("/api/session/{session_id}/data")
    def get_session_data(session_id: str):
        """Full session data (blocks + churn + meta + turn_map) for the dashboard.

        This replaces the need for `ccscope build` — returns all the data
        the v3 dashboard needs to render charts and inspector.
        """
        _validate_session_id(session_id)

        from context_tracker.ccscope.parse_transcript import build_turn_map
        from context_tracker.ccscope.reconcile import find_session_paths as find_paths
        from context_tracker.ccscope.reconcile import reconcile

        try:
            blocks, churn, subagents = reconcile(
                session_id, projects_dir=transcript_dir, trace_dir=trace_dir,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Session transcript not found") from exc

        # Build turn map
        paths = find_paths(session_id, projects_dir=transcript_dir, trace_dir=trace_dir)
        turn_map = []
        if paths.get("transcript"):
            turn_map = build_turn_map(Path(paths["transcript"]))

        return {
            "blocks": blocks,
            "churn": churn,
            "meta": {"session_id": session_id},
            "turn_map": turn_map,
        }

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

    @app.get("/api/session/{session_id}/health")
    def get_session_health(session_id: str):
        """Context health score with signals and recommendations."""
        _validate_session_id(session_id)
        transcript_path = _find_transcript(session_id, transcript_dir)
        if transcript_path is None:
            raise HTTPException(status_code=404, detail="Transcript not found")

        messages, _ = parse_raw_transcript(transcript_path)
        hook_events = read_events(session_id, trace_dir=trace_dir)
        turns, snapshots, content_store, epochs, _, block_registry = (
            reconstruct_session(messages, hook_events)
        )

        # Detect model
        model = "unknown"
        for msg in messages:
            if msg.model:
                model = msg.model
                break

        from context_tracker.analysis.config import HealthConfig

        health_config = HealthConfig()
        staleness_config = StalenessConfig()

        signals = build_health_signals(
            turns=turns,
            snapshots=snapshots,
            block_registry=block_registry,
            content_store=content_store,
            model=model,
            config=health_config,
            staleness_config=staleness_config,
        )

        urgency = compute_urgency(signals, health_config)
        classification = classify_recommendation(urgency, health_config)
        health_score = round(1.0 - urgency, 4)

        recommendations = generate_recommendations(
            signals=signals,
            block_registry=block_registry,
            snapshots=snapshots,
            config=health_config,
            staleness_config=staleness_config,
        )

        return {
            "health_score": health_score,
            "urgency_score": round(urgency, 4),
            "classification": classification,
            "signals": {
                "turn_number": signals.turn_number,
                "dead_weight_ratio": signals.dead_weight_ratio,
                "context_utilization": signals.context_utilization,
                "cache_efficiency": signals.cache_efficiency,
                "cache_efficiency_trend": signals.cache_efficiency_trend,
                "repeated_reads": signals.repeated_reads,
                "error_rate": signals.error_rate,
                "error_rate_spike": signals.error_rate_spike,
                "output_inflation": signals.output_inflation,
                "edit_churn": signals.edit_churn,
                "compaction_count": signals.compaction_count,
                "cost_this_turn": signals.cost_this_turn,
                "cost_cumulative": signals.cost_cumulative,
            },
            "recommendations": recommendations,
        }

    @app.get("/api/session/{session_id}/dead_weight")
    def get_session_dead_weight(session_id: str):
        """Per-turn dead weight data and top stale blocks."""
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

        resource_last_used: dict[str, int] = {}
        blocks_seen_so_far: list[ContextBlock] = []

        per_turn = []
        all_stale_blocks: list[dict] = []

        for snap in snapshots:
            for bid in snap.blocks_entered_ids:
                block = block_registry.get(bid)
                if block:
                    blocks_seen_so_far.append(block)
                    if block.resource:
                        resource_last_used[block.resource] = snap.turn_number

            superseded = detect_superseded(blocks_seen_so_far)

            active_tokens = 0
            stale_tokens = 0
            dead_weight_tokens = 0
            total_tokens = 0

            for bid in snap.block_ids:
                block = block_registry.get(bid)
                if not block:
                    continue

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
                total_tokens += tokens
                if label in ("active", "warm", "pinned"):
                    active_tokens += tokens
                elif label == "stale":
                    stale_tokens += tokens
                else:
                    dead_weight_tokens += tokens

            total = active_tokens + stale_tokens + dead_weight_tokens
            dead_pct = dead_weight_tokens / total if total > 0 else 0.0

            per_turn.append({
                "turn": snap.turn_number,
                "dead_weight_tokens": dead_weight_tokens,
                "dead_weight_pct": round(dead_pct, 4),
                "stale_tokens": stale_tokens,
                "active_tokens": active_tokens,
            })

        # Top stale blocks at the final snapshot
        if snapshots:
            final_snap = snapshots[-1]
            last_turn = final_snap.turn_number
            superseded_final = detect_superseded(list(block_registry.values()))

            for bid in final_snap.block_ids:
                block = block_registry.get(bid)
                if not block:
                    continue

                messages_since = []
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
                    superseded_map=superseded_final,
                )

                if label in ("stale", "dead_weight"):
                    all_stale_blocks.append({
                        "block_id": block.block_id,
                        "block_type": block.block_type.value,
                        "resource": block.resource,
                        "size_tokens_est": block.size_tokens_est,
                        "staleness_score": round(score_val, 3),
                        "staleness_label": label,
                    })

        # Sort by size descending, take top 15
        all_stale_blocks.sort(key=lambda b: b["size_tokens_est"], reverse=True)
        top_blocks = all_stale_blocks[:15]

        # Summary
        dead_pcts = [t["dead_weight_pct"] for t in per_turn]
        peak_dead = max(dead_pcts) if dead_pcts else 0.0
        avg_dead = sum(dead_pcts) / len(dead_pcts) if dead_pcts else 0.0

        return {
            "summary": {
                "peak_dead_weight_pct": round(peak_dead, 4),
                "avg_dead_weight_pct": round(avg_dead, 4),
            },
            "top_blocks": top_blocks,
            "per_turn": per_turn,
        }

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

        if not transcript_path.exists():
            raise HTTPException(status_code=404, detail="Transcript not found")

        target_entries = _walk_transcript_for_range(
            transcript_path, call_index, call_index,
        )

        if not target_entries:
            raise HTTPException(status_code=404, detail=f"API call {call_index} not found")

        messages_out = _flatten_entries_to_messages(target_entries)

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

    @app.get("/api/session/{session_id}/conv_turn/{conv_turn}/content")
    def get_conv_turn_content(session_id: str, conv_turn: int):
        """Full content for a conversation turn (all API calls in the turn).

        conv_turn is 1-based. Returns all messages across all API calls
        that belong to this conversation turn.
        """
        _validate_session_id(session_id)

        # Find transcript first
        transcript_path = _find_transcript(session_id, transcript_dir)
        if transcript_path is None or not transcript_path.exists():
            raise HTTPException(status_code=404, detail="Transcript not found")

        # Build turn_map on the fly (no ccscope build required)
        from context_tracker.ccscope.parse_transcript import build_turn_map
        turn_map_data = build_turn_map(transcript_path)

        # Find the entry for this conv_turn
        turn_entry = None
        for entry in turn_map_data:
            if entry["conv_turn"] == conv_turn:
                turn_entry = entry
                break
        if turn_entry is None:
            raise HTTPException(status_code=404, detail=f"Conversation turn {conv_turn} not found")

        first_call = turn_entry["first_call"]
        last_call = turn_entry["last_call"]

        entries_raw = _walk_transcript_for_range(
            transcript_path, first_call, last_call,
        )

        if not entries_raw:
            raise HTTPException(status_code=404, detail=f"No entries found for conversation turn {conv_turn}")

        # Compute usage from assistant entries
        total_usage = {"input_tokens": 0, "output_tokens": 0, "cache_read": 0, "cache_creation": 0}
        for entry in entries_raw:
            if entry.get("type") == "assistant":
                usage = entry.get("message", {}).get("usage", {})
                total_usage["input_tokens"] += usage.get("input_tokens", 0)
                total_usage["output_tokens"] += usage.get("output_tokens", 0)
                total_usage["cache_read"] += usage.get("cache_read_input_tokens", 0)
                total_usage["cache_creation"] += usage.get("cache_creation_input_tokens", 0)

        messages_out = _flatten_entries_to_messages(entries_raw)

        return {
            "conv_turn": conv_turn,
            "first_call": first_call,
            "last_call": last_call,
            "api_call_count": last_call - first_call + 1,
            "messages": messages_out,
            "usage": total_usage,
        }

    @app.get("/api/session/{session_id}/conv_turn/{conv_turn}/image/{msg_index}/{img_index}")
    def get_conv_turn_image(
        session_id: str, conv_turn: int, msg_index: int, img_index: int,
    ):
        """Serve a specific image from a conversation turn.

        msg_index is the 0-based index into the flattened messages array
        (same order as returned by get_conv_turn_content).
        img_index is the 0-based index into that message's images[] array.
        """
        _validate_session_id(session_id)

        transcript_path = _find_transcript(session_id, transcript_dir)
        if transcript_path is None or not transcript_path.exists():
            raise HTTPException(status_code=404, detail="Transcript not found")

        from context_tracker.ccscope.parse_transcript import build_turn_map
        turn_map_data = build_turn_map(transcript_path)

        turn_entry = None
        for entry in turn_map_data:
            if entry["conv_turn"] == conv_turn:
                turn_entry = entry
                break
        if turn_entry is None:
            raise HTTPException(status_code=404, detail=f"Conversation turn {conv_turn} not found")

        first_call = turn_entry["first_call"]
        last_call = turn_entry["last_call"]

        entries_raw = _walk_transcript_for_range(
            transcript_path, first_call, last_call,
        )
        if not entries_raw:
            raise HTTPException(status_code=404, detail="No entries found for this turn")

        # Flatten using the same logic to get consistent msg_index
        # But we need the RAW content blocks, not the truncated messages.
        # Walk entries the same way as _flatten_entries_to_messages but
        # track which raw image sub-blocks correspond to each msg_index.
        flat_msg_idx = -1
        for entry in entries_raw:
            message = entry.get("message", {})
            content = message.get("content", "")

            if isinstance(content, str) and content:
                flat_msg_idx += 1
                if flat_msg_idx == msg_index:
                    raise HTTPException(
                        status_code=404,
                        detail="Message at this index has no images",
                    )
                continue

            if not isinstance(content, list):
                continue

            for block_item in content:
                if not isinstance(block_item, dict):
                    continue
                block_type = block_item.get("type", "")

                if block_type in ("text", "thinking", "tool_use"):
                    flat_msg_idx += 1
                    if flat_msg_idx == msg_index:
                        raise HTTPException(
                            status_code=404,
                            detail="Message at this index has no images",
                        )

                elif block_type == "tool_result":
                    flat_msg_idx += 1
                    if flat_msg_idx == msg_index:
                        # Find the img_index-th image in this tool_result
                        result_content = block_item.get("content", "")
                        if not isinstance(result_content, list):
                            raise HTTPException(
                                status_code=404,
                                detail="tool_result has no image sub-blocks",
                            )
                        current_img = -1
                        for sub in result_content:
                            if not isinstance(sub, dict):
                                continue
                            if sub.get("type") == "image":
                                current_img += 1
                                if current_img == img_index:
                                    return _serve_image_source(sub.get("source", {}))
                        raise HTTPException(
                            status_code=404,
                            detail=f"Image index {img_index} not found in message",
                        )

                elif block_type == "image":
                    flat_msg_idx += 1
                    if flat_msg_idx == msg_index:
                        if img_index != 0:
                            raise HTTPException(
                                status_code=404,
                                detail=f"Image index {img_index} not found",
                            )
                        return _serve_image_source(block_item.get("source", {}))

        raise HTTPException(
            status_code=404,
            detail=f"Message index {msg_index} not found",
        )

    @app.get("/sessions")
    def serve_sessions_page():
        """Cross-session overview page."""
        sessions_html = static_dir / "sessions.html"
        if sessions_html.exists():
            return FileResponse(str(sessions_html))
        return HTMLResponse("<h1>Sessions</h1><p>sessions.html not found</p>")

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

    @app.get("/turn_map.json")
    def get_turn_map_json():
        path = static_dir / "turn_map.json"
        if path.exists():
            return FileResponse(str(path), media_type="application/json")
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
