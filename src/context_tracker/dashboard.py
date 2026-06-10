"""FastAPI dashboard server for context analysis."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
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
    SubagentApiCallRecord,
    SubagentRecord,
    TurnRecord,
    WorkflowRunRecord,
    get_engine,
    get_session_factory,
)
from context_tracker.ingest import ingest_session
from context_tracker.storage import DEFAULT_TRACE_DIR, list_sessions, read_events
from context_tracker.transcript_parser import parse_raw_transcript

logger = logging.getLogger(__name__)

# Reuse the same session ID pattern as storage.py
_SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


# L3: Self-correction regex patterns
SELF_CORRECTION_HIGH = [
    re.compile(r"I (?:made|introduced) (?:an?|the) error", re.IGNORECASE),
    re.compile(r"(?:that|this) (?:was|is) (?:wrong|incorrect|a bug)", re.IGNORECASE),
    re.compile(r"I (?:accidentally|mistakenly)", re.IGNORECASE),
    re.compile(
        r"(?:let me|I(?:'ll| will)) (?:fix|correct|revert) (?:that|this)",
        re.IGNORECASE,
    ),
]
SELF_CORRECTION_MEDIUM = [
    re.compile(r"I apologize", re.IGNORECASE),
    re.compile(r"(?:actually|wait),? I (?:need|should) to", re.IGNORECASE),
    re.compile(r"I (?:forgot|missed|overlooked)", re.IGNORECASE),
    re.compile(
        r"(?:that|the previous) (?:approach|change) (?:didn't|won't) work",
        re.IGNORECASE,
    ),
]


def _detect_self_corrections(
    text: str,
) -> tuple[bool, str, str]:
    """Test text against self-correction patterns.

    Returns (matched, confidence, pattern_str).
    """
    for pat in SELF_CORRECTION_HIGH:
        m = pat.search(text)
        if m:
            return True, "high", m.group(0)
    for pat in SELF_CORRECTION_MEDIUM:
        m = pat.search(text)
        if m:
            return True, "medium", m.group(0)
    return False, "", ""


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
        existing: SessionRecord | None = db.get(SessionRecord, session_id)
        if existing:
            return existing
    # Not in DB — try to ingest
    return ingest_session(
        session_id,
        trace_dir=trace_dir,
        db_path=db_path,
        projects_dir=projects_dir,
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
) -> list[dict]:
    """Extract message dicts from raw content, including image metadata.

    This shared helper is used by both get_call_content and
    get_conv_turn_content to ensure consistent flattening order.

    Returns a list of message dicts ready for the API response.
    """
    messages: list[dict] = []

    if isinstance(content, str) and content:
        messages.append(
            {
                "type": "user" if entry_type == "user" else "assistant",
                "role": "user" if entry_type == "user" else "assistant",
                "content": content[:8000],
                "size_chars": len(content),
                "is_truncated": len(content) > 8000,
                "timestamp": timestamp,
            }
        )
        return messages

    if not isinstance(content, list):
        return messages

    for block_item in content:
        if not isinstance(block_item, dict):
            continue
        block_type = block_item.get("type", "")

        if block_type == "text":
            text = block_item.get("text", "")
            messages.append(
                {
                    "type": "assistant_text" if entry_type == "assistant" else "user_text",
                    "role": entry_type,
                    "content": text[:8000],
                    "size_chars": len(text),
                    "is_truncated": len(text) > 8000,
                    "timestamp": timestamp,
                }
            )

        elif block_type == "thinking":
            text = block_item.get("thinking", "")
            messages.append(
                {
                    "type": "thinking",
                    "role": "assistant",
                    "content": text[:8000],
                    "size_chars": len(text),
                    "is_truncated": len(text) > 8000,
                    "timestamp": timestamp,
                }
            )

        elif block_type == "tool_use":
            tool_input = block_item.get("input", {})
            input_str = json.dumps(tool_input, indent=2) if isinstance(tool_input, dict) else str(tool_input)
            tool_name = block_item.get("name", "unknown")
            resource = ""
            if tool_name in ("Read", "Edit", "Write"):
                resource = tool_input.get("file_path", "") if isinstance(tool_input, dict) else ""
            elif tool_name == "Bash":
                resource = tool_input.get("command", "")[:100] if isinstance(tool_input, dict) else ""
            messages.append(
                {
                    "type": "tool_use",
                    "role": "assistant",
                    "tool_name": tool_name,
                    "resource": resource,
                    "content": input_str[:8000],
                    "size_chars": len(input_str),
                    "is_truncated": len(input_str) > 8000,
                    "tool_use_id": block_item.get("id", ""),
                    "timestamp": timestamp,
                }
            )

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
            messages.append(
                {
                    "type": "image",
                    "role": entry_type,
                    "content": f"[image: {meta['media_type']} {meta['width']}x{meta['height']}]",
                    "size_chars": 0,
                    "is_truncated": False,
                    "timestamp": timestamp,
                    "images": [meta],
                }
            )

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
                "file-history-snapshot",
                "last-prompt",
                "pr-link",
                "queue-operation",
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
            content,
            entry_type,
            timestamp,
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
    def health_check() -> dict:
        return {"status": "ok"}

    @app.get("/api/sessions")
    def get_sessions() -> list:
        """List all sessions with summary stats from SQLite."""
        session_ids = list_sessions(trace_dir=trace_dir, projects_dir=transcript_dir)
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
                first_turn = db.query(TurnRecord).filter_by(session_id=sid, turn_number=0).first()
                first_prompt = ""
                if first_turn and first_turn.prompt_preview:
                    first_prompt = first_turn.prompt_preview[:80]

                results.append(
                    {
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
                    }
                )
        return results

    @app.get("/api/session/{session_id}/summary")
    def get_session_summary(session_id: str) -> dict:
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

            block_types: dict[str, int] = {}
            for b in db.query(BlockRecord).filter_by(session_id=session_id):
                block_types[b.block_type] = block_types.get(b.block_type, 0) + 1

            hook_types: dict[str, int] = {}
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
    def get_session_trends() -> dict:
        """Cross-session aggregation for trend analysis."""
        session_ids = list_sessions(trace_dir=trace_dir, projects_dir=transcript_dir)
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
                trends.append(
                    {
                        "session_id": rec.session_id,
                        "model": rec.model,
                        "total_turns": rec.total_turns,
                        "total_api_calls": rec.total_api_calls,
                        "peak_context_tokens": rec.peak_context_tokens,
                        "total_cache_read": rec.total_cache_read,
                        "total_output_tokens": rec.total_output_tokens,
                        "total_cost_usd": rec.total_cost_usd,
                        "source_mtime": rec.source_mtime,
                    }
                )

        return {
            "session_count": len(trends),
            "total_cost": round(sum(t["total_cost_usd"] for t in trends), 2),
            "total_cache_read": sum(t["total_cache_read"] for t in trends),
            "total_api_calls": sum(t["total_api_calls"] for t in trends),
            "sessions": trends,
        }

    @app.get("/api/session/{session_id}/data")
    def get_session_data(session_id: str) -> dict:
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
                session_id,
                projects_dir=transcript_dir,
                trace_dir=trace_dir,
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

    # ------------------------------------------------------------------
    # Tool Intelligence endpoint
    # ------------------------------------------------------------------
    @app.get("/api/session/{session_id}/tool-intelligence")
    def get_tool_intelligence(session_id: str) -> dict:
        """Classified tool breakdown for composition donut and tools tab."""
        _validate_session_id(session_id)
        transcript_path = _find_transcript(session_id, transcript_dir)
        if transcript_path is None:
            raise HTTPException(status_code=404, detail="Transcript not found")

        messages, _ = parse_raw_transcript(transcript_path)
        hook_events = read_events(session_id, trace_dir=trace_dir)
        _turns, snapshots, content_store, _epochs, _, block_registry = reconstruct_session(messages, hook_events)

        # ---- helpers ----
        def _classify(tool_name: str) -> str:
            if not tool_name:
                return "builtin"
            if tool_name.startswith("mcp__"):
                return "mcp"
            if tool_name == "Skill":
                return "skill"
            if tool_name in ("Agent", "Task"):
                return "agent"
            return "builtin"

        def _parse_mcp(tool_name: str) -> tuple[str, str]:
            parts = tool_name.split("__", 2)
            server = parts[1] if len(parts) > 1 else "unknown"
            func = parts[2] if len(parts) > 2 else "unknown"
            return server, func

        def _extract_skill_name(block_id: str) -> str:
            try:
                raw = content_store.get_content(block_id)
            except Exception:
                return "unknown"
            if raw:
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        return str(parsed.get("skill", "unknown"))
                except (json.JSONDecodeError, TypeError):
                    pass
            return "unknown"

        # ---- first pass: classify TOOL_USE blocks ----
        tool_use_id_map: dict[str, dict] = {}  # tool_use_id -> {category, tool_name, ...}
        # Accumulators
        category_chars: dict[str, int] = {
            "system_prefix": 0,
            "conversation": 0,
            "regular_tool": 0,
            "mcp_tool": 0,
            "skill": 0,
            "agent": 0,
        }
        mcp_servers: dict[str, dict] = {}  # server -> {total_chars, call_count, functions: {fn: count}}
        skills: dict[str, dict] = {}  # skill_name -> {chars, count}
        regular_tools: dict[str, dict] = {}  # tool_name -> {chars, count}
        agents: dict[str, dict] = {}  # tool_name -> {chars, count}

        for bid, block in block_registry.items():
            if block.block_type == BlockType.TOOL_USE:
                tn = block.tool_name or ""
                cat = _classify(tn)
                info: dict = {"category": cat, "tool_name": tn}

                if cat == "mcp":
                    srv, fn = _parse_mcp(tn)
                    info["mcp_server"] = srv
                    info["mcp_function"] = fn
                    if srv not in mcp_servers:
                        mcp_servers[srv] = {"total_chars": 0, "call_count": 0, "functions": {}}
                    mcp_servers[srv]["total_chars"] += block.size_chars
                    mcp_servers[srv]["call_count"] += 1
                    mcp_servers[srv]["functions"][fn] = mcp_servers[srv]["functions"].get(fn, 0) + 1
                    category_chars["mcp_tool"] += block.size_chars

                elif cat == "skill":
                    sname = _extract_skill_name(bid)
                    info["skill_name"] = sname
                    if sname not in skills:
                        skills[sname] = {"chars": 0, "count": 0}
                    skills[sname]["chars"] += block.size_chars
                    skills[sname]["count"] += 1
                    category_chars["skill"] += block.size_chars

                elif cat == "agent":
                    aname = tn  # "Agent" or "Task"
                    if aname not in agents:
                        agents[aname] = {"chars": 0, "count": 0}
                    agents[aname]["chars"] += block.size_chars
                    agents[aname]["count"] += 1
                    category_chars["agent"] += block.size_chars

                else:  # builtin
                    if tn not in regular_tools:
                        regular_tools[tn] = {"chars": 0, "count": 0}
                    regular_tools[tn]["chars"] += block.size_chars
                    regular_tools[tn]["count"] += 1
                    category_chars["regular_tool"] += block.size_chars

                if block.tool_use_id:
                    tool_use_id_map[block.tool_use_id] = info

            elif block.block_type == BlockType.TOOL_RESULT:
                # Classify via parent_block_id lookup
                parent = block_registry.get(block.parent_block_id or "") if block.parent_block_id else None
                cat = "builtin"
                tn = ""
                if parent and parent.tool_name:
                    tn = parent.tool_name
                    cat = _classify(tn)
                elif block.tool_use_id and block.tool_use_id in tool_use_id_map:
                    info = tool_use_id_map[block.tool_use_id]
                    cat = info["category"]
                    tn = info.get("tool_name", "")

                if cat == "mcp":
                    srv, fn = _parse_mcp(tn)
                    if srv in mcp_servers:
                        mcp_servers[srv]["total_chars"] += block.size_chars
                    category_chars["mcp_tool"] += block.size_chars
                elif cat == "skill":
                    sname = tool_use_id_map.get(block.tool_use_id or "", {}).get("skill_name", "unknown")
                    if sname in skills:
                        skills[sname]["chars"] += block.size_chars
                    category_chars["skill"] += block.size_chars
                elif cat == "agent":
                    if tn in agents:
                        agents[tn]["chars"] += block.size_chars
                    category_chars["agent"] += block.size_chars
                else:
                    rkey = tn or "unknown"
                    if rkey not in regular_tools:
                        regular_tools[rkey] = {"chars": 0, "count": 0}
                    regular_tools[rkey]["chars"] += block.size_chars
                    category_chars["regular_tool"] += block.size_chars

            elif block.is_pinned:
                category_chars["system_prefix"] += block.size_chars

            elif block.block_type in (BlockType.USER_PROMPT, BlockType.ASSISTANT_TEXT):
                category_chars["conversation"] += block.size_chars

            elif block.block_type == BlockType.SYSTEM:
                category_chars["system_prefix"] += block.size_chars

            elif block.block_type == BlockType.COMPACTION_SUMMARY:
                category_chars["conversation"] += block.size_chars

        # ---- proportion chars to tokens ----
        total_chars = sum(category_chars.values())
        # Use last snapshot's actual_context_tokens as total token reference
        total_tokens = 0
        if snapshots:
            last_snap = snapshots[-1]
            total_tokens = last_snap.actual_context_tokens or last_snap.total_tokens_est
        if total_tokens == 0:
            total_tokens = max(1, total_chars // 4)

        def _chars_to_tokens(chars: int) -> int:
            if total_chars == 0:
                return 0
            return round(chars / total_chars * total_tokens)

        composition = {
            "system_prefix_tokens": _chars_to_tokens(category_chars["system_prefix"]),
            "conversation_tokens": _chars_to_tokens(category_chars["conversation"]),
            "regular_tool_tokens": _chars_to_tokens(category_chars["regular_tool"]),
            "mcp_tool_tokens": _chars_to_tokens(category_chars["mcp_tool"]),
            "skill_tokens": _chars_to_tokens(category_chars["skill"]),
            "agent_tokens": _chars_to_tokens(category_chars["agent"]),
        }

        mcp_list = sorted(
            [
                {
                    "server": srv,
                    "total_tokens": _chars_to_tokens(d["total_chars"]),
                    "call_count": d["call_count"],
                    "functions": [
                        {"name": fn, "count": cnt} for fn, cnt in sorted(d["functions"].items(), key=lambda x: -x[1])
                    ],
                }
                for srv, d in mcp_servers.items()
            ],
            key=lambda x: -x["total_tokens"],
        )

        skills_list = sorted(
            [{"name": sn, "tokens": _chars_to_tokens(d["chars"]), "count": d["count"]} for sn, d in skills.items()],
            key=lambda x: -x["tokens"],
        )

        regular_list = sorted(
            [
                {"name": tn, "tokens": _chars_to_tokens(d["chars"]), "count": d["count"]}
                for tn, d in regular_tools.items()
            ],
            key=lambda x: -x["tokens"],
        )

        agents_list = sorted(
            [{"name": tn, "tokens": _chars_to_tokens(d["chars"]), "count": d["count"]} for tn, d in agents.items()],
            key=lambda x: -x["tokens"],
        )

        return {
            "composition": composition,
            "mcp_servers": mcp_list,
            "skills": skills_list,
            "regular_tools": regular_list,
            "agents": agents_list,
        }

    # ------------------------------------------------------------------
    # Subagents endpoint
    # ------------------------------------------------------------------
    @app.get("/api/session/{session_id}/subagents")
    def get_subagents(session_id: str) -> dict:
        """Expose SubagentRecord + SubagentApiCallRecord data from DB."""
        _validate_session_id(session_id)

        # Ensure session is ingested into DB (v3 may load via /data without DB)
        _ensure_ingested(session_id, trace_dir, db_path, transcript_dir)

        # Build agent_id -> launch conv_turn mapping from transcript.
        # Match Agent tool_use blocks in the transcript to SubagentRecords
        # by description text (tool_use_id != agent_id, so we can't match by ID).
        agent_launch_turns: dict[str, int] = {}
        transcript_path = _find_transcript(session_id, transcript_dir)
        if transcript_path and transcript_path.exists():
            from context_tracker.ccscope.parse_transcript import build_turn_map

            turn_map_data = build_turn_map(transcript_path)
            # Collect all Agent tool_use blocks with their descriptions and conv_turns
            agent_launches: list[dict[str, object]] = []
            api_call_idx = -1
            with open(transcript_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if entry.get("type") != "assistant":
                        continue
                    message = entry.get("message", {})
                    if message.get("stop_reason") is None:
                        continue
                    usage = message.get("usage", {})
                    if usage.get("output_tokens", 0) == 0:
                        continue
                    if message.get("model") == "synthetic":
                        continue
                    api_call_idx += 1
                    content = message.get("content", [])
                    if not isinstance(content, list):
                        continue
                    for block_item in content:
                        if not isinstance(block_item, dict):
                            continue
                        if block_item.get("type") == "tool_use" and block_item.get("name") in ("Agent", "Task"):
                            inp = block_item.get("input", {})
                            desc = inp.get("description", inp.get("prompt", "")) if isinstance(inp, dict) else ""
                            # Find conv_turn for this api_call_idx
                            conv_turn = None
                            for tm in turn_map_data:
                                if tm["first_call"] <= api_call_idx <= tm["last_call"]:
                                    conv_turn = tm["conv_turn"]
                                    break
                            if conv_turn is not None:
                                agent_launches.append({"desc": str(desc)[:80], "conv_turn": conv_turn})

        engine = get_engine(db_path)
        factory = get_session_factory(engine)
        with factory() as db:
            subagent_recs = db.query(SubagentRecord).filter_by(session_id=session_id).all()

            subagents_out = []
            total_peak = 0
            for sa in subagent_recs:
                # Load api calls
                api_calls = (
                    db.query(SubagentApiCallRecord)
                    .filter_by(subagent_id=sa.id)
                    .order_by(SubagentApiCallRecord.call_index)
                    .all()
                )

                # Compute total_output_tokens from churn sum (not populated by parse_subagents)
                computed_output = sum(c.output_tokens for c in api_calls)

                churn_data = [
                    {
                        "call_index": c.call_index,
                        "input_tokens": c.input_tokens,
                        "output_tokens": c.output_tokens,
                        "cache_read": c.cache_read,
                        "cache_creation": c.cache_creation,
                    }
                    for c in api_calls
                ]

                total_peak += sa.peak_resident

                # Match this subagent to its launch turn by description
                launch_turn = agent_launch_turns.get(sa.agent_id)
                if launch_turn is None and agent_launches:
                    sa_desc = (sa.description or "")[:80]
                    for al in agent_launches:
                        if sa_desc and al["desc"] and sa_desc in str(al["desc"]):
                            launch_turn = int(str(al["conv_turn"]))
                            agent_launches.remove(al)
                            break

                subagents_out.append(
                    {
                        "agent_id": sa.agent_id,
                        "agent_type": sa.agent_type or "unknown",
                        "description": sa.description or "",
                        "peak_resident": sa.peak_resident,
                        "total_cache_read": sa.total_cache_read,
                        "total_api_calls": sa.total_api_calls,
                        "total_output_tokens": computed_output or sa.total_output_tokens,
                        "churn": churn_data,
                        "launch_turn": launch_turn,
                    }
                )

        return {
            "count": len(subagents_out),
            "total_peak_tokens": total_peak,
            "subagents": subagents_out,
        }

    # ------------------------------------------------------------------
    # Workflows endpoint (multi-agent workflow runs)
    # ------------------------------------------------------------------
    @app.get("/api/session/{session_id}/workflows")
    def get_workflows(session_id: str) -> dict:
        """Expose WorkflowRunRecord data grouped run -> phase -> agents.

        Mirrors the /subagents endpoint's token-computation idioms (per-agent
        output is summed from SubagentApiCallRecord churn).
        """
        _validate_session_id(session_id)
        _ensure_ingested(session_id, trace_dir, db_path, transcript_dir)

        engine = get_engine(db_path)
        factory = get_session_factory(engine)
        with factory() as db:
            run_recs = db.query(WorkflowRunRecord).filter_by(session_id=session_id).all()

            workflows_out = []
            for run in run_recs:
                agent_recs = (
                    db.query(SubagentRecord)
                    .filter_by(session_id=session_id, workflow_id=run.id)
                    .all()
                )

                # Group agents by phase (preserve first-seen order of phases).
                phases: dict[str, dict] = {}
                phase_order: list[str] = []

                run_total_output = 0
                run_peak_resident = 0
                run_agent_count = 0

                for sa in agent_recs:
                    api_calls = (
                        db.query(SubagentApiCallRecord)
                        .filter_by(subagent_id=sa.id)
                        .order_by(SubagentApiCallRecord.call_index)
                        .all()
                    )
                    computed_output = sum(c.output_tokens for c in api_calls)
                    churn_data = [
                        {
                            "call_index": c.call_index,
                            "input_tokens": c.input_tokens,
                            "output_tokens": c.output_tokens,
                            "cache_read": c.cache_read,
                            "cache_creation": c.cache_creation,
                        }
                        for c in api_calls
                    ]

                    output_tokens = computed_output or sa.total_output_tokens
                    run_total_output += output_tokens
                    run_peak_resident = max(run_peak_resident, sa.peak_resident)
                    run_agent_count += 1

                    agent_out = {
                        "agent_id": sa.agent_id,
                        "agent_type": sa.agent_type or "unknown",
                        "description": sa.description or "",
                        "label": sa.label,
                        "peak_resident": sa.peak_resident,
                        "total_cache_read": sa.total_cache_read,
                        "total_api_calls": sa.total_api_calls,
                        "total_output_tokens": output_tokens,
                        "churn": churn_data,
                    }

                    phase_key = sa.phase or "(unphased)"
                    if phase_key not in phases:
                        phases[phase_key] = {
                            "phase": phase_key,
                            "label": sa.label,
                            "agents": [],
                        }
                        phase_order.append(phase_key)
                    bucket = phases[phase_key]
                    bucket["agents"].append(agent_out)
                    # Surface a human-readable label for the phase if available.
                    if bucket["label"] is None and sa.label:
                        bucket["label"] = sa.label

                # Parallelism: agents grouped under the same phase run together;
                # report the largest phase fan-out as a simple indication.
                max_phase_fanout = max((len(p["agents"]) for p in phases.values()), default=0)

                workflows_out.append(
                    {
                        "wf_id": run.wf_id,
                        "name": run.name or run.wf_id,
                        "started_at": run.started_at,
                        "ended_at": run.ended_at,
                        "total_agents": run_agent_count,
                        "total_phases": len(phase_order),
                        "total_output_tokens": run_total_output,
                        "peak_resident": run_peak_resident,
                        "max_parallelism": max_phase_fanout,
                        "phases": [phases[k] for k in phase_order],
                    }
                )

        return {
            "count": len(workflows_out),
            "workflows": workflows_out,
        }

    @app.get("/api/session/{session_id}/turns")
    def get_session_turns(session_id: str) -> dict:
        """Per-turn summary data for sediment chart and scorecards."""
        _validate_session_id(session_id)
        transcript_path = _find_transcript(session_id, transcript_dir)
        if transcript_path is None:
            raise HTTPException(status_code=404, detail="Transcript not found")

        messages, warnings = parse_raw_transcript(transcript_path)
        hook_events = read_events(session_id, trace_dir=trace_dir)
        turns, snapshots, content_store, epochs, recon_warnings, block_registry = reconstruct_session(
            messages, hook_events
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
                entered_blk: ContextBlock | None = block_registry.get(bid)
                if entered_blk and entered_blk.resource:
                    resource_last_used[entered_blk.resource] = snap.turn_number
                # P2-2: Accumulate blocks for incremental superseded map
                if entered_blk:
                    blocks_seen_so_far.append(entered_blk)

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
                maybe_block: ContextBlock | None = block_registry.get(bid)
                if not maybe_block:
                    continue
                block = maybe_block

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

            # Count errors and tool results entering this turn
            turn_error_count = 0
            turn_tool_result_count = 0
            for bid in snap.blocks_entered_ids:
                blk = block_registry.get(bid)
                if blk and blk.block_type == BlockType.TOOL_RESULT:
                    turn_tool_result_count += 1
                    if blk.is_error:
                        turn_error_count += 1

            turn_data.append(
                {
                    "turn": snap.turn_number,
                    "system_tokens": system_tokens,
                    "active_tokens": active_tokens,
                    "stale_tokens": stale_tokens + dead_weight_tokens,
                    "total_tokens": total,
                    "actual_context_tokens": snap.actual_context_tokens,
                    "block_count": block_count,
                    "stale_block_count": stale_count + dead_count,
                    "error_count": turn_error_count,
                    "tool_result_count": turn_tool_result_count,
                    "input_tokens": snap.input_tokens,
                    "output_tokens": snap.output_tokens,
                    "cache_read_tokens": snap.cache_read_tokens,
                    "cache_creation_tokens": snap.cache_creation_tokens,
                    "compaction_detected": snap.compaction_detected,
                    "epoch": snap.epoch,
                    "api_call_count": snap.api_call_count,
                }
            )

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
    def get_session_blocks(session_id: str) -> dict:
        """Block metadata for context tape. No content (lazy loaded)."""
        _validate_session_id(session_id)
        transcript_path = _find_transcript(session_id, transcript_dir)
        if transcript_path is None:
            raise HTTPException(status_code=404, detail="Transcript not found")

        messages, _ = parse_raw_transcript(transcript_path)
        hook_events = read_events(session_id, trace_dir=trace_dir)
        turns, snapshots, content_store, epochs, _, block_registry = reconstruct_session(messages, hook_events)

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
        for bid in snapshots[-1].block_ids if snapshots else []:
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
            blocks_out.append(
                {
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
                }
            )

        return {"session_id": session_id, "blocks": blocks_out}

    @app.get("/api/session/{session_id}/health")
    def get_session_health(session_id: str) -> dict:
        """Context health score with signals and recommendations."""
        _validate_session_id(session_id)
        transcript_path = _find_transcript(session_id, transcript_dir)
        if transcript_path is None:
            raise HTTPException(status_code=404, detail="Transcript not found")

        messages, _ = parse_raw_transcript(transcript_path)
        hook_events = read_events(session_id, trace_dir=trace_dir)
        turns, snapshots, content_store, epochs, _, block_registry = reconstruct_session(messages, hook_events)

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

    @app.get("/api/session/{session_id}/errors")
    def get_session_errors(session_id: str) -> dict:
        """Error analysis: tool failures, retry patterns, self-corrections."""
        _validate_session_id(session_id)
        transcript_path = _find_transcript(session_id, transcript_dir)
        if transcript_path is None:
            raise HTTPException(status_code=404, detail="Transcript not found")

        messages, _ = parse_raw_transcript(transcript_path)
        hook_events = read_events(session_id, trace_dir=trace_dir)
        turns, snapshots, content_store, epochs, _, block_registry = reconstruct_session(messages, hook_events)

        # --- L1: Tool failures ---
        # Count errors per turn, collect error block details
        per_turn_errors: dict[int, list[dict]] = defaultdict(list)
        total_tool_results = 0
        total_errors = 0

        for bid, block in block_registry.items():
            if block.block_type == BlockType.TOOL_RESULT:
                total_tool_results += 1
                if block.is_error:
                    total_errors += 1
                    # Look up tool_name via parent_block_id
                    tool_name = None
                    if block.parent_block_id:
                        parent = block_registry.get(block.parent_block_id)
                        if parent:
                            tool_name = parent.tool_name
                    per_turn_errors[block.turn_entered].append(
                        {
                            "block_id": bid,
                            "tool_name": tool_name or "unknown",
                            "size_chars": block.size_chars,
                        }
                    )

        error_rate = total_errors / total_tool_results if total_tool_results > 0 else 0.0

        per_turn_list = []
        for tn in sorted(per_turn_errors.keys()):
            per_turn_list.append(
                {
                    "turn": tn,
                    "error_count": len(per_turn_errors[tn]),
                    "errors": per_turn_errors[tn],
                }
            )

        # --- L2: Retry patterns (3-turn sliding window) ---
        # Collect (tool_name, turn) pairs for error blocks
        error_tool_turns: list[tuple[str, int]] = []
        for tn, errs in per_turn_errors.items():
            for err in errs:
                error_tool_turns.append((err["tool_name"], tn))

        # Group by turn
        errors_by_turn: dict[int, list[str]] = defaultdict(list)
        for tool_name, tn in error_tool_turns:
            errors_by_turn[tn].append(tool_name)

        all_error_turns = sorted(errors_by_turn.keys())
        retry_patterns = []
        seen_retries: set[tuple[str, int]] = set()  # (tool_name, window_start)

        for i, start_turn in enumerate(all_error_turns):
            # Gather tool names within a 3-turn window
            window_tools: list[tuple[str, int]] = []
            for tn in all_error_turns[i:]:
                if tn > start_turn + 2:
                    break
                for tname in errors_by_turn[tn]:
                    window_tools.append((tname, tn))

            # Group by tool_name
            tool_counts: dict[str, int] = defaultdict(int)
            for tname, _ in window_tools:
                tool_counts[tname] += 1

            for tname, count in tool_counts.items():
                if count >= 2:
                    key = (tname, start_turn)
                    if key not in seen_retries:
                        seen_retries.add(key)
                        retry_patterns.append(
                            {
                                "tool_name": tname,
                                "window_start_turn": start_turn,
                                "retry_count": count,
                            }
                        )

        # --- L3: Self-correction detection ---
        # Iterate raw transcript messages to find assistant visible text
        # (skip thinking blocks)
        self_corrections: list[dict] = []
        # Build a positional index of assistant messages for fallback turn assignment
        assistant_indices: list[int] = []
        for idx, msg in enumerate(messages):
            if msg.entry_type == "assistant":
                assistant_indices.append(idx)

        for msg in messages:
            if msg.entry_type != "assistant":
                continue
            # Find turn number for this message
            # Use timestamp matching against turns
            msg_turn = 0
            for turn in turns:
                if turn.timestamp is not None and msg.timestamp is not None:
                    if turn.timestamp <= msg.timestamp:
                        msg_turn = turn.turn_number
            if msg_turn == 0:
                # Fallback: if timestamp is null, infer turn from message position.
                # Use the 1-based position among assistant messages, capped to turn count.
                if msg.timestamp is None:
                    pos = assistant_indices.index(messages.index(msg)) + 1
                    msg_turn = min(pos, len(turns)) if turns else 0
                    if msg_turn > 0:
                        logger.warning(
                            "Null timestamp on assistant message; inferred turn %d from position",
                            msg_turn,
                        )
                if msg_turn == 0:
                    continue

            for cb in msg.content_blocks:
                # Only scan visible text, skip thinking blocks
                if cb.block_type != "text":
                    continue
                text = cb.content or ""
                if not text:
                    continue

                matched, confidence, pattern_str = _detect_self_corrections(text)
                if matched:
                    preview = text[:120].replace("\n", " ")
                    self_corrections.append(
                        {
                            "turn": msg_turn,
                            "confidence": confidence,
                            "pattern": pattern_str,
                            "preview": preview,
                        }
                    )

        # --- Cluster detection: consecutive turns with errors ---
        clusters: list[dict] = []
        if all_error_turns:
            cluster_start = all_error_turns[0]
            cluster_turns = [cluster_start]

            for j in range(1, len(all_error_turns)):
                if all_error_turns[j] <= all_error_turns[j - 1] + 1:
                    cluster_turns.append(all_error_turns[j])
                else:
                    if len(cluster_turns) >= 2:
                        cluster_total = sum(len(per_turn_errors[t]) for t in cluster_turns)
                        clusters.append(
                            {
                                "start_turn": cluster_turns[0],
                                "end_turn": cluster_turns[-1],
                                "turn_count": len(cluster_turns),
                                "total_errors": cluster_total,
                            }
                        )
                    cluster_start = all_error_turns[j]
                    cluster_turns = [cluster_start]

            # Final cluster
            if len(cluster_turns) >= 2:
                cluster_total = sum(len(per_turn_errors[t]) for t in cluster_turns)
                clusters.append(
                    {
                        "start_turn": cluster_turns[0],
                        "end_turn": cluster_turns[-1],
                        "turn_count": len(cluster_turns),
                        "total_errors": cluster_total,
                    }
                )

        # --- Build recommendations ---
        error_recommendations: list[dict] = []
        for cluster in clusters:
            error_recommendations.append(
                {
                    "priority": "critical",
                    "code": "error_cluster",
                    "title": (f"Error cluster: Turns {cluster['start_turn']}-{cluster['end_turn']}"),
                    "detail": (
                        f"{cluster['total_errors']} tool errors across {cluster['turn_count']} consecutive turns"
                    ),
                    "action": "Review the error cluster for systemic issues",
                    "tokens_recoverable": 0,
                    "target_turn": cluster["start_turn"],
                }
            )
        for rp in retry_patterns:
            error_recommendations.append(
                {
                    "priority": "warning",
                    "code": "retry_pattern",
                    "title": f"Retry pattern: {rp['tool_name']}",
                    "detail": (f"{rp['retry_count']} errors within 3 turns starting at turn {rp['window_start_turn']}"),
                    "action": "Check if the tool approach needs adjustment",
                    "tokens_recoverable": 0,
                    "target_turn": rp["window_start_turn"],
                }
            )
        if self_corrections:
            high_count = sum(1 for sc in self_corrections if sc["confidence"] == "high")
            med_count = sum(1 for sc in self_corrections if sc["confidence"] == "medium")
            sc_target = self_corrections[0]["turn"] if self_corrections else (len(turns) if turns else 1)
            error_recommendations.append(
                {
                    "priority": "warning",
                    "code": "self_correction",
                    "title": "Self-correction detected",
                    "detail": (
                        f"{len(self_corrections)} patterns found ({high_count} high / {med_count} medium confidence)"
                    ),
                    "action": "Review self-corrections for recurring issues",
                    "tokens_recoverable": 0,
                    "target_turn": sc_target,
                }
            )

        return {
            "session_id": session_id,
            "total_errors": total_errors,
            "total_tool_results": total_tool_results,
            "error_rate": round(error_rate, 4),
            "per_turn": per_turn_list,
            "clusters": clusters,
            "retry_patterns": retry_patterns,
            "self_corrections": self_corrections,
            "recommendations": error_recommendations,
        }

    @app.get("/api/session/{session_id}/dead_weight")
    def get_session_dead_weight(session_id: str) -> dict:
        """Per-turn dead weight data and top stale blocks."""
        _validate_session_id(session_id)
        transcript_path = _find_transcript(session_id, transcript_dir)
        if transcript_path is None:
            raise HTTPException(status_code=404, detail="Transcript not found")

        messages, _ = parse_raw_transcript(transcript_path)
        hook_events = read_events(session_id, trace_dir=trace_dir)
        turns, snapshots, content_store, epochs, _, block_registry = reconstruct_session(messages, hook_events)

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

            per_turn.append(
                {
                    "turn": snap.turn_number,
                    "dead_weight_tokens": dead_weight_tokens,
                    "dead_weight_pct": round(dead_pct, 4),
                    "stale_tokens": stale_tokens,
                    "active_tokens": active_tokens,
                }
            )

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
                    all_stale_blocks.append(
                        {
                            "block_id": block.block_id,
                            "block_type": block.block_type.value,
                            "resource": block.resource,
                            "size_tokens_est": block.size_tokens_est,
                            "staleness_score": round(score_val, 3),
                            "staleness_label": label,
                        }
                    )

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
    def get_turn_messages(session_id: str, turn_number: int) -> dict:
        """Full message content for a specific turn (drilldown)."""
        _validate_session_id(session_id)
        transcript_path = _find_transcript(session_id, transcript_dir)
        if transcript_path is None:
            raise HTTPException(status_code=404, detail="Transcript not found")

        messages, _ = parse_raw_transcript(transcript_path)
        hook_events = read_events(session_id, trace_dir=trace_dir)
        turns, snapshots, content_store, epochs, _, block_registry = reconstruct_session(messages, hook_events)

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
            msgs_out.append(
                {
                    "block_id": bid,
                    "block_type": block.block_type.value,
                    "tool_name": block.tool_name,
                    "resource": block.resource,
                    "size_chars": block.size_chars,
                    "size_tokens_est": block.size_tokens_est,
                    "content": content[:5000],
                    "is_truncated": len(content) > 5000,
                }
            )

        return {"turn": turn_number, "messages": msgs_out}

    # ------------------------------------------------------------------
    # Tool classification helper (shared by call + conv_turn content)
    # ------------------------------------------------------------------
    def _classify_tool_name(tool_name: str) -> tuple[str, str]:
        """Classify a tool name and return (category, display_name).

        Categories: 'mcp', 'skill', 'agent', 'builtin'.
        """
        if not tool_name or tool_name == "unknown":
            return "builtin", tool_name or "unknown"
        if tool_name.startswith("mcp__"):
            parts = tool_name.split("__", 2)
            server = parts[1] if len(parts) > 1 else "unknown"
            func = parts[2] if len(parts) > 2 else "unknown"
            return "mcp", f"{server}.{func}"
        if tool_name == "Skill":
            return "skill", "Skill"
        if tool_name in ("Agent", "Task"):
            return "agent", tool_name
        return "builtin", tool_name

    def _enrich_tool_fields(
        tool_name: str,
        tool_input: dict | None,
    ) -> dict:
        """Return tool_category + tool_display_name for a tool_use block."""
        cat, display = _classify_tool_name(tool_name)
        if cat == "skill" and isinstance(tool_input, dict):
            sname = tool_input.get("skill", "")
            if sname:
                display = sname
        elif cat == "agent" and isinstance(tool_input, dict):
            prompt = tool_input.get("prompt", "")
            if prompt:
                display = prompt[:40]
        return {"tool_category": cat, "tool_display_name": display}

    @app.get("/api/session/{session_id}/call/{call_index}/content")
    def get_call_content(session_id: str, call_index: int) -> dict:
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
            transcript_path,
            call_index,
            call_index,
        )

        if not target_entries:
            raise HTTPException(status_code=404, detail=f"API call {call_index} not found")

        messages_out = _flatten_entries_to_messages(target_entries)

        # Enrich tool_use / tool_result with category info (#43)
        _tu_cat_map: dict[str, dict] = {}
        for msg_item in messages_out:
            if msg_item.get("type") == "tool_use":
                tool_name = msg_item.get("tool_name", "unknown")
                tool_input = None
                try:
                    raw = msg_item.get("content", "")
                    tool_input = json.loads(raw) if raw else None
                except (json.JSONDecodeError, TypeError):
                    pass
                enriched = _enrich_tool_fields(tool_name, tool_input)
                msg_item.update(enriched)
                tu_id = msg_item.get("tool_use_id", "")
                if tu_id:
                    _tu_cat_map[tu_id] = enriched
            elif msg_item.get("type") == "tool_result":
                tu_id = msg_item.get("tool_use_id", "")
                result_enriched = _tu_cat_map.get(
                    tu_id,
                    {"tool_category": "builtin", "tool_display_name": "unknown"},
                )
                msg_item.update(result_enriched)

        # Add server-computed error flags to messages (#41)
        # Note: retry detection requires multi-turn context (comparing tool names
        # across consecutive turns), which is not available in this single-call
        # scope. All messages are marked is_retry=False here.
        for msg_item in messages_out:
            msg_item["is_retry"] = False

            msg_type = msg_item.get("type", "")
            if msg_type == "assistant_text":
                text = msg_item.get("content", "")
                matched, confidence, _ = _detect_self_corrections(text)
                msg_item["is_self_correction"] = matched
                msg_item["self_correction_confidence"] = confidence if matched else None
            else:
                msg_item["is_self_correction"] = False
                msg_item["self_correction_confidence"] = None

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
    def get_conv_turn_content(session_id: str, conv_turn: int) -> dict:
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
            transcript_path,
            first_call,
            last_call,
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

        # Enrich tool_use / tool_result with category info (#43)
        _ct_cat_map: dict[str, dict] = {}
        for msg_item in messages_out:
            if msg_item.get("type") == "tool_use":
                tool_name = msg_item.get("tool_name", "unknown")
                tool_input = None
                try:
                    raw = msg_item.get("content", "")
                    tool_input = json.loads(raw) if raw else None
                except (json.JSONDecodeError, TypeError):
                    pass
                enriched = _enrich_tool_fields(tool_name, tool_input)
                msg_item.update(enriched)
                tu_id = msg_item.get("tool_use_id", "")
                if tu_id:
                    _ct_cat_map[tu_id] = enriched
            elif msg_item.get("type") == "tool_result":
                tu_id = msg_item.get("tool_use_id", "")
                result_enriched = _ct_cat_map.get(
                    tu_id,
                    {"tool_category": "builtin", "tool_display_name": "unknown"},
                )
                msg_item.update(result_enriched)

        # --- Server-computed error flags (#41) ---
        # Build tool_use_id -> tool_name map from tool_use entries
        tu_id_to_name: dict[str, str] = {}
        for msg_item in messages_out:
            if msg_item.get("type") == "tool_use" and msg_item.get("tool_use_id"):
                tu_id_to_name[msg_item["tool_use_id"]] = msg_item.get("tool_name", "unknown")

        # Collect error tool_use_ids with their tool_names in this turn
        error_tool_names_this_turn: list[str] = []
        for msg_item in messages_out:
            if msg_item.get("type") == "tool_result" and msg_item.get("is_error"):
                tuid = msg_item.get("tool_use_id", "")
                tname = tu_id_to_name.get(tuid, "unknown")
                error_tool_names_this_turn.append(tname)

        # Lightweight retry detection: scan nearby turns for errors
        # with the same tool_name (3-turn window centered on this turn)
        # Re-scan transcript for error tool_names in turns [conv_turn-1, conv_turn+1]
        nearby_error_tools: list[str] = list(error_tool_names_this_turn)
        neighbor_turns = set()
        for t_entry in turn_map_data:
            ct = t_entry["conv_turn"]
            if abs(ct - conv_turn) <= 1 and ct != conv_turn:
                neighbor_turns.add((t_entry["first_call"], t_entry["last_call"]))

        if neighbor_turns:
            # Re-scan transcript for neighbor turns
            neighbor_api_idx = -1
            neighbor_tu_map: dict[str, str] = {}
            with open(transcript_path) as f2:
                for nline in f2:
                    nline = nline.strip()
                    if not nline:
                        continue
                    try:
                        nentry = json.loads(nline)
                    except json.JSONDecodeError:
                        continue
                    ntype = nentry.get("type", "")
                    if ntype in ("file-history-snapshot", "last-prompt", "pr-link", "queue-operation"):
                        continue
                    if ntype == "user":
                        ncontent = nentry.get("message", {}).get("content", "")
                        if isinstance(ncontent, list):
                            for ni in ncontent:
                                if isinstance(ni, dict) and ni.get("type") == "tool_result":
                                    tuid = ni.get("tool_use_id", "")
                                    if ni.get("is_error") and tuid in neighbor_tu_map:
                                        nearby_error_tools.append(neighbor_tu_map[tuid])
                        continue
                    if ntype == "assistant":
                        nmsg = nentry.get("message", {})
                        nusage = nmsg.get("usage", {})
                        if nmsg.get("stop_reason") is None or nusage.get("output_tokens", 0) == 0:
                            continue
                        if nmsg.get("model", "") == "synthetic":
                            continue
                        neighbor_api_idx += 1
                        in_neighbor = any(fc <= neighbor_api_idx <= lc for fc, lc in neighbor_turns)
                        if in_neighbor:
                            ncontent = nmsg.get("content", "")
                            if isinstance(ncontent, list):
                                for ni in ncontent:
                                    if isinstance(ni, dict):
                                        if ni.get("type") == "tool_use":
                                            neighbor_tu_map[ni.get("id", "")] = ni.get("name", "unknown")
                                        elif ni.get("type") == "tool_result":
                                            tuid = ni.get("tool_use_id", "")
                                            if ni.get("is_error") and tuid in neighbor_tu_map:
                                                nearby_error_tools.append(neighbor_tu_map[tuid])

        # Count error tool_names -- retry if same name appears 2+ times
        retry_tool_names = set()
        tool_name_counts = Counter(nearby_error_tools)
        for tname, cnt in tool_name_counts.items():
            if cnt >= 2:
                retry_tool_names.add(tname)

        # Annotate messages_out with is_retry, is_self_correction
        for msg_item in messages_out:
            msg_type = msg_item.get("type", "")

            if msg_type == "tool_result":
                tuid = msg_item.get("tool_use_id", "")
                tname = tu_id_to_name.get(tuid, "unknown")
                msg_item["is_retry"] = bool(msg_item.get("is_error") and tname in retry_tool_names)
            else:
                msg_item["is_retry"] = False

            if msg_type == "assistant_text":
                text = msg_item.get("content", "")
                matched, confidence, _ = _detect_self_corrections(text)
                msg_item["is_self_correction"] = matched
                msg_item["self_correction_confidence"] = confidence if matched else None
            else:
                msg_item["is_self_correction"] = False
                msg_item["self_correction_confidence"] = None

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
        session_id: str,
        conv_turn: int,
        msg_index: int,
        img_index: int,
    ) -> dict:  # type: ignore[return-value]
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
            transcript_path,
            first_call,
            last_call,
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
    def serve_sessions_page() -> Response:
        """Cross-session overview page."""
        sessions_html = static_dir / "sessions.html"
        if sessions_html.exists():
            return FileResponse(str(sessions_html))
        return HTMLResponse("<h1>Sessions</h1><p>sessions.html not found</p>")

    @app.get("/workflows")
    def serve_workflows_page() -> Response:
        """Multi-agent workflow run viewer."""
        workflows_html = static_dir / "workflows.html"
        if workflows_html.exists():
            return FileResponse(str(workflows_html))
        return HTMLResponse("<h1>Workflows</h1><p>workflows.html not found</p>")

    @app.get("/")
    def serve_dashboard() -> Response:
        # Prefer v3 dashboard, fall back to v2
        v3 = static_dir / "dashboard-v3.html"
        if v3.exists():
            return FileResponse(str(v3))
        v2 = static_dir / "context-scope.html"
        if v2.exists():
            return FileResponse(str(v2))
        return HTMLResponse("<h1>Context Analyzer</h1><p>Run ccscope build first.</p>")

    @app.get("/blocks.json")
    def get_blocks_json() -> Response:
        blocks_path = static_dir / "blocks.json"
        if blocks_path.exists():
            return FileResponse(str(blocks_path), media_type="application/json")
        raise HTTPException(status_code=404, detail="Run ccscope build first")

    @app.get("/churn.json")
    def get_churn_json() -> Response:
        churn_path = static_dir / "churn.json"
        if churn_path.exists():
            return FileResponse(str(churn_path), media_type="application/json")
        raise HTTPException(status_code=404, detail="Run ccscope build first")

    @app.get("/meta.json")
    def get_meta_json() -> Response:
        meta_path = static_dir / "meta.json"
        if meta_path.exists():
            return FileResponse(str(meta_path), media_type="application/json")
        raise HTTPException(status_code=404, detail="No meta.json")

    @app.get("/turn_map.json")
    def get_turn_map_json() -> Response:
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
