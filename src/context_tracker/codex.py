"""Parse OpenAI Codex CLI session rollouts into Context Scope blocks + churn.

Codex CLI stores each session as a JSONL "rollout" file under
``~/.codex/sessions/YYYY/MM/DD/rollout-<timestamp>-<uuid>.jsonl``.

Record types (top-level ``type`` field):

- ``session_meta``: session id, cwd, cli_version, instructions (AGENTS.md)
- ``turn_context``: per-turn model / sandbox settings
- ``response_item``: conversation items -- payload.type in
  {message, reasoning, function_call, function_call_output, ...}
- ``event_msg``: UI events -- payload.type ``token_count`` carries usage info
  (``last_token_usage``: input_tokens, cached_input_tokens, output_tokens)

The parser is tolerant of unknown record types: they are skipped, never fatal.
Output uses the same blocks/churn dict format as
:mod:`context_tracker.ccscope.parse_transcript` so the existing SQLite schema
and dashboard render Codex sessions unchanged.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from context_tracker.ccscope.parse_transcript import (
    COMPACTION_THRESHOLD,
    _distribute_tokens,
    _mark_compaction,
)

DEFAULT_CODEX_SESSIONS_DIR = Path.home() / ".codex" / "sessions"

MAX_CONTENT_CHARS = 500

# UUID suffix in rollout filenames: rollout-<timestamp>-<uuid>.jsonl
_ROLLOUT_RE = re.compile(r"^rollout-.*-([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$")

# User-message prefixes that are injected context, not typed prompts.
_META_PREFIXES = (
    "<user_instructions>",
    "<environment_context>",
    "<ENVIRONMENT_CONTEXT>",
    "<permissions instructions>",
    "<turn_aborted>",
)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def list_codex_sessions(codex_dir: Path = DEFAULT_CODEX_SESSIONS_DIR) -> list[str]:
    """List Codex session IDs, newest first (by rollout file mtime)."""
    if not codex_dir.exists():
        return []
    sessions: dict[str, float] = {}
    for f in codex_dir.rglob("rollout-*.jsonl"):
        m = _ROLLOUT_RE.match(f.stem)
        if not m:
            continue
        sid = m.group(1)
        mtime = f.stat().st_mtime
        if sid not in sessions or mtime > sessions[sid]:
            sessions[sid] = mtime
    return sorted(sessions, key=lambda s: sessions[s], reverse=True)


def find_codex_rollout(session_id: str, codex_dir: Path = DEFAULT_CODEX_SESSIONS_DIR) -> Path | None:
    """Find the rollout JSONL for a Codex session ID. Newest wins if several."""
    if not codex_dir.exists():
        return None
    matches = [f for f in codex_dir.rglob(f"rollout-*{session_id}.jsonl") if _ROLLOUT_RE.match(f.stem)]
    if not matches:
        return None
    return max(matches, key=lambda f: f.stat().st_mtime)


# ---------------------------------------------------------------------------
# Content helpers
# ---------------------------------------------------------------------------


def _message_text(payload: dict) -> str:
    """Extract plain text from a Codex message payload."""
    content = payload.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text", "")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return ""


def _reasoning_text(payload: dict) -> str:
    """Extract summary text from a Codex reasoning payload."""
    summary = payload.get("summary")
    if isinstance(summary, list):
        parts = []
        for item in summary:
            if isinstance(item, dict):
                text = item.get("text", "")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return ""


def _is_meta_user_text(text: str) -> bool:
    return text.lstrip().startswith(_META_PREFIXES)


def _function_call_label(name: str, arguments: str) -> str:
    """Build a human-readable label for a Codex function_call."""
    resource = ""
    try:
        args = json.loads(arguments) if arguments else {}
    except (json.JSONDecodeError, TypeError):
        args = {}
    if isinstance(args, dict):
        cmd = args.get("command")
        if isinstance(cmd, list):
            # e.g. ["bash", "-lc", "grep -n foo"] -> the actual command string
            tail = [str(c) for c in cmd if str(c) not in ("bash", "-lc", "sh", "-c")]
            resource = " ".join(tail)[:60]
        elif isinstance(cmd, str):
            resource = cmd[:60]
        elif args.get("path"):
            resource = str(args["path"])[:60]
    if resource:
        return f"{name} {resource}"
    return name


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------


def parse_codex_rollout(rollout_path: Path) -> dict[str, Any]:
    """Parse a Codex rollout JSONL into Context Scope-compatible session data.

    Returns a dict with:
        session_id, model, cwd, cli_version, started_at, ended_at,
        blocks (list), churn (list), turn_map (list)

    Unknown record types and malformed lines are skipped.
    """
    session_id = ""
    model: str | None = None
    cwd: str | None = None
    cli_version: str | None = None
    started_at: str | None = None
    ended_at: str | None = None

    blocks: list[dict[str, Any]] = []
    churn: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []  # blocks awaiting an API-call index
    call_index = 0
    prev_resident = 0
    call_names: dict[str, str] = {}  # call_id -> function name
    # (first_call_index, prompt) per real user prompt
    turn_boundaries: list[tuple[int, str]] = []
    pending_prompts: list[str] = []

    with open(rollout_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue

            ts = record.get("timestamp")
            if isinstance(ts, str) and ts:
                if started_at is None:
                    started_at = ts
                ended_at = ts

            rtype = record.get("type")
            payload = record.get("payload")
            if not isinstance(payload, dict):
                payload = {}

            if rtype == "session_meta":
                session_id = str(payload.get("id", "")) or session_id
                cwd = payload.get("cwd") or cwd
                cli_version = payload.get("cli_version") or cli_version
                instructions = payload.get("instructions") or ""
                if instructions:
                    blocks.append(
                        {
                            "id": "instructions",
                            "type": "system",
                            "label": "instructions (AGENTS.md)",
                            "tokens": max(1, len(instructions) // 4),
                            "enter": 0,
                            "exit": None,
                            "cached": True,
                            "ref": True,
                            "content": instructions[:MAX_CONTENT_CHARS],
                        }
                    )

            elif rtype == "turn_context":
                if model is None and payload.get("model"):
                    model = str(payload["model"])

            elif rtype == "response_item":
                block = _parse_response_item(payload, call_names)
                if block is not None:
                    pending.append(block)
                    if block["type"] == "user" and not block.pop("_meta", False):
                        pending_prompts.append(block["content"][:200])

            elif rtype == "event_msg":
                if payload.get("type") != "token_count":
                    continue  # other UI events carry no context data
                info = payload.get("info")
                if not isinstance(info, dict):
                    continue  # rate-limit-only updates
                usage = info.get("last_token_usage") or info.get("total_token_usage")
                if not isinstance(usage, dict):
                    continue

                input_total = int(usage.get("input_tokens", 0) or 0)
                cached = int(usage.get("cached_input_tokens", 0) or 0)
                output = int(usage.get("output_tokens", 0) or 0)
                churn.append(
                    {
                        "turn": call_index,
                        "cache_read": cached,
                        "cache_creation": 0,
                        "input": max(input_total - cached, 0),
                        "output": output,
                    }
                )

                resident = input_total
                if call_index > 0 and prev_resident > 0 and resident < prev_resident * COMPACTION_THRESHOLD:
                    _mark_compaction(blocks, call_index)
                growth = resident - prev_resident if call_index > 0 else 0

                _finalize_pending(pending, call_index)
                _distribute_tokens(pending, growth)
                blocks.extend(pending)
                pending = []

                for prompt in pending_prompts:
                    turn_boundaries.append((call_index, prompt))
                pending_prompts = []

                prev_resident = resident
                call_index += 1

            # Unknown record types: skip silently (forward compatibility).

    # Trailing response items with no closing token_count.
    if pending:
        _finalize_pending(pending, call_index)
        for b in pending:
            chars = b.pop("_char_count", 0)
            b.pop("_meta", None)
            b["tokens"] = max(1, chars // 4) if chars > 0 else 1
        blocks.extend(pending)
        for prompt in pending_prompts:
            turn_boundaries.append((call_index, prompt))

    turn_map = _build_turn_map(turn_boundaries, total_calls=len(churn))

    if not session_id:
        m = _ROLLOUT_RE.match(rollout_path.stem)
        if m:
            session_id = m.group(1)

    return {
        "session_id": session_id,
        "model": model,
        "cwd": cwd,
        "cli_version": cli_version,
        "started_at": started_at,
        "ended_at": ended_at,
        "blocks": blocks,
        "churn": churn,
        "turn_map": turn_map,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_response_item(payload: dict, call_names: dict[str, str]) -> dict[str, Any] | None:
    """Convert one Codex response_item payload into a pending block dict.

    Returns None for unknown/empty payloads (skipped, not fatal).
    Pending blocks carry ``_char_count`` (for token distribution), ``_meta``
    (injected context flag) and ``_suffix`` (id suffix), all removed later.
    """
    ptype = payload.get("type")

    if ptype == "message":
        text = _message_text(payload)
        role = payload.get("role", "user")
        if role == "assistant":
            btype, label = "assistant", "assistant"
            meta = False
        else:
            btype = "user"
            meta = _is_meta_user_text(text)
            label = "meta prompt (instructions / environment)" if meta else "user prompt"
        return {
            "id": "",
            "type": btype,
            "label": label,
            "tokens": 0,
            "enter": 0,
            "exit": None,
            "cached": False,
            "ref": True,
            "content": text[:MAX_CONTENT_CHARS],
            "_char_count": len(text),
            "_meta": meta,
        }

    if ptype == "reasoning":
        text = _reasoning_text(payload)
        return {
            "id": "",
            "type": "thinking",
            "label": "reasoning",
            "tokens": 0,
            "enter": 0,
            "exit": None,
            "cached": False,
            "ref": True,
            "content": text[:MAX_CONTENT_CHARS],
            "_char_count": max(len(text), len(payload.get("encrypted_content") or "") // 8),
        }

    if ptype == "function_call":
        name = str(payload.get("name", "tool"))
        arguments = payload.get("arguments", "")
        call_id = str(payload.get("call_id", ""))
        if call_id:
            call_names[call_id] = name
        content = f"{name}: {arguments}" if arguments else name
        return {
            "id": "",
            "type": "tool_call",
            "label": _function_call_label(name, str(arguments)),
            "tokens": 0,
            "enter": 0,
            "exit": None,
            "cached": False,
            "ref": True,
            "content": content[:MAX_CONTENT_CHARS],
            "_char_count": len(content),
            "_suffix": call_id or None,
        }

    if ptype == "function_call_output":
        call_id = str(payload.get("call_id", ""))
        output = payload.get("output", "")
        if not isinstance(output, str):
            output = json.dumps(output, default=str)
        name = call_names.get(call_id, "tool")
        return {
            "id": "",
            "type": "tool_result",
            "label": f"{name} → result",
            "tokens": 0,
            "enter": 0,
            "exit": None,
            "cached": False,
            "ref": True,
            "content": output[:MAX_CONTENT_CHARS],
            "_char_count": len(output),
            "_suffix": call_id or None,
        }

    # Unknown response_item type: skip.
    return None


def _finalize_pending(pending: list[dict[str, Any]], call_index: int) -> None:
    """Assign enter turn + unique block IDs to pending blocks."""
    type_seq: dict[str, int] = {}
    for b in pending:
        b["enter"] = call_index
        btype = b["type"]
        suffix = b.pop("_suffix", None)
        b.pop("_meta", None)
        if suffix:
            b["id"] = f"t{call_index}-{btype}-{suffix}"
        else:
            seq = type_seq.get(btype, 0)
            b["id"] = f"t{call_index}-{btype}-{seq}"
            type_seq[btype] = seq + 1


def _build_turn_map(turn_boundaries: list[tuple[int, str]], total_calls: int) -> list[dict[str, Any]]:
    """Build conversation turn map from (first_call, prompt) boundaries."""
    if not turn_boundaries:
        return []

    boundaries = list(turn_boundaries)
    # API calls before the first real prompt belong to the first turn.
    if boundaries[0][0] > 0:
        boundaries[0] = (0, boundaries[0][1])

    last_index = max(total_calls - 1, 0)
    turn_map: list[dict[str, Any]] = []
    for i, (first_call, prompt) in enumerate(boundaries):
        if i + 1 < len(boundaries):
            next_first = boundaries[i + 1][0]
            last_call = next_first - 1 if next_first > first_call else first_call
        else:
            last_call = last_index
        turn_map.append(
            {
                "conv_turn": i + 1,
                "first_call": first_call,
                "last_call": max(first_call, last_call),
                "user_prompt": prompt,
            }
        )
    return turn_map
