"""Parse a Claude Code transcript JSONL into Context Scope blocks and churn series.

Produces:
- blocks: one dict per content block (system, tooldef, skill, user, assistant,
  thinking, tool_call, tool_result, etc.) in Context Scope format.
- churn: one dict per API call with token usage directly from message.usage.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .tokens import (
    DEFAULT_SYSTEM_PROMPT_TOKENS,
    char_count_of_block,
)

# Max characters stored in the content field of each block.
MAX_CONTENT_CHARS = 500

# Compaction detection threshold: if total resident drops below this
# fraction of previous total, we treat it as a compaction event.
COMPACTION_THRESHOLD = 0.5

# Entry types we skip entirely.
_SKIP_TYPES = frozenset(
    {
        "system",
        "file-history-snapshot",
        "last-prompt",
        "pr-link",
        "queue-operation",
    }
)


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------


def _tool_resource(name: str, inp: dict) -> str:
    """Extract a human-readable resource string from a tool_use input."""
    if name in ("Read", "Edit", "Write"):
        fp = str(inp.get("file_path", ""))
        return os.path.basename(fp) if fp else ""
    if name == "Bash":
        cmd = str(inp.get("command", ""))
        # First meaningful word of the command
        first = cmd.strip().split()[0] if cmd.strip() else ""
        return first
    if name == "Grep":
        return str(inp.get("pattern", ""))[:40]
    if name == "Glob":
        return str(inp.get("pattern", ""))[:40]
    # Fallback: tool name only
    return ""


def _tool_call_label(name: str, inp: dict) -> str:
    """Build a descriptive label for a tool_use block."""
    resource = _tool_resource(name, inp)
    if resource:
        return f"{name} {resource}"
    return name


def _tool_result_label(tool_use_id: str, tool_use_map: dict[str, dict]) -> str:
    """Build a descriptive label for a tool_result block."""
    info = tool_use_map.get(tool_use_id, {})
    name = info.get("name", "tool")
    inp = info.get("input", {})
    resource = _tool_resource(name, inp)
    if resource:
        return f"{name} \u2192 {resource}"
    return f"{name} \u2192 result"


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------


def _block_content_str(block: dict, block_type: str) -> str:
    """Extract the text content of a block, truncated to MAX_CONTENT_CHARS."""
    raw: str
    if block_type in ("text", "assistant"):
        raw = str(block.get("text", ""))
    elif block_type == "thinking":
        raw = str(block.get("thinking", ""))
    elif block_type in ("tool_use", "tool_call"):
        name = block.get("name", "")
        inp = block.get("input", {})
        raw = f"{name}: {json.dumps(inp, default=str)}"
    elif block_type == "tool_result":
        c = block.get("content", "")
        if isinstance(c, str):
            raw = c
        elif isinstance(c, list):
            parts = [sub.get("text", "") for sub in c]
            raw = "\n".join(parts)
        else:
            raw = str(c)
    else:
        raw = str(block)
    return raw[:MAX_CONTENT_CHARS]


def _user_content_str(content: str | list) -> str:
    """Extract text from a user message content field."""
    if isinstance(content, str):
        return content[:MAX_CONTENT_CHARS]
    if isinstance(content, list):
        parts = []
        for block in content:
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)[:MAX_CONTENT_CHARS]
    return ""


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------


def parse_transcript_to_blocks(
    transcript_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse a Claude Code transcript JSONL into blocks and churn.

    Args:
        transcript_path: Path to the .jsonl transcript file.

    Returns:
        (blocks, churn) where blocks is a list of Context Scope block dicts
        and churn is a list of per-API-call usage dicts.
    """
    entries = _load_entries(transcript_path)
    if not entries:
        return [], []

    blocks: list[dict[str, Any]] = []
    churn: list[dict[str, Any]] = []
    tool_use_map: dict[str, dict] = {}  # tool_use_id -> {name, input}
    api_call_index = 0
    prev_total_resident = 0

    # --------------- First pass: find prefix size from first completed call ------
    first_usage = _find_first_completed_usage(entries)
    prefix_tokens = first_usage.get("cache_creation_input_tokens", 0) if first_usage else 0

    # Create pinned prefix blocks
    if prefix_tokens > 0:
        sys_tokens = min(DEFAULT_SYSTEM_PROMPT_TOKENS, prefix_tokens)
        skills_tokens = prefix_tokens - sys_tokens

        blocks.append(
            {
                "id": "sys",
                "type": "system",
                "label": "system prompt",
                "tokens": sys_tokens,
                "enter": 0,
                "exit": None,
                "cached": True,
                "ref": True,
                "content": f"Claude Code system prompt (~{sys_tokens} tokens)",
            }
        )
        if skills_tokens > 0:
            blocks.append(
                {
                    "id": "skills",
                    "type": "skill",
                    "label": "CLAUDE.md + skills + tools",
                    "tokens": skills_tokens,
                    "enter": 0,
                    "exit": None,
                    "cached": True,
                    "ref": True,
                    "content": f"Project config, skills, tool definitions (~{skills_tokens} tokens)",
                }
            )

    # --------------- Walk all entries in order --------------------------------
    # Collect user content blocks that precede each API call so we can
    # assign them the correct api_call_index when the next assistant fires.
    pending_user_blocks: list[dict[str, Any]] = []

    for entry in entries:
        etype = entry.get("type", "")

        if etype in _SKIP_TYPES:
            continue

        if etype == "user":
            msg = entry.get("message", {})
            content = msg.get("content", "")
            new_blocks = _parse_user_content(content, tool_use_map, entry.get("isMeta", False))
            pending_user_blocks.extend(new_blocks)
            continue

        if etype == "assistant":
            msg = entry.get("message", {})

            # Always index tool_use blocks from ALL assistant entries
            # (including streaming), so tool_result can resolve labels.
            for blk in msg.get("content", []):
                if blk.get("type") == "tool_use":
                    tid = blk.get("id", "")
                    if tid and tid not in tool_use_map:
                        tool_use_map[tid] = {
                            "name": blk.get("name", ""),
                            "input": blk.get("input", {}),
                        }

            if not _is_completed_assistant(msg):
                continue

            usage = msg.get("usage", {})
            content_blocks_raw = msg.get("content", [])

            # Record churn
            churn_entry = {
                "turn": api_call_index,
                "cache_read": usage.get("cache_read_input_tokens", 0),
                "cache_creation": usage.get("cache_creation_input_tokens", 0),
                "input": usage.get("input_tokens", 0),
                "output": usage.get("output_tokens", 0),
            }
            churn.append(churn_entry)

            # Compute total resident for this call
            total_resident = (
                usage.get("input_tokens", 0)
                + usage.get("cache_read_input_tokens", 0)
                + usage.get("cache_creation_input_tokens", 0)
            )

            # Detect compaction
            if (
                api_call_index > 0
                and prev_total_resident > 0
                and total_resident < prev_total_resident * COMPACTION_THRESHOLD
            ):
                _mark_compaction(blocks, api_call_index)

            # Compute context growth for token sizing
            growth = total_resident - prev_total_resident if api_call_index > 0 else 0

            # Finalize pending user blocks with correct api_call_index
            # Use a per-type counter to ensure unique IDs when multiple
            # user entries precede the same assistant response.
            _type_seq: dict[str, int] = {}
            for ub in pending_user_blocks:
                ub["enter"] = api_call_index
                btype = ub["type"]
                tool_id = ub.get("_tool_id")
                if tool_id:
                    # tool_result blocks use tool_use_id as suffix (already unique)
                    ub["id"] = _make_block_id(api_call_index, btype, tool_id=tool_id)
                else:
                    seq = _type_seq.get(btype, 0)
                    ub["id"] = _make_block_id(api_call_index, btype, seq)
                    _type_seq[btype] = seq + 1
                ub.pop("_seq", None)
                ub.pop("_tool_id", None)

            # Parse assistant content blocks
            asst_blocks = _parse_assistant_content(content_blocks_raw, api_call_index, tool_use_map)

            # Distribute tokens proportionally
            all_new_blocks = pending_user_blocks + asst_blocks
            _distribute_tokens(all_new_blocks, growth)

            blocks.extend(pending_user_blocks)
            blocks.extend(asst_blocks)
            pending_user_blocks = []

            prev_total_resident = total_resident
            api_call_index += 1

    # Any remaining pending user blocks (no final assistant) — still add them
    if pending_user_blocks:
        _type_seq_final: dict[str, int] = {}
        for ub in pending_user_blocks:
            ub["enter"] = api_call_index
            btype = ub["type"]
            tool_id = ub.get("_tool_id")
            if tool_id:
                ub["id"] = _make_block_id(api_call_index, btype, tool_id=tool_id)
            else:
                seq = _type_seq_final.get(btype, 0)
                ub["id"] = _make_block_id(api_call_index, btype, seq)
                _type_seq_final[btype] = seq + 1
            ub.pop("_seq", None)
            ub.pop("_tool_id", None)
            chars = ub.pop("_char_count", 0)
            ub["tokens"] = max(1, chars // 4) if chars > 0 else 1
        blocks.extend(pending_user_blocks)

    return blocks, churn


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_entries(path: Path) -> list[dict]:
    """Load all JSONL entries from the transcript file."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def _is_completed_assistant(msg: dict) -> bool:
    """Check if an assistant message is a completed API response."""
    if msg.get("model") == "synthetic":
        return False
    if msg.get("stop_reason") is None:
        return False
    usage = msg.get("usage", {})
    if usage.get("output_tokens", 0) <= 0:
        return False
    return True


def _find_first_completed_usage(entries: list[dict]) -> dict[str, Any] | None:
    """Find the usage dict from the first completed assistant message."""
    for entry in entries:
        if entry.get("type") != "assistant":
            continue
        msg = entry.get("message", {})
        if _is_completed_assistant(msg):
            usage = msg.get("usage", {})
            return dict(usage) if isinstance(usage, dict) else {}
    return None


def _make_block_id(
    api_call_index: int,
    block_type: str,
    seq: int = 0,
    tool_id: str | None = None,
) -> str:
    """Generate a unique block ID.

    Format: t{api_call_index}-{type}-{suffix}
    """
    if tool_id:
        return f"t{api_call_index}-{block_type}-{tool_id}"
    return f"t{api_call_index}-{block_type}-{seq}"


def _parse_user_content(
    content: str | list,
    tool_use_map: dict[str, dict],
    is_meta: bool = False,
) -> list[dict[str, Any]]:
    """Parse user message content into pending block dicts.

    Returns blocks with placeholder enter/id (filled in when next assistant fires).
    """
    pending: list[dict[str, Any]] = []
    seq = 0

    if isinstance(content, str):
        # Plain text user prompt
        btype = "user"
        label = "user prompt"
        if is_meta:
            label = "meta prompt (CLAUDE.md / skills)"
        pending.append(
            {
                "id": "",  # placeholder
                "type": btype,
                "label": label,
                "tokens": 0,  # placeholder
                "enter": 0,  # placeholder
                "exit": None,
                "cached": False,
                "ref": True,
                "content": content[:MAX_CONTENT_CHARS],
                "_seq": seq,
                "_char_count": len(content),
            }
        )
        return pending

    if isinstance(content, list):
        for block in content:
            btype_raw = block.get("type", "")

            if btype_raw == "tool_result":
                tool_use_id = block.get("tool_use_id", "")
                label = _tool_result_label(tool_use_id, tool_use_map)
                raw_content = _block_content_str(block, "tool_result")
                chars = char_count_of_block(block)
                pending.append(
                    {
                        "id": "",
                        "type": "tool_result",
                        "label": label,
                        "tokens": 0,
                        "enter": 0,
                        "exit": None,
                        "cached": False,
                        "ref": True,
                        "content": raw_content[:MAX_CONTENT_CHARS],
                        "_seq": seq,
                        "_tool_id": tool_use_id,
                        "_char_count": chars,
                    }
                )
            elif btype_raw == "text":
                text = block.get("text", "")
                label = "user prompt"
                if is_meta:
                    label = "meta prompt (CLAUDE.md / skills)"
                pending.append(
                    {
                        "id": "",
                        "type": "user",
                        "label": label,
                        "tokens": 0,
                        "enter": 0,
                        "exit": None,
                        "cached": False,
                        "ref": True,
                        "content": text[:MAX_CONTENT_CHARS],
                        "_seq": seq,
                        "_char_count": len(text),
                    }
                )
            else:
                # Other block types in user content (rare)
                pending.append(
                    {
                        "id": "",
                        "type": "user",
                        "label": f"user ({btype_raw})",
                        "tokens": 0,
                        "enter": 0,
                        "exit": None,
                        "cached": False,
                        "ref": True,
                        "content": _block_content_str(block, btype_raw)[:MAX_CONTENT_CHARS],
                        "_seq": seq,
                        "_char_count": char_count_of_block(block),
                    }
                )
            seq += 1

    return pending


def _parse_assistant_content(
    content_blocks: list[dict],
    api_call_index: int,
    tool_use_map: dict[str, dict],
) -> list[dict[str, Any]]:
    """Parse assistant message content blocks into block dicts."""
    result: list[dict[str, Any]] = []
    seq = 0

    for block in content_blocks:
        btype_raw = block.get("type", "")

        if btype_raw == "text":
            text = block.get("text", "")
            result.append(
                {
                    "id": _make_block_id(api_call_index, "assistant", seq),
                    "type": "assistant",
                    "label": "assistant",
                    "tokens": 0,
                    "enter": api_call_index,
                    "exit": None,
                    "cached": False,
                    "ref": True,
                    "content": text[:MAX_CONTENT_CHARS],
                    "_char_count": len(text),
                }
            )
            seq += 1

        elif btype_raw == "thinking":
            thinking = block.get("thinking", "")
            result.append(
                {
                    "id": _make_block_id(api_call_index, "thinking", seq),
                    "type": "thinking",
                    "label": "thinking",
                    "tokens": 0,
                    "enter": api_call_index,
                    "exit": None,
                    "cached": False,
                    "ref": True,
                    "content": thinking[:MAX_CONTENT_CHARS],
                    "_char_count": len(thinking),
                }
            )
            seq += 1

        elif btype_raw == "tool_use":
            tool_id = block.get("id", "")
            tool_name = block.get("name", "")
            tool_input = block.get("input", {})

            # Record in map for later tool_result labeling
            tool_use_map[tool_id] = {"name": tool_name, "input": tool_input}

            label = _tool_call_label(tool_name, tool_input)
            content_str = _block_content_str(block, "tool_use")
            chars = char_count_of_block(block)

            result.append(
                {
                    "id": _make_block_id(api_call_index, "tool_call", tool_id=tool_id),
                    "type": "tool_call",
                    "label": label,
                    "tokens": 0,
                    "enter": api_call_index,
                    "exit": None,
                    "cached": False,
                    "ref": True,
                    "content": content_str[:MAX_CONTENT_CHARS],
                    "_char_count": chars,
                }
            )
            seq += 1

        else:
            # Unknown block type — still record it
            result.append(
                {
                    "id": _make_block_id(api_call_index, btype_raw, seq),
                    "type": btype_raw,
                    "label": btype_raw,
                    "tokens": 0,
                    "enter": api_call_index,
                    "exit": None,
                    "cached": False,
                    "ref": True,
                    "content": _block_content_str(block, btype_raw)[:MAX_CONTENT_CHARS],
                    "_char_count": char_count_of_block(block),
                }
            )
            seq += 1

    return result


def _distribute_tokens(blocks: list[dict[str, Any]], growth: int) -> None:
    """Distribute token growth proportionally by character count across blocks.

    If growth is zero or negative, fall back to chars/4 estimation.
    Removes the internal _char_count key after sizing.
    """
    total_chars = sum(b.get("_char_count", 0) for b in blocks)

    for b in blocks:
        chars = b.pop("_char_count", 0)
        if growth > 0 and total_chars > 0:
            b["tokens"] = max(1, int(growth * chars / total_chars))
        else:
            # Fallback: estimate from character count
            b["tokens"] = max(1, chars // 4) if chars > 0 else 1


def _mark_compaction(blocks: list[dict[str, Any]], compaction_turn: int) -> None:
    """Mark all non-pinned pre-compaction blocks as evicted."""
    for b in blocks:
        if b.get("cached"):
            continue  # Pinned prefix blocks survive compaction
        if b.get("exit") is not None:
            continue  # Already evicted
        b["exit"] = compaction_turn


# ---------------------------------------------------------------------------
# Conversation turn mapping
# ---------------------------------------------------------------------------


def _is_injected_content(text: str) -> bool:
    """Detect system-injected content that isn't a real user prompt.

    Claude Code injects skill expansions, command handlers, and system
    reminders as user messages. These look like user text but aren't
    typed by the user.
    """
    t = text.strip()
    if not t:
        return True
    # Skill/command invocations
    if "<command-message>" in t or "<command-name>" in t:
        return True
    # Skill expansion content (starts with "Base directory for this skill:")
    if t.startswith("Base directory for this skill:"):
        return True
    # Local command output injected into context
    if "<local-command-caveat>" in t or "<local-command-stdout>" in t:
        return True
    # System reminders injected as user messages
    if t.startswith("<system-reminder>"):
        return True
    return False


def _has_user_text_block(content: str | list) -> bool:
    """Check if user message content contains a real user text prompt.

    Filters out system-injected content (skill expansions, command handlers).
    """
    if isinstance(content, str):
        return bool(content.strip()) and not _is_injected_content(content)
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text.strip() and not _is_injected_content(text):
                    return True
    return False


def _extract_user_prompt(content: str | list) -> str:
    """Extract the user prompt text from message content."""
    if isinstance(content, str):
        return content[:200]
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = str(block.get("text", ""))
                if text.strip():
                    return text[:200]
    return ""


def build_turn_map(transcript_path: Path) -> list[dict[str, Any]]:
    """Map conversation turns to API call ranges.

    A conversation turn = one user text prompt + all API calls until the
    next user text prompt.

    Returns list of:
    {
        "conv_turn": 1,           # Conversation turn number (1-based)
        "first_call": 0,          # First API call index in this turn
        "last_call": 5,           # Last API call index in this turn
        "user_prompt": "Fix ...", # User prompt text (truncated)
    }
    """
    entries = _load_entries(transcript_path)
    if not entries:
        return []

    # Walk entries the same way as parse_transcript_to_blocks to track
    # api_call_index, but also detect user text prompts.
    api_call_index = 0
    # Each item: (api_call_index_of_first_call_after_prompt, user_prompt_text)
    turn_boundaries: list[tuple[int, str]] = []

    # Buffer ALL user text prompts until the next completed assistant entry.
    # Multiple prompts can arrive before one API call (e.g., user sends
    # several messages rapidly). Each is its own conversation turn.
    pending_prompts: list[str] = []

    for entry in entries:
        etype = entry.get("type", "")

        if etype in _SKIP_TYPES:
            continue

        if etype == "user":
            msg = entry.get("message", {})
            content = msg.get("content", "")
            if _has_user_text_block(content):
                pending_prompts.append(_extract_user_prompt(content))
            continue

        if etype == "assistant":
            msg = entry.get("message", {})
            if not _is_completed_assistant(msg):
                continue

            # This is a completed API call at api_call_index.
            # Record ALL pending user prompts as separate turn boundaries.
            # The first N-1 prompts share this API call index (they had
            # no API calls of their own), the last one "owns" this call.
            for prompt in pending_prompts:
                turn_boundaries.append((api_call_index, prompt))
            pending_prompts = []

            api_call_index += 1

    total_api_calls = api_call_index

    if not turn_boundaries:
        return []

    # Include API calls before the first real prompt (injected skill content)
    # by pushing the first boundary's first_call to 0
    if turn_boundaries[0][0] > 0:
        turn_boundaries[0] = (0, turn_boundaries[0][1])

    # Build turn_map from boundaries.
    # Handle merged prompts: when multiple prompts share the same first_call,
    # earlier ones get an empty API call range (first_call == last_call == the
    # shared call index, and only the last prompt in the group "owns" the calls).
    turn_map: list[dict[str, Any]] = []
    for i, (first_call, prompt) in enumerate(turn_boundaries):
        conv_turn = i + 1  # 1-based
        if i + 1 < len(turn_boundaries):
            next_first = turn_boundaries[i + 1][0]
            if next_first > first_call:
                last_call = next_first - 1
            else:
                # Merged prompt: this prompt shares the API call with the next.
                # Give it the same call index (one API call range).
                last_call = first_call
        else:
            last_call = total_api_calls - 1

        turn_map.append(
            {
                "conv_turn": conv_turn,
                "first_call": first_call,
                "last_call": max(first_call, last_call),
                "user_prompt": prompt,
            }
        )

    return turn_map
