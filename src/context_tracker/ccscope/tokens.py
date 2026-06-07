"""Token estimation utilities for ccscope."""

from __future__ import annotations

import base64
import struct

# Approximate ratio of characters to tokens for Claude models.
# Claude tokenizer averages ~4 characters per token for English text.
CHARS_PER_TOKEN = 4

# Default system prompt size in tokens (Claude Code system prompt).
DEFAULT_SYSTEM_PROMPT_TOKENS = 6300

# Fallback dimensions when image headers can't be parsed.
_FALLBACK_WIDTH = 1024
_FALLBACK_HEIGHT = 1024


def image_dimensions(b64_data: str, media_type: str) -> tuple[int, int]:
    """Parse actual dimensions from base64-encoded image data.

    Only decodes the first 256 bytes (enough for PNG/JPEG headers).
    Returns (width, height) or fallback (1024, 1024) on failure.
    """
    if not b64_data:
        return _FALLBACK_WIDTH, _FALLBACK_HEIGHT
    try:
        # Only need first ~256 bytes of the decoded data for headers
        raw = base64.b64decode(b64_data[:344])  # 344 base64 chars ~ 256 bytes
    except Exception:
        return _FALLBACK_WIDTH, _FALLBACK_HEIGHT

    if media_type == "image/png":
        if len(raw) >= 24:
            try:
                w, h = struct.unpack(">II", raw[16:24])
                if w > 0 and h > 0:
                    return w, h
            except struct.error:
                pass
    elif media_type in ("image/jpeg", "image/jpg"):
        # Scan for SOF markers (SOF0-SOF3: baseline, extended, progressive, lossless)
        i = 0
        while i < len(raw) - 8:
            if raw[i] == 0xFF and raw[i + 1] in (0xC0, 0xC1, 0xC2, 0xC3):
                try:
                    h = struct.unpack(">H", raw[i + 5 : i + 7])[0]
                    w = struct.unpack(">H", raw[i + 7 : i + 9])[0]
                    if w > 0 and h > 0:
                        return w, h
                except struct.error:
                    pass
            i += 1

    return _FALLBACK_WIDTH, _FALLBACK_HEIGHT


def image_tokens(w: int, h: int) -> int:
    """Compute Anthropic image token cost from dimensions."""
    return (w * h) // 750


def estimate_tokens(text: str) -> int:
    """Estimate token count from character count.

    Uses a simple chars/4 heuristic. Not precise, but good enough
    for proportional sizing when we don't have exact counts.
    """
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def char_count_of_block(block: dict) -> int:
    """Return the character count of a content block's payload.

    Works for both assistant-side blocks (text, thinking, tool_use)
    and user-side blocks (text, tool_result, image).
    """
    btype = block.get("type", "")
    if btype == "text":
        return len(block.get("text", ""))
    if btype == "thinking":
        return len(block.get("thinking", ""))
    if btype == "tool_use":
        import json

        inp = block.get("input", {})
        return len(json.dumps(inp, default=str))
    if btype == "image":
        # Top-level image block
        source = block.get("source", {})
        w, h = image_dimensions(source.get("data", ""), source.get("media_type", "image/png"))
        return image_tokens(w, h) * CHARS_PER_TOKEN
    if btype == "tool_result":
        content = block.get("content", "")
        if isinstance(content, str):
            return len(content)
        if isinstance(content, list):
            total = 0
            for sub in content:
                if not isinstance(sub, dict):
                    continue
                if sub.get("type") == "image":
                    source = sub.get("source", {})
                    w, h = image_dimensions(
                        source.get("data", ""),
                        source.get("media_type", "image/png"),
                    )
                    total += image_tokens(w, h) * CHARS_PER_TOKEN
                else:
                    total += len(sub.get("text", ""))
            return total
        return 0
    # Fallback for unknown types
    return len(str(block))
