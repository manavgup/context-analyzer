"""Token estimation utilities for ccscope."""

# Approximate ratio of characters to tokens for Claude models.
# Claude tokenizer averages ~4 characters per token for English text.
CHARS_PER_TOKEN = 4

# Default system prompt size in tokens (Claude Code system prompt).
DEFAULT_SYSTEM_PROMPT_TOKENS = 6300


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
    and user-side blocks (text, tool_result).
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
    if btype == "tool_result":
        content = block.get("content", "")
        if isinstance(content, str):
            return len(content)
        if isinstance(content, list):
            total = 0
            for sub in content:
                total += len(sub.get("text", ""))
            return total
        return 0
    # Fallback for unknown types
    return len(str(block))
