"""Resolve tool-result offloads: resident vs spilled token sizes."""

from __future__ import annotations

import re
from pathlib import Path

# Pattern matching Claude Code's persisted-output format:
#   Full output saved to: /path/to/tool-results/<id>.txt
_PERSISTED_PATH_RE = re.compile(r"tool-results/([^./\s]+)\.txt")


def resolve_offloads(
    blocks: list[dict],
    tool_results_dir: Path,
) -> list[dict]:
    """Adjust blocks for tool-result offloads.

    When Claude Code spills a large tool output to disk, the in-window
    content is truncated. The full content is in tool-results/<id>.txt.

    For each tool_result block with a matching offload file:
    - tokens stays as resident (already correct from usage-based sizing)
    - spilled_tokens added with full file's estimated token count
    - content annotated with offload note

    Matching strategy (in priority order):
    1. The offload file ID appears in block content (via <persisted-output> tag)
    2. The offload file ID appears in the block's id field
    """
    if not tool_results_dir.exists():
        return blocks

    # Build index of offload files: filename (without .txt) -> size in bytes
    offload_files: dict[str, int] = {}
    for f in tool_results_dir.glob("*.txt"):
        offload_files[f.stem] = f.stat().st_size

    if not offload_files:
        return blocks

    for block in blocks:
        if block.get("type") != "tool_result":
            continue

        content = block.get("content", "")
        block_id = block.get("id", "")

        matched_file_id: str | None = None

        # Priority 1: parse the persisted-output path from content
        m = _PERSISTED_PATH_RE.search(content)
        if m:
            candidate = m.group(1)
            if candidate in offload_files:
                matched_file_id = candidate

        # Priority 2: file ID appears anywhere in block id
        if matched_file_id is None:
            for file_id in offload_files:
                if file_id in block_id:
                    matched_file_id = file_id
                    break

        if matched_file_id is None:
            continue

        file_size = offload_files[matched_file_id]
        spilled_tokens = max(1, file_size // 4)
        block["spilled_tokens"] = spilled_tokens

        resident_tokens = block.get("tokens", 0)
        if spilled_tokens > resident_tokens * 2:
            block["content"] = (
                f"[OFFLOADED: full output {spilled_tokens:,} est. tokens "
                f"in tool-results/{matched_file_id}.txt, "
                f"resident preview {resident_tokens:,} tokens]\n"
                + content[:300]
            )

    return blocks
