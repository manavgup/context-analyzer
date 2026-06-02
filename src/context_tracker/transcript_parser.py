"""Raw transcript parser — extracts full message content from Claude Code transcripts."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from context_tracker.analysis.models import ContentBlock, DataQualityWarning

logger = logging.getLogger(__name__)

SYNTHETIC_MODEL = "synthetic"


@dataclass(frozen=True)
class TranscriptMessage:
    """A single parsed message from a Claude Code transcript."""
    message_id: str
    sequence_index: int
    entry_type: str           # "user", "assistant", "system"
    timestamp: str | None
    session_id: str
    content_blocks: list[ContentBlock] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    stop_reason: str | None = None
    model: str | None = None


def _parse_content_blocks(content: str | list | None) -> list[ContentBlock]:
    """Extract ContentBlock list from a message's content field."""
    if content is None:
        return []

    if isinstance(content, str):
        if not content:
            return []
        return [ContentBlock(
            block_type="text",
            content=content,
            size_chars=len(content),
        )]

    if not isinstance(content, list):
        return []

    blocks: list[ContentBlock] = []
    for item in content:
        if not isinstance(item, dict):
            continue

        block_type = item.get("type", "")

        if block_type == "text":
            text = item.get("text", "")
            blocks.append(ContentBlock(
                block_type="text",
                content=text,
                size_chars=len(text),
            ))

        elif block_type == "thinking":
            text = item.get("thinking", "")
            blocks.append(ContentBlock(
                block_type="thinking",
                content=text,
                size_chars=len(text),
            ))

        elif block_type == "tool_use":
            tool_input = item.get("input", {})
            input_str = json.dumps(tool_input) if isinstance(tool_input, dict) else str(tool_input)
            blocks.append(ContentBlock(
                block_type="tool_use",
                content=input_str,
                size_chars=len(input_str),
                tool_use_id=item.get("id"),
                tool_name=item.get("name"),
                tool_input=tool_input if isinstance(tool_input, dict) else None,
            ))

        elif block_type == "tool_result":
            result_content = item.get("content", "")
            if isinstance(result_content, list):
                result_content = "\n".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in result_content
                )
            elif not isinstance(result_content, str):
                result_content = str(result_content)
            blocks.append(ContentBlock(
                block_type="tool_result",
                content=result_content,
                size_chars=len(result_content),
                tool_use_id=item.get("tool_use_id"),
                is_error=bool(item.get("is_error", False)),
            ))

    return blocks


def parse_raw_transcript(
    transcript_path: Path,
) -> tuple[list[TranscriptMessage], list[DataQualityWarning]]:
    """Parse a Claude Code transcript JSONL into structured messages.

    Returns (messages, warnings). Timestamps come from the transcript, not
    generated at parse time. Malformed lines produce warnings, not silent drops.
    """
    if not transcript_path.exists():
        return [], []

    messages: list[TranscriptMessage] = []
    warnings: list[DataQualityWarning] = []
    sequence_index = 0

    with open(transcript_path, encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                entry = json.loads(raw_line)
            except json.JSONDecodeError:
                warnings.append(DataQualityWarning(
                    line_number=line_number,
                    warning_type="malformed_json",
                    description=f"Could not parse JSON: {raw_line[:80]}",
                ))
                continue

            entry_type = entry.get("type", "")

            # Skip non-message entry types
            if entry_type in ("file-history-snapshot", "last-prompt", "pr-link", "queue-operation"):
                continue

            session_id = entry.get("sessionId", "unknown")
            timestamp = entry.get("timestamp")
            message_id = entry.get("uuid", f"gen-{line_number}")

            if entry_type == "system":
                messages.append(TranscriptMessage(
                    message_id=message_id,
                    sequence_index=sequence_index,
                    entry_type="system",
                    timestamp=timestamp,
                    session_id=session_id,
                ))
                sequence_index += 1
                continue

            message = entry.get("message")
            if not isinstance(message, dict):
                continue

            if entry_type == "assistant":
                # Skip streaming chunks — only keep completed API calls
                stop_reason = message.get("stop_reason")
                if stop_reason is None:
                    continue

                usage = message.get("usage", {})
                output_tokens = usage.get("output_tokens", 0)
                if output_tokens == 0:
                    continue

                model = message.get("model", "unknown")
                if model == SYNTHETIC_MODEL:
                    continue

                content_blocks = _parse_content_blocks(message.get("content"))
                messages.append(TranscriptMessage(
                    message_id=message_id,
                    sequence_index=sequence_index,
                    entry_type="assistant",
                    timestamp=timestamp,
                    session_id=session_id,
                    content_blocks=content_blocks,
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=output_tokens,
                    cache_read_tokens=usage.get("cache_read_input_tokens", 0),
                    cache_creation_tokens=usage.get("cache_creation_input_tokens", 0),
                    stop_reason=stop_reason,
                    model=model,
                ))
                sequence_index += 1

            elif entry_type == "user":
                content_blocks = _parse_content_blocks(message.get("content"))
                messages.append(TranscriptMessage(
                    message_id=message_id,
                    sequence_index=sequence_index,
                    entry_type="user",
                    timestamp=timestamp,
                    session_id=session_id,
                    content_blocks=content_blocks,
                ))
                sequence_index += 1

    return messages, warnings
