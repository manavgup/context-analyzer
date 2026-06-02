"""Parse Claude Code transcript JSONL files for exact API token usage."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from context_tracker.models import ApiTurnEvent

logger = logging.getLogger(__name__)

# Claude Code uses this model name for synthetic/internal messages
SYNTHETIC_MODEL = "synthetic"


def parse_transcript(transcript_path: Path) -> list[ApiTurnEvent]:
    """Extract API turn events from a Claude Code transcript JSONL file.

    Claude Code emits multiple assistant entries per API call as streaming
    chunks arrive. We only keep entries that have:
    - type == "assistant"
    - a non-null stop_reason (marks a completed API call)
    - a usage object with output_tokens > 0
    """
    if not transcript_path.exists():
        return []

    events: list[ApiTurnEvent] = []
    turn_number = 0

    with open(transcript_path, encoding="utf-8") as f:
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

            message = entry.get("message")
            if not isinstance(message, dict):
                continue

            usage = message.get("usage")
            if not isinstance(usage, dict):
                continue

            stop_reason = message.get("stop_reason")
            if stop_reason is None:
                continue

            output_tokens = usage.get("output_tokens", 0)
            if output_tokens == 0:
                continue

            model = message.get("model", "unknown")
            if model == SYNTHETIC_MODEL:
                continue

            session_id = entry.get("sessionId", "unknown")
            turn_number += 1

            events.append(
                ApiTurnEvent(
                    session_id=session_id,
                    turn_number=turn_number,
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=output_tokens,
                    cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
                    cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
                    model=model,
                    stop_reason=stop_reason,
                )
            )

    return events
