"""Hook processor: reads Claude Code HookInput from stdin, writes JSONL events."""

from __future__ import annotations

import json
import logging
import sys

from context_tracker.models import (
    BaseEvent,
    InstructionsLoadedEvent,
    PostCompactEvent,
    PostToolUseEvent,
    PostToolUseFailureEvent,
    PreCompactEvent,
    SessionEndEvent,
    SessionStartEvent,
    SubagentStartEvent,
    SubagentStopEvent,
    UserPromptEvent,
)
from context_tracker.storage import append_event

logger = logging.getLogger(__name__)


def _safe_len(value: object) -> int:
    """Get string length of a value, serializing dicts/lists to JSON first."""
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    if isinstance(value, (dict, list)):
        return len(json.dumps(value))
    return len(str(value))


def process_hook_input(raw_json: str) -> BaseEvent | None:
    """Parse a Claude Code HookInput JSON string into a typed event."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        logger.warning("Malformed hook input: %s", raw_json[:80])
        return None

    hook_event = data.get("hook_event_name")
    session_id = data.get("session_id", "unknown")

    if hook_event == "PostToolUse":
        return PostToolUseEvent(
            session_id=session_id,
            tool_name=data.get("tool_name", "unknown"),
            input_payload_chars=_safe_len(data.get("tool_input")),
            output_payload_chars=_safe_len(data.get("tool_response")),
            tool_use_id=data.get("tool_use_id", ""),
        )

    if hook_event == "PostToolUseFailure":
        return PostToolUseFailureEvent(
            session_id=session_id,
            tool_name=data.get("tool_name", "unknown"),
            input_payload_chars=_safe_len(data.get("tool_input")),
            error_length=_safe_len(data.get("error")),
            tool_use_id=data.get("tool_use_id", ""),
        )

    if hook_event == "SessionStart":
        return SessionStartEvent(
            session_id=session_id,
            source=data.get("source", "startup"),
            model=data.get("model", "unknown"),
        )

    if hook_event == "SessionEnd":
        return SessionEndEvent(
            session_id=session_id,
            reason=data.get("reason", "unknown"),
        )

    if hook_event == "UserPromptSubmit":
        return UserPromptEvent(
            session_id=session_id,
            prompt_length_chars=_safe_len(data.get("prompt")),
        )

    if hook_event == "PreCompact":
        return PreCompactEvent(
            session_id=session_id,
            trigger=data.get("trigger", "auto"),
        )

    if hook_event == "PostCompact":
        return PostCompactEvent(
            session_id=session_id,
            trigger=data.get("trigger", "auto"),
            compact_summary_length=_safe_len(data.get("compact_summary")),
        )

    if hook_event == "SubagentStart":
        return SubagentStartEvent(
            session_id=session_id,
            agent_id=data.get("agent_id", ""),
            agent_type=data.get("agent_type", ""),
        )

    if hook_event == "SubagentStop":
        return SubagentStopEvent(
            session_id=session_id,
            agent_id=data.get("agent_id", ""),
            agent_type=data.get("agent_type", ""),
            agent_transcript_path=data.get("agent_transcript_path", ""),
        )

    if hook_event == "InstructionsLoaded":
        return InstructionsLoadedEvent(
            session_id=session_id,
            file_path=data.get("file_path", ""),
            memory_type=data.get("memory_type", ""),
            load_reason=data.get("load_reason", ""),
        )

    logger.debug("Unhandled hook event: %s", hook_event)
    return None


def main() -> None:
    """Entry point: read HookInput JSON from stdin, write event to JSONL."""
    raw = sys.stdin.read().strip()
    if not raw:
        return

    event = process_hook_input(raw)
    if event is not None:
        append_event(event)


if __name__ == "__main__":
    main()
