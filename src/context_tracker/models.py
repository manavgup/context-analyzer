"""Pydantic models for all context tracker event types."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BaseEvent(BaseModel):
    """Base fields shared by all events."""

    session_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_jsonl(self) -> str:
        return self.model_dump_json()


class PostToolUseEvent(BaseEvent):
    event: Literal["post_tool_use"] = "post_tool_use"
    tool_name: str
    input_payload_chars: int
    output_payload_chars: int
    tool_use_id: str


class PostToolUseFailureEvent(BaseEvent):
    event: Literal["post_tool_use_failure"] = "post_tool_use_failure"
    tool_name: str
    input_payload_chars: int
    error_length: int
    tool_use_id: str


class PreCompactEvent(BaseEvent):
    event: Literal["pre_compact"] = "pre_compact"
    trigger: Literal["auto", "manual"]


class PostCompactEvent(BaseEvent):
    event: Literal["post_compact"] = "post_compact"
    trigger: Literal["auto", "manual"]
    compact_summary_length: int


class SessionStartEvent(BaseEvent):
    event: Literal["session_start"] = "session_start"
    source: Literal["startup", "resume", "clear", "compact"]
    model: str


class SessionEndEvent(BaseEvent):
    event: Literal["session_end"] = "session_end"
    reason: str


class UserPromptEvent(BaseEvent):
    event: Literal["user_prompt"] = "user_prompt"
    prompt_length_chars: int


class SubagentStartEvent(BaseEvent):
    event: Literal["subagent_start"] = "subagent_start"
    agent_id: str
    agent_type: str


class SubagentStopEvent(BaseEvent):
    event: Literal["subagent_stop"] = "subagent_stop"
    agent_id: str
    agent_type: str
    agent_transcript_path: str


class InstructionsLoadedEvent(BaseEvent):
    event: Literal["instructions_loaded"] = "instructions_loaded"
    file_path: str
    memory_type: str
    load_reason: str


class ApiTurnEvent(BaseEvent):
    event: Literal["api_turn"] = "api_turn"
    turn_number: int
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    model: str
    stop_reason: str | None = None


EVENT_TYPE_MAP: dict[str, type[BaseEvent]] = {
    "post_tool_use": PostToolUseEvent,
    "post_tool_use_failure": PostToolUseFailureEvent,
    "pre_compact": PreCompactEvent,
    "post_compact": PostCompactEvent,
    "session_start": SessionStartEvent,
    "session_end": SessionEndEvent,
    "user_prompt": UserPromptEvent,
    "subagent_start": SubagentStartEvent,
    "subagent_stop": SubagentStopEvent,
    "instructions_loaded": InstructionsLoadedEvent,
    "api_turn": ApiTurnEvent,
}

TrackerEvent = (
    PostToolUseEvent
    | PostToolUseFailureEvent
    | PreCompactEvent
    | PostCompactEvent
    | SessionStartEvent
    | SessionEndEvent
    | UserPromptEvent
    | SubagentStartEvent
    | SubagentStopEvent
    | InstructionsLoadedEvent
    | ApiTurnEvent
)


def parse_event(line: str) -> TrackerEvent | None:
    """Parse a JSONL line into a typed event. Returns None for unknown or malformed lines."""
    try:
        raw = json.loads(line)
    except json.JSONDecodeError:
        logger.warning("Malformed JSON line: %s", line[:80])
        return None

    event_type = raw.get("event")
    model_class = EVENT_TYPE_MAP.get(event_type)
    if model_class is None:
        logger.debug("Unknown event type: %s", event_type)
        return None

    try:
        return model_class.model_validate(raw)
    except Exception:
        logger.warning("Failed to parse event: %s", line[:80])
        return None
