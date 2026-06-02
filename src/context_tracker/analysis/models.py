"""Immutable data models for context analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class BlockType(StrEnum):
    SYSTEM = "system"
    USER_PROMPT = "user_prompt"
    ASSISTANT_TEXT = "assistant_text"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    COMPACTION_SUMMARY = "compaction_summary"


@dataclass(frozen=True)
class ContentBlock:
    """A single content block within a transcript message."""
    block_type: str              # "text", "tool_use", "tool_result", "thinking"
    content: str
    size_chars: int
    tool_use_id: str | None = None
    tool_name: str | None = None
    tool_input: dict | None = None
    is_error: bool = False


@dataclass(frozen=True)
class ContextBlock:
    """An immutable block in the context window. Content stored separately."""
    block_id: str
    turn_entered: int
    api_call_entered: int
    epoch_entered: int
    block_type: BlockType
    resource: str | None = None
    resource_type: str | None = None
    size_chars: int = 0
    size_tokens_est: int = 0
    content_hash: str = ""
    tool_name: str | None = None
    tool_use_id: str | None = None
    parent_block_id: str | None = None
    is_error: bool = False
    is_pinned: bool = False
    timestamp: str | None = None


@dataclass(frozen=True)
class BlockStateAtTurn:
    """Per-block staleness state at a specific turn. Immutable overlay."""
    block_id: str
    staleness_score: float
    staleness_label: str       # active, warm, stale, dead_weight, pinned
    is_superseded: bool = False
    superseded_by: str | None = None


@dataclass
class ContextEpoch:
    """A context epoch — compaction creates a new one."""
    epoch_number: int
    started_at_turn: int
    compaction_summary_size: int | None = None
    blocks_before_compaction: int = 0


@dataclass(frozen=True)
class ApiCall:
    """A single API round-trip within a conversation turn."""
    api_call_index: int
    conversation_turn: int
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    stop_reason: str | None = None
    timestamp: str | None = None
    blocks_entered: list[str] = field(default_factory=list)


@dataclass
class ConversationTurn:
    """One user prompt + all API calls until next user prompt."""
    turn_number: int
    timestamp: str | None
    user_prompt_text: str
    api_calls: list[ApiCall] = field(default_factory=list)
    epoch: int = 0


@dataclass
class TurnSnapshot:
    """State of the context window at a specific turn."""
    turn_number: int
    timestamp: str | None
    epoch: int
    block_ids: list[str]
    block_states: list[BlockStateAtTurn]
    blocks_entered_ids: list[str]
    blocks_exited_ids: list[str]
    total_tokens_est: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    actual_context_tokens: int = 0  # From API: input + cache_read + cache_create (ground truth)
    compaction_detected: bool = False
    api_call_count: int = 0


class ContentStore:
    """Stores full content indexed by block_id. Lazy loading for drilldowns."""

    def __init__(self) -> None:
        self._content: dict[str, str] = {}

    def add(self, block_id: str, content: str) -> None:
        self._content[block_id] = content

    def get_content(self, block_id: str) -> str:
        return self._content.get(block_id, "")

    def get_preview(self, block_id: str, max_chars: int = 200) -> str:
        return self._content.get(block_id, "")[:max_chars]

    def has(self, block_id: str) -> bool:
        return block_id in self._content

    def __len__(self) -> int:
        return len(self._content)


@dataclass
class DataQualityWarning:
    """Warning for data quality issues during parsing."""
    line_number: int
    warning_type: str       # malformed_json, missing_field, unexpected_type
    description: str
