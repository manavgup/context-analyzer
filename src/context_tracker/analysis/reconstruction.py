"""Context window reconstruction from parsed transcript + hook events.

Rebuilds the context window state at each conversation turn, producing
immutable ContextBlocks, TurnSnapshots, and a ContentStore.
"""

from __future__ import annotations

import hashlib
import logging
import re

from context_tracker.analysis.config import StalenessConfig
from context_tracker.analysis.models import (
    ApiCall,
    BlockType,
    ContentStore,
    ContextBlock,
    ContextEpoch,
    ConversationTurn,
    DataQualityWarning,
    TurnSnapshot,
)
from context_tracker.models import PostCompactEvent, TrackerEvent
from context_tracker.transcript_parser import TranscriptMessage

logger = logging.getLogger(__name__)

# Tokens per char rough estimate (English text, ~4 chars/token)
_CHARS_PER_TOKEN = 4

# If cache_creation_input_tokens > this fraction of total input tokens,
# it indicates a compaction event (cached prefix was rebuilt).
COMPACTION_CACHE_CREATE_THRESHOLD = 0.5

# Prefixes stripped before extracting the primary program
_CD_PREFIX_RE = re.compile(r"^cd\s+\S+\s*&&\s*")
_ENV_PREFIX_RE = re.compile(r"^env\s+\S+=\S+\s+")
_RUNNER_PREFIXES = {"uv", "pipx", "npx"}


def _extract_bash_program(command: str) -> str:
    """Extract the primary program name from a bash command string.

    Handles cd prefixes, env prefixes, runner wrappers (uv run, pipx run, npx),
    and pipelines (takes first command).
    """
    cmd = command.strip()
    if not cmd:
        return ""

    # Strip cd ... && prefix
    cmd = _CD_PREFIX_RE.sub("", cmd).strip()

    # Strip env VAR=val prefix (may repeat)
    while _ENV_PREFIX_RE.match(cmd):
        cmd = _ENV_PREFIX_RE.sub("", cmd, count=1).strip()

    # Take first command in a pipeline
    if "|" in cmd:
        cmd = cmd.split("|")[0].strip()

    # Split into tokens
    tokens = cmd.split()
    if not tokens:
        return ""

    program = tokens[0]

    # Handle runner prefixes: uv run ..., pipx run ..., npx ...
    if program in _RUNNER_PREFIXES and len(tokens) >= 2:
        if program == "npx":
            # npx <program>
            return tokens[1]
        if len(tokens) >= 3 and tokens[1] == "run":
            # uv run <program>, pipx run <program>
            return tokens[2]

    return program


def extract_resource(
    tool_name: str,
    tool_input: dict,
) -> tuple[str | None, str | None]:
    """Extract (resource, resource_type) from a tool_use input dict.

    Returns (None, None) for unrecognized tools.
    """
    if tool_name in ("Read", "Edit", "Write"):
        file_path = tool_input.get("file_path")
        if file_path:
            return file_path, "file"
        return None, None

    if tool_name == "Bash":
        command = tool_input.get("command", "")
        program = _extract_bash_program(command)
        return program or None, "command" if program else None

    if tool_name == "Grep":
        pattern = tool_input.get("pattern", "")
        path = tool_input.get("path", "")
        return f"{pattern}@{path}", "pattern"

    if tool_name == "Glob":
        pattern = tool_input.get("pattern", "")
        return pattern or None, "pattern" if pattern else None

    if tool_name == "Agent":
        prompt = tool_input.get("prompt", "")
        return prompt[:80] if prompt else None, "agent" if prompt else None

    return None, None


def _is_tool_result_only(msg: TranscriptMessage) -> bool:
    """Check if a user message contains only tool_result blocks."""
    if not msg.content_blocks:
        return False
    return all(b.block_type == "tool_result" for b in msg.content_blocks)


def group_into_turns(
    messages: list[TranscriptMessage],
) -> list[ConversationTurn]:
    """Group transcript messages into conversation turns.

    A new turn starts when a user message with at least one text block arrives.
    User messages with only tool_result blocks continue the current turn.
    """
    turns: list[ConversationTurn] = []
    current_turn: ConversationTurn | None = None

    for msg in messages:
        if msg.entry_type == "user":
            if _is_tool_result_only(msg):
                # Tool results continue the current turn
                continue

            # Extract user prompt text from text blocks
            prompt_parts: list[str] = []
            for block in msg.content_blocks:
                if block.block_type == "text":
                    prompt_parts.append(block.content)

            prompt_text = "\n".join(prompt_parts)
            if not prompt_text:
                continue

            current_turn = ConversationTurn(
                turn_number=len(turns) + 1,
                timestamp=msg.timestamp,
                user_prompt_text=prompt_text,
            )
            turns.append(current_turn)

        elif msg.entry_type == "assistant" and current_turn is not None:
            # Create an ApiCall for this assistant response
            api_call = ApiCall(
                api_call_index=len(current_turn.api_calls),
                conversation_turn=current_turn.turn_number,
                input_tokens=msg.input_tokens,
                output_tokens=msg.output_tokens,
                cache_read_tokens=msg.cache_read_tokens,
                cache_creation_tokens=msg.cache_creation_tokens,
                stop_reason=msg.stop_reason,
                timestamp=msg.timestamp,
            )
            current_turn.api_calls.append(api_call)

    return turns


def _content_hash(content: str) -> str:
    """SHA256 hash of content, truncated to 16 hex chars."""
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:16]


def _estimate_tokens(size_chars: int) -> int:
    """Rough token estimate from character count."""
    return max(1, size_chars // _CHARS_PER_TOKEN)


def _block_type_for_content_block(
    block_type_str: str,
    entry_type: str,
) -> BlockType:
    """Map content block type + message entry type to BlockType enum."""
    if block_type_str == "tool_use":
        return BlockType.TOOL_USE
    if block_type_str == "tool_result":
        return BlockType.TOOL_RESULT
    if entry_type == "user":
        return BlockType.USER_PROMPT
    if entry_type == "assistant":
        return BlockType.ASSISTANT_TEXT
    return BlockType.ASSISTANT_TEXT


def _detect_compactions_from_api(
    turns: list[ConversationTurn],
) -> list[int]:
    """Detect compaction events from cache_creation spikes.

    Returns list of turn numbers where compaction was detected.
    A compaction is detected when cache_creation_input_tokens > 50% of
    total input tokens (after the first API call), indicating the cached
    prefix was rebuilt.
    """
    compaction_turns: list[int] = []
    first_call_seen = False

    for turn in turns:
        for api_call in turn.api_calls:
            total_in = api_call.input_tokens + api_call.cache_read_tokens + api_call.cache_creation_tokens
            if total_in == 0:
                continue

            if not first_call_seen:
                first_call_seen = True
                continue  # Skip first API call (always 100% cache_create)

            cache_create_ratio = api_call.cache_creation_tokens / total_in
            if cache_create_ratio > COMPACTION_CACHE_CREATE_THRESHOLD:
                compaction_turns.append(turn.turn_number)
                break  # Only count once per turn

    return compaction_turns


def reconstruct_session(
    messages: list[TranscriptMessage],
    hook_events: list[TrackerEvent],
    config: StalenessConfig | None = None,
) -> tuple[
    list[ConversationTurn],
    list[TurnSnapshot],
    ContentStore,
    list[ContextEpoch],
    list[DataQualityWarning],
    dict[str, ContextBlock],
]:
    """Reconstruct context window state from transcript messages and hook events.

    Returns:
        turns: grouped conversation turns
        snapshots: per-turn context window snapshots
        content_store: block content indexed by block_id
        epochs: context epochs (compaction boundaries)
        warnings: data quality warnings
        block_registry: block_id -> ContextBlock mapping for all blocks
    """
    if config is None:
        config = StalenessConfig()

    warnings: list[DataQualityWarning] = []
    content_store = ContentStore()

    # Index compaction events by timestamp for matching
    compaction_events: list[PostCompactEvent] = [e for e in hook_events if isinstance(e, PostCompactEvent)]

    # Group messages into turns
    turns = group_into_turns(messages)

    # Initialize epoch tracking
    current_epoch = ContextEpoch(epoch_number=0, started_at_turn=1)
    epochs: list[ContextEpoch] = [current_epoch]

    # Accumulated context blocks and snapshots
    all_block_ids: list[str] = []  # ordered block IDs in context
    all_blocks: dict[str, ContextBlock] = {}  # block_id -> block
    tool_use_map: dict[str, tuple[str | None, str | None]] = {}  # tool_use_id -> (resource, resource_type)
    snapshots: list[TurnSnapshot] = []

    # Global counters
    api_call_global_index = 0
    block_counter = 0

    # Process each message in order, building context blocks
    for msg in messages:
        if msg.entry_type == "system":
            # Create pinned blocks for system messages
            for cb in msg.content_blocks:
                block_counter += 1
                block_id = f"b{block_counter:06d}"
                content = cb.content or ""
                chash = _content_hash(content)
                size_tokens = _estimate_tokens(cb.size_chars)

                context_block = ContextBlock(
                    block_id=block_id,
                    turn_entered=1,
                    api_call_entered=0,
                    epoch_entered=0,
                    block_type=BlockType.SYSTEM,
                    size_chars=cb.size_chars,
                    size_tokens_est=size_tokens,
                    content_hash=chash,
                    is_pinned=True,
                    timestamp=msg.timestamp,
                )
                all_blocks[block_id] = context_block
                all_block_ids.append(block_id)
                content_store.add(block_id, content)
            continue

        # Determine which turn this message belongs to
        turn_number = _find_turn_for_message(msg, turns)
        if turn_number == 0:
            # Message before first user prompt or unmatched
            continue

        for cb in msg.content_blocks:
            block_counter += 1
            block_id = f"b{block_counter:06d}"

            block_type = _block_type_for_content_block(cb.block_type, msg.entry_type)

            resource: str | None = None
            resource_type: str | None = None
            tool_name: str | None = None
            tool_use_id: str | None = None
            parent_block_id: str | None = None

            if cb.block_type == "tool_use":
                tool_name = cb.tool_name
                tool_use_id = cb.tool_use_id
                if cb.tool_input:
                    resource, resource_type = extract_resource(
                        cb.tool_name or "",
                        cb.tool_input,
                    )
                # Store for pairing with tool_result
                if tool_use_id:
                    tool_use_map[tool_use_id] = (resource, resource_type)

            elif cb.block_type == "tool_result":
                tool_use_id = cb.tool_use_id
                # Copy resource from paired tool_use
                if tool_use_id and tool_use_id in tool_use_map:
                    resource, resource_type = tool_use_map[tool_use_id]
                # Find parent tool_use block
                if tool_use_id:
                    for bid in reversed(all_block_ids):
                        blk = all_blocks[bid]
                        if blk.tool_use_id == tool_use_id and blk.block_type == BlockType.TOOL_USE:
                            parent_block_id = bid
                            break

            content = cb.content or ""
            chash = _content_hash(content)
            size_tokens = _estimate_tokens(cb.size_chars)

            context_block = ContextBlock(
                block_id=block_id,
                turn_entered=turn_number,
                api_call_entered=api_call_global_index,
                epoch_entered=current_epoch.epoch_number,
                block_type=block_type,
                resource=resource,
                resource_type=resource_type,
                size_chars=cb.size_chars,
                size_tokens_est=size_tokens,
                content_hash=chash,
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                parent_block_id=parent_block_id,
                is_error=cb.is_error,
                timestamp=msg.timestamp,
            )

            all_blocks[block_id] = context_block
            all_block_ids.append(block_id)
            content_store.add(block_id, content)

        # Track API calls from assistant messages
        if msg.entry_type == "assistant" and msg.output_tokens > 0:
            api_call_global_index += 1

    # Create synthetic pinned blocks from first-turn cache_creation data
    # if no explicit system blocks were found.
    has_system_blocks = any(b.block_type == BlockType.SYSTEM for b in all_blocks.values())
    if not has_system_blocks and turns:
        first_turn = turns[0]
        if first_turn.api_calls:
            first_call = first_turn.api_calls[0]
            cache_creation = first_call.cache_creation_tokens
            if cache_creation > 0:
                # Estimate: ~6K tokens for system prompt (Claude Code default)
                system_prompt_est = min(6000, cache_creation)
                remainder = cache_creation - system_prompt_est

                block_counter += 1
                sp_block_id = f"b{block_counter:06d}"
                sp_block = ContextBlock(
                    block_id=sp_block_id,
                    turn_entered=1,
                    api_call_entered=0,
                    epoch_entered=0,
                    block_type=BlockType.SYSTEM,
                    size_chars=system_prompt_est * _CHARS_PER_TOKEN,
                    size_tokens_est=system_prompt_est,
                    content_hash=_content_hash("__system_prompt__"),
                    is_pinned=True,
                    timestamp=first_turn.timestamp,
                )
                all_blocks[sp_block_id] = sp_block
                all_block_ids.insert(0, sp_block_id)
                content_store.add(sp_block_id, "[System Prompt — estimated from cache data]")

                if remainder > 0:
                    block_counter += 1
                    claude_md_block_id = f"b{block_counter:06d}"
                    claude_md_block = ContextBlock(
                        block_id=claude_md_block_id,
                        turn_entered=1,
                        api_call_entered=0,
                        epoch_entered=0,
                        block_type=BlockType.SYSTEM,
                        size_chars=remainder * _CHARS_PER_TOKEN,
                        size_tokens_est=remainder,
                        content_hash=_content_hash("__claude_md_skills__"),
                        is_pinned=True,
                        timestamp=first_turn.timestamp,
                    )
                    all_blocks[claude_md_block_id] = claude_md_block
                    all_block_ids.insert(1, claude_md_block_id)
                    content_store.add(
                        claude_md_block_id,
                        "[CLAUDE.md + Skills — estimated from cache data]",
                    )

    # ---- Detect compaction events ----
    # Primary: detect from API cache-creation spikes (works without hook events)
    api_compaction_turn_numbers = set(_detect_compactions_from_api(turns))

    # Secondary: also detect from hook events (PostCompactEvent)
    hook_compaction_events = list(compaction_events)  # copy, will consume

    # ---- Build per-turn snapshots with epoch-based block inclusion ----
    epoch_number = 0
    current_epoch_start_turn = 1
    compaction_summary_block_id: str | None = None

    previous_block_ids: set[str] = set()

    for turn in turns:
        tn = turn.turn_number

        # Check if compaction happened at this turn
        compaction_detected = tn in api_compaction_turn_numbers

        # Also check hook events (timestamp-based)
        hook_compaction_size: int | None = None
        for ce in hook_compaction_events:
            if turn.timestamp and ce.timestamp and ce.timestamp <= turn.timestamp:
                compaction_detected = True
                hook_compaction_size = ce.compact_summary_length
                hook_compaction_events.remove(ce)
                break

        if compaction_detected and tn > 1:
            # Count blocks from previous epochs that are being compacted out
            blocks_before = len(
                [bid for bid in all_block_ids if all_blocks[bid].turn_entered < tn and not all_blocks[bid].is_pinned]
            )

            epoch_number += 1
            current_epoch_start_turn = tn

            # Estimate compaction summary size from API data or hook event
            summary_size: int | None = hook_compaction_size
            if summary_size is None:
                # Estimate: after compaction the entire context is rebuilt into cache.
                # The compaction summary replaces old content, so estimate from
                # the difference between actual context and new-epoch blocks.
                last_api = turn.api_calls[-1] if turn.api_calls else None
                if last_api:
                    actual_total = last_api.input_tokens + last_api.cache_read_tokens + last_api.cache_creation_tokens
                    # New-epoch blocks (pinned + just entered) token estimate
                    new_blocks_est = sum(
                        all_blocks[bid].size_tokens_est
                        for bid in all_block_ids
                        if all_blocks[bid].is_pinned or all_blocks[bid].turn_entered >= tn
                    )
                    summary_size = max(0, actual_total - new_blocks_est)

            current_epoch = ContextEpoch(
                epoch_number=epoch_number,
                started_at_turn=tn,
                compaction_summary_size=summary_size,
                blocks_before_compaction=blocks_before,
            )
            epochs.append(current_epoch)

            # Create a synthetic compaction summary block
            block_counter += 1
            compaction_summary_block_id = f"b{block_counter:06d}"
            est_summary_tokens = summary_size if summary_size else 0
            summary_block = ContextBlock(
                block_id=compaction_summary_block_id,
                turn_entered=tn,
                api_call_entered=api_call_global_index,
                epoch_entered=epoch_number,
                block_type=BlockType.COMPACTION_SUMMARY,
                size_chars=est_summary_tokens * _CHARS_PER_TOKEN,
                size_tokens_est=est_summary_tokens,
                content_hash=_content_hash(f"__compaction_summary_epoch_{epoch_number}__"),
                is_pinned=False,
                timestamp=turn.timestamp,
            )
            all_blocks[compaction_summary_block_id] = summary_block
            all_block_ids.append(compaction_summary_block_id)
            content_store.add(
                compaction_summary_block_id,
                f"[Compaction summary — epoch {epoch_number}, "
                f"~{est_summary_tokens} tokens, "
                f"replaced {blocks_before} blocks]",
            )

        turn.epoch = epoch_number

        # ---- Determine blocks in context at this turn ----
        # Blocks in context = pinned blocks + blocks entered in CURRENT epoch
        # + compaction summary block (if past epoch 0)
        current_block_ids: list[str] = []

        for bid in all_block_ids:
            block = all_blocks[bid]
            if block.is_pinned:
                current_block_ids.append(bid)
            elif (
                block.turn_entered >= current_epoch_start_turn
                and block.turn_entered <= tn
                and block.block_type != BlockType.COMPACTION_SUMMARY
            ):
                current_block_ids.append(bid)

        # Add compaction summary block at the front (after pinned) if past epoch 0
        if epoch_number > 0 and compaction_summary_block_id:
            # Insert after pinned blocks
            insert_pos = 0
            for i, bid in enumerate(current_block_ids):
                if all_blocks[bid].is_pinned:
                    insert_pos = i + 1
                else:
                    break
            current_block_ids.insert(insert_pos, compaction_summary_block_id)

        current_set = set(current_block_ids)
        entered = current_set - previous_block_ids
        exited = previous_block_ids - current_set

        # Aggregate token counts from API calls in this turn
        total_input = sum(ac.input_tokens for ac in turn.api_calls)
        total_output = sum(ac.output_tokens for ac in turn.api_calls)
        total_cache_read = sum(ac.cache_read_tokens for ac in turn.api_calls)
        total_cache_create = sum(ac.cache_creation_tokens for ac in turn.api_calls)

        # Estimate total tokens from block sizes
        total_tokens_est = sum(all_blocks[bid].size_tokens_est for bid in current_block_ids)

        # REAL context size from the last API call in this turn (ground truth)
        actual_context_tokens = 0
        if turn.api_calls:
            last_api = turn.api_calls[-1]
            actual_context_tokens = last_api.input_tokens + last_api.cache_read_tokens + last_api.cache_creation_tokens

        snapshot = TurnSnapshot(
            turn_number=tn,
            timestamp=turn.timestamp,
            epoch=turn.epoch,
            block_ids=current_block_ids,
            block_states=[],  # Populated by staleness detection
            blocks_entered_ids=sorted(entered),
            blocks_exited_ids=sorted(exited),
            total_tokens_est=total_tokens_est,
            input_tokens=total_input,
            output_tokens=total_output,
            cache_read_tokens=total_cache_read,
            cache_creation_tokens=total_cache_create,
            actual_context_tokens=actual_context_tokens,
            compaction_detected=compaction_detected,
            api_call_count=len(turn.api_calls),
        )
        snapshots.append(snapshot)
        previous_block_ids = current_set

    return turns, snapshots, content_store, epochs, warnings, all_blocks


def _find_turn_for_message(
    msg: TranscriptMessage,
    turns: list[ConversationTurn],
) -> int:
    """Find which turn number a message belongs to.

    Returns 0 if the message is before the first turn.
    """
    if not turns:
        return 0

    # For user messages, match by timestamp and prompt text
    if msg.entry_type == "user":
        for turn in turns:
            if turn.timestamp == msg.timestamp:
                # Check if prompt text matches
                for cb in msg.content_blocks:
                    if cb.block_type == "text" and cb.content in turn.user_prompt_text:
                        return turn.turn_number
                # Tool result messages belong to the preceding turn
                if all(cb.block_type == "tool_result" for cb in msg.content_blocks):
                    # Find the most recent turn at or before this message
                    return _find_nearest_turn(msg, turns)
        # Fallback: find nearest turn
        return _find_nearest_turn(msg, turns)

    # For assistant messages, find the turn based on sequence ordering
    return _find_nearest_turn(msg, turns)


def _find_nearest_turn(
    msg: TranscriptMessage,
    turns: list[ConversationTurn],
) -> int:
    """Find the nearest turn at or before this message's sequence index."""
    # Use sequence index to find which turn this message falls into
    # Turns are ordered; find the last turn whose first message seq <= msg seq
    best_turn = 0
    for turn in turns:
        # Compare timestamps or infer from ordering
        if turn.timestamp is not None and msg.timestamp is not None:
            if turn.timestamp <= msg.timestamp:
                best_turn = turn.turn_number
            else:
                break
        else:
            # Fallback: just assign to latest turn
            best_turn = turn.turn_number

    return best_turn
