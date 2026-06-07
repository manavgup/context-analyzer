"""Four-layer additive-capped staleness scoring engine.

EXPERIMENTAL HEURISTIC — all weights and thresholds require calibration
against real conversation sessions.

Scoring model (additive-capped, NOT multiplicative):
    age  = base_decay()      * 0.35
    res  = resource_factor() * 0.25
    ref  = reference_factor()* 0.25
    ctx  = max(0, ((group + task) / 2 - 0.75)) * 0.15
    score = clamp(age + res + ref + ctx, 0.0, 1.0)

Labels:
    <0.3  = active
    <0.6  = warm
    <0.8  = stale
    >=0.8 = dead_weight
"""

from __future__ import annotations

import math
import os
import re
from collections import defaultdict

from context_tracker.analysis.config import StalenessConfig
from context_tracker.analysis.models import BlockType, ContextBlock, ConversationTurn

# Common identifiers that produce false-positive reference matches
_FALSE_POSITIVE_IDENTIFIERS: frozenset[str] = frozenset(
    {
        "self",
        "cls",
        "args",
        "kwargs",
        "None",
        "True",
        "False",
        "return",
        "import",
        "from",
        "class",
        "def",
        "if",
        "else",
        "for",
        "while",
        "try",
        "except",
        "with",
        "as",
        "in",
        "is",
        "not",
        "and",
        "or",
        "the",
        "a",
        "an",
        "to",
        "of",
        "it",
        "this",
        "that",
        "get",
        "set",
        "data",
        "value",
        "result",
        "name",
        "type",
        "key",
        "id",
        "str",
        "int",
        "list",
        "dict",
        "file",
        "path",
        "test",
        "init",
        "main",
        "run",
        "new",
    }
)


def detect_superseded(blocks: list[ContextBlock]) -> dict[str, str]:
    """Detect blocks superseded by newer reads of the same resource.

    Only considers TOOL_RESULT blocks with a non-None resource.
    When the same resource appears multiple times, all older copies
    are superseded by the newest one.

    Returns:
        Dict mapping old_block_id -> new_block_id that supersedes it.
    """
    # Group tool-result blocks by resource
    resource_blocks: dict[str, list[ContextBlock]] = defaultdict(list)
    for block in blocks:
        if block.block_type == BlockType.TOOL_RESULT and block.resource is not None:
            resource_blocks[block.resource].append(block)

    superseded: dict[str, str] = {}
    for _resource, res_blocks in resource_blocks.items():
        if len(res_blocks) < 2:
            continue
        # Sort by turn_entered so the last entry is the newest
        sorted_blocks = sorted(res_blocks, key=lambda b: b.turn_entered)
        newest = sorted_blocks[-1]
        for older in sorted_blocks[:-1]:
            superseded[older.block_id] = newest.block_id

    return superseded


def base_decay(turns_since_entry: int, window: int) -> float:
    """Age-based decay curve.

    - 0.0 for turns_since_entry <= 2 (grace period)
    - Linear ramp from 0.0 to 0.5 between 2 and window
    - Slow logarithmic climb toward 1.0 after window

    Args:
        turns_since_entry: How many turns since the block entered context.
        window: The decay window (number of turns for half-life).

    Returns:
        Decay value in [0.0, 1.0].
    """
    if turns_since_entry <= 2:
        return 0.0

    if window <= 2:
        # Edge case: very small window
        window = 3

    if turns_since_entry <= window:
        # Linear ramp from 0.0 at turn 2 to 0.5 at turn window
        return 0.5 * (turns_since_entry - 2) / (window - 2)

    # Past window: slow logarithmic climb from 0.5 toward 1.0
    # Uses log curve that approaches 1.0 asymptotically
    overshoot = turns_since_entry - window
    # 0.5 + 0.5 * (1 - 1 / (1 + log(1 + overshoot/window)))
    extra = 0.5 * (1.0 - 1.0 / (1.0 + math.log(1.0 + overshoot / window)))
    return min(0.5 + extra, 1.0)


def resource_factor(
    block: ContextBlock,
    resource_last_used: dict[str, int],
    current_turn: int,
    window: int,
) -> float:
    """Resource recency factor.

    Returns 0.0 if the block's resource was used within the window,
    gradual decay to 1.0 after.

    Args:
        block: The context block.
        resource_last_used: Maps resource path -> last turn it was referenced.
        current_turn: Current conversation turn number.
        window: Resource staleness window.

    Returns:
        Factor in [0.0, 1.0].
    """
    if block.resource is None:
        return 0.5  # No resource info — neutral

    last_used = resource_last_used.get(block.resource)
    if last_used is None:
        return 1.0  # Never used again after entry

    turns_since_use = current_turn - last_used
    if turns_since_use <= 0:
        return 0.0

    if turns_since_use >= window:
        return 1.0

    return turns_since_use / window


def _extract_identifiers(block: ContextBlock) -> set[str]:
    """Extract searchable identifiers from a block's resource and metadata.

    Extracts:
    - File name (without extension) and full path components from resource
    - Function/class names via simple regex patterns
    """
    identifiers: set[str] = set()

    if block.resource:
        # Full resource path
        identifiers.add(block.resource)
        # File name without extension
        basename = os.path.basename(block.resource)
        name_no_ext = os.path.splitext(basename)[0]
        if name_no_ext and len(name_no_ext) > 2:
            identifiers.add(name_no_ext)
        # Basename with extension
        if basename and len(basename) > 2:
            identifiers.add(basename)

    if block.tool_name:
        identifiers.add(block.tool_name)

    # Filter out common false positives
    identifiers -= _FALSE_POSITIVE_IDENTIFIERS

    return identifiers


def reference_factor(
    block: ContextBlock,
    messages_since_block: list[str],
    scan_window: int,
) -> float:
    """Reference recency factor.

    Returns 0.0 if the block's identifiers appear in recent messages
    (word-boundary matching), 1.0 if not referenced.

    Args:
        block: The context block.
        messages_since_block: List of message texts after this block entered.
        scan_window: Number of recent messages to scan.

    Returns:
        Factor in [0.0, 1.0].
    """
    identifiers = _extract_identifiers(block)
    if not identifiers:
        return 0.5  # No identifiers to check — neutral

    # Only scan the most recent messages within the window
    recent = messages_since_block[-scan_window:] if messages_since_block else []
    if not recent:
        # No messages to search. Could mean the block is very new (no opportunity
        # for reference yet) or very old with no recorded messages. We rely on
        # the caller to provide an empty list for truly new blocks.
        # Return a neutral value that doesn't dominate the score.
        return 0.5

    # Check for word-boundary matches
    for ident in identifiers:
        pattern = re.compile(re.escape(ident))
        for msg in recent:
            if pattern.search(msg):
                return 0.0  # Found a reference

    return 1.0


def group_factor(
    block: ContextBlock,
    active_resources: set[str],
) -> float:
    """Group/directory cohesion factor.

    Simplified: always returns 1.0.
    Directory-based grouping was found to be too broad (Codex feedback).
    Import-based grouping to be added later.

    Args:
        block: The context block.
        active_resources: Set of currently active resource paths.

    Returns:
        Always 1.0 for now.
    """
    return 1.0


def task_factor(
    block: ContextBlock,
    task_boundaries: list[int],
    current_turn: int,
) -> float:
    """Task boundary factor.

    Returns 1.5 if a task boundary was crossed since the block entered,
    1.0 otherwise. Task boundaries indicate the user shifted to a new
    sub-task, making older context less relevant.

    Args:
        block: The context block.
        task_boundaries: List of turn numbers where task boundaries were detected.
        current_turn: Current conversation turn number.

    Returns:
        1.5 if boundary crossed, 1.0 otherwise.
    """
    for boundary_turn in task_boundaries:
        if block.turn_entered < boundary_turn <= current_turn:
            return 1.5

    return 1.0


def label_staleness(score: float) -> str:
    """Convert a staleness score to a human-readable label.

    Args:
        score: Staleness score in [0.0, 1.0].

    Returns:
        One of: "active", "warm", "stale", "dead_weight".
    """
    if score < 0.3:
        return "active"
    if score < 0.6:
        return "warm"
    if score < 0.8:
        return "stale"
    return "dead_weight"


def compute_staleness(
    block: ContextBlock,
    current_turn: int,
    config: StalenessConfig,
    resource_last_used: dict[str, int],
    messages_since_block: list[str],
    active_resources: set[str],
    task_boundaries: list[int],
    superseded_map: dict[str, str],
) -> tuple[float, str]:
    """Compute staleness score and label for a single block.

    Uses the four-layer additive-capped model:
        age  = base_decay()      * 0.35
        res  = resource_factor() * 0.25
        ref  = reference_factor()* 0.25
        ctx  = max(0, ((group + task) / 2 - 0.75)) * 0.15
        score = clamp(age + res + ref + ctx, 0.0, 1.0)

    Special cases:
        - Pinned blocks always return (0.0, "pinned")
        - Superseded blocks always return (0.9, "dead_weight")

    Args:
        block: The context block to score.
        current_turn: Current conversation turn number.
        config: Staleness configuration thresholds.
        resource_last_used: Maps resource path -> last turn it was referenced.
        messages_since_block: Message texts after this block entered.
        active_resources: Set of currently active resource paths.
        task_boundaries: Turn numbers where task boundaries were detected.
        superseded_map: Maps old_block_id -> new_block_id for superseded blocks.

    Returns:
        Tuple of (score, label) where score is in [0.0, 1.0] and label is
        one of "pinned", "active", "warm", "stale", "dead_weight".
    """
    # Special case: pinned blocks are always fresh
    if block.is_pinned:
        return (0.0, "pinned")

    # Special case: superseded blocks are dead weight
    if block.block_id in superseded_map:
        return (0.9, "dead_weight")

    # Layer 1: Age-based decay
    turns_since = current_turn - block.turn_entered
    age = base_decay(turns_since, config.decay_window) * 0.35

    # Layer 2: Resource recency
    res = resource_factor(block, resource_last_used, current_turn, config.resource_window) * 0.25

    # Layer 3: Reference recency
    ref = reference_factor(block, messages_since_block, config.reference_scan_window) * 0.25

    # Layer 4: Context (group + task)
    grp = group_factor(block, active_resources)
    tsk = task_factor(block, task_boundaries, current_turn)
    ctx = max(0.0, ((grp + tsk) / 2.0 - 0.75)) * 0.15

    # Additive-capped score
    score = age + res + ref + ctx
    score = max(0.0, min(1.0, score))

    return (score, label_staleness(score))


def detect_task_boundaries(
    turns: list[ConversationTurn],
    config: StalenessConfig,
) -> list[int]:
    """Detect task boundaries based on time gaps and keyword overlap.

    A boundary is detected when:
    1. The time gap between consecutive turns exceeds config.task_boundary_time_gap minutes
    2. Both prompts are at least config.min_prompt_length_for_boundary characters long
    3. The keyword overlap between consecutive prompts is below config.task_boundary_overlap

    Returns:
        List of turn numbers where task boundaries were detected.
    """
    from datetime import datetime

    boundaries: list[int] = []

    for i in range(1, len(turns)):
        prev = turns[i - 1]
        curr = turns[i]

        # Need timestamps to detect time gap
        if not prev.timestamp or not curr.timestamp:
            continue

        try:
            t1 = datetime.fromisoformat(prev.timestamp.replace("Z", "+00:00"))
            t2 = datetime.fromisoformat(curr.timestamp.replace("Z", "+00:00"))
            gap_minutes = (t2 - t1).total_seconds() / 60
        except (ValueError, TypeError):
            continue

        if gap_minutes <= config.task_boundary_time_gap:
            continue

        # Only check overlap on long prompts
        min_len = config.min_prompt_length_for_boundary
        if len(curr.user_prompt_text) < min_len or len(prev.user_prompt_text) < min_len:
            continue

        # Simple keyword overlap (Jaccard)
        prev_words = set(prev.user_prompt_text.lower().split())
        curr_words = set(curr.user_prompt_text.lower().split())
        if not prev_words or not curr_words:
            continue
        overlap = len(prev_words & curr_words) / max(len(prev_words | curr_words), 1)

        if overlap < config.task_boundary_overlap:
            boundaries.append(curr.turn_number)

    return boundaries
