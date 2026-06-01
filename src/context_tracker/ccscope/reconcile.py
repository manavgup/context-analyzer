"""Reconcile all sources into Context Scope blocks + churn format."""

from __future__ import annotations

import json
from pathlib import Path

from context_tracker.ccscope.parse_transcript import parse_transcript_to_blocks
from context_tracker.ccscope.offload import resolve_offloads
from context_tracker.ccscope.subagents import parse_subagents
from context_tracker.storage import DEFAULT_TRACE_DIR


def find_session_paths(
    session_id: str,
    projects_dir: Path | None = None,
    trace_dir: Path | None = None,
) -> dict:
    """Discover all data paths for a session.

    Returns dict with keys:
        transcript: Path to parent transcript JSONL (Source A)
        hook_events: Path to hook event JSONL (Source B), or None
        tool_results: Path to tool-results directory (Source D), or None
        subagents: Path to subagents directory (Source E), or None
    """
    if projects_dir is None:
        projects_dir = Path.home() / ".claude" / "projects"
    if trace_dir is None:
        trace_dir = DEFAULT_TRACE_DIR

    result = {
        "transcript": None,
        "hook_events": None,
        "tool_results": None,
        "subagents": None,
    }

    # Find transcript (Source A) - search all project dirs
    for jsonl in projects_dir.rglob(f"{session_id}.jsonl"):
        result["transcript"] = jsonl
        break

    # Find session folder (contains tool-results/ and subagents/)
    for d in projects_dir.rglob(session_id):
        if d.is_dir():
            tr_dir = d / "tool-results"
            if tr_dir.exists():
                result["tool_results"] = tr_dir
            sa_dir = d / "subagents"
            if sa_dir.exists():
                result["subagents"] = sa_dir
            break

    # Find hook events (Source B)
    hook_path = trace_dir / f"{session_id}.jsonl"
    if hook_path.exists():
        result["hook_events"] = hook_path

    return result


def reconcile(
    session_id: str,
    projects_dir: Path | None = None,
    trace_dir: Path | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Reconcile all sources into Context Scope format.

    Returns:
        blocks: list of block dicts (Context Scope format)
        churn: list of churn dicts (per API call)
        subagent_summaries: list of subagent summary dicts
    """
    paths = find_session_paths(session_id, projects_dir, trace_dir)

    if paths["transcript"] is None:
        raise FileNotFoundError(f"No transcript found for session {session_id}")

    # 1. A is the spine -- parse transcript into blocks + churn
    blocks, churn = parse_transcript_to_blocks(paths["transcript"])

    # 2. Overlay B for failures/timing (hook events)
    if paths["hook_events"]:
        blocks = overlay_hook_events(blocks, paths["hook_events"])

    # 3. Resolve D for resident sizes (tool-results offload)
    if paths["tool_results"]:
        blocks = resolve_offloads(blocks, paths["tool_results"])

    # 4. Attach E as collapsed blocks (subagents)
    subagent_summaries = []
    if paths["subagents"]:
        subagent_summaries = parse_subagents(paths["subagents"])
        for sa in subagent_summaries:
            blocks.append(sa["parent_block"])

    return blocks, churn, subagent_summaries


def overlay_hook_events(blocks: list[dict], hook_events_path: Path) -> list[dict]:
    """Overlay hook event data onto blocks.

    Adds failure information from PostToolUseFailure events.
    Hook events provide real-time entry timing and failure details
    that the transcript may not capture.
    """
    from context_tracker.models import parse_event, PostToolUseFailureEvent

    # Read hook events
    failures = {}  # tool_use_id -> error info
    with open(hook_events_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = parse_event(line)
            if isinstance(event, PostToolUseFailureEvent):
                failures[event.tool_use_id] = {
                    "error_length": event.error_length,
                    "timestamp": event.timestamp,
                }

    # Annotate blocks with failure info
    for block in blocks:
        if block.get("type") != "tool_result":
            continue

        # Extract tool_use_id from block ID (format: t{N}-tool_result-{tool_use_id})
        block_id = block.get("id", "")
        parts = block_id.split("-", 2)
        if len(parts) >= 3:
            possible_tu_id = parts[2]
            if possible_tu_id in failures:
                block["failed"] = True
                block["label"] = "\u26a0 " + block.get("label", "")

    return blocks


def write_output(
    blocks: list[dict],
    churn: list[dict],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write blocks.json and churn.json to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    blocks_path = output_dir / "blocks.json"
    churn_path = output_dir / "churn.json"

    blocks_path.write_text(json.dumps(blocks, indent=None), encoding="utf-8")
    churn_path.write_text(json.dumps(churn, indent=None), encoding="utf-8")

    return blocks_path, churn_path
