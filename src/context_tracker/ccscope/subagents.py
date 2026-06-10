"""Parse subagent transcripts for child window blocks + churn.

Each subagent in a Claude Code session runs in its own context window.
This module reads their transcripts (same JSONL format as the parent)
and produces summary stats, a collapsed block for the parent timeline,
and per-call churn entries.
"""

from __future__ import annotations

import json
from pathlib import Path


def parse_subagents(subagents_dir: Path) -> list[dict]:
    """Parse all subagent transcripts in a directory.

    Each subagent gets:
    - Summary stats (peak resident, total cache_read, api_calls)
    - A collapsed block for the parent timeline
    - Its own blocks + churn (for nested/expanded view)

    Args:
        subagents_dir: Path to the subagents directory containing
            agent-{id}.jsonl and agent-{id}.meta.json files.

    Returns:
        List of dicts, one per subagent:
        {
            "agent_id": str,
            "agent_type": str,
            "description": str,
            "peak_resident": int,
            "total_cache_read": int,
            "api_calls": int,
            "parent_block": dict,   # Context Scope block for parent timeline
            "churn": list[dict],    # per-call churn for this subagent
        }
    """
    if not subagents_dir.exists():
        return []

    # Discover subagent IDs from .jsonl files
    agent_ids = _discover_agent_ids(subagents_dir)
    if not agent_ids:
        return []

    results = []
    for agent_id in sorted(agent_ids):
        result = _parse_single_subagent(subagents_dir, agent_id)
        results.append(result)

    return results


def parse_workflows(subagents_dir: Path) -> list[dict]:
    """Parse all multi-agent workflow runs under <subagents>/workflows/.

    A workflow run lives at ``<subagents>/workflows/wf_<runid>/`` and contains:
      - ``agent-<id>.jsonl`` / ``agent-<id>.meta.json`` (same format as plain
        subagents — token data is reconstructed by the existing machinery)
      - ``journal.jsonl`` — orchestration log. "started" lines map a phase
        ``key`` to an ``agentId``; "result" lines carry a human-readable label
        (e.g. ``result.dimension``).

    Args:
        subagents_dir: The session's ``subagents/`` directory.

    Returns:
        List of workflow-run dicts, one per ``wf_<runid>``:
        {
            "wf_id": str,
            "name": str | None,
            "agents": [ <subagent dict with extra phase/label keys>, ... ],
        }
        Each agent dict has the same shape as :func:`parse_subagents` entries
        plus ``"phase"`` and ``"label"`` (either may be None for a still-running
        or partially-journaled run).
    """
    workflows_dir = subagents_dir / "workflows"
    if not workflows_dir.exists():
        return []

    runs: list[dict] = []
    for run_dir in sorted(workflows_dir.glob("wf_*")):
        if not run_dir.is_dir():
            continue
        run = _parse_workflow_run(run_dir)
        if run["agents"]:
            runs.append(run)
    return runs


def _parse_workflow_run(run_dir: Path) -> dict:
    """Parse a single workflow run directory."""
    wf_id = run_dir.name  # "wf_<runid>"

    phases, labels = _parse_journal(run_dir / "journal.jsonl")

    agent_ids = _discover_agent_ids(run_dir)
    agents: list[dict] = []
    for agent_id in sorted(agent_ids):
        agent = _parse_single_subagent(run_dir, agent_id)
        agent["phase"] = phases.get(agent_id)
        agent["label"] = labels.get(agent_id)
        agents.append(agent)

    # No explicit workflow name on disk yet — derive a friendly default.
    name = wf_id

    return {"wf_id": wf_id, "name": name, "agents": agents}


def _parse_journal(journal_path: Path) -> tuple[dict[str, str], dict[str, str | None]]:
    """Parse journal.jsonl into per-agent phase + label maps.

    Returns ``(phases, labels)`` keyed by agentId:
      - phases[agentId] = the "started"/"result" line's ``key`` (phase grouping)
      - labels[agentId] = a human-readable label, preferring ``result.dimension``
    Tolerant of partial journals (still-running workflows) and malformed lines.
    """
    phases: dict[str, str] = {}
    labels: dict[str, str | None] = {}
    if not journal_path.exists():
        return phases, labels

    with open(journal_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            agent_id = entry.get("agentId")
            if not agent_id:
                continue
            key = entry.get("key")
            if key is not None and agent_id not in phases:
                phases[agent_id] = key
            result = entry.get("result")
            if isinstance(result, dict):
                label = result.get("dimension") or result.get("summary")
                if label:
                    labels[agent_id] = str(label)[:200]
    return phases, labels


def _discover_agent_ids(subagents_dir: Path) -> list[str]:
    """Find all unique agent IDs from .jsonl files in the directory."""
    ids = []
    for path in subagents_dir.glob("agent-*.jsonl"):
        # Extract ID from "agent-{id}.jsonl"
        name = path.stem  # "agent-{id}"
        agent_id = name[len("agent-") :]
        if agent_id:
            ids.append(agent_id)
    return ids


def _parse_single_subagent(subagents_dir: Path, agent_id: str) -> dict:
    """Parse a single subagent's meta + transcript."""
    # Read metadata
    agent_type, description = _read_meta(subagents_dir, agent_id)

    # Read and parse transcript
    jsonl_path = subagents_dir / f"agent-{agent_id}.jsonl"
    entries = _load_entries(jsonl_path)

    # Walk entries, collect stats from completed assistant messages
    peak_resident = 0
    total_cache_read = 0
    api_calls = 0
    churn: list[dict] = []

    for entry in entries:
        if entry.get("type") != "assistant":
            continue
        msg = entry.get("message", {})
        if not _is_completed_assistant(msg):
            continue

        usage = msg.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        cache_creation = usage.get("cache_creation_input_tokens", 0)
        cache_read = usage.get("cache_read_input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        resident = input_tokens + cache_read + cache_creation
        if resident > peak_resident:
            peak_resident = resident

        total_cache_read += cache_read

        churn.append(
            {
                "turn": api_calls,
                "cache_read": cache_read,
                "cache_creation": cache_creation,
                "input": input_tokens,
                "output": output_tokens,
            }
        )

        api_calls += 1

    # Build parent block
    parent_block = _build_parent_block(
        agent_id,
        agent_type,
        description,
        peak_resident,
        total_cache_read,
        api_calls,
    )

    return {
        "agent_id": agent_id,
        "agent_type": agent_type,
        "description": description,
        "peak_resident": peak_resident,
        "total_cache_read": total_cache_read,
        "api_calls": api_calls,
        "parent_block": parent_block,
        "churn": churn,
    }


def _read_meta(subagents_dir: Path, agent_id: str) -> tuple[str, str]:
    """Read agent type and description from meta.json.

    Returns (agent_type, description) with sensible defaults if missing.
    """
    meta_path = subagents_dir / f"agent-{agent_id}.meta.json"
    if not meta_path.exists():
        return "unknown", ""
    try:
        data = json.loads(meta_path.read_text())
        return (
            data.get("agentType", "unknown"),
            data.get("description", ""),
        )
    except (json.JSONDecodeError, OSError):
        return "unknown", ""


def _load_entries(path: Path) -> list[dict]:
    """Load all JSONL entries from a transcript file."""
    entries: list[dict] = []
    if not path.exists():
        return entries
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def _is_completed_assistant(msg: dict) -> bool:
    """Check if an assistant message is a completed API response.

    Same logic as parse_transcript.py — skip synthetic and incomplete entries.
    """
    if msg.get("model") == "synthetic":
        return False
    if msg.get("stop_reason") is None:
        return False
    usage = msg.get("usage", {})
    if usage.get("output_tokens", 0) <= 0:
        return False
    return True


def _build_parent_block(
    agent_id: str,
    agent_type: str,
    description: str,
    peak_resident: int,
    total_cache_read: int,
    api_calls: int,
) -> dict:
    """Build a Context Scope block representing this subagent in the parent timeline.

    enter/exit are placeholders (0/None) — the reconciler (Task 5) will fix
    them by matching subagent spawn times to parent API calls.
    """
    label = f"{agent_type}: {description[:60]}"
    content = (
        f"Subagent {agent_type} ({api_calls} calls, peak {peak_resident:,} tok, churn {total_cache_read:,} cache_read)"
    )

    return {
        "id": f"subagent-{agent_id}",
        "type": "tool_result",
        "label": label,
        "tokens": peak_resident,
        "enter": 0,
        "exit": None,
        "cached": False,
        "ref": True,
        "content": content,
    }
