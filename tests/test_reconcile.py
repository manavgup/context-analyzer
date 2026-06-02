"""Tests for ccscope reconciliation — merging all sources into blocks + churn."""

import json
from pathlib import Path

import pytest

from context_tracker.ccscope.reconcile import (
    find_session_paths,
    overlay_hook_events,
    reconcile,
    write_output,
)

# ---------------------------------------------------------------------------
# Real data paths
# ---------------------------------------------------------------------------

REAL_SESSION_ID = "81dc8a2f-2bc6-4241-81bb-9dea09f45a68"
REAL_PROJECTS_DIR = Path.home() / ".claude" / "projects"
REAL_TRANSCRIPT = (
    REAL_PROJECTS_DIR
    / "-Users-mg-Downloads-claude-src"
    / f"{REAL_SESSION_ID}.jsonl"
)
REAL_SUBAGENTS_DIR = (
    REAL_PROJECTS_DIR
    / "-Users-mg-Downloads-claude-src"
    / REAL_SESSION_ID
    / "subagents"
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic data
# ---------------------------------------------------------------------------


def _user_entry(content="hello", uuid="u1"):
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": None,
        "timestamp": "2026-01-01T00:00:00.000Z",
        "message": {"content": content},
    }


def _assistant_entry(
    content_blocks=None,
    usage=None,
    stop_reason="end_turn",
    uuid="a1",
    model="claude-opus-4-6",
):
    if content_blocks is None:
        content_blocks = [{"type": "text", "text": "response"}]
    if usage is None:
        usage = {
            "input_tokens": 100,
            "cache_creation_input_tokens": 5000,
            "cache_read_input_tokens": 0,
            "output_tokens": 50,
        }
    return {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": "u1",
        "timestamp": "2026-01-01T00:00:01.000Z",
        "message": {
            "model": model,
            "stop_reason": stop_reason,
            "content": content_blocks,
            "usage": usage,
        },
    }


def _write_transcript(path: Path, entries: list[dict]):
    """Write a synthetic transcript JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def _write_subagent(subagents_dir: Path, agent_id: str, meta: dict, entries: list[dict]):
    """Write a synthetic subagent meta.json + JSONL."""
    subagents_dir.mkdir(parents=True, exist_ok=True)
    meta_path = subagents_dir / f"agent-{agent_id}.meta.json"
    meta_path.write_text(json.dumps(meta))
    jsonl_path = subagents_dir / f"agent-{agent_id}.jsonl"
    with jsonl_path.open("w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def _write_hook_events(path: Path, events: list[dict]):
    """Write synthetic hook events JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")


# ---------------------------------------------------------------------------
# Test: find_session_paths
# ---------------------------------------------------------------------------


class TestFindSessionPaths:
    def test_discovers_all_paths(self, tmp_path):
        """Discovers transcript, hook events, tool-results, and subagents."""
        session_id = "test-session-001"
        projects_dir = tmp_path / "projects"
        trace_dir = tmp_path / "trace"

        # Create transcript
        project = projects_dir / "myproject"
        project.mkdir(parents=True)
        transcript = project / f"{session_id}.jsonl"
        transcript.write_text("{}\n")

        # Create session directory with tool-results and subagents
        session_dir = project / session_id
        session_dir.mkdir()
        (session_dir / "tool-results").mkdir()
        (session_dir / "subagents").mkdir()

        # Create hook events
        trace_dir.mkdir(parents=True)
        hook = trace_dir / f"{session_id}.jsonl"
        hook.write_text("{}\n")

        result = find_session_paths(session_id, projects_dir, trace_dir)

        assert result["transcript"] == transcript
        assert result["hook_events"] == hook
        assert result["tool_results"] == session_dir / "tool-results"
        assert result["subagents"] == session_dir / "subagents"

    def test_handles_missing_session(self, tmp_path):
        """Returns None for all paths when session doesn't exist."""
        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()
        trace_dir = tmp_path / "trace"
        trace_dir.mkdir()

        result = find_session_paths("nonexistent", projects_dir, trace_dir)

        assert result["transcript"] is None
        assert result["hook_events"] is None
        assert result["tool_results"] is None
        assert result["subagents"] is None

    def test_transcript_only(self, tmp_path):
        """Finds transcript even without hook events or session dir."""
        session_id = "only-transcript"
        projects_dir = tmp_path / "projects"
        trace_dir = tmp_path / "trace"
        trace_dir.mkdir()

        project = projects_dir / "myproject"
        project.mkdir(parents=True)
        transcript = project / f"{session_id}.jsonl"
        transcript.write_text("{}\n")

        result = find_session_paths(session_id, projects_dir, trace_dir)

        assert result["transcript"] == transcript
        assert result["hook_events"] is None
        assert result["tool_results"] is None
        assert result["subagents"] is None


# ---------------------------------------------------------------------------
# Test: reconcile with transcript only
# ---------------------------------------------------------------------------


class TestReconcileTranscriptOnly:
    def test_basic_reconcile(self, tmp_path):
        """Works with just Source A (transcript)."""
        session_id = "basic-session"
        projects_dir = tmp_path / "projects"
        trace_dir = tmp_path / "trace"
        trace_dir.mkdir()

        project = projects_dir / "myproject"
        project.mkdir(parents=True)
        transcript = project / f"{session_id}.jsonl"

        entries = [
            _user_entry("What is 2+2?"),
            _assistant_entry(
                [{"type": "text", "text": "2+2 = 4"}],
                {
                    "input_tokens": 100,
                    "cache_creation_input_tokens": 5000,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 50,
                },
            ),
        ]
        _write_transcript(transcript, entries)

        blocks, churn, subagents = reconcile(session_id, projects_dir, trace_dir)

        assert len(blocks) > 0
        assert len(churn) == 1
        assert subagents == []

        # Churn entry should have correct fields
        c = churn[0]
        assert c["turn"] == 0
        assert c["cache_creation"] == 5000
        assert c["input"] == 100
        assert c["output"] == 50

    def test_missing_transcript_raises(self, tmp_path):
        """Raises FileNotFoundError when no transcript exists."""
        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()
        trace_dir = tmp_path / "trace"
        trace_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="No transcript found"):
            reconcile("nonexistent-session", projects_dir, trace_dir)


# ---------------------------------------------------------------------------
# Test: reconcile with offloads (Source A + D)
# ---------------------------------------------------------------------------


class TestReconcileWithOffloads:
    def test_offloads_add_spilled_tokens(self, tmp_path):
        """Source A + D: blocks gain spilled_tokens from offloaded files."""
        session_id = "offload-session"
        projects_dir = tmp_path / "projects"
        trace_dir = tmp_path / "trace"
        trace_dir.mkdir()

        project = projects_dir / "myproject"
        project.mkdir(parents=True)

        # Create session dir with tool-results
        session_dir = project / session_id
        tool_results = session_dir / "tool-results"
        tool_results.mkdir(parents=True)

        # Create offload file
        file_id = "toolu_offload123"
        offload = tool_results / f"{file_id}.txt"
        offload.write_text("x" * 200_000)  # 200KB -> ~50K tokens

        # Create transcript with a tool_use + tool_result referencing the offload
        entries = [
            _user_entry("Run the command"),
            _assistant_entry(
                [
                    {
                        "type": "tool_use",
                        "id": file_id,
                        "name": "Bash",
                        "input": {"command": "ls -la"},
                    }
                ],
                {
                    "input_tokens": 100,
                    "cache_creation_input_tokens": 5000,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 50,
                },
                stop_reason="tool_use",
            ),
            {
                "type": "user",
                "uuid": "u2",
                "parentUuid": "a1",
                "timestamp": "2026-01-01T00:00:02.000Z",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": file_id,
                            "content": (
                                f"<persisted-output>\n"
                                f"Output too large. Full output saved to: "
                                f"{tool_results}/{file_id}.txt\n"
                                f"Preview: some content\n"
                                f"</persisted-output>"
                            ),
                        }
                    ]
                },
            },
            _assistant_entry(
                [{"type": "text", "text": "Done!"}],
                {
                    "input_tokens": 200,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 5000,
                    "output_tokens": 30,
                },
                uuid="a2",
            ),
        ]
        transcript = project / f"{session_id}.jsonl"
        _write_transcript(transcript, entries)

        blocks, churn, subagents = reconcile(session_id, projects_dir, trace_dir)

        # Find the tool_result block
        tr_blocks = [b for b in blocks if b.get("type") == "tool_result"]
        assert len(tr_blocks) >= 1

        # At least one should have spilled_tokens
        spilled = [b for b in tr_blocks if b.get("spilled_tokens")]
        assert len(spilled) >= 1
        assert spilled[0]["spilled_tokens"] == 50_000  # 200000 // 4


# ---------------------------------------------------------------------------
# Test: reconcile with subagents (Source A + E)
# ---------------------------------------------------------------------------


class TestReconcileWithSubagents:
    def test_subagents_append_collapsed_blocks(self, tmp_path):
        """Source A + E: subagent parent_blocks appended to blocks list."""
        session_id = "subagent-session"
        projects_dir = tmp_path / "projects"
        trace_dir = tmp_path / "trace"
        trace_dir.mkdir()

        project = projects_dir / "myproject"
        project.mkdir(parents=True)

        # Create transcript
        entries = [
            _user_entry("Build the feature"),
            _assistant_entry(
                [{"type": "text", "text": "Starting subagent..."}],
                {
                    "input_tokens": 100,
                    "cache_creation_input_tokens": 5000,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 50,
                },
            ),
        ]
        transcript = project / f"{session_id}.jsonl"
        _write_transcript(transcript, entries)

        # Create subagent data
        subagents_dir = project / session_id / "subagents"
        _write_subagent(
            subagents_dir,
            "agent001",
            {"agentType": "general-purpose", "description": "Implement parser"},
            [
                _user_entry("implement it"),
                _assistant_entry(
                    [{"type": "text", "text": "done"}],
                    {
                        "input_tokens": 200,
                        "cache_creation_input_tokens": 3000,
                        "cache_read_input_tokens": 1000,
                        "output_tokens": 100,
                    },
                ),
            ],
        )

        blocks, churn, subagents = reconcile(session_id, projects_dir, trace_dir)

        # Subagent summary returned
        assert len(subagents) == 1
        assert subagents[0]["agent_id"] == "agent001"
        assert subagents[0]["total_cache_read"] == 1000

        # Subagent block appended to blocks
        sa_blocks = [b for b in blocks if "subagent" in b.get("id", "")]
        assert len(sa_blocks) == 1
        assert sa_blocks[0]["id"] == "subagent-agent001"
        assert sa_blocks[0]["type"] == "tool_result"


# ---------------------------------------------------------------------------
# Test: overlay_hook_events
# ---------------------------------------------------------------------------


class TestOverlayHookEvents:
    def test_annotates_failures(self, tmp_path):
        """PostToolUseFailure events annotate matching blocks."""
        hook_path = tmp_path / "hooks.jsonl"
        tool_use_id = "toolu_fail123"
        events = [
            {
                "event": "post_tool_use_failure",
                "session_id": "test",
                "timestamp": "2026-01-01T00:00:00Z",
                "tool_name": "Bash",
                "input_payload_chars": 100,
                "error_length": 500,
                "tool_use_id": tool_use_id,
            }
        ]
        _write_hook_events(hook_path, events)

        blocks = [
            {
                "id": f"t0-tool_result-{tool_use_id}",
                "type": "tool_result",
                "label": "Bash -> result",
                "tokens": 10,
            }
        ]

        result = overlay_hook_events(blocks, hook_path)

        assert result[0].get("failed") is True
        assert result[0]["label"].startswith("\u26a0")

    def test_non_matching_blocks_unchanged(self, tmp_path):
        """Blocks without matching failure events are unchanged."""
        hook_path = tmp_path / "hooks.jsonl"
        events = [
            {
                "event": "post_tool_use_failure",
                "session_id": "test",
                "timestamp": "2026-01-01T00:00:00Z",
                "tool_name": "Bash",
                "input_payload_chars": 100,
                "error_length": 500,
                "tool_use_id": "toolu_other",
            }
        ]
        _write_hook_events(hook_path, events)

        blocks = [
            {
                "id": "t0-tool_result-toolu_nope",
                "type": "tool_result",
                "label": "Bash -> result",
                "tokens": 10,
            }
        ]

        result = overlay_hook_events(blocks, hook_path)

        assert "failed" not in result[0]


# ---------------------------------------------------------------------------
# Test: write_output
# ---------------------------------------------------------------------------


class TestWriteOutput:
    def test_writes_valid_json_files(self, tmp_path):
        """Writes blocks.json and churn.json with valid JSON."""
        blocks = [
            {"id": "sys", "type": "system", "tokens": 1000},
            {"id": "t0-user-0", "type": "user", "tokens": 50},
        ]
        churn = [
            {"turn": 0, "cache_read": 0, "cache_creation": 5000, "input": 100, "output": 50}
        ]

        blocks_path, churn_path = write_output(blocks, churn, tmp_path / "output")

        assert blocks_path.exists()
        assert churn_path.exists()

        loaded_blocks = json.loads(blocks_path.read_text())
        loaded_churn = json.loads(churn_path.read_text())

        assert loaded_blocks == blocks
        assert loaded_churn == churn

    def test_creates_output_dir(self, tmp_path):
        """Creates output directory if it doesn't exist."""
        output_dir = tmp_path / "deep" / "nested" / "dir"
        write_output([], [], output_dir)
        assert output_dir.exists()


# ---------------------------------------------------------------------------
# Test: real session data
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not REAL_TRANSCRIPT.exists(),
    reason="Real session data not available",
)
class TestRealSession:
    """Integration tests against the real 81dc8a2f session data.

    Expected: ~754 parent blocks + 17 subagent blocks = ~771 total,
    314 churn entries, 17 subagents, ~94M parent + ~11.7M sub = ~106M combined churn.
    """

    def test_block_count(self):
        blocks, churn, subagents = reconcile(
            REAL_SESSION_ID, REAL_PROJECTS_DIR
        )
        # Individual blocks from transcript should be > 700
        parent_block_count = len(blocks) - len(subagents)
        assert parent_block_count > 700, (
            f"Expected >700 parent blocks, got {parent_block_count}"
        )

    def test_churn_entries(self):
        blocks, churn, subagents = reconcile(
            REAL_SESSION_ID, REAL_PROJECTS_DIR
        )
        assert len(churn) == 314, f"Expected 314 churn entries, got {len(churn)}"

    def test_subagent_count(self):
        blocks, churn, subagents = reconcile(
            REAL_SESSION_ID, REAL_PROJECTS_DIR
        )
        assert len(subagents) == 17, f"Expected 17 subagents, got {len(subagents)}"

    def test_subagent_blocks_appended(self):
        blocks, churn, subagents = reconcile(
            REAL_SESSION_ID, REAL_PROJECTS_DIR
        )
        sa_blocks = [b for b in blocks if "subagent" in b.get("id", "")]
        assert len(sa_blocks) == 17, (
            f"Expected 17 subagent blocks in timeline, got {len(sa_blocks)}"
        )

    def test_churn_totals(self):
        blocks, churn, subagents = reconcile(
            REAL_SESSION_ID, REAL_PROJECTS_DIR
        )
        # Parent churn
        total_cr = sum(c["cache_read"] for c in churn)
        assert total_cr >= 80_000_000, (
            f"Expected parent cache_read >= 80M, got {total_cr:,}"
        )
        # Subagent churn
        sub_churn = sum(s["total_cache_read"] for s in subagents)
        assert sub_churn >= 8_000_000, (
            f"Expected subagent cache_read >= 8M, got {sub_churn:,}"
        )
        # Combined
        combined = total_cr + sub_churn
        assert combined >= 90_000_000, (
            f"Expected combined >= 90M, got {combined:,}"
        )

    def test_spilled_tokens(self):
        blocks, churn, subagents = reconcile(
            REAL_SESSION_ID, REAL_PROJECTS_DIR
        )
        spilled = [b for b in blocks if b.get("spilled_tokens")]
        assert len(spilled) > 0, "Expected some blocks with spilled_tokens"

    def test_total_block_count(self):
        blocks, churn, subagents = reconcile(
            REAL_SESSION_ID, REAL_PROJECTS_DIR
        )
        # Total should be parent blocks + subagent blocks
        assert len(blocks) > 750, (
            f"Expected >750 total blocks, got {len(blocks)}"
        )

    def test_write_output_roundtrip(self, tmp_path):
        blocks, churn, subagents = reconcile(
            REAL_SESSION_ID, REAL_PROJECTS_DIR
        )
        blocks_path, churn_path = write_output(blocks, churn, tmp_path)

        loaded_blocks = json.loads(blocks_path.read_text())
        loaded_churn = json.loads(churn_path.read_text())

        assert len(loaded_blocks) == len(blocks)
        assert len(loaded_churn) == len(churn)
