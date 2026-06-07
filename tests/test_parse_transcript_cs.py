"""Tests for ccscope transcript parser (Context Scope format)."""

import json

from context_tracker.ccscope.parse_transcript import parse_transcript_to_blocks
from context_tracker.ccscope.tokens import char_count_of_block, estimate_tokens

# ---------------------------------------------------------------------------
# Helpers to build synthetic transcript JSONL
# ---------------------------------------------------------------------------


def _user_entry(content, uuid="u1", parent=None, is_meta=False):
    """Build a user entry dict."""
    entry = {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent,
        "timestamp": "2026-01-01T00:00:00.000Z",
        "message": {"content": content},
    }
    if is_meta:
        entry["isMeta"] = True
    return entry


def _assistant_entry(
    content_blocks,
    usage,
    stop_reason="end_turn",
    uuid="a1",
    parent="u1",
    model="claude-opus-4-6",
):
    """Build a completed assistant entry dict."""
    return {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": parent,
        "timestamp": "2026-01-01T00:00:01.000Z",
        "message": {
            "model": model,
            "stop_reason": stop_reason,
            "content": content_blocks,
            "usage": usage,
        },
    }


def _streaming_assistant_entry(uuid="s1", parent="u1"):
    """Build a streaming (incomplete) assistant entry."""
    return {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": parent,
        "timestamp": "2026-01-01T00:00:01.000Z",
        "message": {
            "model": "claude-opus-4-6",
            "stop_reason": None,
            "content": [{"type": "text", "text": "partial..."}],
            "usage": {
                "input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 32,
            },
        },
    }


def _system_entry():
    """Build a system metadata entry (should be skipped)."""
    return {"type": "system", "uuid": "sys1", "message": {"type": "turn_duration"}}


def _write_jsonl(entries, path):
    """Write entries as JSONL file."""
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Token estimation tests
# ---------------------------------------------------------------------------


class TestTokenEstimation:
    def test_estimate_tokens_basic(self):
        assert estimate_tokens("hello world") == 2  # 11 chars / 4 = 2

    def test_estimate_tokens_empty(self):
        assert estimate_tokens("") == 0

    def test_estimate_tokens_short(self):
        assert estimate_tokens("ab") == 1  # min 1

    def test_char_count_text_block(self):
        block = {"type": "text", "text": "Hello world"}
        assert char_count_of_block(block) == 11

    def test_char_count_thinking_block(self):
        block = {"type": "thinking", "thinking": "Let me think..."}
        assert char_count_of_block(block) == 15

    def test_char_count_tool_use_block(self):
        block = {
            "type": "tool_use",
            "name": "Read",
            "id": "toolu_01",
            "input": {"file_path": "/tmp/test.py"},
        }
        chars = char_count_of_block(block)
        assert chars > 0

    def test_char_count_tool_result_string(self):
        block = {"type": "tool_result", "content": "file contents here"}
        assert char_count_of_block(block) == 18

    def test_char_count_tool_result_list(self):
        block = {
            "type": "tool_result",
            "content": [{"type": "text", "text": "result line 1"}, {"type": "text", "text": "result line 2"}],
        }
        assert char_count_of_block(block) == 26


# ---------------------------------------------------------------------------
# Minimal synthetic transcript tests
# ---------------------------------------------------------------------------


class TestParseTranscriptMinimal:
    """Test with minimal synthetic data."""

    def _make_simple_transcript(self, tmp_path):
        """Create a minimal transcript with one user message and one assistant reply."""
        entries = [
            _user_entry("Hello, help me with code", uuid="u1"),
            _assistant_entry(
                content_blocks=[
                    {"type": "text", "text": "Sure, I can help you with that!"},
                ],
                usage={
                    "input_tokens": 5,
                    "cache_creation_input_tokens": 10000,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 20,
                },
                uuid="a1",
                parent="u1",
            ),
        ]
        path = tmp_path / "test.jsonl"
        _write_jsonl(entries, path)
        return path

    def test_returns_blocks_and_churn(self, tmp_path):
        path = self._make_simple_transcript(tmp_path)
        blocks, churn = parse_transcript_to_blocks(path)
        assert isinstance(blocks, list)
        assert isinstance(churn, list)

    def test_churn_count(self, tmp_path):
        path = self._make_simple_transcript(tmp_path)
        blocks, churn = parse_transcript_to_blocks(path)
        assert len(churn) == 1

    def test_churn_values_exact(self, tmp_path):
        path = self._make_simple_transcript(tmp_path)
        blocks, churn = parse_transcript_to_blocks(path)
        c = churn[0]
        assert c["turn"] == 0
        assert c["cache_read"] == 0
        assert c["cache_creation"] == 10000
        assert c["input"] == 5
        assert c["output"] == 20

    def test_has_prefix_blocks(self, tmp_path):
        path = self._make_simple_transcript(tmp_path)
        blocks, churn = parse_transcript_to_blocks(path)
        pinned = [b for b in blocks if b.get("cached")]
        assert len(pinned) >= 1  # At least system prompt

    def test_has_user_block(self, tmp_path):
        path = self._make_simple_transcript(tmp_path)
        blocks, churn = parse_transcript_to_blocks(path)
        user_blocks = [b for b in blocks if b["type"] == "user"]
        assert len(user_blocks) >= 1

    def test_has_assistant_block(self, tmp_path):
        path = self._make_simple_transcript(tmp_path)
        blocks, churn = parse_transcript_to_blocks(path)
        asst_blocks = [b for b in blocks if b["type"] == "assistant"]
        assert len(asst_blocks) >= 1

    def test_block_ids_unique(self, tmp_path):
        path = self._make_simple_transcript(tmp_path)
        blocks, churn = parse_transcript_to_blocks(path)
        ids = [b["id"] for b in blocks]
        assert len(ids) == len(set(ids)), f"Duplicate block IDs: {ids}"

    def test_block_fields(self, tmp_path):
        """Each block should have required fields."""
        path = self._make_simple_transcript(tmp_path)
        blocks, churn = parse_transcript_to_blocks(path)
        required_fields = {"id", "type", "label", "tokens", "enter", "exit", "cached", "ref", "content"}
        for b in blocks:
            missing = required_fields - set(b.keys())
            assert not missing, f"Block {b.get('id')} missing fields: {missing}"


class TestParseTranscriptStreaming:
    """Test streaming dedup — incomplete assistant entries are skipped."""

    def test_streaming_entries_skipped(self, tmp_path):
        entries = [
            _user_entry("Hello", uuid="u1"),
            _streaming_assistant_entry(uuid="s1", parent="u1"),
            _streaming_assistant_entry(uuid="s2", parent="u1"),
            _assistant_entry(
                content_blocks=[{"type": "text", "text": "Final answer"}],
                usage={
                    "input_tokens": 2,
                    "cache_creation_input_tokens": 8000,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 15,
                },
                uuid="a1",
                parent="u1",
            ),
        ]
        path = tmp_path / "test.jsonl"
        _write_jsonl(entries, path)
        blocks, churn = parse_transcript_to_blocks(path)
        # Only one API call should be recorded
        assert len(churn) == 1

    def test_synthetic_model_skipped(self, tmp_path):
        entries = [
            _user_entry("Hello", uuid="u1"),
            _assistant_entry(
                content_blocks=[{"type": "text", "text": "Synthetic reply"}],
                usage={
                    "input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 10,
                },
                uuid="synth1",
                parent="u1",
                model="synthetic",
            ),
            _assistant_entry(
                content_blocks=[{"type": "text", "text": "Real reply"}],
                usage={
                    "input_tokens": 2,
                    "cache_creation_input_tokens": 8000,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 15,
                },
                uuid="a1",
                parent="u1",
            ),
        ]
        path = tmp_path / "test.jsonl"
        _write_jsonl(entries, path)
        blocks, churn = parse_transcript_to_blocks(path)
        assert len(churn) == 1


class TestParseTranscriptSystemSkipped:
    """System metadata entries should be ignored."""

    def test_system_entries_skipped(self, tmp_path):
        entries = [
            _system_entry(),
            _user_entry("Hello", uuid="u1"),
            _system_entry(),
            _assistant_entry(
                content_blocks=[{"type": "text", "text": "Hi"}],
                usage={
                    "input_tokens": 2,
                    "cache_creation_input_tokens": 5000,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 5,
                },
                uuid="a1",
                parent="u1",
            ),
        ]
        path = tmp_path / "test.jsonl"
        _write_jsonl(entries, path)
        blocks, churn = parse_transcript_to_blocks(path)
        assert len(churn) == 1
        # No system-metadata blocks
        sys_blocks = [b for b in blocks if b["type"] == "system_metadata"]
        assert len(sys_blocks) == 0


class TestParseTranscriptToolBlocks:
    """Test tool_use and tool_result create proper blocks."""

    def _make_tool_transcript(self, tmp_path):
        entries = [
            _user_entry("Read server.py", uuid="u1"),
            _assistant_entry(
                content_blocks=[
                    {"type": "thinking", "thinking": "I should read the file"},
                    {
                        "type": "tool_use",
                        "id": "toolu_01ABC",
                        "name": "Read",
                        "input": {"file_path": "/tmp/server.py"},
                    },
                ],
                usage={
                    "input_tokens": 2,
                    "cache_creation_input_tokens": 10000,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 50,
                },
                stop_reason="tool_use",
                uuid="a1",
                parent="u1",
            ),
            _user_entry(
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01ABC",
                        "content": "def main():\n    pass\n",
                    }
                ],
                uuid="u2",
                parent="a1",
            ),
            _assistant_entry(
                content_blocks=[{"type": "text", "text": "The file contains a main function."}],
                usage={
                    "input_tokens": 2,
                    "cache_creation_input_tokens": 200,
                    "cache_read_input_tokens": 10000,
                    "output_tokens": 30,
                },
                uuid="a2",
                parent="u2",
            ),
        ]
        path = tmp_path / "test.jsonl"
        _write_jsonl(entries, path)
        return path

    def test_tool_use_block_created(self, tmp_path):
        path = self._make_tool_transcript(tmp_path)
        blocks, churn = parse_transcript_to_blocks(path)
        tc_blocks = [b for b in blocks if b["type"] == "tool_call"]
        assert len(tc_blocks) == 1
        assert "server.py" in tc_blocks[0]["label"]

    def test_tool_result_block_created(self, tmp_path):
        path = self._make_tool_transcript(tmp_path)
        blocks, churn = parse_transcript_to_blocks(path)
        tr_blocks = [b for b in blocks if b["type"] == "tool_result"]
        assert len(tr_blocks) == 1

    def test_thinking_block_created(self, tmp_path):
        path = self._make_tool_transcript(tmp_path)
        blocks, churn = parse_transcript_to_blocks(path)
        think_blocks = [b for b in blocks if b["type"] == "thinking"]
        assert len(think_blocks) == 1

    def test_two_churn_entries(self, tmp_path):
        path = self._make_tool_transcript(tmp_path)
        blocks, churn = parse_transcript_to_blocks(path)
        assert len(churn) == 2

    def test_churn_values_accurate(self, tmp_path):
        path = self._make_tool_transcript(tmp_path)
        blocks, churn = parse_transcript_to_blocks(path)
        # Second call should have cache_read = 10000
        assert churn[1]["cache_read"] == 10000
        assert churn[1]["cache_creation"] == 200


class TestToolLabels:
    """Test that tool labels are descriptive."""

    def _parse_with_tool(self, tmp_path, tool_name, tool_input):
        entries = [
            _user_entry("do it", uuid="u1"),
            _assistant_entry(
                content_blocks=[
                    {
                        "type": "tool_use",
                        "id": "toolu_01X",
                        "name": tool_name,
                        "input": tool_input,
                    },
                ],
                usage={
                    "input_tokens": 2,
                    "cache_creation_input_tokens": 8000,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 30,
                },
                stop_reason="tool_use",
                uuid="a1",
                parent="u1",
            ),
        ]
        path = tmp_path / "test.jsonl"
        _write_jsonl(entries, path)
        blocks, _ = parse_transcript_to_blocks(path)
        tc_blocks = [b for b in blocks if b["type"] == "tool_call"]
        assert len(tc_blocks) == 1
        return tc_blocks[0]["label"]

    def test_read_label(self, tmp_path):
        label = self._parse_with_tool(tmp_path, "Read", {"file_path": "/tmp/foo/bar.py"})
        assert "Read" in label
        assert "bar.py" in label

    def test_edit_label(self, tmp_path):
        label = self._parse_with_tool(
            tmp_path, "Edit", {"file_path": "/tmp/models.py", "old_string": "x", "new_string": "y"}
        )
        assert "Edit" in label
        assert "models.py" in label

    def test_bash_label(self, tmp_path):
        label = self._parse_with_tool(tmp_path, "Bash", {"command": "git status --short"})
        assert "Bash" in label
        assert "git" in label

    def test_grep_label(self, tmp_path):
        label = self._parse_with_tool(tmp_path, "Grep", {"pattern": "def main"})
        assert "Grep" in label
        assert "def main" in label

    def test_write_label(self, tmp_path):
        label = self._parse_with_tool(tmp_path, "Write", {"file_path": "/tmp/new_file.py", "content": "x"})
        assert "Write" in label
        assert "new_file.py" in label

    def test_unknown_tool_label(self, tmp_path):
        label = self._parse_with_tool(tmp_path, "CustomTool", {"data": "value"})
        assert "CustomTool" in label


class TestToolResultLabels:
    """Test that tool_result blocks get descriptive labels traced from tool_use."""

    def test_tool_result_label_has_arrow(self, tmp_path):
        entries = [
            _user_entry("read it", uuid="u1"),
            _assistant_entry(
                content_blocks=[
                    {
                        "type": "tool_use",
                        "id": "toolu_01X",
                        "name": "Read",
                        "input": {"file_path": "/tmp/server.py"},
                    },
                ],
                usage={
                    "input_tokens": 2,
                    "cache_creation_input_tokens": 8000,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 30,
                },
                stop_reason="tool_use",
                uuid="a1",
                parent="u1",
            ),
            _user_entry(
                [{"type": "tool_result", "tool_use_id": "toolu_01X", "content": "file data"}],
                uuid="u2",
                parent="a1",
            ),
            _assistant_entry(
                content_blocks=[{"type": "text", "text": "Done"}],
                usage={
                    "input_tokens": 2,
                    "cache_creation_input_tokens": 200,
                    "cache_read_input_tokens": 8000,
                    "output_tokens": 10,
                },
                uuid="a2",
                parent="u2",
            ),
        ]
        path = tmp_path / "test.jsonl"
        _write_jsonl(entries, path)
        blocks, _ = parse_transcript_to_blocks(path)
        tr_blocks = [b for b in blocks if b["type"] == "tool_result"]
        assert len(tr_blocks) == 1
        label = tr_blocks[0]["label"]
        # Should contain arrow notation
        assert "\u2192" in label or "->" in label
        assert "server.py" in label


class TestCompactionDetection:
    """Test compaction detection sets exit on pre-compaction blocks."""

    def test_compaction_sets_exit(self, tmp_path):
        entries = [
            _user_entry("Hello", uuid="u1"),
            # First call: large context
            _assistant_entry(
                content_blocks=[{"type": "text", "text": "First reply"}],
                usage={
                    "input_tokens": 2,
                    "cache_creation_input_tokens": 100000,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 50,
                },
                uuid="a1",
                parent="u1",
            ),
            _user_entry("Continue", uuid="u2", parent="a1"),
            # Second call: much smaller context (compaction)
            _assistant_entry(
                content_blocks=[{"type": "text", "text": "After compaction"}],
                usage={
                    "input_tokens": 2,
                    "cache_creation_input_tokens": 20000,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 30,
                },
                uuid="a2",
                parent="u2",
            ),
        ]
        path = tmp_path / "test.jsonl"
        _write_jsonl(entries, path)
        blocks, churn = parse_transcript_to_blocks(path)
        # Pre-compaction non-pinned blocks should have exit set
        pre_compaction = [b for b in blocks if b["enter"] == 0 and not b.get("cached")]
        for b in pre_compaction:
            assert b["exit"] is not None, f"Block {b['id']} should have exit set after compaction"


class TestContentTruncation:
    """Test that content field is truncated to 500 chars."""

    def test_content_max_500(self, tmp_path):
        long_text = "x" * 2000
        entries = [
            _user_entry(long_text, uuid="u1"),
            _assistant_entry(
                content_blocks=[{"type": "text", "text": "ok"}],
                usage={
                    "input_tokens": 2,
                    "cache_creation_input_tokens": 8000,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 5,
                },
                uuid="a1",
                parent="u1",
            ),
        ]
        path = tmp_path / "test.jsonl"
        _write_jsonl(entries, path)
        blocks, _ = parse_transcript_to_blocks(path)
        for b in blocks:
            assert len(b["content"]) <= 500, f"Block {b['id']} content too long: {len(b['content'])}"


class TestBlockIdFormat:
    """Test block ID format: t{api_call_index}-{type}-{index}."""

    def test_id_format(self, tmp_path):
        entries = [
            _user_entry("Hello", uuid="u1"),
            _assistant_entry(
                content_blocks=[
                    {"type": "thinking", "thinking": "hmm"},
                    {"type": "text", "text": "Hi there!"},
                    {
                        "type": "tool_use",
                        "id": "toolu_01ABC",
                        "name": "Bash",
                        "input": {"command": "ls"},
                    },
                ],
                usage={
                    "input_tokens": 2,
                    "cache_creation_input_tokens": 8000,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 40,
                },
                stop_reason="tool_use",
                uuid="a1",
                parent="u1",
            ),
        ]
        path = tmp_path / "test.jsonl"
        _write_jsonl(entries, path)
        blocks, _ = parse_transcript_to_blocks(path)
        # Find non-pinned blocks
        non_pinned = [b for b in blocks if not b.get("cached")]
        ids = [b["id"] for b in non_pinned]
        # Should contain t0- prefix for first API call
        t0_ids = [i for i in ids if i.startswith("t0-")]
        assert len(t0_ids) > 0, f"Expected t0- prefixed IDs, got: {ids}"


class TestMultipleApiCalls:
    """Test a transcript with multiple API calls."""

    def test_multiple_calls(self, tmp_path):
        entries = [
            _user_entry("First question", uuid="u1"),
            _assistant_entry(
                content_blocks=[{"type": "text", "text": "First answer"}],
                usage={
                    "input_tokens": 2,
                    "cache_creation_input_tokens": 10000,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 20,
                },
                uuid="a1",
                parent="u1",
            ),
            _user_entry("Second question", uuid="u2", parent="a1"),
            _assistant_entry(
                content_blocks=[{"type": "text", "text": "Second answer"}],
                usage={
                    "input_tokens": 2,
                    "cache_creation_input_tokens": 500,
                    "cache_read_input_tokens": 10000,
                    "output_tokens": 25,
                },
                uuid="a2",
                parent="u2",
            ),
            _user_entry("Third question", uuid="u3", parent="a2"),
            _assistant_entry(
                content_blocks=[{"type": "text", "text": "Third answer"}],
                usage={
                    "input_tokens": 2,
                    "cache_creation_input_tokens": 300,
                    "cache_read_input_tokens": 10500,
                    "output_tokens": 15,
                },
                uuid="a3",
                parent="u3",
            ),
        ]
        path = tmp_path / "test.jsonl"
        _write_jsonl(entries, path)
        blocks, churn = parse_transcript_to_blocks(path)
        assert len(churn) == 3
        # Block enter values should be monotonically assigned
        enters = sorted(set(b["enter"] for b in blocks if not b.get("cached")))
        assert 0 in enters
        assert 1 in enters or 2 in enters  # At least some blocks from later calls


class TestIsMetaUserEntries:
    """Test that isMeta user entries are handled (they carry skill/CLAUDE.md content)."""

    def test_meta_user_entry_processed(self, tmp_path):
        entries = [
            _user_entry("Hello", uuid="u1"),
            _user_entry(
                [{"type": "text", "text": "CLAUDE.md content here with project rules..."}],
                uuid="u2",
                parent="u1",
                is_meta=True,
            ),
            _assistant_entry(
                content_blocks=[{"type": "text", "text": "Got it"}],
                usage={
                    "input_tokens": 2,
                    "cache_creation_input_tokens": 10000,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 10,
                },
                uuid="a1",
                parent="u2",
            ),
        ]
        path = tmp_path / "test.jsonl"
        _write_jsonl(entries, path)
        blocks, churn = parse_transcript_to_blocks(path)
        # Should have blocks for the meta user content
        assert len(blocks) > 0
        assert len(churn) == 1
