"""Tests for ccscope subagent transcript parsing."""

import json
import tempfile
from pathlib import Path

import pytest

from context_tracker.ccscope.subagents import parse_subagents


# ---------------------------------------------------------------------------
# Helpers to build synthetic subagent data
# ---------------------------------------------------------------------------

def _meta_json(agent_type: str = "general-purpose", description: str = "Test agent") -> str:
    return json.dumps({"agentType": agent_type, "description": description})


def _user_entry(content: str = "hello", agent_id: str = "abc123"):
    return {
        "type": "user",
        "agentId": agent_id,
        "message": {"role": "user", "content": content},
        "uuid": "u1",
        "timestamp": "2026-01-01T00:00:00.000Z",
    }


def _assistant_entry(
    usage: dict,
    content_blocks: list | None = None,
    stop_reason: str = "end_turn",
    agent_id: str = "abc123",
    model: str = "claude-opus-4-6",
):
    if content_blocks is None:
        content_blocks = [{"type": "text", "text": "response"}]
    return {
        "type": "assistant",
        "agentId": agent_id,
        "message": {
            "model": model,
            "role": "assistant",
            "stop_reason": stop_reason,
            "content": content_blocks,
            "usage": usage,
        },
        "uuid": "a1",
        "timestamp": "2026-01-01T00:00:01.000Z",
    }


def _streaming_assistant(agent_id: str = "abc123"):
    """An incomplete/streaming assistant entry — no stop_reason."""
    return {
        "type": "assistant",
        "agentId": agent_id,
        "message": {
            "model": "claude-opus-4-6",
            "role": "assistant",
            "stop_reason": None,
            "content": [{"type": "text", "text": "partial..."}],
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
        "uuid": "s1",
        "timestamp": "2026-01-01T00:00:01.000Z",
    }


def _write_subagent(tmp_dir: Path, agent_id: str, meta: str, entries: list[dict]):
    """Write a meta.json and JSONL for a single subagent."""
    meta_path = tmp_dir / f"agent-{agent_id}.meta.json"
    meta_path.write_text(meta)
    jsonl_path = tmp_dir / f"agent-{agent_id}.jsonl"
    with jsonl_path.open("w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Test: empty / nonexistent directory
# ---------------------------------------------------------------------------

class TestEmptyDirectory:
    def test_nonexistent_dir(self, tmp_path):
        result = parse_subagents(tmp_path / "nonexistent")
        assert result == []

    def test_empty_dir(self, tmp_path):
        result = parse_subagents(tmp_path)
        assert result == []

    def test_dir_with_no_meta_files(self, tmp_path):
        """JSONL without meta should still be handled gracefully."""
        (tmp_path / "agent-abc.jsonl").write_text("{}\n")
        result = parse_subagents(tmp_path)
        # Should parse it but with default metadata
        assert len(result) == 1
        assert result[0]["agent_id"] == "abc"


# ---------------------------------------------------------------------------
# Test: single subagent parsing
# ---------------------------------------------------------------------------

class TestSingleSubagent:
    def test_basic_parse(self, tmp_path):
        entries = [
            _user_entry("hello"),
            _assistant_entry({
                "input_tokens": 100,
                "cache_creation_input_tokens": 5000,
                "cache_read_input_tokens": 0,
                "output_tokens": 50,
            }),
        ]
        _write_subagent(tmp_path, "test1", _meta_json(), entries)
        results = parse_subagents(tmp_path)
        assert len(results) == 1
        r = results[0]
        assert r["agent_id"] == "test1"
        assert r["agent_type"] == "general-purpose"
        assert r["description"] == "Test agent"
        assert r["api_calls"] == 1

    def test_meta_fields(self, tmp_path):
        meta = _meta_json("code-review", "Review PR #42")
        entries = [
            _user_entry("review this"),
            _assistant_entry({
                "input_tokens": 50,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 20,
            }),
        ]
        _write_subagent(tmp_path, "rev1", meta, entries)
        r = parse_subagents(tmp_path)[0]
        assert r["agent_type"] == "code-review"
        assert r["description"] == "Review PR #42"

    def test_skips_streaming_entries(self, tmp_path):
        """Streaming/incomplete assistant entries should not count as API calls."""
        entries = [
            _user_entry("hello"),
            _streaming_assistant(),
            _assistant_entry({
                "input_tokens": 100,
                "cache_creation_input_tokens": 5000,
                "cache_read_input_tokens": 0,
                "output_tokens": 50,
            }),
        ]
        _write_subagent(tmp_path, "s1", _meta_json(), entries)
        r = parse_subagents(tmp_path)[0]
        assert r["api_calls"] == 1

    def test_skips_synthetic_model(self, tmp_path):
        entries = [
            _user_entry("hello"),
            _assistant_entry(
                {"input_tokens": 100, "cache_creation_input_tokens": 0,
                 "cache_read_input_tokens": 0, "output_tokens": 50},
                model="synthetic",
            ),
            _assistant_entry({
                "input_tokens": 200,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 30,
            }),
        ]
        _write_subagent(tmp_path, "syn1", _meta_json(), entries)
        r = parse_subagents(tmp_path)[0]
        assert r["api_calls"] == 1


# ---------------------------------------------------------------------------
# Test: peak_resident computation
# ---------------------------------------------------------------------------

class TestPeakResident:
    def test_peak_is_max_across_calls(self, tmp_path):
        entries = [
            _user_entry("1"),
            _assistant_entry({
                "input_tokens": 100,
                "cache_creation_input_tokens": 5000,
                "cache_read_input_tokens": 200,
                "output_tokens": 50,
            }, stop_reason="tool_use"),
            _user_entry("2"),
            _assistant_entry({
                "input_tokens": 300,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 8000,
                "output_tokens": 100,
            }),
        ]
        _write_subagent(tmp_path, "pk1", _meta_json(), entries)
        r = parse_subagents(tmp_path)[0]
        # Call 1: 100 + 5000 + 200 = 5300
        # Call 2: 300 + 0 + 8000 = 8300
        assert r["peak_resident"] == 8300

    def test_peak_with_single_call(self, tmp_path):
        entries = [
            _user_entry("x"),
            _assistant_entry({
                "input_tokens": 500,
                "cache_creation_input_tokens": 10000,
                "cache_read_input_tokens": 3000,
                "output_tokens": 200,
            }),
        ]
        _write_subagent(tmp_path, "pk2", _meta_json(), entries)
        r = parse_subagents(tmp_path)[0]
        assert r["peak_resident"] == 13500  # 500 + 10000 + 3000


# ---------------------------------------------------------------------------
# Test: total_cache_read computation
# ---------------------------------------------------------------------------

class TestTotalCacheRead:
    def test_sums_cache_read(self, tmp_path):
        entries = [
            _user_entry("1"),
            _assistant_entry({
                "input_tokens": 100,
                "cache_creation_input_tokens": 5000,
                "cache_read_input_tokens": 1000,
                "output_tokens": 50,
            }, stop_reason="tool_use"),
            _user_entry("2"),
            _assistant_entry({
                "input_tokens": 100,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 6000,
                "output_tokens": 50,
            }),
        ]
        _write_subagent(tmp_path, "cr1", _meta_json(), entries)
        r = parse_subagents(tmp_path)[0]
        assert r["total_cache_read"] == 7000

    def test_zero_cache_read(self, tmp_path):
        entries = [
            _user_entry("x"),
            _assistant_entry({
                "input_tokens": 100,
                "cache_creation_input_tokens": 5000,
                "cache_read_input_tokens": 0,
                "output_tokens": 50,
            }),
        ]
        _write_subagent(tmp_path, "cr2", _meta_json(), entries)
        r = parse_subagents(tmp_path)[0]
        assert r["total_cache_read"] == 0


# ---------------------------------------------------------------------------
# Test: parent_block format (Context Scope contract)
# ---------------------------------------------------------------------------

class TestParentBlock:
    def test_parent_block_shape(self, tmp_path):
        entries = [
            _user_entry("x"),
            _assistant_entry({
                "input_tokens": 100,
                "cache_creation_input_tokens": 5000,
                "cache_read_input_tokens": 2000,
                "output_tokens": 50,
            }, stop_reason="tool_use"),
            _user_entry("y"),
            _assistant_entry({
                "input_tokens": 200,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 6000,
                "output_tokens": 100,
            }),
        ]
        meta = _meta_json("general-purpose", "Build the feature")
        _write_subagent(tmp_path, "blk1", meta, entries)
        r = parse_subagents(tmp_path)[0]
        pb = r["parent_block"]

        # Required Context Scope fields
        assert pb["id"] == "subagent-blk1"
        assert pb["type"] == "tool_result"
        assert "general-purpose" in pb["label"]
        assert "Build the feature" in pb["label"]
        assert pb["tokens"] == r["peak_resident"]
        assert pb["enter"] == 0  # placeholder
        assert pb["exit"] is None  # placeholder
        assert pb["cached"] is False
        assert pb["ref"] is True
        assert "2 calls" in pb["content"]
        assert "Subagent" in pb["content"]

    def test_label_truncates_description(self, tmp_path):
        long_desc = "A" * 100
        entries = [
            _user_entry("x"),
            _assistant_entry({
                "input_tokens": 100,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 50,
            }),
        ]
        _write_subagent(tmp_path, "trunc", _meta_json("general-purpose", long_desc), entries)
        r = parse_subagents(tmp_path)[0]
        # Description in label should be truncated to 60 chars
        label = r["parent_block"]["label"]
        assert len(label) <= len("general-purpose: ") + 60


# ---------------------------------------------------------------------------
# Test: churn entries
# ---------------------------------------------------------------------------

class TestChurn:
    def test_churn_entries(self, tmp_path):
        entries = [
            _user_entry("1"),
            _assistant_entry({
                "input_tokens": 100,
                "cache_creation_input_tokens": 5000,
                "cache_read_input_tokens": 1000,
                "output_tokens": 50,
            }, stop_reason="tool_use"),
            _user_entry("2"),
            _assistant_entry({
                "input_tokens": 200,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 6000,
                "output_tokens": 100,
            }),
        ]
        _write_subagent(tmp_path, "ch1", _meta_json(), entries)
        r = parse_subagents(tmp_path)[0]
        assert len(r["churn"]) == 2

        c0 = r["churn"][0]
        assert c0["turn"] == 0
        assert c0["cache_read"] == 1000
        assert c0["cache_creation"] == 5000
        assert c0["input"] == 100
        assert c0["output"] == 50

        c1 = r["churn"][1]
        assert c1["turn"] == 1
        assert c1["cache_read"] == 6000
        assert c1["cache_creation"] == 0
        assert c1["input"] == 200
        assert c1["output"] == 100


# ---------------------------------------------------------------------------
# Test: multiple subagents
# ---------------------------------------------------------------------------

class TestMultipleSubagents:
    def test_parses_all_subagents(self, tmp_path):
        for i in range(3):
            entries = [
                _user_entry(f"hello {i}"),
                _assistant_entry({
                    "input_tokens": 100 * (i + 1),
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 1000 * (i + 1),
                    "output_tokens": 50,
                }),
            ]
            _write_subagent(tmp_path, f"multi{i}", _meta_json(), entries)
        results = parse_subagents(tmp_path)
        assert len(results) == 3
        ids = {r["agent_id"] for r in results}
        assert ids == {"multi0", "multi1", "multi2"}

    def test_empty_transcript_jsonl(self, tmp_path):
        """Subagent with an empty JSONL should have 0 calls."""
        _write_subagent(tmp_path, "empty1", _meta_json(), [])
        r = parse_subagents(tmp_path)[0]
        assert r["api_calls"] == 0
        assert r["peak_resident"] == 0
        assert r["total_cache_read"] == 0
        assert r["churn"] == []


# ---------------------------------------------------------------------------
# Test: real data (17 subagents)
# ---------------------------------------------------------------------------

REAL_SUBAGENTS_DIR = Path.home() / ".claude/projects/-Users-mg-Downloads-claude-src/81dc8a2f-2bc6-4241-81bb-9dea09f45a68/subagents"


@pytest.mark.skipif(
    not REAL_SUBAGENTS_DIR.exists(),
    reason="Real subagent data not available",
)
class TestRealData:
    def test_parses_17_subagents(self):
        results = parse_subagents(REAL_SUBAGENTS_DIR)
        assert len(results) == 17

    def test_known_subagent_abd91(self):
        """agent-abd91345670ff042f: 72 calls, ~3.6M cache_read, peak ~73K."""
        results = parse_subagents(REAL_SUBAGENTS_DIR)
        r = next(r for r in results if r["agent_id"] == "abd91345670ff042f")
        assert r["api_calls"] >= 60  # ~72 expected
        assert r["total_cache_read"] >= 3_000_000  # ~3.6M expected
        assert r["peak_resident"] >= 50_000  # ~73K expected

    def test_known_subagent_af488(self):
        """agent-af4882881110367e0: 28 calls, ~1.5M cache_read, peak ~90K."""
        results = parse_subagents(REAL_SUBAGENTS_DIR)
        r = next(r for r in results if r["agent_id"] == "af4882881110367e0")
        assert r["api_calls"] >= 20  # ~28 expected
        assert r["total_cache_read"] >= 1_000_000  # ~1.5M expected
        assert r["peak_resident"] >= 70_000  # ~90K expected

    def test_known_subagent_ab656(self):
        """agent-ab6563be46739295e: 13 calls, ~581K cache_read, peak ~68K."""
        results = parse_subagents(REAL_SUBAGENTS_DIR)
        r = next(r for r in results if r["agent_id"] == "ab6563be46739295e")
        assert r["api_calls"] >= 10  # ~13 expected
        assert r["total_cache_read"] >= 400_000  # ~581K expected
        assert r["peak_resident"] >= 50_000  # ~68K expected

    def test_total_churn_across_all(self):
        """Total subagent cache_read should be ~11.6M."""
        results = parse_subagents(REAL_SUBAGENTS_DIR)
        total = sum(r["total_cache_read"] for r in results)
        assert total >= 8_000_000  # at least 8M (conservative)

    def test_all_have_parent_blocks(self):
        results = parse_subagents(REAL_SUBAGENTS_DIR)
        for r in results:
            pb = r["parent_block"]
            assert pb["id"].startswith("subagent-")
            assert pb["type"] == "tool_result"
            assert pb["enter"] == 0
            assert pb["exit"] is None
            assert isinstance(pb["tokens"], int)
            assert isinstance(pb["content"], str)

    def test_churn_entries_match_api_calls(self):
        results = parse_subagents(REAL_SUBAGENTS_DIR)
        for r in results:
            assert len(r["churn"]) == r["api_calls"]
