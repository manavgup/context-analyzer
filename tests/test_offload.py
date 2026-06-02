"""Tests for tool-result offload resolution."""

from pathlib import Path

from context_tracker.ccscope.offload import resolve_offloads

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool_result_block(tool_use_id: str, content: str, tokens: int = 10) -> dict:
    return {
        "id": f"t0-tool_result-{tool_use_id}",
        "type": "tool_result",
        "label": "Bash -> result",
        "tokens": tokens,
        "enter": 1,
        "exit": None,
        "cached": False,
        "ref": True,
        "content": content,
    }


def _persisted_output_content(file_id: str, dir_path: Path, size_kb: str = "300KB") -> str:
    """Build a content string matching Claude Code's offload format."""
    return (
        f"<persisted-output>\n"
        f"Output too large ({size_kb}). Full output saved to: "
        f"{dir_path}/{file_id}.txt\n\n"
        f"Preview (first 2KB):\nsome preview content here\n"
        f"</persisted-output>"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_resolve_offloads_no_dir(tmp_path):
    """No tool-results dir -> blocks unchanged."""
    missing_dir = tmp_path / "nonexistent"
    blocks = [_tool_result_block("toolu_001", "some output")]
    result = resolve_offloads(blocks, missing_dir)
    assert result == blocks
    assert "spilled_tokens" not in result[0]


def test_resolve_offloads_empty_dir(tmp_path):
    """Empty tool-results dir -> blocks unchanged."""
    tool_results_dir = tmp_path / "tool-results"
    tool_results_dir.mkdir()
    blocks = [_tool_result_block("toolu_001", "some output")]
    result = resolve_offloads(blocks, tool_results_dir)
    assert result == blocks
    assert "spilled_tokens" not in result[0]


def test_resolve_offloads_matching_file(tmp_path):
    """Matching offload file adds spilled_tokens field."""
    tool_results_dir = tmp_path / "tool-results"
    tool_results_dir.mkdir()

    # Create offload file ~400KB
    file_id = "abc123xyz"
    offload_file = tool_results_dir / f"{file_id}.txt"
    offload_content = "x" * 400_000
    offload_file.write_text(offload_content)

    content = _persisted_output_content(file_id, tool_results_dir)
    blocks = [_tool_result_block("toolu_001", content, tokens=500)]

    result = resolve_offloads(blocks, tool_results_dir)

    assert len(result) == 1
    b = result[0]
    # spilled_tokens should be added
    assert "spilled_tokens" in b
    # estimated from file size: 400000 // 4 = 100000
    assert b["spilled_tokens"] == 100_000
    # resident tokens unchanged
    assert b["tokens"] == 500
    # content annotated
    assert "OFFLOADED" in b["content"]
    assert file_id in b["content"]


def test_resolve_offloads_no_match(tmp_path):
    """Non-matching blocks are unchanged."""
    tool_results_dir = tmp_path / "tool-results"
    tool_results_dir.mkdir()

    # Create an offload file for a different ID
    (tool_results_dir / "zzz999.txt").write_text("x" * 100_000)

    # Block whose content doesn't reference any offload file
    content = "Normal tool output without offload reference"
    blocks = [_tool_result_block("toolu_001", content, tokens=20)]

    result = resolve_offloads(blocks, tool_results_dir)

    assert "spilled_tokens" not in result[0]
    assert result[0]["content"] == content
    assert result[0]["tokens"] == 20


def test_resolve_offloads_preserves_resident_tokens(tmp_path):
    """Resident tokens field stays unchanged (already correct from usage)."""
    tool_results_dir = tmp_path / "tool-results"
    tool_results_dir.mkdir()

    file_id = "res123test"
    (tool_results_dir / f"{file_id}.txt").write_text("y" * 300_000)

    content = _persisted_output_content(file_id, tool_results_dir)
    original_tokens = 42
    blocks = [_tool_result_block("toolu_007", content, tokens=original_tokens)]

    result = resolve_offloads(blocks, tool_results_dir)

    assert result[0]["tokens"] == original_tokens  # resident tokens unchanged
    assert result[0]["spilled_tokens"] == 75_000   # 300000 // 4


def test_resolve_offloads_non_tool_result_blocks_unchanged(tmp_path):
    """Non-tool_result blocks are not modified."""
    tool_results_dir = tmp_path / "tool-results"
    tool_results_dir.mkdir()

    file_id = "xyz456abc"
    (tool_results_dir / f"{file_id}.txt").write_text("z" * 200_000)

    # An assistant block that happens to contain the file_id in its content
    assistant_block = {
        "id": "t0-assistant-0",
        "type": "assistant",
        "label": "assistant",
        "tokens": 100,
        "content": f"I saved the output to {file_id}.txt",
    }
    tool_result_block = _tool_result_block(
        "toolu_abc",
        _persisted_output_content(file_id, tool_results_dir),
        tokens=50,
    )
    blocks = [assistant_block, tool_result_block]

    result = resolve_offloads(blocks, tool_results_dir)

    # Assistant block unchanged
    assert "spilled_tokens" not in result[0]
    assert result[0]["tokens"] == 100
    # tool_result block gets spilled_tokens
    assert "spilled_tokens" in result[1]


def test_resolve_offloads_multiple_blocks_multiple_files(tmp_path):
    """Multiple offloaded blocks in the same call are all resolved."""
    tool_results_dir = tmp_path / "tool-results"
    tool_results_dir.mkdir()

    ids_and_sizes = [("file_aaa", 400_000), ("file_bbb", 200_000), ("file_ccc", 100_000)]
    for fid, size in ids_and_sizes:
        (tool_results_dir / f"{fid}.txt").write_text("x" * size)

    blocks = [
        _tool_result_block(
            f"toolu_{i:03d}",
            _persisted_output_content(fid, tool_results_dir),
            tokens=50 + i * 10,
        )
        for i, (fid, _) in enumerate(ids_and_sizes)
    ]

    result = resolve_offloads(blocks, tool_results_dir)

    assert result[0]["spilled_tokens"] == 100_000   # 400000 // 4
    assert result[1]["spilled_tokens"] == 50_000    # 200000 // 4
    assert result[2]["spilled_tokens"] == 25_000    # 100000 // 4
    # Resident tokens unchanged
    assert result[0]["tokens"] == 50
    assert result[1]["tokens"] == 60
    assert result[2]["tokens"] == 70


def test_resolve_offloads_id_in_block_id(tmp_path):
    """Offload matched via file_id appearing in the block id field."""
    tool_results_dir = tmp_path / "tool-results"
    tool_results_dir.mkdir()

    file_id = "direct_id_match"
    (tool_results_dir / f"{file_id}.txt").write_text("w" * 80_000)

    # Block whose ID contains the file_id (fallback matching)
    block = {
        "id": f"t1-tool_result-{file_id}",
        "type": "tool_result",
        "label": "Bash -> result",
        "tokens": 5,
        "enter": 1,
        "exit": None,
        "cached": False,
        "ref": True,
        "content": "Normal output without persisted-output tag",
    }

    result = resolve_offloads([block], tool_results_dir)

    assert "spilled_tokens" in result[0]
    assert result[0]["spilled_tokens"] == 20_000   # 80000 // 4
