"""Tests for ccscope CLI commands."""

import argparse
import json
from unittest.mock import patch

import pytest

from context_tracker.ccscope.cli import cmd_build, cmd_list, cmd_open, main


@pytest.fixture
def projects_dir_with_sessions(tmp_path):
    """Create a projects dir with fake session JSONL files."""
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()

    # Create a session file
    session_id = "abcdef12-3456-7890-abcd-ef1234567890"
    jsonl_path = projects_dir / f"{session_id}.jsonl"
    jsonl_path.write_text(json.dumps({"type": "assistant", "message": {"content": "hi"}}) + "\n")

    # Create another in a subdirectory
    sub_dir = projects_dir / "my-project"
    sub_dir.mkdir()
    session_id2 = "12345678-abcd-ef12-3456-7890abcdef12"
    jsonl2 = sub_dir / f"{session_id2}.jsonl"
    jsonl2.write_text("{}\n")

    return projects_dir


def test_cmd_list_with_sessions(projects_dir_with_sessions, capsys):
    args = argparse.Namespace(projects_dir=str(projects_dir_with_sessions))
    result = cmd_list(args)
    assert result == 0
    captured = capsys.readouterr()
    assert "SESSION ID" in captured.out


def test_cmd_list_no_sessions(tmp_path, capsys):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    args = argparse.Namespace(projects_dir=str(empty_dir))
    result = cmd_list(args)
    assert result == 0
    captured = capsys.readouterr()
    assert "No sessions found" in captured.out


def test_cmd_list_dir_not_found(tmp_path, capsys):
    args = argparse.Namespace(projects_dir=str(tmp_path / "nonexistent"))
    result = cmd_list(args)
    assert result == 1
    captured = capsys.readouterr()
    assert "not found" in captured.err


def test_cmd_list_skips_subagents(tmp_path, capsys):
    """Subagent transcripts should be skipped."""
    projects_dir = tmp_path / "projects"
    sub_dir = projects_dir / "session1" / "subagents"
    sub_dir.mkdir(parents=True)
    (sub_dir / "12345678-abcd-1234-5678-abcdef123456.jsonl").write_text("{}\n")
    args = argparse.Namespace(projects_dir=str(projects_dir))
    result = cmd_list(args)
    assert result == 0
    captured = capsys.readouterr()
    assert "No sessions found" in captured.out


def test_cmd_list_skips_short_names(tmp_path, capsys):
    """Short filenames should be skipped."""
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    (projects_dir / "ab.jsonl").write_text("{}\n")
    args = argparse.Namespace(projects_dir=str(projects_dir))
    result = cmd_list(args)
    assert result == 0
    captured = capsys.readouterr()
    assert "No sessions found" in captured.out


def test_cmd_list_with_extras(tmp_path, capsys):
    """Sessions with subagents and offloads should show extras."""
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    session_id = "abcdef12-3456-7890-abcd-ef1234567890"
    (projects_dir / f"{session_id}.jsonl").write_text("{}\n")
    session_dir = projects_dir / session_id
    (session_dir / "subagents").mkdir(parents=True)
    (session_dir / "tool-results").mkdir(parents=True)

    args = argparse.Namespace(projects_dir=str(projects_dir))
    result = cmd_list(args)
    assert result == 0
    captured = capsys.readouterr()
    assert "subagents" in captured.out
    assert "offloads" in captured.out


def test_cmd_build_no_session(capsys):
    args = argparse.Namespace(session=None, output=None, projects_dir=None)
    result = cmd_build(args)
    assert result == 1
    captured = capsys.readouterr()
    assert "Usage" in captured.err


def test_cmd_build_file_not_found(capsys):
    args = argparse.Namespace(session="nonexistent", output=None, projects_dir=None)
    result = cmd_build(args)
    assert result == 1
    captured = capsys.readouterr()
    assert "Error" in captured.err


def test_cmd_build_success(tmp_path, capsys):
    """Test successful build with mocked reconcile."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_blocks = [{"id": "b1", "label": "sys", "tokens": 100, "cached": True}]
    mock_churn = [{"cache_read": 1000, "input": 200, "cache_creation": 50}]
    mock_subagents = [{"total_cache_read": 500}]

    out_ret = (output_dir / "blocks.json", output_dir / "churn.json")
    transcript_str = str(tmp_path / "test.jsonl")
    turn_map_ret = [{"conv_turn": 1, "first_call": 0, "last_call": 0}]

    with (
        patch("context_tracker.ccscope.cli.reconcile", return_value=(mock_blocks, mock_churn, mock_subagents)),
        patch("context_tracker.ccscope.cli.write_output", return_value=out_ret),
        patch("context_tracker.ccscope.cli.find_session_paths", return_value={"transcript": transcript_str}),
        patch("context_tracker.ccscope.cli.build_turn_map", return_value=turn_map_ret),
    ):
        # Create the transcript path so it's found
        (tmp_path / "test.jsonl").write_text("{}\n")

        args = argparse.Namespace(session="test-session", output=str(output_dir), projects_dir=None)
        result = cmd_build(args)
        assert result == 0
        captured = capsys.readouterr()
        assert "Done!" in captured.out
        assert "Subagents" in captured.out


def test_cmd_build_no_transcript(tmp_path, capsys):
    """Build succeeds even without transcript (no turn_map)."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    out_ret = (output_dir / "blocks.json", output_dir / "churn.json")
    with (
        patch("context_tracker.ccscope.cli.reconcile", return_value=([], [], [])),
        patch("context_tracker.ccscope.cli.write_output", return_value=out_ret),
        patch("context_tracker.ccscope.cli.find_session_paths", return_value={"transcript": None}),
    ):
        args = argparse.Namespace(session="test-session", output=str(output_dir), projects_dir=None)
        result = cmd_build(args)
        assert result == 0


def test_cmd_open_no_session(capsys):
    args = argparse.Namespace(session=None, output=None, projects_dir=None)
    result = cmd_open(args)
    assert result == 1


def test_cmd_open_build_fails(capsys):
    """If build fails, open should also fail."""
    args = argparse.Namespace(session="nonexistent", output=None, projects_dir=None)
    result = cmd_open(args)
    assert result == 1


def test_cmd_open_success(tmp_path, capsys):
    """Test successful open (mocked build + serve)."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    out_ret = (output_dir / "blocks.json", output_dir / "churn.json")
    with (
        patch("context_tracker.ccscope.cli.reconcile", return_value=([], [], [])),
        patch("context_tracker.ccscope.cli.write_output", return_value=out_ret),
        patch("context_tracker.ccscope.cli.find_session_paths", return_value={"transcript": None}),
        patch("webbrowser.open"),
        patch("uvicorn.run"),
    ):
        args = argparse.Namespace(
            session="test-session",
            output=None,
            projects_dir=None,
            host="127.0.0.1",
            port=9201,
        )
        result = cmd_open(args)
        assert result == 0


def test_main_no_command(capsys):
    """main() with no command prints help and exits."""
    with (
        patch("sys.argv", ["ccscope"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()
    assert exc_info.value.code == 1


def test_main_list(capsys):
    with (
        patch("sys.argv", ["ccscope", "list", "--projects-dir", "/tmp/nonexistent"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()
    assert exc_info.value.code == 1  # dir not found


def test_main_build_no_session(capsys):
    with (
        patch("sys.argv", ["ccscope", "build"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()
    assert exc_info.value.code == 1


def test_main_open_no_session(capsys):
    with (
        patch("sys.argv", ["ccscope", "open"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()
    assert exc_info.value.code == 1
