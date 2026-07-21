"""Tests for `context-tracker audit-headroom` (one-command ceiling audit, #94).

Hermetic like tests/test_retrospective.py: headroom, tiktoken and onnxruntime
are NEVER imported — dependency checks are exercised by monkeypatching
`_module_available`, and the audit itself runs with injected fake
compressor/tokenizer functions over synthetic transcripts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from context_tracker import headroom_audit

SESSION_A = "aaaabbbb-cccc-dddd-eeee-ffff00001111"
SESSION_B = "99998888-7777-6666-5555-444433332222"


# ---------------------------------------------------------------------------
# Fakes (same shapes as tests/test_retrospective.py)
# ---------------------------------------------------------------------------


def fake_count(text: str) -> int:
    return max(1, len(text) // 4)


def fake_compress_half(text: str, tool_name: str, tool_input: dict):
    return text[: len(text) // 2], ["fake:half"]


# ---------------------------------------------------------------------------
# Synthetic transcripts that the REAL ingest pipeline can consume
# ---------------------------------------------------------------------------


def _entry_user(content, uuid, parent=None):
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent,
        "timestamp": "2026-01-01T00:00:00.000Z",
        "message": {"content": content},
    }


def _entry_assistant(content_blocks, usage, uuid, parent):
    return {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": parent,
        "timestamp": "2026-01-01T00:00:01.000Z",
        "message": {
            "model": "claude-opus-4-6",
            "stop_reason": "end_turn",
            "content": content_blocks,
            "usage": usage,
        },
    }


def _session_entries(tool_use_id: str, payload: str):
    """Two API calls; one tool_use/tool_result pair carrying `payload`."""
    return [
        _entry_user("please run the thing", "u1"),
        _entry_assistant(
            [{"type": "tool_use", "id": tool_use_id, "name": "Bash", "input": {"command": "do-it"}}],
            {
                "input_tokens": 100,
                "cache_creation_input_tokens": 1000,
                "cache_read_input_tokens": 0,
                "output_tokens": 10,
            },
            "a1",
            "u1",
        ),
        _entry_user([{"type": "tool_result", "tool_use_id": tool_use_id, "content": payload}], "u2", "a1"),
        _entry_assistant(
            [{"type": "text", "text": "done"}],
            {
                "input_tokens": 50,
                "cache_creation_input_tokens": 100,
                "cache_read_input_tokens": 2000,
                "output_tokens": 20,
            },
            "a2",
            "u2",
        ),
    ]


@pytest.fixture()
def projects_dir(tmp_path: Path) -> Path:
    """Fake ~/.claude/projects with two ingestable sessions."""
    proj = tmp_path / "projects" / "-my-project"
    proj.mkdir(parents=True)
    for sid, tid, payload in (
        (SESSION_A, "toolu_aaa", "x" * 800),
        (SESSION_B, "toolu_bbb", "y" * 400),
    ):
        with open(proj / f"{sid}.jsonl", "w") as f:
            for e in _session_entries(tid, payload):
                f.write(json.dumps(e) + "\n")
    return tmp_path / "projects"


@pytest.fixture()
def isolated_dirs(tmp_path: Path, monkeypatch):
    """Redirect the live-DB path and --keep-db dir away from the real home."""
    fake_live_db = tmp_path / "live" / "analyzer.db"
    keep_dir = tmp_path / "keep" / "audit-headroom"
    monkeypatch.setattr(headroom_audit, "DEFAULT_DB_PATH", fake_live_db)
    monkeypatch.setattr(headroom_audit, "KEEP_DB_DIR", keep_dir)
    return fake_live_db, keep_dir


# ---------------------------------------------------------------------------
# Dependency UX
# ---------------------------------------------------------------------------


def test_missing_headroom_prints_exact_install_command(monkeypatch, capsys):
    monkeypatch.setattr(headroom_audit, "_module_available", lambda name: False)
    rc = headroom_audit.run_audit_headroom()
    assert rc == 1
    err = capsys.readouterr().err
    assert "pip install --no-deps headroom-ai==0.32.1 && pip install tiktoken" in err
    assert "--no-deps" in err  # the macOS/litellm rationale is stated
    assert "litellm" in err


def test_missing_tiktoken_only_still_prints_install_command(monkeypatch, capsys):
    monkeypatch.setattr(headroom_audit, "_module_available", lambda name: name == "headroom")
    rc = headroom_audit.run_audit_headroom()
    assert rc == 1
    assert "pip install --no-deps headroom-ai==0.32.1" in capsys.readouterr().err


def test_max_profile_missing_onnxruntime(monkeypatch, capsys):
    monkeypatch.setattr(headroom_audit, "_module_available", lambda name: name != "onnxruntime")
    rc = headroom_audit.run_audit_headroom(profile="max", yes=True)
    assert rc == 1
    err = capsys.readouterr().err
    assert "pip install onnxruntime" in err


def test_defaults_profile_does_not_require_onnxruntime(monkeypatch, projects_dir, tmp_path, isolated_dirs):
    """defaults profile with deps 'installed' but fakes injected runs fine sans onnxruntime."""
    monkeypatch.setattr(headroom_audit, "_module_available", lambda name: name != "onnxruntime")
    out = tmp_path / "r.md"
    rc = headroom_audit.run_audit_headroom(
        out=out,
        projects_dir=projects_dir,
        compress_fn=fake_compress_half,
        count_fn=fake_count,
    )
    assert rc == 0


def test_max_profile_requires_confirmation_for_model_download(monkeypatch, capsys):
    """Without --yes and with the confirmation declined, exit 1 before any work."""
    monkeypatch.setattr(headroom_audit, "_module_available", lambda name: True)
    rc = headroom_audit.run_audit_headroom(profile="max", yes=False, confirm_fn=lambda prompt: False)
    assert rc == 1
    err = capsys.readouterr().err
    assert "--yes" in err
    assert "261" in err  # the ~261MB download size is disclosed


def test_max_profile_non_tty_defaults_to_abort(monkeypatch, capsys):
    """Non-interactive stdin + no --yes must not silently start a 261MB download."""
    monkeypatch.setattr(headroom_audit, "_module_available", lambda name: True)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    rc = headroom_audit.run_audit_headroom(profile="max", yes=False)
    assert rc == 1


# ---------------------------------------------------------------------------
# The command itself (fakes injected — no optional deps needed)
# ---------------------------------------------------------------------------


def test_report_written_and_headline_printed(projects_dir, tmp_path, isolated_dirs, capsys):
    out = tmp_path / "ceiling.md"
    rc = headroom_audit.run_audit_headroom(
        out=out,
        projects_dir=projects_dir,
        compress_fn=fake_compress_half,
        count_fn=fake_count,
    )
    assert rc == 0

    # Full report on disk with the standard sections.
    report = out.read_text()
    assert "## Headline" in report
    assert "Kill criterion" in report
    assert "Sessions analyzed: **2**" in report

    # Headline on stdout: ceiling % and $, kill criterion, top content types,
    # and the numbers-only privacy reminder.
    stdout = capsys.readouterr().out
    assert "Ceiling:" in stdout
    assert "Kill criterion" in stdout
    assert "Top content types" in stdout
    assert str(out) in stdout
    assert "numbers only" in stdout


def test_scratch_db_isolation_live_db_never_written(projects_dir, tmp_path, isolated_dirs):
    fake_live_db, keep_dir = isolated_dirs
    out = tmp_path / "r.md"
    rc = headroom_audit.run_audit_headroom(
        out=out,
        projects_dir=projects_dir,
        compress_fn=fake_compress_half,
        count_fn=fake_count,
    )
    assert rc == 0
    assert not fake_live_db.exists()  # live analyzer DB never created/touched
    assert not (keep_dir / "corpus.db").exists()  # no --keep-db -> temp dir, cleaned up


def test_keep_db_persists_scratch_corpus(projects_dir, tmp_path, isolated_dirs):
    fake_live_db, keep_dir = isolated_dirs
    rc = headroom_audit.run_audit_headroom(
        out=tmp_path / "r.md",
        keep_db=True,
        projects_dir=projects_dir,
        compress_fn=fake_compress_half,
        count_fn=fake_count,
    )
    assert rc == 0
    assert (keep_dir / "corpus.db").exists()
    assert not fake_live_db.exists()


def test_build_scratch_corpus_refuses_live_db(isolated_dirs, projects_dir):
    fake_live_db, _ = isolated_dirs
    fake_live_db.parent.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="live analyzer DB"):
        headroom_audit.build_scratch_corpus(fake_live_db, projects_dir)


def test_limit_restricts_sessions(projects_dir, tmp_path, isolated_dirs):
    out = tmp_path / "r.md"
    rc = headroom_audit.run_audit_headroom(
        out=out,
        limit=1,
        projects_dir=projects_dir,
        compress_fn=fake_compress_half,
        count_fn=fake_count,
    )
    assert rc == 0
    assert "Sessions analyzed: **1**" in out.read_text()


def test_no_transcripts_is_a_clean_error(tmp_path, isolated_dirs, capsys):
    empty = tmp_path / "empty-projects"
    empty.mkdir()
    rc = headroom_audit.run_audit_headroom(
        projects_dir=empty,
        compress_fn=fake_compress_half,
        count_fn=fake_count,
    )
    assert rc == 1
    assert "No Claude Code session transcripts found" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# CLI dispatch (context-tracker audit-headroom ...)
# ---------------------------------------------------------------------------


def test_cli_dispatches_audit_headroom(monkeypatch):
    from context_tracker import cli

    calls: list[dict] = []
    monkeypatch.setattr(headroom_audit, "run_audit_headroom", lambda **kw: calls.append(kw) or 0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "context-tracker",
            "audit-headroom",
            "--profile",
            "max",
            "--out",
            "my-report.md",
            "--limit",
            "3",
            "--keep-db",
            "--yes",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    assert calls == [
        {
            "profile": "max",
            "out": Path("my-report.md"),
            "limit": 3,
            "keep_db": True,
            "yes": True,
        }
    ]


def test_cli_audit_headroom_defaults(monkeypatch):
    from context_tracker import server

    calls: list[dict] = []
    monkeypatch.setattr(headroom_audit, "run_audit_headroom", lambda **kw: calls.append(kw) or 0)
    monkeypatch.setattr(sys, "argv", ["context-tracker", "audit-headroom"])
    with pytest.raises(SystemExit) as exc:
        server.main()
    assert exc.value.code == 0
    assert calls == [
        {
            "profile": "defaults",
            "out": Path("headroom-ceiling-report.md"),
            "limit": None,
            "keep_db": False,
            "yes": False,
        }
    ]
