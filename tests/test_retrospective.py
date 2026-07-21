"""Tests for experiments/headroom/retrospective.py (Stage 0 of #94).

Hermetic: headroom and tiktoken are NEVER imported — a fake compressor and a
fake token counter are injected. A synthetic transcript + fixture DB give
known expected math for the residency-weighted model, and a no-leakage test
proves the report generator emits no transcript content.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parent.parent / "experiments" / "headroom" / "retrospective.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("retrospective", _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["retrospective"] = mod
    spec.loader.exec_module(mod)
    return mod


retro = _load_module()

# Distinctive markers that must never leak into the report (not credentials).
SECRET_A = "XSECRETMARKER-ALPHA-7f3e-do-not-leakX"  # noqa: S105
SECRET_B = "XSECRETMARKER-BRAVO-9c1d-do-not-leakX"  # noqa: S105
SESSION_ID = "aaaabbbb-cccc-dddd-eeee-ffff00001111"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def fake_count(text: str) -> int:
    """Deterministic fake tokenizer: 1 token per 4 characters."""
    return max(1, len(text) // 4)


def fake_compress_half(text: str, tool_name: str, tool_input: dict):
    """Fake compressor: keeps exactly the first half of the text."""
    return text[: len(text) // 2], ["fake:half"]


def fake_compress_noop(text: str, tool_name: str, tool_input: dict):
    return text, []


def fake_compress_error(text: str, tool_name: str, tool_input: dict):
    raise RuntimeError("compressor exploded")


# ---------------------------------------------------------------------------
# Fixtures: synthetic transcript + fixture DB with known math
# ---------------------------------------------------------------------------


@pytest.fixture()
def fixture_paths(tmp_path: Path) -> tuple[Path, Path]:
    """Build a transcript (2 tool results) and a matching analyzer-style DB.

    Session: 10 API calls total.
      - tool_result toolu_aaa: enters at call 2, never exits -> residency 8.
        DB block tokens 1000. Content SECRET_A (800 chars -> 200 fake tokens).
      - tool_result toolu_bbb: enters at call 4, exits at call 9 -> residency 5.
        DB block tokens 400. Content SECRET_B (400 chars -> 100 fake tokens).

    With fake_compress_half (ratio exactly 0.5 under fake_count):
      saved_token_calls = 1000*0.5*8 + 400*0.5*5 = 4000 + 1000 = 5000
    Denominator (resident token-calls) = 20000 (set in sessions row).
      ceiling_pct = 0.25
    Input-side cost at fixed rates:
      input=1_000_000 -> $15.00 ; cache_read=8_000_000 -> $15.00 ;
      cache_creation=... 0 => $30.00 ; ceiling_usd = 7.50
    """
    projects = tmp_path / "projects" / "-proj-x"
    projects.mkdir(parents=True)
    transcript = projects / f"{SESSION_ID}.jsonl"

    content_a = (SECRET_A + " padding ") * 20  # deterministic length
    content_a = content_a[:800]
    content_b = (SECRET_B + " padding ") * 20
    content_b = content_b[:400]
    assert SECRET_A in content_a and SECRET_B in content_b

    entries = [
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "id": "toolu_aaa", "name": "Bash", "input": {"command": "run-a"}},
                ]
            },
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_aaa", "content": content_a},
                ]
            },
        },
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "id": "toolu_bbb", "name": "Read", "input": {"file_path": "/x/y.py"}},
                ]
            },
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_bbb",
                        "content": [{"type": "text", "text": content_b}],
                    },
                ]
            },
        },
    ]
    with open(transcript, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    db_path = tmp_path / "fixture.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY, agent TEXT, total_api_calls INTEGER,
            total_input_tokens INTEGER, total_output_tokens INTEGER,
            total_cache_read INTEGER, total_cache_creation INTEGER,
            total_cost_usd FLOAT, source_mtime FLOAT
        );
        CREATE TABLE blocks (
            id INTEGER PRIMARY KEY, session_id TEXT, block_id TEXT,
            block_type TEXT, label TEXT, tokens INTEGER,
            enter_turn INTEGER, exit_turn INTEGER
        );
        """
    )
    conn.execute(
        "INSERT INTO sessions VALUES (?, 'claude-code', 10, 1000000, 50000, 8000000, 11000000, 123.45, 1750000000.0)",
        (SESSION_ID,),
    )
    # resident token-calls denominator = 1e6 + 8e6 + 11e6 = 20e6... — see test
    conn.executemany(
        "INSERT INTO blocks (session_id, block_id, block_type, label, tokens, enter_turn, exit_turn) "
        "VALUES (?, ?, 'tool_result', ?, ?, ?, ?)",
        [
            (SESSION_ID, "t2-tool_result-toolu_aaa", "Bash → run-a", 1000, 2, None),
            (SESSION_ID, "t4-tool_result-toolu_bbb", "Read → y.py", 400, 4, 9),
        ],
    )
    conn.commit()
    conn.close()
    return db_path, tmp_path / "projects"


# ---------------------------------------------------------------------------
# Unit tests: helpers
# ---------------------------------------------------------------------------


def test_tool_use_id_from_block():
    assert retro.tool_use_id_from_block("t4-tool_result-toolu_xyz") == "toolu_xyz"
    assert retro.tool_use_id_from_block("t0-user-0") is None
    assert retro.tool_use_id_from_block("tool_result-") is None


def test_classify_content():
    assert retro.classify_content(json.dumps([{"a": 1}] * 30)) == "json"
    code = "\n".join(f"def f{i}(x):\n    return x + {i}" for i in range(10))
    assert retro.classify_content(code) == "code"
    log = "\n".join(f"2026-07-21 12:00:{i:02d} INFO something happened" for i in range(20))
    assert retro.classify_content(log) == "log"
    prose = "This is a plain English explanation of the system. " * 20
    assert retro.classify_content(prose) == "prose"
    assert retro.classify_content("") == "other"


def test_bucket_tool():
    assert retro.bucket_tool("Bash") == "Bash"
    assert retro.bucket_tool("Agent") == "Task"
    assert retro.bucket_tool("SomethingElse") == "other"


def test_tool_result_text_shapes():
    assert retro._tool_result_text("abc") == "abc"
    assert retro._tool_result_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a\nb"
    assert retro._tool_result_text(None) == ""


# ---------------------------------------------------------------------------
# The residency-weighted model: known expected math
# ---------------------------------------------------------------------------


def test_residency_weighted_math(fixture_paths):
    db_path, projects_dir = fixture_paths
    corpus = retro.run_audit(
        db_path,
        projects_dir,
        compress_fn=fake_compress_half,
        count_fn=fake_count,
        headroom_version="fake-1.0",
    )
    assert len(corpus.sessions) == 1
    sess = corpus.sessions[0]

    # residency: toolu_aaa = 10 - 2 = 8 ; toolu_bbb = 9 - 4 = 5
    # fake ratio is exactly 0.5 (fake_count = len//4, compressed = half length)
    expected_saved = 1000 * 0.5 * 8 + 400 * 0.5 * 5  # 5000.0
    assert sess.saved_token_calls == pytest.approx(expected_saved)

    # denominator from sessions row: 1e6 + 8e6 + 11e6
    assert sess.resident_token_calls == pytest.approx(20_000_000)
    assert sess.ceiling_pct == pytest.approx(5000 / 20_000_000)

    # input-side cost at the ingest-fixed rates:
    # 1e6*15/1e6 + 8e6*1.875/1e6 + 11e6*18.75/1e6 = 15 + 15 + 206.25 = 236.25
    assert sess.input_side_cost == pytest.approx(236.25)
    assert sess.ceiling_usd == pytest.approx(sess.ceiling_pct * 236.25)

    assert sess.items_total == 2
    assert sess.items_compressed == 2
    assert sess.items_failed == 0
    assert sess.items_missing_content == 0
    assert sess.items_fallback_residency == 0


def test_noop_compressor_saves_nothing(fixture_paths):
    db_path, projects_dir = fixture_paths
    corpus = retro.run_audit(
        db_path,
        projects_dir,
        compress_fn=fake_compress_noop,
        count_fn=fake_count,
    )
    assert corpus.total_saved_token_calls == 0
    assert corpus.overall_ceiling_pct == 0


def test_compressor_error_counts_incompressible(fixture_paths):
    db_path, projects_dir = fixture_paths
    corpus = retro.run_audit(
        db_path,
        projects_dir,
        compress_fn=fake_compress_error,
        count_fn=fake_count,
    )
    sess = corpus.sessions[0]
    assert sess.items_failed == 2
    assert corpus.total_saved_token_calls == 0


def test_limit_zero_processes_nothing(fixture_paths):
    db_path, projects_dir = fixture_paths
    corpus = retro.run_audit(
        db_path,
        projects_dir,
        limit=0,
        compress_fn=fake_compress_half,
        count_fn=fake_count,
    )
    assert corpus.sessions == []


def test_db_opened_readonly(fixture_paths):
    db_path, projects_dir = fixture_paths
    conn = retro.open_db_readonly(db_path)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("INSERT INTO sessions (session_id) VALUES ('nope')")
    conn.close()


# ---------------------------------------------------------------------------
# No-leakage: report must contain numbers only
# ---------------------------------------------------------------------------


def test_report_has_no_content_leakage(fixture_paths):
    db_path, projects_dir = fixture_paths
    corpus = retro.run_audit(
        db_path,
        projects_dir,
        compress_fn=fake_compress_half,
        count_fn=fake_count,
    )
    report = retro.build_report(corpus)

    assert SECRET_A not in report
    assert SECRET_B not in report
    assert "SECRETMARKER" not in report
    # no tool_use ids, no transcript paths, no full session id
    assert "toolu_aaa" not in report
    assert "toolu_bbb" not in report
    assert str(projects_dir) not in report
    assert SESSION_ID not in report
    # session id appears only as its first 8 chars
    assert SESSION_ID[:8] in report


def test_report_structure(fixture_paths):
    db_path, projects_dir = fixture_paths
    corpus = retro.run_audit(
        db_path,
        projects_dir,
        compress_fn=fake_compress_half,
        count_fn=fake_count,
        headroom_version="fake-1.0",
    )
    report = retro.build_report(corpus)
    assert "fake-1.0" in report
    assert "## Headline" in report
    assert "## By content type" in report
    assert "## By tool" in report
    assert "## Limitations" in report
    assert "Kill criterion" in report
    # ceiling on this fixture is 0.025% -> kill criterion met
    assert "MET — ceiling is below 10%" in report


def test_fallback_residency_for_unjoined_items(tmp_path):
    """A transcript tool_result with no DB block uses session-average residency."""
    projects = tmp_path / "projects" / "-p"
    projects.mkdir(parents=True)
    sid = "11112222-3333-4444-5555-666677778888"
    entries = [
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "id": "toolu_j", "name": "Bash", "input": {}},
                    {"type": "tool_use", "id": "toolu_orphan", "name": "Grep", "input": {}},
                ]
            },
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_j", "content": "x" * 400},
                    {"type": "tool_result", "tool_use_id": "toolu_orphan", "content": "y" * 400},
                ]
            },
        },
    ]
    with open(projects / f"{sid}.jsonl", "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    db_path = tmp_path / "f.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE sessions (session_id TEXT PRIMARY KEY, agent TEXT,
            total_api_calls INTEGER, total_input_tokens INTEGER,
            total_output_tokens INTEGER, total_cache_read INTEGER,
            total_cache_creation INTEGER, total_cost_usd FLOAT, source_mtime FLOAT);
        CREATE TABLE blocks (id INTEGER PRIMARY KEY, session_id TEXT,
            block_id TEXT, block_type TEXT, tokens INTEGER,
            enter_turn INTEGER, exit_turn INTEGER);
        """
    )
    conn.execute(
        "INSERT INTO sessions VALUES (?, 'claude-code', 10, 100000, 1000, 0, 0, 1.0, 0)",
        (sid,),
    )
    conn.execute(
        "INSERT INTO blocks (session_id, block_id, block_type, tokens, enter_turn, exit_turn) "
        "VALUES (?, 't2-tool_result-toolu_j', 'tool_result', 100, 2, None)".replace("None", "NULL"),
        (sid,),
    )
    conn.commit()
    conn.close()

    corpus = retro.run_audit(
        db_path,
        tmp_path / "projects",
        compress_fn=fake_compress_half,
        count_fn=fake_count,
    )
    sess = corpus.sessions[0]
    assert sess.items_total == 2
    assert sess.items_fallback_residency == 1
    # joined item: tokens=100, ratio 0.5, residency 10-2=8 -> 400
    # orphan: fake_count(400 chars)=100 tokens, ratio 0.5, avg residency=8 -> 400
    assert sess.saved_token_calls == pytest.approx(800.0)
