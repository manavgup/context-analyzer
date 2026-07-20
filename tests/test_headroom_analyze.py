"""Tests for experiments/headroom/analyze.py against the real DB schema.

The fixture DB is created via context_tracker.db.get_engine, so these tests
prove analyze.py's raw SQL matches the schema as it actually exists.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from context_tracker.db import ApiCallRecord, BlockRecord, SessionRecord, get_engine, get_session_factory

ANALYZE_PATH = Path(__file__).parent.parent / "experiments" / "headroom" / "analyze.py"


def _load_analyze():
    spec = importlib.util.spec_from_file_location("headroom_analyze", ANALYZE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec: dataclass field resolution looks up sys.modules[__module__].
    sys.modules["headroom_analyze"] = module
    spec.loader.exec_module(module)
    return module


analyze = _load_analyze()


def _add_session(
    db,
    session_id: str,
    input_tokens: int,
    output_tokens: int,
    cache_read: int,
    cache_creation: int,
    cost: float,
    peak: int,
    retrieval_calls: int = 0,
) -> None:
    db.add(
        SessionRecord(
            session_id=session_id,
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            total_cache_read=cache_read,
            total_cache_creation=cache_creation,
            total_cost_usd=cost,
            peak_context_tokens=peak,
            total_api_calls=3,
        )
    )
    db.add(
        ApiCallRecord(
            session_id=session_id,
            call_index=0,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read=cache_read,
            cache_creation=cache_creation,
        )
    )
    for i in range(retrieval_calls):
        db.add(
            BlockRecord(
                session_id=session_id,
                block_id=f"t1-tool_call-{i}",
                block_type="tool_call",
                label="mcp__headroom__headroom_retrieve chunk-42",
                tokens=100,
            )
        )
    # A non-retrieval tool call that must NOT be counted.
    db.add(
        BlockRecord(
            session_id=session_id,
            block_id="t1-tool_call-read",
            block_type="tool_call",
            label="Read /tmp/foo.py",
            tokens=50,
        )
    )


def _manifest_row(
    task_id: str,
    pair: int,
    arm: str,
    session_id: str,
    success: bool,
    cost_usd: float | None = None,
) -> dict:
    return {
        "task_id": task_id,
        "pair": pair,
        "arm": arm,
        "order_position": 1 if arm == "plain" else 2,
        "order": "PH",
        "session_id": session_id,
        "success": success,
        "cost_usd": cost_usd,
        "model": "claude-sonnet-4-5",
        "claude_version": "test",
        "headroom_version": "test",
    }


@pytest.fixture
def fixture_env(tmp_path: Path) -> tuple[Path, Path]:
    """Create a small fixture DB (real schema) and a matching manifest."""
    db_path = tmp_path / "analyzer.db"
    engine = get_engine(db_path)
    factory = get_session_factory(engine)
    with factory() as db:
        # Pair 1: headroom saves input tokens but busts the cache and retrieves twice.
        _add_session(db, "sess-plain-1", 100_000, 5_000, 900_000, 50_000, 1.20, 140_000)
        _add_session(db, "sess-head-1", 60_000, 6_000, 300_000, 200_000, 1.40, 90_000, retrieval_calls=2)
        # Pair 2: headroom arm failed the task.
        _add_session(db, "sess-plain-2", 80_000, 4_000, 500_000, 30_000, 0.80, 110_000)
        _add_session(db, "sess-head-2", 70_000, 4_500, 400_000, 60_000, 0.85, 100_000, retrieval_calls=1)
        db.commit()

    manifest = tmp_path / "manifest.jsonl"
    # Manifest costs are the model-correct (Sonnet) envelope values — deliberately
    # ~5x lower than the fixed-Opus-rate DB costs so tests can tell them apart.
    rows = [
        _manifest_row("code-search", 1, "plain", "sess-plain-1", True, cost_usd=0.24),
        _manifest_row("code-search", 1, "headroom", "sess-head-1", True, cost_usd=0.28),
        _manifest_row("test-fix", 1, "plain", "sess-plain-2", True, cost_usd=0.16),
        _manifest_row("test-fix", 1, "headroom", "sess-head-2", False, cost_usd=0.17),
    ]
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return db_path, manifest


class TestFetchSessionMetrics:
    def test_reads_real_schema_columns(self, fixture_env):
        import sqlite3

        db_path, _ = fixture_env
        conn = sqlite3.connect(db_path)
        m = analyze.fetch_session_metrics(conn, "sess-head-1")
        conn.close()
        assert m is not None
        assert m.input_tokens == 60_000
        assert m.cache_read == 300_000
        assert m.cost_usd == 1.40
        assert m.peak_context == 90_000

    def test_counts_only_retrieval_tool_calls(self, fixture_env):
        import sqlite3

        db_path, _ = fixture_env
        conn = sqlite3.connect(db_path)
        head = analyze.fetch_session_metrics(conn, "sess-head-1")
        plain = analyze.fetch_session_metrics(conn, "sess-plain-1")
        conn.close()
        assert head is not None and head.retrievals == 2
        assert plain is not None and plain.retrievals == 0

    def test_missing_session_returns_none(self, fixture_env):
        import sqlite3

        db_path, _ = fixture_env
        conn = sqlite3.connect(db_path)
        assert analyze.fetch_session_metrics(conn, "no-such-session") is None
        conn.close()

    def test_cache_hit_rate(self):
        m = analyze.SessionMetrics(
            session_id="s",
            input_tokens=100,
            output_tokens=10,
            cache_read=800,
            cache_creation=100,
            cost_usd=0.1,
            peak_context=1,
            api_calls=1,
            retrievals=0,
        )
        assert m.cache_hit_rate == pytest.approx(0.8)


class TestReport:
    def test_produces_markdown_tables(self, fixture_env, capsys):
        db_path, manifest = fixture_env
        rc = analyze.main(["--manifest", str(manifest), "--db", str(db_path)])
        out = capsys.readouterr().out
        assert rc == 0
        # Per-pair table with both arms' values and the delta.
        assert "## Per-pair results" in out
        assert "| Task | Pair | Metric | Plain | Headroom | Delta | Delta % |" in out
        assert "| code-search | 1 | Input tok | 100,000 | 60,000 | -40,000 | -40.0% |" in out
        # Aggregate excludes the parity-failed pair and says so.
        assert "## Aggregate" in out
        assert "usable (both arms succeeded, both ingested): 1" in out
        assert "Outcome-parity failures (excluded): test-fix#1" in out
        # Retrieval round-trips surfaced.
        assert "Retrievals" in out

    def test_manifest_cost_preferred_over_db(self, fixture_env, capsys):
        """Cost rows use the model-correct manifest cost, not the fixed-Opus-rate DB cost."""
        db_path, manifest = fixture_env
        rc = analyze.main(["--manifest", str(manifest), "--db", str(db_path)])
        captured = capsys.readouterr()
        assert rc == 0
        # Manifest costs (0.24 / 0.28), not DB costs (1.20 / 1.40).
        assert "| code-search | 1 | Cost $ | 0.2400 | 0.2800 | +0.0400 | +16.7% |" in captured.out
        assert "1.2000" not in captured.out
        assert "1.4000" not in captured.out
        # No fallback warning when the manifest carries costs.
        assert "fixed" not in captured.err.lower()

    def test_db_cost_fallback_warns_about_fixed_opus_rates(self, fixture_env, capsys):
        """Without manifest cost_usd, the DB cost is used with an explicit caveat."""
        db_path, manifest = fixture_env
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
        for r in rows:
            del r["cost_usd"]  # simulate a manifest that predates cost capture
        manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        rc = analyze.main(["--manifest", str(manifest), "--db", str(db_path)])
        captured = capsys.readouterr()
        assert rc == 0
        # Falls back to the DB costs...
        assert "| code-search | 1 | Cost $ | 1.2000 | 1.4000 | +0.2000 | +16.7% |" in captured.out
        # ...and names the fixed-rate caveat.
        assert "no cost_usd in manifest" in captured.err
        assert "fixed Opus rates" in captured.err

    def test_failed_pair_excluded_from_metrics_but_listed_as_parity_failure(self, fixture_env, capsys):
        """A pair where either arm failed emits NO metric rows anywhere; it is only
        listed in the outcome-parity failures section (per METHODOLOGY.md)."""
        db_path, manifest = fixture_env
        rc = analyze.main(["--manifest", str(manifest), "--db", str(db_path)])
        out = capsys.readouterr().out
        assert rc == 0
        # No per-pair metric row for the failed pair — even though both sessions
        # are fully ingested in the DB.
        assert "| test-fix | 1 | Input tok |" not in out
        assert "| test-fix | 1 | Cost $ |" not in out
        # Listed in the dedicated parity-failures section with per-arm outcomes.
        assert "## Outcome-parity failures" in out
        assert "| test-fix | 1 | yes | NO |" in out

    def test_no_parity_failures_reported_when_all_pairs_pass(self, fixture_env, capsys):
        db_path, manifest = fixture_env
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
        for r in rows:
            r["success"] = True
        manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        rc = analyze.main(["--manifest", str(manifest), "--db", str(db_path)])
        out = capsys.readouterr().out
        assert rc == 0
        assert "None — both arms passed success_check in every pair." in out
        assert "| test-fix | 1 | Input tok | 80,000 | 70,000 | -10,000 | -12.5% |" in out
        assert "usable (both arms succeeded, both ingested): 2" in out

    def test_out_file_written(self, fixture_env, tmp_path, capsys):
        db_path, manifest = fixture_env
        out_file = tmp_path / "results.md"
        rc = analyze.main(["--manifest", str(manifest), "--db", str(db_path), "--out", str(out_file)])
        capsys.readouterr()
        assert rc == 0
        assert out_file.exists()
        assert "# Headroom experiment results" in out_file.read_text(encoding="utf-8")

    def test_missing_manifest_errors(self, tmp_path, capsys):
        rc = analyze.main(["--manifest", str(tmp_path / "nope.jsonl"), "--db", str(tmp_path / "nope.db")])
        capsys.readouterr()
        assert rc == 1

    def test_build_pairs_skips_incomplete(self):
        rows = [
            _manifest_row("t", 1, "plain", "a", True),
            _manifest_row("t", 1, "headroom", "b", True),
            _manifest_row("t", 2, "plain", "c", True),  # missing headroom arm
        ]
        pairs = analyze.build_pairs(rows)
        assert len(pairs) == 1
        assert pairs[0].task_id == "t"
