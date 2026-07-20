#!/usr/bin/env python3
"""Analyze headroom experiment results (issue #81).

Joins the run_pair.sh manifest (arm -> session_id mapping) against the
context-analyzer SQLite database and emits per-pair and aggregate markdown
comparison tables for the write-up.

Usage:
    python analyze.py --manifest manifest.jsonl [--db ~/.context-analyzer/analyzer.db]
                      [--ingest] [--out results.md]

Stdlib-only against the DB (sqlite3); --ingest optionally imports
context_tracker.ingest to pull sessions from ~/.claude/projects transcripts
first (requires the project venv).

Schema dependency (src/context_tracker/db.py):
    sessions:  total_input_tokens, total_output_tokens, total_cache_read,
               total_cache_creation, total_cost_usd, peak_context_tokens,
               total_api_calls
    blocks:    block_type, label  (retrieval round-trips = tool_call blocks
               whose label matches the headroom retrieval tool)

Cost source: the manifest's per-arm `cost_usd` (captured by run_pair.sh from the
`claude -p --output-format json` envelope's `total_cost_usd`) is authoritative —
it is model-correct. `sessions.total_cost_usd` is only a fallback: ingest_session
computes it with fixed Opus rates, so it is ~5x high for Sonnet and wrong for any
non-Opus model; a warning is emitted whenever the fallback is used.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

DEFAULT_DB = Path.home() / ".context-analyzer" / "analyzer.db"

# blocks.label for a tool_call starts with the tool name (see
# ccscope/parse_transcript.py::_tool_call_label). Headroom's retrieval tool is
# exposed via MCP, so the label contains e.g. "headroom_retrieve" or
# "mcp__headroom__retrieve" — the default pattern matches both.
DEFAULT_RETRIEVAL_PATTERN = "%headroom%retriev%"

METRICS: list[tuple[str, str]] = [
    # (key, table header)
    ("input_tokens", "Input tok"),
    ("output_tokens", "Output tok"),
    ("cache_read", "Cache read"),
    ("cache_creation", "Cache creation"),
    ("cache_hit_rate", "Cache hit %"),
    ("cost_usd", "Cost $"),
    ("peak_context", "Peak ctx"),
    ("api_calls", "API calls"),
    ("retrievals", "Retrievals"),
]


@dataclass
class SessionMetrics:
    session_id: str
    input_tokens: int
    output_tokens: int
    cache_read: int
    cache_creation: int
    cost_usd: float
    peak_context: int
    api_calls: int
    retrievals: int

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of prompt-side tokens served from the prefix cache."""
        denom = self.input_tokens + self.cache_read + self.cache_creation
        return self.cache_read / denom if denom else 0.0

    def get(self, key: str) -> float:
        if key == "cache_hit_rate":
            return self.cache_hit_rate
        return float(getattr(self, key))


@dataclass
class Pair:
    task_id: str
    pair: int
    plain: dict  # manifest row
    headroom: dict  # manifest row
    plain_metrics: SessionMetrics | None = None
    headroom_metrics: SessionMetrics | None = None

    @property
    def both_succeeded(self) -> bool:
        return bool(self.plain.get("success")) and bool(self.headroom.get("success"))

    @property
    def complete(self) -> bool:
        return self.plain_metrics is not None and self.headroom_metrics is not None


def load_manifest(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_pairs(rows: list[dict]) -> list[Pair]:
    """Group manifest rows into (task_id, pair) matched pairs."""
    grouped: dict[tuple[str, int], dict[str, dict]] = {}
    for row in rows:
        key = (row["task_id"], int(row["pair"]))
        grouped.setdefault(key, {})[row["arm"]] = row

    pairs = []
    for (task_id, pair_idx), arms in sorted(grouped.items()):
        if "plain" not in arms or "headroom" not in arms:
            print(f"WARNING: pair {task_id}#{pair_idx} is missing an arm; skipping", file=sys.stderr)
            continue
        pairs.append(Pair(task_id=task_id, pair=pair_idx, plain=arms["plain"], headroom=arms["headroom"]))
    return pairs


def fetch_session_metrics(
    conn: sqlite3.Connection,
    session_id: str,
    retrieval_pattern: str = DEFAULT_RETRIEVAL_PATTERN,
) -> SessionMetrics | None:
    row = conn.execute(
        """
        SELECT total_input_tokens, total_output_tokens, total_cache_read,
               total_cache_creation, total_cost_usd, peak_context_tokens,
               total_api_calls
        FROM sessions WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    if row is None:
        return None

    (retrievals,) = conn.execute(
        """
        SELECT COUNT(*) FROM blocks
        WHERE session_id = ? AND block_type = 'tool_call'
          AND lower(label) LIKE ?
        """,
        (session_id, retrieval_pattern),
    ).fetchone()

    return SessionMetrics(
        session_id=session_id,
        input_tokens=row[0] or 0,
        output_tokens=row[1] or 0,
        cache_read=row[2] or 0,
        cache_creation=row[3] or 0,
        cost_usd=row[4] or 0.0,
        peak_context=row[5] or 0,
        api_calls=row[6] or 0,
        retrievals=retrievals,
    )


def _apply_manifest_cost(pair: Pair, arm: str, row: dict, metrics: SessionMetrics | None) -> None:
    """Prefer the model-correct manifest cost over the DB's fixed-rate cost.

    run_pair.sh records `cost_usd` from the `claude -p --output-format json`
    envelope (`total_cost_usd`), which uses the actual model's rates. The DB's
    sessions.total_cost_usd is computed by ingest_session with fixed Opus rates
    and is wrong for other models (~5x high for Sonnet), so it is only used as
    a fallback, with an explicit warning.
    """
    if metrics is None:
        return
    cost = row.get("cost_usd")
    if cost is not None:
        metrics.cost_usd = float(cost)
    else:
        print(
            f"WARNING: {pair.task_id}#{pair.pair} arm={arm}: no cost_usd in manifest; "
            "falling back to DB sessions.total_cost_usd, which is computed with fixed "
            "Opus rates and is wrong for other models (~5x high for Sonnet). "
            "Re-run run_pair.sh (it captures total_cost_usd from the print-mode JSON "
            "envelope) for model-correct costs.",
            file=sys.stderr,
        )


def attach_metrics(pairs: list[Pair], db_path: Path, retrieval_pattern: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        for p in pairs:
            p.plain_metrics = fetch_session_metrics(conn, p.plain["session_id"], retrieval_pattern)
            p.headroom_metrics = fetch_session_metrics(conn, p.headroom["session_id"], retrieval_pattern)
            for arm, row, m in (("plain", p.plain, p.plain_metrics), ("headroom", p.headroom, p.headroom_metrics)):
                if m is None:
                    print(
                        f"WARNING: session for {p.task_id}#{p.pair} arm={arm} not in DB "
                        f"(id={row['session_id']}); run with --ingest or ingest manually",
                        file=sys.stderr,
                    )
                else:
                    _apply_manifest_cost(p, arm, row, m)
    finally:
        conn.close()


def _fmt(key: str, value: float) -> str:
    if key == "cost_usd":
        return f"{value:.4f}"
    if key == "cache_hit_rate":
        return f"{value * 100:.1f}"
    return f"{value:,.0f}"


def _pct_delta(plain: float, headroom: float) -> str:
    if plain == 0:
        return "n/a"
    return f"{(headroom - plain) / plain * 100:+.1f}%"


def render_per_pair_table(pairs: list[Pair]) -> str:
    """Per-pair metric rows — only for pairs where BOTH arms passed success_check.

    METHODOLOGY.md: token comparisons are only reported when both arms completed
    the task. Pairs with divergent outcomes appear only in the outcome-parity
    failures section, never as metric rows.
    """
    lines = [
        "## Per-pair results",
        "",
        "Only pairs where both arms passed success_check are shown "
        "(see Outcome-parity failures below for the rest).",
        "",
        "| Task | Pair | Metric | Plain | Headroom | Delta | Delta % |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for p in pairs:
        if not p.complete or not p.both_succeeded:
            continue
        assert p.plain_metrics is not None and p.headroom_metrics is not None
        for key, header in METRICS:
            pv = p.plain_metrics.get(key)
            hv = p.headroom_metrics.get(key)
            delta = hv - pv
            lines.append(
                f"| {p.task_id} | {p.pair} | {header} | {_fmt(key, pv)} | {_fmt(key, hv)} "
                f"| {_fmt(key, delta) if key != 'cost_usd' else f'{delta:+.4f}'} "
                f"| {_pct_delta(pv, hv)} |"
            )
    return "\n".join(lines)


def render_parity_failures(pairs: list[Pair]) -> str:
    """Pairs where either arm failed success_check — excluded from all metrics."""
    failed = [p for p in pairs if not p.both_succeeded]
    lines = ["## Outcome-parity failures", ""]
    if not failed:
        lines.append("None — both arms passed success_check in every pair.")
        return "\n".join(lines)
    lines += [
        "These pairs are excluded from every metric table and aggregate "
        "(METHODOLOGY.md: savings on a failed task are worthless). They are a "
        "finding in their own right — outcome divergence between arms.",
        "",
        "| Task | Pair | Plain success | Headroom success |",
        "| --- | --- | :---: | :---: |",
    ]

    def _s(v: object) -> str:
        if v is True:
            return "yes"
        if v is False:
            return "NO"
        return "unknown"

    for p in failed:
        lines.append(f"| {p.task_id} | {p.pair} | {_s(p.plain.get('success'))} | {_s(p.headroom.get('success'))} |")
    return "\n".join(lines)


def render_aggregate_table(pairs: list[Pair]) -> str:
    """Aggregate over pairs where both arms succeeded AND both sessions are in the DB."""
    usable = [p for p in pairs if p.complete and p.both_succeeded]
    lines = [
        "## Aggregate (pairs with outcome parity only)",
        "",
        f"Pairs total: {len(pairs)} | usable (both arms succeeded, both ingested): {len(usable)}",
        "",
    ]
    parity_fail = [p for p in pairs if not p.both_succeeded]
    if parity_fail:
        detail = ", ".join(f"{p.task_id}#{p.pair}" for p in parity_fail)
        lines.append(f"Outcome-parity failures (excluded): {detail}")
        lines.append("")
    if not usable:
        lines.append("No usable pairs — nothing to aggregate.")
        return "\n".join(lines)

    lines += [
        "| Metric | Mean plain | Mean headroom | Mean delta | Mean delta % | Headroom lower / higher / tied |",
        "| --- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for key, header in METRICS:
        pvs = [p.plain_metrics.get(key) for p in usable]  # type: ignore[union-attr]
        hvs = [p.headroom_metrics.get(key) for p in usable]  # type: ignore[union-attr]
        deltas = [h - v for v, h in zip(pvs, hvs, strict=True)]
        pct = [(h - v) / v * 100 for v, h in zip(pvs, hvs, strict=True) if v]
        lower = sum(1 for d in deltas if d < 0)
        higher = sum(1 for d in deltas if d > 0)
        tied = len(deltas) - lower - higher
        lines.append(
            f"| {header} | {_fmt(key, mean(pvs))} | {_fmt(key, mean(hvs))} "
            f"| {_fmt(key, mean(deltas)) if key != 'cost_usd' else f'{mean(deltas):+.4f}'} "
            f"| {f'{mean(pct):+.1f}%' if pct else 'n/a'} "
            f"| {lower} / {higher} / {tied} |"
        )

    lines += [
        "",
        "Sign counts are the honest summary at small n; do not report p-values below "
        "n = 10 pairs (see METHODOLOGY.md).",
    ]
    return "\n".join(lines)


def render_report(pairs: list[Pair]) -> str:
    return "\n\n".join(
        [
            "# Headroom experiment results",
            render_per_pair_table(pairs),
            render_parity_failures(pairs),
            render_aggregate_table(pairs),
        ]
    )


def ingest_sessions(rows: list[dict], db_path: Path) -> None:
    """Ingest manifest sessions from ~/.claude/projects transcripts (best effort)."""
    try:
        from context_tracker.ingest import ingest_session
    except ImportError:
        print(
            "ERROR: --ingest requires the project venv (run `make install-dev` at repo "
            "root and use .venv/bin/python)",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    for row in rows:
        sid = row["session_id"]
        if sid in ("dry-run", "unknown"):
            continue
        result = ingest_session(sid, db_path=db_path)
        status = "ok" if result is not None else "NOT FOUND"
        print(f"ingest {sid}: {status}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path(__file__).parent / "manifest.jsonl")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB, help="context-analyzer SQLite DB path")
    ap.add_argument("--ingest", action="store_true", help="ingest manifest sessions into the DB first")
    ap.add_argument("--retrieval-pattern", default=DEFAULT_RETRIEVAL_PATTERN, help="SQL LIKE pattern for the headroom retrieval tool label")
    ap.add_argument("--out", type=Path, default=None, help="write markdown here as well as stdout")
    args = ap.parse_args(argv)

    if not args.manifest.exists():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    rows = load_manifest(args.manifest)
    if args.ingest:
        ingest_sessions(rows, args.db)
    if not args.db.exists():
        print(f"ERROR: DB not found: {args.db}", file=sys.stderr)
        return 1

    pairs = build_pairs(rows)
    attach_metrics(pairs, args.db, args.retrieval_pattern)
    report = render_report(pairs)
    print(report)
    if args.out:
        args.out.write_text(report + "\n", encoding="utf-8")
        print(f"\nwritten to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
