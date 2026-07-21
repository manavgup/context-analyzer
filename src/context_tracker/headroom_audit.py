"""Headroom ceiling audit — retrospective compression replay (#94).

Replays the full content of REAL historical Claude Code sessions through
headroom's actual local compressors — offline, zero API calls — and produces
a defensible ceiling report: the maximum headroom could have saved on this
workload. Exposed as ``context-tracker audit-headroom`` so anyone can
reproduce the audit on their own transcripts with one command.

Methodology (residency-weighted):
    A compressed tool output does not save tokens once; it saves them on
    EVERY subsequent API call it stays resident in. For each tool_result
    block we compute:

        saved_token_calls(block) =
            block_tokens_db x compression_ratio x residency_calls(block)

    where residency_calls = (exit_turn or total_api_calls) - enter_turn,
    block_tokens_db is the API-usage-derived token size of the block from
    the analyzer DB, and compression_ratio is measured with a neutral
    tokenizer (tiktoken o200k_base) on original vs compressed text.
    Applying the *relative* ratio to DB-derived absolute tokens keeps the
    numerator and denominator in the same units (API-reported tokens) and
    makes the result robust to tokenizer choice.

    ceiling_% (session) = saved_token_calls / total resident token-call
    volume, where the denominator is input + cache_read + cache_creation
    summed over all API calls (each call's full resident context, counted
    once per call) — straight from the DB's sessions table.

Dollarization: the DB's total_cost_usd is computed at fixed Opus rates
(input $15/M, output $75/M, cache_read $1.875/M, cache_creation $18.75/M —
see context_tracker/ingest.py). We derive each session's input-side cost
share from the DB's token columns at those same fixed rates and apply the
ceiling percentage to it. No fresh rate assumptions are introduced.

This is an UPPER BOUND by construction: zero retrieval clawback (CCR
round-trips), zero prompt-cache damage from rewritten prefixes, zero answer
degradation, and compression applied from turn 0.

READ-ONLY: any analyzer DB handed to :func:`run_audit` is opened with a
mode=ro sqlite URI; transcripts are never modified. The one-command path
(:func:`run_audit_headroom`) builds its OWN scratch corpus DB and never
writes to the live analyzer DB.

Dependencies: stdlib + tiktoken + headroom-ai (both imported lazily, so the
module itself — and its tests, which inject fakes — stay hermetic).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import re
import sqlite3
import sys
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from context_tracker.db import DEFAULT_DB_DIR, DEFAULT_DB_PATH

logger = logging.getLogger("headroom_audit")

# Fixed rates ($ per Mtok) — MUST match context_tracker/ingest.py so the
# dollarization is internally consistent with the DB's recorded costs.
RATE_INPUT = 15.0
RATE_OUTPUT = 75.0
RATE_CACHE_READ = 1.875
RATE_CACHE_CREATION = 18.75

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

# Canonical tool buckets for the by-tool breakdown.
_TOOL_BUCKETS = (
    "Bash",
    "Read",
    "Grep",
    "Glob",
    "Edit",
    "Write",
    "WebFetch",
    "WebSearch",
    "Task",
    "other",
)

# Per-item compression ratios outside this range indicate a bug (or a
# pathological item) and are logged for investigation.
SUSPICIOUS_RATIO = 0.95

# One-command reproduction (audit-headroom subcommand) constants.
HEADROOM_PIN = "0.32.1"
DEFAULT_REPORT_PATH = Path("headroom-ceiling-report.md")
KEEP_DB_DIR = DEFAULT_DB_DIR / "audit-headroom"
KOMPRESS_DOWNLOAD_MB = 261  # one-time HuggingFace model download (--profile max)

_INSTALL_INSTRUCTIONS = f"""\
audit-headroom needs two extra packages that are NOT installed by default
(they are optional — nothing else in context-tracker requires them):

    pip install --no-deps headroom-ai=={HEADROOM_PIN} && pip install tiktoken

Why --no-deps: headroom's litellm dependency has no wheels on macOS and is
only needed for headroom's proxy paths, which this offline audit never uses.

Then re-run:  context-tracker audit-headroom
"""

_ONNX_INSTRUCTIONS = """\
--profile max additionally needs onnxruntime for the Kompress ML prose model:

    pip install onnxruntime

Then re-run:  context-tracker audit-headroom --profile max
"""


# ---------------------------------------------------------------------------
# Transcript parsing (full content — the DB only stores 500-char previews)
# ---------------------------------------------------------------------------


def iter_transcript_items(transcript_path: Path) -> tuple[dict[str, dict], dict[str, str]]:
    """Extract full tool_result content and tool_use metadata from a transcript.

    Returns:
        tool_use_map: tool_use_id -> {"name": str, "input": dict}
        results: tool_use_id -> full text content of the tool_result
    """
    tool_use_map: dict[str, dict] = {}
    results: dict[str, str] = {}

    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = entry.get("type", "")
            if etype == "assistant":
                for blk in entry.get("message", {}).get("content", []) or []:
                    if isinstance(blk, dict) and blk.get("type") == "tool_use":
                        tid = blk.get("id", "")
                        if tid and tid not in tool_use_map:
                            tool_use_map[tid] = {
                                "name": blk.get("name", ""),
                                "input": blk.get("input", {}),
                            }
            elif etype == "user":
                content = entry.get("message", {}).get("content", "")
                if not isinstance(content, list):
                    continue
                for blk in content:
                    if not isinstance(blk, dict) or blk.get("type") != "tool_result":
                        continue
                    tid = blk.get("tool_use_id", "")
                    if not tid:
                        continue
                    results[tid] = _tool_result_text(blk.get("content", ""))

    return tool_use_map, results


def _tool_result_text(content: Any) -> str:
    """Flatten a tool_result content field (str or list of blocks) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for sub in content:
            if isinstance(sub, dict) and sub.get("type") == "text":
                parts.append(sub.get("text", ""))
        return "\n".join(parts)
    return str(content) if content is not None else ""


def find_transcripts(projects_dir: Path) -> dict[str, Path]:
    """Map session_id -> transcript path, skipping subagent transcripts."""
    out: dict[str, Path] = {}
    if not projects_dir.exists():
        return out
    for f in projects_dir.rglob("*.jsonl"):
        if "subagents" in f.parts:
            continue
        if _UUID_RE.match(f.stem):
            out[f.stem] = f
    return out


# ---------------------------------------------------------------------------
# Content classification (simple heuristics; numbers-only reporting)
# ---------------------------------------------------------------------------

_CODE_TOKENS = (
    "def ",
    "class ",
    "import ",
    "function ",
    "const ",
    "return ",
    "#include",
    "=> ",
    "});",
    "public ",
    "fn ",
)
_LOG_LINE_RE = re.compile(r"^\s*(\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}|INFO|WARN|ERROR|DEBUG|TRACE|\[\d+\]|\S+\.\w+:\d+)")


def classify_content(text: str) -> str:
    """Classify content as json / code / log / prose / other via heuristics."""
    s = text.strip()
    if not s:
        return "other"
    if s[0] in "[{":
        try:
            json.loads(s)
            return "json"
        except (json.JSONDecodeError, ValueError):
            pass
    lines = s.splitlines()
    if len(lines) >= 5:
        log_like = sum(1 for ln in lines[:50] if _LOG_LINE_RE.match(ln))
        if log_like >= max(3, len(lines[:50]) // 2):
            return "log"
    code_hits = sum(1 for tok in _CODE_TOKENS if tok in s)
    semi_density = s.count(";") / max(1, len(lines))
    indent_lines = sum(1 for ln in lines[:80] if ln.startswith(("    ", "\t")))
    if code_hits >= 2 or semi_density > 0.5 or (len(lines) > 4 and indent_lines > len(lines[:80]) // 2):
        return "code"
    alpha = sum(c.isalpha() or c.isspace() for c in s[:2000]) / max(1, len(s[:2000]))
    if alpha > 0.85:
        return "prose"
    return "other"


def bucket_tool(name: str) -> str:
    """Map a tool name onto one of the canonical report buckets."""
    if name in _TOOL_BUCKETS:
        return name
    if name in ("Task", "Agent", "dispatch_agent"):
        return "Task"
    return "other"


# ---------------------------------------------------------------------------
# Compressor / tokenizer factories (lazy imports; injectable for tests)
# ---------------------------------------------------------------------------


def make_tiktoken_counter() -> Callable[[str], int]:
    """Neutral token counter (tiktoken o200k_base)."""
    import tiktoken

    enc = tiktoken.get_encoding("o200k_base")

    def count(text: str) -> int:
        return len(enc.encode(text, disallowed_special=()))

    return count


def make_headroom_compressor(
    workspace_dir: Path | None = None, profile: str = "defaults"
) -> Callable[..., tuple[str, list[str]]]:
    """Build the real headroom compressor function.

    Returns fn(text, tool_name, tool_input) -> (compressed_text, transforms).

    Profiles:
      - "defaults": headroom's shipped-default local pipeline (Kompress ML
        prose model disabled, AST code compression off — both are headroom's
        own defaults on the library path).
      - "max": maximum capability. Loads the Kompress ML prose model
        (chopratejas/kompress-v2-base, one-time HuggingFace download at
        headroom's own pinned revision) and enables AST code compression via
        ContentRouterConfig(enable_code_aware=True). The library path exposes
        no public switch for enable_code_aware (CompressConfig doesn't carry
        it; HEADROOM_CODE_AWARE_ENABLED is read only by the proxy), so the
        singleton pipeline's ContentRouter is replaced with an identically
        built one whose config sets enable_code_aware=True.

    Environment hardening (all documented in the report):
      - HEADROOM_OFFLINE=1: no egress (telemetry, update check, HF
        downloads). In the "max" profile this is set immediately AFTER the
        one-time model load, so the audit itself still runs with zero egress.
      - HEADROOM_DETECT_BACKEND=python: the native Magika/ONNX content
        detector deadlocks on this macOS host; headroom's own escape hatch
        routes to its pure-Python regex detector.
      - HEADROOM_WORKSPACE_DIR: keep headroom's runtime state (CCR store)
        out of ~/.headroom so the audit leaves no traces on the machine.
      - Kompress canary/time-budget knobs (max profile only): the shipped 5s
        startup canary and 20s per-call budget protect a live proxy from a
        slow model; in an offline batch audit they would silently turn slow
        items into passthroughs, so they are relaxed.
    """
    if profile not in ("defaults", "max"):
        raise ValueError(f"unknown profile: {profile!r}")
    os.environ.setdefault("HEADROOM_UPDATE_CHECK", "off")
    os.environ.setdefault("HEADROOM_DETECT_BACKEND", "python")
    if workspace_dir is not None:
        workspace_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HEADROOM_WORKSPACE_DIR", str(workspace_dir))
        os.environ.setdefault("HEADROOM_CONFIG_DIR", str(workspace_dir / "config"))

    if profile == "max":
        os.environ.setdefault("HEADROOM_KOMPRESS_CANARY_SECONDS", "0")
        os.environ.setdefault("HEADROOM_KOMPRESS_TIME_BUDGET_SECONDS", "600")
        os.environ.setdefault("HEADROOM_KOMPRESS_ACQUIRE_TIMEOUT_SECONDS", "600")

        from headroom.transforms.kompress_compressor import HF_MODEL_ID, _load_kompress

        t0 = time.time()
        _model, _tok, backend = _load_kompress(HF_MODEL_ID)
        logger.info("kompress loaded: model=%s backend=%s (%.1fs)", HF_MODEL_ID, backend, time.time() - t0)

    # From here on: no egress. (In the max profile the model is now cached.)
    os.environ.setdefault("HEADROOM_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from headroom import CompressConfig, compress

    if profile == "max":
        import sys as _sys

        from headroom.transforms.content_router import ContentRouter, ContentRouterConfig

        _hc_module = _sys.modules["headroom.compress"]
        pipeline = _hc_module._get_pipeline()
        pipeline.transforms = [
            ContentRouter(ContentRouterConfig(enable_code_aware=True)) if type(t).__name__ == "ContentRouter" else t
            for t in pipeline.transforms
        ]
        # protect_recent=0: we compress each historical item in isolation; by
        # the time residency matters the item is no longer "recent".
        # kompress_model=None -> headroom's default ML prose model.
        cfg = CompressConfig(protect_recent=0)
    else:
        # protect_recent=0: as above.
        # kompress_model="disabled": skips the ML prose compressor (would
        # require a HuggingFace model download; prose compression is NOT
        # simulated in this profile).
        # Everything else is headroom's shipped default (including its own
        # defaults of protecting file Reads verbatim and disabling AST code
        # compression).
        cfg = CompressConfig(protect_recent=0, kompress_model="disabled")

    def compress_item(text: str, tool_name: str, tool_input: dict) -> tuple[str, list[str]]:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "continue the task"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_retro",
                        "name": tool_name or "Bash",
                        "input": tool_input or {},
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_retro", "content": text}],
            },
        ]
        result = compress(messages, model="claude-sonnet-4-5-20250929", config=cfg)
        compressed = _extract_compressed(result.messages, fallback=text)
        return compressed, list(result.transforms_applied)

    return compress_item


def _extract_compressed(messages: list[dict], fallback: str) -> str:
    """Pull the (possibly compressed) tool_result text back out of messages."""
    for msg in reversed(messages):
        content = msg.get("content")
        if isinstance(content, list):
            for blk in content:
                if isinstance(blk, dict) and blk.get("type") == "tool_result":
                    return _tool_result_text(blk.get("content", ""))
        elif isinstance(content, str) and msg.get("role") in ("user", "tool"):
            return content
    return fallback


# ---------------------------------------------------------------------------
# DB access (READ-ONLY)
# ---------------------------------------------------------------------------


def open_db_readonly(db_path: Path) -> sqlite3.Connection:
    """Open an analyzer-style SQLite DB with a mode=ro URI (writes fail)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def load_sessions(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """All Claude Code sessions, most expensive first."""
    return conn.execute(
        """
        SELECT session_id, total_api_calls, total_input_tokens,
               total_output_tokens, total_cache_read, total_cache_creation,
               total_cost_usd, source_mtime
        FROM sessions
        WHERE agent = 'claude-code'
        ORDER BY total_cost_usd DESC
        """
    ).fetchall()


def load_tool_result_blocks(conn: sqlite3.Connection, session_id: str) -> list[sqlite3.Row]:
    """All tool_result blocks for one session."""
    return conn.execute(
        """
        SELECT block_id, tokens, enter_turn, exit_turn
        FROM blocks
        WHERE session_id = ? AND block_type = 'tool_result'
        """,
        (session_id,),
    ).fetchall()


def tool_use_id_from_block(block_id: str) -> str | None:
    """block_id format: t{enter}-tool_result-{tool_use_id}."""
    marker = "tool_result-"
    idx = block_id.find(marker)
    if idx < 0:
        return None
    return block_id[idx + len(marker) :] or None


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


@dataclass
class GroupStat:
    """Aggregate for one content-type or tool bucket."""

    items: int = 0
    orig_tokens: int = 0  # DB-derived block tokens (one-shot, not residency-weighted)
    saved_tokens: int = 0  # one-shot saved tokens (DB units)
    saved_token_calls: float = 0.0  # residency-weighted
    resident_token_calls: float = 0.0  # residency-weighted volume of these blocks

    @property
    def ratio(self) -> float:
        return self.saved_tokens / self.orig_tokens if self.orig_tokens else 0.0


@dataclass
class SessionResult:
    """Per-session audit outcome."""

    session_id: str
    date: str
    api_calls: int
    recorded_cost: float
    input_side_cost: float
    resident_token_calls: float  # denominator: input + cache_read + cache_creation
    saved_token_calls: float = 0.0
    items_total: int = 0
    items_compressed: int = 0  # ratio > 0
    items_failed: int = 0  # compressor raised -> counted incompressible
    items_missing_content: int = 0  # DB block with no transcript content
    items_fallback_residency: int = 0  # transcript item missing from DB blocks
    suspicious_items: int = 0  # per-item ratio > SUSPICIOUS_RATIO

    @property
    def ceiling_pct(self) -> float:
        if self.resident_token_calls <= 0:
            return 0.0
        return min(1.0, self.saved_token_calls / self.resident_token_calls)

    @property
    def ceiling_usd(self) -> float:
        return self.ceiling_pct * self.input_side_cost


@dataclass
class CorpusResult:
    """Whole-corpus audit outcome."""

    headroom_version: str
    tokenizer_name: str
    profile: str = "defaults"
    sessions: list[SessionResult] = field(default_factory=list)
    by_content: dict[str, GroupStat] = field(default_factory=lambda: defaultdict(GroupStat))
    by_tool: dict[str, GroupStat] = field(default_factory=lambda: defaultdict(GroupStat))
    transforms_seen: Counter = field(default_factory=Counter)
    wall_time_s: float = 0.0
    unique_items_compressed: int = 0

    @property
    def total_resident_token_calls(self) -> float:
        return sum(s.resident_token_calls for s in self.sessions)

    @property
    def total_saved_token_calls(self) -> float:
        return sum(s.saved_token_calls for s in self.sessions)

    @property
    def overall_ceiling_pct(self) -> float:
        denom = self.total_resident_token_calls
        return (self.total_saved_token_calls / denom) if denom > 0 else 0.0

    @property
    def total_recorded_cost(self) -> float:
        return sum(s.recorded_cost for s in self.sessions)

    @property
    def total_input_side_cost(self) -> float:
        return sum(s.input_side_cost for s in self.sessions)

    @property
    def total_ceiling_usd(self) -> float:
        return sum(s.ceiling_usd for s in self.sessions)


def input_side_cost(row: sqlite3.Row) -> float:
    """Input-side share of the recorded cost, at the same fixed rates as ingest."""
    return (
        float(
            row["total_input_tokens"] * RATE_INPUT
            + row["total_cache_read"] * RATE_CACHE_READ
            + row["total_cache_creation"] * RATE_CACHE_CREATION
        )
        / 1e6
    )


def analyze_session(
    session_row: sqlite3.Row,
    transcript_path: Path,
    conn: sqlite3.Connection,
    compress_fn: Callable[..., tuple[str, list[str]]],
    count_fn: Callable[[str], int],
    corpus: CorpusResult,
    cache: dict[str, tuple[float, list[str]]],
) -> SessionResult:
    """Run the residency-weighted compression model for one session."""
    sid = session_row["session_id"]
    total_api_calls = session_row["total_api_calls"] or 0
    mtime = session_row["source_mtime"] or 0
    date = datetime.fromtimestamp(mtime, tz=UTC).strftime("%Y-%m-%d") if mtime else "?"

    result = SessionResult(
        session_id=sid,
        date=date,
        api_calls=total_api_calls,
        recorded_cost=session_row["total_cost_usd"] or 0.0,
        input_side_cost=input_side_cost(session_row),
        resident_token_calls=float(
            (session_row["total_input_tokens"] or 0)
            + (session_row["total_cache_read"] or 0)
            + (session_row["total_cache_creation"] or 0)
        ),
    )

    tool_use_map, contents = iter_transcript_items(transcript_path)
    blocks = load_tool_result_blocks(conn, sid)

    matched_ids: set[str] = set()
    residencies: list[float] = []

    # Pass 1: blocks that join cleanly to transcript content.
    joined: list[tuple[str, int, float]] = []  # (tool_use_id, block_tokens, residency)
    for blk in blocks:
        tid = tool_use_id_from_block(blk["block_id"])
        if tid is None:
            continue
        enter = blk["enter_turn"]
        if enter is None:
            continue
        exit_turn = blk["exit_turn"] if blk["exit_turn"] is not None else total_api_calls
        residency = max(0.0, float(exit_turn) - float(enter))
        if tid not in contents:
            result.items_missing_content += 1
            continue
        matched_ids.add(tid)
        residencies.append(residency)
        joined.append((tid, int(blk["tokens"] or 0), residency))

    # Pass 2 (stated fallback): transcript tool_results absent from DB blocks
    # get the session-average residency and a tokenizer-estimated size.
    avg_residency = sum(residencies) / len(residencies) if residencies else total_api_calls / 2.0
    for tid in contents:
        if tid in matched_ids:
            continue
        result.items_fallback_residency += 1
        joined.append((tid, -1, avg_residency))  # -1 -> size via count_fn

    for tid, block_tokens, residency in joined:
        text = contents[tid]
        result.items_total += 1
        if not text:
            continue

        info = tool_use_map.get(tid, {})
        tool = bucket_tool(info.get("name", ""))
        ctype = classify_content(text)

        key = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
        if key in cache:
            ratio, transforms = cache[key]
        else:
            try:
                orig_tok = count_fn(text)
                if orig_tok <= 0:
                    ratio, transforms = 0.0, []
                else:
                    compressed_text, transforms = compress_fn(text, info.get("name", ""), info.get("input", {}))
                    comp_tok = count_fn(compressed_text)
                    ratio = 1.0 - (comp_tok / orig_tok)
            except Exception as exc:  # compressor failure -> incompressible
                logger.warning("compressor failed on item %s...: %s", tid[:12], type(exc).__name__)
                result.items_failed += 1
                ratio, transforms = 0.0, ["error"]
            if ratio < 0.0:
                ratio = 0.0  # inflation -> would never be applied
            corpus.unique_items_compressed += 1
            cache[key] = (ratio, transforms)

        if ratio > SUSPICIOUS_RATIO:
            result.suspicious_items += 1
            logger.warning(
                "suspicious ratio %.3f on item %s... (%d chars, tool=%s type=%s)",
                ratio,
                tid[:12],
                len(text),
                tool,
                ctype,
            )

        if block_tokens < 0:  # fallback item: estimate size with the tokenizer
            block_tokens = count_fn(text)

        saved_tokens = block_tokens * ratio
        saved_tc = saved_tokens * residency
        resident_tc = block_tokens * residency

        result.saved_token_calls += saved_tc
        if ratio > 0:
            result.items_compressed += 1

        for group, keyname in ((corpus.by_content, ctype), (corpus.by_tool, tool)):
            g = group[keyname]
            g.items += 1
            g.orig_tokens += block_tokens
            g.saved_tokens += int(saved_tokens)
            g.saved_token_calls += saved_tc
            g.resident_token_calls += resident_tc

        for t in transforms:
            corpus.transforms_seen[t] += 1

    return result


def run_audit(
    db_path: Path,
    projects_dir: Path,
    limit: int | None = None,
    compress_fn: Callable[..., tuple[str, list[str]]] | None = None,
    count_fn: Callable[[str], int] | None = None,
    headroom_version: str = "",
    profile: str = "defaults",
) -> CorpusResult:
    """Replay every session in db_path through the compressor and aggregate."""
    t0 = time.time()
    if compress_fn is None:
        compress_fn = make_headroom_compressor(
            workspace_dir=Path(db_path).parent / "headroom-workspace", profile=profile
        )
        if not headroom_version:
            import headroom

            headroom_version = getattr(headroom, "__version__", "unknown")
    if count_fn is None:
        count_fn = make_tiktoken_counter()

    corpus = CorpusResult(
        headroom_version=headroom_version or "injected-fake",
        tokenizer_name="tiktoken o200k_base",
        profile=profile,
    )

    conn = open_db_readonly(db_path)
    try:
        sessions = load_sessions(conn)
        transcripts = find_transcripts(projects_dir)
        cache: dict[str, tuple[float, list[str]]] = {}

        processed = 0
        for row in sessions:
            if limit is not None and processed >= limit:
                break
            path = transcripts.get(row["session_id"])
            if path is None:
                logger.info("no transcript on disk for %s... — skipped", row["session_id"][:8])
                continue
            logger.info("session %s... (%s api calls)", row["session_id"][:8], row["total_api_calls"])
            corpus.sessions.append(analyze_session(row, path, conn, compress_fn, count_fn, corpus, cache))
            processed += 1
    finally:
        conn.close()

    corpus.wall_time_s = time.time() - t0
    return corpus


# ---------------------------------------------------------------------------
# Report generation — NUMBERS ONLY. No prompts, paths, or content strings.
# ---------------------------------------------------------------------------


def _fmt_tok(n: float) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return f"{n:.0f}"


def build_report(corpus: CorpusResult) -> str:
    """Render the ceiling report. Contains numbers only — zero content text."""
    s = corpus
    total_items = sum(sess.items_total for sess in s.sessions)
    failed = sum(sess.items_failed for sess in s.sessions)
    missing = sum(sess.items_missing_content for sess in s.sessions)
    fallback = sum(sess.items_fallback_residency for sess in s.sessions)
    suspicious = sum(sess.suspicious_items for sess in s.sessions)
    kill = s.overall_ceiling_pct < 0.10

    lines: list[str] = []
    a = lines.append
    a("# Retrospective Compression Audit — Ceiling Report (Stage 0, #94)")
    a("")
    a(f"- headroom-ai version: **{s.headroom_version}**")
    if s.profile == "max":
        a(
            "- Configuration profile: **maximum capability** — Kompress ML prose "
            "model (chopratejas/kompress-v2-base) active + AST code compression "
            "enabled (ContentRouterConfig(enable_code_aware=True))"
        )
    else:
        a(
            "- Configuration profile: **shipped defaults** — Kompress ML prose "
            "model disabled, AST code compression off (headroom's own defaults)"
        )
    a(
        f"- Tokenizer for ratios: {s.tokenizer_name} (relative ratios are primary; "
        "absolute token units come from the analyzer DB's API-usage-derived block sizes)"
    )
    a(f"- Sessions analyzed: **{len(s.sessions)}**")
    a(f"- tool_result items analyzed: **{total_items}** (unique contents compressed: {s.unique_items_compressed})")
    a(
        f"- Recorded corpus cost (fixed Opus rates): **${s.total_recorded_cost:,.2f}** "
        f"(input-side share: ${s.total_input_side_cost:,.2f})"
    )
    a(f"- Total resident token-call volume: {_fmt_tok(s.total_resident_token_calls)} tokens")
    a(f"- Wall time: {s.wall_time_s:,.1f}s")
    a("")
    a("## Headline")
    a("")
    a(
        f"**Ceiling: {s.overall_ceiling_pct * 100:.2f}% of resident token-call volume "
        f"= ${s.total_ceiling_usd:,.2f} of the recorded ${s.total_recorded_cost:,.2f} "
        "was theoretically compressible.**"
    )
    a("")
    a(
        f"Kill criterion (#94: ceiling < 10% of spend → publish and stop): "
        f"**{'MET — ceiling is below 10%' if kill else 'NOT met — ceiling is at/above 10%'}** "
        f"({s.overall_ceiling_pct * 100:.2f}%)."
    )
    a("")
    denom = s.total_resident_token_calls or 1.0
    tool_result_share = sum(g.resident_token_calls for g in s.by_content.values()) / denom * 100
    a(f"tool_result blocks account for {tool_result_share:.1f}% of resident token-call volume.")
    a("")
    a("## By content type")
    a("")
    a(
        "| Type | Items | Block tokens | One-shot ratio | Saved token-calls "
        "| Resident share | Saved share of corpus volume |"
    )
    a("|---|---:|---:|---:|---:|---:|---:|")
    for name, g in sorted(s.by_content.items(), key=lambda kv: -kv[1].saved_token_calls):
        a(
            f"| {name} | {g.items} | {_fmt_tok(g.orig_tokens)} | {g.ratio * 100:.1f}% "
            f"| {_fmt_tok(g.saved_token_calls)} | {g.resident_token_calls / denom * 100:.2f}% "
            f"| {g.saved_token_calls / denom * 100:.2f}% |"
        )
    a("")
    a("## By tool")
    a("")
    a(
        "| Tool | Items | Block tokens | One-shot ratio | Saved token-calls "
        "| Resident share | Saved share of corpus volume |"
    )
    a("|---|---:|---:|---:|---:|---:|---:|")
    for name, g in sorted(s.by_tool.items(), key=lambda kv: -kv[1].saved_token_calls):
        a(
            f"| {name} | {g.items} | {_fmt_tok(g.orig_tokens)} | {g.ratio * 100:.1f}% "
            f"| {_fmt_tok(g.saved_token_calls)} | {g.resident_token_calls / denom * 100:.2f}% "
            f"| {g.saved_token_calls / denom * 100:.2f}% |"
        )
    a("")
    a("## Top 10 most-compressible sessions")
    a("")
    a("| Date | Session | API calls | Recorded $ | Input-side $ | Ceiling % | Ceiling $ |")
    a("|---|---|---:|---:|---:|---:|---:|")
    for sess in sorted(s.sessions, key=lambda x: -x.ceiling_usd)[:10]:
        a(
            f"| {sess.date} | {sess.session_id[:8]} | {sess.api_calls} "
            f"| ${sess.recorded_cost:,.2f} | ${sess.input_side_cost:,.2f} "
            f"| {sess.ceiling_pct * 100:.2f}% | ${sess.ceiling_usd:,.2f} |"
        )
    a("")
    a("## Data quality")
    a("")
    a(f"- Items where the compressor errored (counted incompressible): {failed}")
    a(f"- DB blocks with no matching transcript content (skipped): {missing}")
    a(f"- Transcript items missing from DB blocks (session-average residency fallback): {fallback}")
    a(f"- Items with per-item ratio > {SUSPICIOUS_RATIO:.0%} (flagged): {suspicious}")
    a("")
    a("## Transforms applied (headroom routing)")
    a("")
    for t, n in s.transforms_seen.most_common(15):
        a(f"- `{t}`: {n}")
    a("")
    a("## Limitations — this is an upper bound by construction")
    a("")
    a(
        "- **Zero retrieval clawback assumed.** Headroom's SmartCrusher emits CCR retrieval "
        "markers; every retrieval in live use costs a round-trip that claws back savings. "
        "None of that cost is modeled here."
    )
    a(
        "- **Zero prompt-cache damage assumed.** Compression rewrites message content, which "
        "breaks prompt-cache prefixes. On this corpus most resident volume is cache reads at "
        "0.125x the input rate; recompressing a prefix converts cheap cache reads into "
        "full-price input tokens and cache re-writes. In live use this can consume a large "
        "share of the theoretical savings — it is not modeled here at all."
    )
    a(
        "- **Zero answer degradation assumed.** Compressed tool outputs may change agent "
        "behavior (extra retries, wrong edits); not modeled."
    )
    a("- **Compression applied from turn 0** of every session, the most favorable timing.")
    a(
        "- **Tokenizer approximation.** Compression ratios are measured with a neutral "
        "tokenizer (tiktoken o200k_base), not Anthropic's tokenizer. Relative ratios are "
        "robust to this choice; absolute token-call numbers inherit the DB's "
        "proportional-attribution block sizing."
    )
    prose_share = s.by_content["prose"].resident_token_calls / denom * 100
    if s.profile == "max":
        a(
            "- **Prose compression is lossy-by-model.** Kompress (ModernBERT) is a "
            "summarization-style token-drop compressor: the compressed prose is NOT "
            "reconstructible from what stays in context, unlike SmartCrusher's "
            "reversible CCR path (original stashed for retrieval). Every prose "
            "point in this ceiling therefore assumes the agent never needed the "
            "dropped tokens — fidelity loss is not modeled. "
            "AST code compression (enable_code_aware=True) was force-enabled for "
            "this run; headroom ships it OFF by default, so the code contribution "
            "counts a capability headroom itself does not enable. File Read "
            "outputs remain protected verbatim by headroom's policy."
        )
    else:
        a(
            "- **Prose ML compression not simulated.** Headroom's Kompress (ModernBERT) model "
            "requires a HuggingFace download; it was disabled (kompress_model='disabled'). "
            f"Sensitivity: prose tool_results are {prose_share:.1f}% of resident volume, so a "
            f"hypothetical prose ratio of R would add up to {prose_share:.1f}×R points to the "  # noqa: RUF001
            "ceiling (e.g. R=50% → +"
            f"{prose_share * 0.5:.1f} points). "
            "AST code compression is disabled by headroom's own default configuration "
            "(enable_code_aware=False) and was left at that default; headroom also protects "
            "file Read outputs verbatim by default. JSON (SmartCrusher) and log/search "
            "compression — headroom's headline compressors — ran as shipped."
        )
    a(
        "- **Native content detector bypassed.** Headroom's native Magika/ONNX detector "
        "deadlocked on this host; per headroom's own escape hatch, "
        "HEADROOM_DETECT_BACKEND=python (its pure-Python regex detector) was used."
    )
    a("- **Single machine, single user's workload.** No generalization claimed.")
    a("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# One-command reproduction: `context-tracker audit-headroom`
# ---------------------------------------------------------------------------


def _module_available(name: str) -> bool:
    """True if a module could be imported (without importing it)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def check_audit_dependencies(profile: str = "defaults") -> str | None:
    """Return exact install instructions if optional audit deps are missing.

    headroom-ai and tiktoken are deliberately NOT project dependencies: the
    audit is an optional, offline reproduction path, and headroom must be
    installed with --no-deps (its litellm dependency has no macOS wheels and
    is only needed for headroom's proxy paths, never touched here).
    """
    missing = [name for name in ("headroom", "tiktoken") if not _module_available(name)]
    if missing:
        return _INSTALL_INSTRUCTIONS
    if profile == "max" and not _module_available("onnxruntime"):
        return _ONNX_INSTRUCTIONS
    return None


def build_scratch_corpus(db_path: Path, projects_dir: Path) -> int:
    """Ingest every top-level Claude Code transcript into a scratch DB.

    Uses the existing ingest pipeline (same parsing and fixed rates as the
    live analyzer DB) but writes ONLY to db_path — never to the live DB.
    Returns the number of sessions ingested.
    """
    from context_tracker.ingest import ingest_session

    if db_path.resolve() == DEFAULT_DB_PATH.resolve():  # defense in depth
        raise ValueError("scratch corpus DB must not be the live analyzer DB")

    count = 0
    for f in sorted(projects_dir.rglob("*.jsonl")):
        if "subagents" in f.parts or not _UUID_RE.match(f.stem):
            continue
        try:
            if ingest_session(f.stem, db_path=db_path, projects_dir=projects_dir) is not None:
                count += 1
        except Exception as exc:  # one bad transcript must not sink the audit
            logger.warning("failed to ingest %s...: %s", f.stem[:8], type(exc).__name__)
    return count


def _default_confirm(prompt: str) -> bool:
    if not sys.stdin.isatty():
        return False
    answer = input(f"{prompt} [y/N] ")
    return answer.strip().lower() in ("y", "yes")


def _print_headline(corpus: CorpusResult, out: Path) -> None:
    """Print the numbers-only headline to stdout."""
    kill = corpus.overall_ceiling_pct < 0.10
    top_types = sorted(corpus.by_content.items(), key=lambda kv: -kv[1].saved_token_calls)[:3]
    total_saved = corpus.total_saved_token_calls or 1.0

    print()
    print(f"Headroom ceiling audit — profile: {corpus.profile} (headroom-ai {corpus.headroom_version})")
    print(f"Sessions analyzed: {len(corpus.sessions)}   wall time: {corpus.wall_time_s:,.1f}s   API spend: $0")
    print()
    print(
        f"  Ceiling: {corpus.overall_ceiling_pct * 100:.2f}% of resident token-call volume "
        f"= ${corpus.total_ceiling_usd:,.2f} of ${corpus.total_recorded_cost:,.2f} recorded spend"
    )
    print(
        f"  Kill criterion (#94: ceiling < 10% of spend): "
        f"{'MET — ceiling is below 10%' if kill else 'NOT met — ceiling is at/above 10%'}"
    )
    if top_types:
        shares = ", ".join(f"{name} ({g.saved_token_calls / total_saved * 100:.0f}%)" for name, g in top_types)
        print(f"  Top content types by saved token-calls: {shares}")
    print()
    print(f"Full report written to {out}")
    print("The report contains numbers only — no prompt text, file paths, or session content.")


def run_audit_headroom(
    profile: str = "defaults",
    out: Path = DEFAULT_REPORT_PATH,
    limit: int | None = None,
    keep_db: bool = False,
    yes: bool = False,
    projects_dir: Path | None = None,
    compress_fn: Callable[..., tuple[str, list[str]]] | None = None,
    count_fn: Callable[[str], int] | None = None,
    confirm_fn: Callable[[str], bool] | None = None,
) -> int:
    """One-command reproduction of the headroom ceiling audit.

    Builds a scratch corpus DB from the user's own transcripts (never
    touching the live analyzer DB), replays it through headroom's local
    compressors offline, prints the headline, and writes the full
    numbers-only markdown report to ``out``.

    compress_fn / count_fn / confirm_fn are injectable for hermetic tests.
    Returns a process exit code (0 = success).
    """
    if profile not in ("defaults", "max"):
        print(f"error: unknown profile {profile!r} (choose 'defaults' or 'max')", file=sys.stderr)
        return 2

    # 1. Dependency UX: exact install instructions, clean exit — only when
    #    the real compressor/tokenizer will actually be used.
    if compress_fn is None or count_fn is None:
        problem = check_audit_dependencies(profile)
        if problem is not None:
            print(problem, file=sys.stderr)
            return 1

    # 2. --profile max downloads a ~261MB HuggingFace model on first run:
    #    require explicit consent (--yes or interactive confirmation).
    if profile == "max" and compress_fn is None and not yes:
        prompt = (
            f"--profile max loads headroom's Kompress prose model — a one-time "
            f"~{KOMPRESS_DOWNLOAD_MB}MB HuggingFace download (cached afterwards; "
            f"the audit itself then runs fully offline). Continue?"
        )
        if not (confirm_fn or _default_confirm)(prompt):
            print(
                f"Aborted. Re-run with --yes to accept the one-time ~{KOMPRESS_DOWNLOAD_MB}MB "
                "model download, or use --profile defaults (no download needed).",
                file=sys.stderr,
            )
            return 1

    resolved_projects = projects_dir if projects_dir is not None else Path.home() / ".claude" / "projects"
    if not find_transcripts(resolved_projects):
        print(
            f"No Claude Code session transcripts found under {resolved_projects}.\n"
            "Run some Claude Code sessions first, then re-run the audit.",
            file=sys.stderr,
        )
        return 1

    # 3. Scratch corpus DB — a temp dir by default, ~/.context-analyzer/
    #    audit-headroom/ with --keep-db. The live analyzer DB is never written.
    tmp: tempfile.TemporaryDirectory[str] | None = None
    if keep_db:
        scratch_dir = KEEP_DB_DIR
        scratch_dir.mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.TemporaryDirectory(prefix="audit-headroom-")
        scratch_dir = Path(tmp.name)

    try:
        scratch_db = scratch_dir / "corpus.db"
        print(f"Building scratch corpus DB at {scratch_db} (your live analyzer DB is not touched)...")
        ingested = build_scratch_corpus(scratch_db, resolved_projects)
        if ingested == 0:
            print("No sessions could be ingested from your transcripts.", file=sys.stderr)
            return 1
        print(f"Ingested {ingested} session(s). Replaying through headroom's local compressors (offline)...")

        corpus = run_audit(
            scratch_db,
            resolved_projects,
            limit=limit,
            compress_fn=compress_fn,
            count_fn=count_fn,
            profile=profile,
        )

        report = build_report(corpus)
        out.write_text(report, encoding="utf-8")
        _print_headline(corpus, out)
        if keep_db:
            print(f"Scratch corpus DB kept at {scratch_db} (--keep-db).")
    finally:
        if tmp is not None:
            tmp.cleanup()

    return 0


# ---------------------------------------------------------------------------
# Standalone CLI (kept for experiments/headroom/retrospective.py parity)
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Standalone entry point: audit an existing analyzer-style DB."""
    parser = argparse.ArgumentParser(description="Retrospective headroom compression audit (Stage 0 of #94)")
    parser.add_argument("--db", type=Path, required=True, help="analyzer sqlite DB (opened read-only)")
    parser.add_argument("--projects-dir", type=Path, default=Path.home() / ".claude" / "projects")
    parser.add_argument("--limit", type=int, default=None, help="smoke: only N sessions (by cost desc)")
    parser.add_argument("--out", type=Path, default=None, help="write markdown report here")
    parser.add_argument(
        "--profile",
        choices=("defaults", "max"),
        default="defaults",
        help="'defaults' = shipped headroom config; 'max' = Kompress ML prose model + AST code compression",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # Always show per-session progress on stderr for long runs.
    logger.setLevel(logging.INFO)

    corpus = run_audit(args.db, args.projects_dir, limit=args.limit, profile=args.profile)
    report = build_report(corpus)

    if args.out:
        args.out.write_text(report, encoding="utf-8")
        print(f"report written to {args.out}", file=sys.stderr)
    print(report)
    print(
        f"\nceiling={corpus.overall_ceiling_pct * 100:.2f}% "
        f"(${corpus.total_ceiling_usd:,.2f} of ${corpus.total_recorded_cost:,.2f} recorded) "
        f"wall={corpus.wall_time_s:.1f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
