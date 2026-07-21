"""Personal stats card — aggregate local SQLite data into shareable numbers.

Implements `context-tracker stats` (issue #82). Everything is computed from
the local analyzer database; nothing is uploaded anywhere (local-first).

Two output modes:

* default — a terminal summary card for the user's own consumption
* ``--share`` — a markdown snippet containing NUMBERS ONLY (no prompt text,
  no file paths, no repo/project names, no session ids), safe to paste
  into X/Reddit.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session as DbSession

from context_tracker.db import DEFAULT_DB_PATH, BlockRecord, SessionRecord, get_session_factory


@dataclass
class StatsCard:
    """Aggregated, privacy-safe personal stats (numbers only)."""

    total_sessions: int = 0
    total_api_calls: int = 0
    total_spend_usd: float = 0.0
    wasted_spend_usd: float = 0.0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    input_tokens: int = 0
    # Most expensive session — identified by date only (never by id/path).
    top_session_cost_usd: float = 0.0
    top_session_peak_context: int = 0
    top_session_date: str = "unknown date"

    @property
    def cache_efficiency(self) -> float:
        """Fraction of prompt tokens served from cache.

        cache_read / (cache_read + cache_creation + input) — the same
        cache-hit-rate formula used in ``server.mcp_get_session_summary``.
        Cache-creation tokens are prompt tokens that were NOT served from
        cache, so they belong in the denominator; omitting them would
        overstate efficiency.
        """
        denom = self.cache_read_tokens + self.cache_creation_tokens + self.input_tokens
        return self.cache_read_tokens / denom if denom else 0.0

    @property
    def wasted_pct_of_spend(self) -> float:
        """Wasted spend as a fraction of total spend."""
        return self.wasted_spend_usd / self.total_spend_usd if self.total_spend_usd else 0.0


def _session_date(rec: SessionRecord) -> str:
    """Date-only label for a session (never leaks id, path, or project)."""
    if rec.started_at:
        try:
            return datetime.fromisoformat(str(rec.started_at).replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            pass
    if rec.source_mtime:
        try:
            return datetime.fromtimestamp(float(rec.source_mtime)).date().isoformat()
        except (ValueError, OSError, OverflowError):
            pass
    return "unknown date"


def _estimate_wasted_spend(db: DbSession, rec: SessionRecord) -> float:
    """Estimate the dollars a session spent carrying dead-weight context.

    Estimation method
    -----------------
    A block is *dead weight* when it was never referenced again after it
    entered the context (``ref == 0``) and was not served from the prefix
    cache (``cached == 0``) — the same definition the MCP staleness tool
    uses (see ``server.mcp_get_staleness_analysis``).

    The cost of a block is proportional not just to its size but to how
    long it stayed resident, because every subsequent API call re-transmits
    it. Using the recorded enter/exit data (``enter_turn`` / ``exit_turn``
    are API-call indices; ``exit_turn IS NULL`` means the block stayed until
    the end of the session):

        residency(block) = (exit_turn or total_api_calls) - enter_turn
        token_calls(block) = tokens * residency(block)

    Blocks still present at session end (``exit_turn IS NULL``) count
    through ``total_api_calls`` — the same effective-exit convention as
    ``analysis.report._detect_stale_content`` — because ``enter_turn`` is
    itself a call where the block is resident. Using ``total_api_calls - 1``
    would undercount every surviving block by one call and give zero
    residency in a one-call session.

    The session's waste fraction is the share of total token-call volume
    attributable to dead-weight blocks, and the wasted spend is that
    fraction of the session's recorded cost:

        waste_fraction = sum(token_calls of dead blocks) / sum(token_calls of all blocks)
        wasted_usd     = waste_fraction * total_cost_usd

    This is an estimate: it attributes cost by resident token volume and
    ignores per-token price differences between cached/uncached/output
    tokens.
    """
    end_turn = int(rec.total_api_calls or 0)

    total_token_calls = 0
    dead_token_calls = 0
    for block in db.query(BlockRecord).filter_by(session_id=rec.session_id):
        tokens = int(block.tokens or 0)
        enter = int(block.enter_turn or 0)
        exit_ = int(block.exit_turn) if block.exit_turn is not None else end_turn
        residency = max(exit_ - enter, 0)
        token_calls = tokens * residency
        total_token_calls += token_calls
        if not block.ref and not block.cached:
            dead_token_calls += token_calls

    if total_token_calls <= 0:
        return 0.0
    waste_fraction = dead_token_calls / total_token_calls
    return waste_fraction * float(rec.total_cost_usd or 0.0)


def compute_stats(db: DbSession) -> StatsCard:
    """Aggregate all sessions in the local DB into a :class:`StatsCard`."""
    card = StatsCard()
    top: SessionRecord | None = None

    for rec in db.query(SessionRecord):
        card.total_sessions += 1
        card.total_api_calls += int(rec.total_api_calls or 0)
        card.total_spend_usd += float(rec.total_cost_usd or 0.0)
        card.cache_read_tokens += int(rec.total_cache_read or 0)
        card.cache_creation_tokens += int(rec.total_cache_creation or 0)
        card.input_tokens += int(rec.total_input_tokens or 0)
        card.wasted_spend_usd += _estimate_wasted_spend(db, rec)
        if top is None or float(rec.total_cost_usd or 0.0) > float(top.total_cost_usd or 0.0):
            top = rec

    if top is not None:
        card.top_session_cost_usd = float(top.total_cost_usd or 0.0)
        card.top_session_peak_context = int(top.peak_context_tokens or 0)
        card.top_session_date = _session_date(top)

    return card


def render_card(card: StatsCard) -> str:
    """Render the terminal summary card (numbers only)."""
    rows = [
        ("Sessions analyzed", f"{card.total_sessions:,}"),
        ("API calls", f"{card.total_api_calls:,}"),
        ("Total spend", f"${card.total_spend_usd:,.2f}"),
        (
            "Wasted on dead-weight context",
            f"${card.wasted_spend_usd:,.2f} ({card.wasted_pct_of_spend:.1%} of spend)",
        ),
        ("Cache efficiency", f"{card.cache_efficiency:.1%}"),
        (
            "Most expensive session",
            f"${card.top_session_cost_usd:,.2f} "
            f"(peak {card.top_session_peak_context:,} tokens) on {card.top_session_date}",
        ),
    ]
    title = "Context Analyzer — Personal Stats"
    label_w = max(len(label) for label, _ in rows)
    body = [f"  {label.ljust(label_w)}  {value}" for label, value in rows]
    width = max(len(title) + 4, *(len(line) for line in body)) + 2
    lines = [
        "┌" + "─" * width + "┐",
        "│" + title.center(width) + "│",
        "├" + "─" * width + "┤",
        *[f"│{line.ljust(width)}│" for line in body],
        "└" + "─" * width + "┘",
        "  All numbers computed locally — nothing leaves your machine.",
    ]
    return "\n".join(lines)


def render_share_markdown(card: StatsCard) -> str:
    """Render the shareable markdown snippet.

    NUMBERS ONLY: no prompt text, no file paths, no repo/project names,
    no session ids. Sessions are identified by date at most.
    """
    return "\n".join(
        [
            "## My Claude Code context stats",
            "",
            f"- **Sessions analyzed:** {card.total_sessions:,}",
            f"- **API calls:** {card.total_api_calls:,}",
            f"- **Total spend:** ${card.total_spend_usd:,.2f}",
            f"- **Wasted on dead-weight context:** ${card.wasted_spend_usd:,.2f}"
            f" ({card.wasted_pct_of_spend:.1%} of spend)",
            f"- **Cache efficiency:** {card.cache_efficiency:.1%}",
            f"- **Most expensive session:** ${card.top_session_cost_usd:,.2f}"
            f" (peak {card.top_session_peak_context:,} tokens) on {card.top_session_date}",
            "",
            "_Measured locally with context-analyzer — local-first, numbers only._",
        ]
    )


def run_stats(share: bool = False, db_path: Path = DEFAULT_DB_PATH) -> int:
    """Entry point for `context-tracker stats`."""
    if not db_path.exists():
        print(f"No analyzer database found at {db_path} — run some sessions first.", file=sys.stderr)
        return 1

    factory = get_session_factory(db_path=db_path)
    with factory() as db:
        card = compute_stats(db)

    if card.total_sessions == 0:
        print("No sessions recorded yet — nothing to summarize.", file=sys.stderr)
        return 1

    print(render_share_markdown(card) if share else render_card(card))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Standalone CLI (also reachable via `context-tracker stats`)."""
    parser = argparse.ArgumentParser(prog="context-tracker stats", description="Print a personal stats card")
    parser.add_argument("--share", action="store_true", help="Emit a shareable markdown snippet (numbers only)")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Path to analyzer SQLite DB")
    args = parser.parse_args(argv)
    return run_stats(share=args.share, db_path=args.db)


if __name__ == "__main__":
    raise SystemExit(main())
