"""CLAUDE.md optimizer — parse, correlate usage, and trim unused instructions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.orm import Session as DbSession

from context_tracker.db import BlockRecord, HookEventRecord, SessionRecord

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Instruction:
    """A single instruction block parsed from CLAUDE.md."""

    line_start: int
    line_end: int
    text: str
    token_count: int  # approximate: len(text) // 4
    category: str  # "directive", "context", "constraint", "example"


@dataclass
class InstructionUsage:
    """An instruction with its cross-session usage evidence."""

    instruction: Instruction
    sessions_active: int = 0
    sessions_total: int = 0
    evidence: list[str] = field(default_factory=list)
    status: str = "unused"  # "active", "rarely_used", "unused"


@dataclass
class ClaudeMdReport:
    """Full analysis report for a CLAUDE.md file."""

    file_path: str
    total_tokens: int
    active_tokens: int
    unused_tokens: int
    instructions: list[InstructionUsage]
    estimated_savings_per_session: float
    optimized_content: str  # trimmed version with only active instructions


# ---------------------------------------------------------------------------
# Categorization keywords
# ---------------------------------------------------------------------------

_DIRECTIVE_PATTERNS = re.compile(
    r"\b(always|never|must|shall|required|ensure|do not skip|every time)\b",
    re.IGNORECASE,
)
_CONSTRAINT_PATTERNS = re.compile(
    r"\b(don'?t|without|avoid|prefer not|should not|forbid|disallow|except)\b",
    re.IGNORECASE,
)
_EXAMPLE_PATTERNS = re.compile(
    r"(```|example|e\.g\.|for instance|such as|like this)",
    re.IGNORECASE,
)


def _categorize(text: str) -> str:
    """Classify an instruction block by keyword analysis."""
    if _DIRECTIVE_PATTERNS.search(text):
        return "directive"
    if _CONSTRAINT_PATTERNS.search(text):
        return "constraint"
    if _EXAMPLE_PATTERNS.search(text):
        return "example"
    return "context"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_claude_md(path: Path) -> list[Instruction]:
    """Parse CLAUDE.md into instruction blocks split on blank lines/headers.

    Each paragraph or bullet-group becomes one Instruction.
    Markdown headers (# ...) start a new block.
    """
    if not path.exists():
        return []

    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return []

    lines = raw.split("\n")
    instructions: list[Instruction] = []
    block_lines: list[str] = []
    block_start = 1  # 1-based line numbers

    def _flush(end_line: int) -> None:
        text = "\n".join(block_lines).strip()
        if text:
            instructions.append(
                Instruction(
                    line_start=block_start,
                    line_end=end_line,
                    text=text,
                    token_count=max(1, len(text) // 4),
                    category=_categorize(text),
                )
            )

    for idx, line in enumerate(lines):
        lineno = idx + 1
        stripped = line.strip()

        # Markdown header starts a new block
        if stripped.startswith("#"):
            _flush(lineno - 1)
            block_lines = [line]
            block_start = lineno
            continue

        # Blank line ends current block
        if not stripped:
            _flush(lineno - 1)
            block_lines = []
            block_start = lineno + 1
            continue

        block_lines.append(line)

    # Flush remainder
    if block_lines:
        _flush(len(lines))

    return instructions


# ---------------------------------------------------------------------------
# Keyword extraction for matching against DB records
# ---------------------------------------------------------------------------

# Common words that should not count as meaningful keywords
_STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "up",
        "as",
        "is",
        "it",
        "be",
        "are",
        "was",
        "not",
        "no",
        "do",
        "if",
        "so",
        "this",
        "that",
        "use",
        "when",
        "will",
        "can",
        "all",
        "any",
        "each",
        "you",
        "your",
        "they",
        "them",
        "their",
        "should",
        "would",
        "could",
        "may",
        "might",
        "must",
        "shall",
        "has",
        "have",
        "had",
        "does",
        "did",
        "been",
        "being",
        "also",
        "than",
        "then",
        "just",
        "only",
        "very",
        "more",
        "most",
        "some",
        "such",
        "like",
        "about",
        "after",
        "before",
        "between",
        "into",
        "through",
        "during",
        "without",
        "within",
        "along",
        "across",
        "above",
        "below",
        "under",
        "over",
        "don",
        "always",
        "never",
        "avoid",
        "prefer",
        "ensure",
        "make",
        "sure",
        "run",
        "file",
        "files",
        "code",
        "using",
        "set",
    }
)

# Patterns that look like tool names, commands, or file paths
_KEYWORD_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_.-]{2,}")


def _extract_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from an instruction for DB matching."""
    candidates = _KEYWORD_RE.findall(text)
    keywords: set[str] = set()
    for word in candidates:
        lower = word.lower()
        if lower not in _STOP_WORDS and len(lower) >= 3:
            keywords.add(lower)
    return keywords


# ---------------------------------------------------------------------------
# Usage correlation
# ---------------------------------------------------------------------------


def correlate_usage(
    instructions: list[Instruction],
    db_session: DbSession,
    min_sessions: int = 3,
) -> list[InstructionUsage]:
    """Cross-reference instructions against observed tool usage across sessions.

    Looks at HookEventRecord.tool_name and BlockRecord.label values to determine
    whether keywords mentioned in each instruction appear in real session data.
    """
    # Gather all sessions
    sessions = db_session.query(SessionRecord).all()
    total_sessions = len(sessions)

    if total_sessions == 0:
        return [
            InstructionUsage(
                instruction=inst,
                sessions_active=0,
                sessions_total=0,
                evidence=[],
                status="unused",
            )
            for inst in instructions
        ]

    # Build per-session sets of keywords from tool names and block labels
    session_keywords: dict[str, set[str]] = {}
    for sess in sessions:
        kws: set[str] = set()
        hooks = db_session.query(HookEventRecord).filter(HookEventRecord.session_id == sess.session_id).all()
        for h in hooks:
            if h.tool_name:
                kws.add(h.tool_name.lower())
                # Also add parts of compound tool names (e.g. "mcp__server__tool")
                for part in h.tool_name.lower().split("__"):
                    if part and part not in _STOP_WORDS:
                        kws.add(part)

        blocks = db_session.query(BlockRecord).filter(BlockRecord.session_id == sess.session_id).all()
        for b in blocks:
            if b.label:
                kws.add(b.label.lower())
                for part in re.split(r"[_\-./\\]", b.label.lower()):
                    if part and part not in _STOP_WORDS:
                        kws.add(part)

        session_keywords[str(sess.session_id)] = kws

    # Correlate each instruction
    results: list[InstructionUsage] = []
    effective_total = min(total_sessions, max(min_sessions, total_sessions))

    for inst in instructions:
        inst_keywords = _extract_keywords(inst.text)
        active_count = 0
        evidence: list[str] = []

        for sid, s_kws in session_keywords.items():
            matched = inst_keywords & s_kws
            if matched:
                active_count += 1
                if len(evidence) < 5:  # cap evidence list
                    evidence.append(f"{sid}: matched {', '.join(sorted(matched)[:3])}")

        # Determine status
        if total_sessions < min_sessions:
            # Not enough data — assume active
            status = "active"
        elif active_count == 0:
            status = "unused"
        elif active_count / effective_total < 0.2:
            status = "rarely_used"
        else:
            status = "active"

        results.append(
            InstructionUsage(
                instruction=inst,
                sessions_active=active_count,
                sessions_total=total_sessions,
                evidence=evidence,
                status=status,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Optimized output generation
# ---------------------------------------------------------------------------


def generate_optimized(instructions: list[InstructionUsage]) -> str:
    """Generate trimmed CLAUDE.md with only active instructions.

    Preserves active instructions verbatim. Rarely-used instructions are
    kept but wrapped in a comment noting low usage. Unused instructions
    are dropped entirely.
    """
    parts: list[str] = []
    for iu in instructions:
        if iu.status == "active":
            parts.append(iu.instruction.text)
        elif iu.status == "rarely_used":
            parts.append(iu.instruction.text)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# High-level analysis entry point
# ---------------------------------------------------------------------------

# Cost per 1K input tokens for Claude Sonnet 4 (representative)
_COST_PER_1K_INPUT = 0.003


def analyze_claude_md(
    path: Path,
    db_session: DbSession,
    min_sessions: int = 3,
) -> ClaudeMdReport:
    """Full analysis pipeline for a single CLAUDE.md file."""
    instructions = parse_claude_md(path)
    usage = correlate_usage(instructions, db_session, min_sessions=min_sessions)
    optimized = generate_optimized(usage)

    total_tokens = sum(iu.instruction.token_count for iu in usage)
    active_tokens = sum(iu.instruction.token_count for iu in usage if iu.status == "active")
    rarely_tokens = sum(iu.instruction.token_count for iu in usage if iu.status == "rarely_used")
    unused_tokens = sum(iu.instruction.token_count for iu in usage if iu.status == "unused")

    # Savings = unused tokens removed per session, priced at input cost
    savings = (unused_tokens / 1000) * _COST_PER_1K_INPUT

    return ClaudeMdReport(
        file_path=str(path),
        total_tokens=total_tokens,
        active_tokens=active_tokens + rarely_tokens,
        unused_tokens=unused_tokens,
        instructions=usage,
        estimated_savings_per_session=round(savings, 6),
        optimized_content=optimized,
    )


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def find_claude_md_files() -> list[Path]:
    """Find CLAUDE.md files in standard locations."""
    candidates = [
        Path.home() / ".claude" / "CLAUDE.md",
        Path(".claude") / "CLAUDE.md",
        Path("CLAUDE.md"),
    ]
    return [p.resolve() for p in candidates if p.exists()]
