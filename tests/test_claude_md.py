"""Tests for CLAUDE.md optimizer — parsing, categorization, correlation, and optimization."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from context_tracker.analysis.claude_md import (
    ClaudeMdReport,
    Instruction,
    InstructionUsage,
    _categorize,
    _extract_keywords,
    analyze_claude_md,
    correlate_usage,
    find_claude_md_files,
    generate_optimized,
    parse_claude_md,
)
from context_tracker.dashboard import create_app
from context_tracker.db import (
    BlockRecord,
    HookEventRecord,
    SessionRecord,
    get_engine,
    get_session_factory,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


SAMPLE_CLAUDE_MD = """\
# Project guidelines

Always run pytest before committing changes.
Use ruff for linting.

## Code style

- Prefer dataclasses over plain dicts
- Never use wildcard imports

## Testing

Run `make test` to execute the full suite.
For example:
```
make test
```

## Deployment

Avoid deploying on Fridays.
"""


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    """Write a sample CLAUDE.md file and return its path."""
    p = tmp_path / "CLAUDE.md"
    p.write_text(SAMPLE_CLAUDE_MD)
    return p


@pytest.fixture
def empty_file(tmp_path: Path) -> Path:
    """Write an empty CLAUDE.md file and return its path."""
    p = tmp_path / "CLAUDE.md"
    p.write_text("")
    return p


@pytest.fixture
def db_session(tmp_path: Path):
    """In-memory SQLite session with test data."""
    db_path = tmp_path / "test.db"
    engine = get_engine(db_path)
    factory = get_session_factory(engine)
    with factory() as session:
        yield session


@pytest.fixture
def db_with_sessions(db_session):
    """DB session pre-populated with sessions and hook events."""
    # Session 1 — uses pytest, ruff
    s1 = SessionRecord(session_id="sess-1", total_turns=5, total_api_calls=10)
    db_session.add(s1)
    db_session.add(HookEventRecord(session_id="sess-1", event_type="post_tool_use", tool_name="Bash"))
    db_session.add(HookEventRecord(session_id="sess-1", event_type="post_tool_use", tool_name="pytest"))
    db_session.add(HookEventRecord(session_id="sess-1", event_type="post_tool_use", tool_name="ruff"))
    db_session.add(
        BlockRecord(
            session_id="sess-1",
            block_id="b1",
            block_type="tool_result",
            label="pytest-output",
            tokens=500,
        )
    )

    # Session 2 — uses pytest only
    s2 = SessionRecord(session_id="sess-2", total_turns=3, total_api_calls=5)
    db_session.add(s2)
    db_session.add(HookEventRecord(session_id="sess-2", event_type="post_tool_use", tool_name="pytest"))
    db_session.add(
        BlockRecord(
            session_id="sess-2",
            block_id="b2",
            block_type="tool_result",
            label="test-run",
            tokens=300,
        )
    )

    # Session 3 — uses deploy
    s3 = SessionRecord(session_id="sess-3", total_turns=2, total_api_calls=3)
    db_session.add(s3)
    db_session.add(HookEventRecord(session_id="sess-3", event_type="post_tool_use", tool_name="deploy"))
    db_session.add(
        BlockRecord(
            session_id="sess-3",
            block_id="b3",
            block_type="tool_result",
            label="deploy-output",
            tokens=200,
        )
    )

    # Session 4 — uses ruff
    s4 = SessionRecord(session_id="sess-4", total_turns=1, total_api_calls=2)
    db_session.add(s4)
    db_session.add(HookEventRecord(session_id="sess-4", event_type="post_tool_use", tool_name="ruff"))

    db_session.commit()
    return db_session


# ---------------------------------------------------------------------------
# parse_claude_md
# ---------------------------------------------------------------------------


class TestParseClaude:
    def test_parse_sample(self, sample_file: Path) -> None:
        instructions = parse_claude_md(sample_file)
        assert len(instructions) > 0
        # Each instruction has valid fields
        for inst in instructions:
            assert inst.line_start >= 1
            assert inst.line_end >= inst.line_start
            assert len(inst.text) > 0
            assert inst.token_count >= 1
            assert inst.category in ("directive", "context", "constraint", "example")

    def test_headers_split_blocks(self, sample_file: Path) -> None:
        instructions = parse_claude_md(sample_file)
        texts = [i.text for i in instructions]
        # "# Project guidelines" should be a separate block from "## Code style"
        header_texts = [t for t in texts if t.startswith("#")]
        assert len(header_texts) >= 3  # Project guidelines, Code style, Testing, Deployment

    def test_paragraphs_split_on_blank_lines(self, sample_file: Path) -> None:
        instructions = parse_claude_md(sample_file)
        # "Always run pytest..." and "Use ruff..." are on consecutive lines (same block)
        # while the bullet list is a separate block
        found_pytest_block = any("pytest" in i.text and "ruff" in i.text for i in instructions)
        assert found_pytest_block, "pytest and ruff should be in the same paragraph block"

    def test_empty_file(self, empty_file: Path) -> None:
        instructions = parse_claude_md(empty_file)
        assert instructions == []

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        instructions = parse_claude_md(tmp_path / "nonexistent.md")
        assert instructions == []


# ---------------------------------------------------------------------------
# Categorization
# ---------------------------------------------------------------------------


class TestCategorize:
    def test_directive_always(self) -> None:
        assert _categorize("Always run tests before committing") == "directive"

    def test_directive_never(self) -> None:
        assert _categorize("Never push to main directly") == "directive"

    def test_directive_must(self) -> None:
        assert _categorize("You must format code with ruff") == "directive"

    def test_constraint_dont(self) -> None:
        assert _categorize("Don't use wildcard imports") == "constraint"

    def test_constraint_avoid(self) -> None:
        assert _categorize("Avoid deploying on Fridays") == "constraint"

    def test_constraint_without(self) -> None:
        assert _categorize("Run without verbose logging") == "constraint"

    def test_example_backtick(self) -> None:
        assert _categorize("For example:\n```\nmake test\n```") == "example"

    def test_example_eg(self) -> None:
        assert _categorize("Use a linter, e.g. ruff") == "example"

    def test_context_default(self) -> None:
        assert _categorize("This project uses Python 3.11") == "context"


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------


class TestKeywordExtraction:
    def test_extracts_tool_names(self) -> None:
        kws = _extract_keywords("Always run pytest before committing")
        assert "pytest" in kws
        assert "committing" in kws

    def test_filters_stop_words(self) -> None:
        kws = _extract_keywords("Always use the ruff linter for all code")
        assert "the" not in kws
        assert "for" not in kws
        assert "all" not in kws

    def test_extracts_compound_names(self) -> None:
        kws = _extract_keywords("Use mcp__server__tool for this")
        assert "mcp__server__tool" in kws


# ---------------------------------------------------------------------------
# Usage correlation
# ---------------------------------------------------------------------------


class TestCorrelateUsage:
    def test_empty_db(self, sample_file: Path, db_session) -> None:
        instructions = parse_claude_md(sample_file)
        result = correlate_usage(instructions, db_session)
        # With 0 sessions, all should be marked unused
        assert all(iu.status == "unused" for iu in result)
        assert all(iu.sessions_total == 0 for iu in result)

    def test_with_sessions(self, sample_file: Path, db_with_sessions) -> None:
        instructions = parse_claude_md(sample_file)
        result = correlate_usage(instructions, db_with_sessions, min_sessions=2)
        assert len(result) == len(instructions)
        # At least one instruction should be active (the pytest one is used in 2/4 sessions)
        statuses = {iu.status for iu in result}
        assert "active" in statuses or "rarely_used" in statuses

    def test_active_never_marked_unused(self, sample_file: Path, db_with_sessions) -> None:
        """Instructions matching active tool usage should not be 'unused'."""
        instructions = parse_claude_md(sample_file)
        result = correlate_usage(instructions, db_with_sessions, min_sessions=2)
        for iu in result:
            if iu.sessions_active > 0:
                assert iu.status != "unused", (
                    f"Instruction with {iu.sessions_active} active sessions "
                    f"should not be 'unused': {iu.instruction.text[:50]}"
                )


# ---------------------------------------------------------------------------
# generate_optimized
# ---------------------------------------------------------------------------


class TestGenerateOptimized:
    def test_preserves_active(self) -> None:
        instructions = [
            InstructionUsage(
                instruction=Instruction(1, 2, "Always run tests", 5, "directive"),
                sessions_active=3,
                sessions_total=5,
                evidence=[],
                status="active",
            ),
            InstructionUsage(
                instruction=Instruction(3, 4, "Avoid deploying on Fridays", 6, "constraint"),
                sessions_active=0,
                sessions_total=5,
                evidence=[],
                status="unused",
            ),
        ]
        optimized = generate_optimized(instructions)
        assert "Always run tests" in optimized
        assert "Avoid deploying on Fridays" not in optimized

    def test_keeps_rarely_used(self) -> None:
        instructions = [
            InstructionUsage(
                instruction=Instruction(1, 2, "Check coverage", 4, "directive"),
                sessions_active=1,
                sessions_total=10,
                evidence=[],
                status="rarely_used",
            ),
        ]
        optimized = generate_optimized(instructions)
        assert "Check coverage" in optimized

    def test_removes_unused(self) -> None:
        instructions = [
            InstructionUsage(
                instruction=Instruction(1, 2, "Legacy rule nobody uses", 6, "context"),
                sessions_active=0,
                sessions_total=10,
                evidence=[],
                status="unused",
            ),
        ]
        optimized = generate_optimized(instructions)
        assert optimized == ""

    def test_empty_input(self) -> None:
        optimized = generate_optimized([])
        assert optimized == ""


# ---------------------------------------------------------------------------
# analyze_claude_md (end-to-end)
# ---------------------------------------------------------------------------


class TestAnalyzeClaude:
    def test_report_structure(self, sample_file: Path, db_with_sessions) -> None:
        report = analyze_claude_md(sample_file, db_with_sessions, min_sessions=2)
        assert isinstance(report, ClaudeMdReport)
        assert report.file_path == str(sample_file)
        assert report.total_tokens > 0
        assert report.active_tokens + report.unused_tokens == report.total_tokens
        assert isinstance(report.optimized_content, str)
        assert isinstance(report.estimated_savings_per_session, float)
        assert len(report.instructions) > 0

    def test_empty_file_report(self, empty_file: Path, db_session) -> None:
        report = analyze_claude_md(empty_file, db_session)
        assert report.total_tokens == 0
        assert report.active_tokens == 0
        assert report.unused_tokens == 0
        assert report.instructions == []
        assert report.optimized_content == ""


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------


class TestOptimizeAPI:
    def test_endpoint_returns_ok(self, tmp_path: Path) -> None:
        """The API endpoint should return a valid response even with no files."""
        app = create_app(
            trace_dir=tmp_path / "traces",
            transcript_dir=tmp_path / "transcripts",
            static_dir=tmp_path / "static",
            db_path=tmp_path / "test.db",
        )
        client = TestClient(app)
        resp = client.get("/api/optimize/claude-md")
        assert resp.status_code == 200
        body = resp.json()
        assert "reports" in body

    def test_optimize_page_route(self, tmp_path: Path) -> None:
        """The /optimize route should return an HTML response."""
        static_dir = tmp_path / "static"
        static_dir.mkdir()
        (static_dir / "optimize.html").write_text("<h1>Optimize</h1>")
        app = create_app(
            trace_dir=tmp_path / "traces",
            transcript_dir=tmp_path / "transcripts",
            static_dir=static_dir,
            db_path=tmp_path / "test.db",
        )
        client = TestClient(app)
        resp = client.get("/optimize")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# find_claude_md_files
# ---------------------------------------------------------------------------


class TestFindClaudeMdFiles:
    def test_finds_root_claude_md(self, tmp_path: Path) -> None:
        (tmp_path / "CLAUDE.md").write_text("# Instructions")
        result = find_claude_md_files(project_dir=tmp_path)
        assert len(result) == 1
        assert result[0] == (tmp_path / "CLAUDE.md").resolve()

    def test_finds_dotclaude_subdir(self, tmp_path: Path) -> None:
        (tmp_path / ".claude").mkdir()
        (tmp_path / ".claude" / "CLAUDE.md").write_text("# Nested")
        result = find_claude_md_files(project_dir=tmp_path)
        assert len(result) == 1
        assert result[0] == (tmp_path / ".claude" / "CLAUDE.md").resolve()

    def test_finds_both(self, tmp_path: Path) -> None:
        (tmp_path / "CLAUDE.md").write_text("# Root")
        (tmp_path / ".claude").mkdir()
        (tmp_path / ".claude" / "CLAUDE.md").write_text("# Nested")
        result = find_claude_md_files(project_dir=tmp_path)
        assert len(result) == 2

    def test_excludes_home_claude_md(self, tmp_path: Path) -> None:
        """User-global ~/.claude/CLAUDE.md should NOT be discovered."""
        # Simulate a home dir file that is outside project_dir
        home_like = tmp_path / "fake_home" / ".claude"
        home_like.mkdir(parents=True)
        (home_like / "CLAUDE.md").write_text("# Private")
        project = tmp_path / "project"
        project.mkdir()
        result = find_claude_md_files(project_dir=project)
        assert len(result) == 0

    def test_empty_project(self, tmp_path: Path) -> None:
        result = find_claude_md_files(project_dir=tmp_path)
        assert result == []
