# Context Analyzer

Context window usage analyzer for Claude Code. An external MCP server that tracks how context is consumed across tools, compaction, skills, and user interactions.

## What it does

- **Hooks** into Claude Code via `~/.claude/settings.json` to capture tool calls, compaction events, session lifecycle
- **Parses transcripts** for exact API token usage (input, output, cache_read, cache_creation)
- **Exposes MCP tools** for querying context usage mid-session
- **Dashboard** for visualizing context composition, growth, cache efficiency, and identifying stale blocks

## Quick start

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Install hooks into Claude Code
python -m context_tracker.installer install

# Run tests
pytest tests/ -v

# Open dashboard
open evidence/dashboard-v2.html
```

## Architecture

```
Claude Code Hooks (shell commands)
  ├── PostToolUse, PostToolUseFailure
  ├── PreCompact, PostCompact
  ├── SessionStart, SessionEnd
  ├── UserPromptSubmit
  ├── SubagentStart, SubagentStop
  └── InstructionsLoaded
       │
       ▼
  ~/.claude/context-trace/<session_id>.jsonl  (hook events)
  ~/.claude/projects/<project>/<session_id>.jsonl  (transcripts)
       │
       ▼
  MCP Server (FastMCP, Python)
  ├── get_session_summary
  ├── get_tool_breakdown
  ├── get_compaction_history
  ├── get_context_hogs
  ├── get_session_history
  ├── get_bloat_events
  └── should_clear
```

## Key findings from real sessions

- Tool I/O consumes 60%+ of used context
- 30% of context becomes stale dead weight within 5 turns
- Cache hit rate is 96-98% when the system prompt prefix stays stable
- A single turn with heavy tool use can add 60K+ tokens

## Roadmap

See [GitHub Issues](https://github.com/manavgup/context-analyzer/issues) for the Phase 4 roadmap: Flask app migration, context tape, staleness detection, health score, drift detection.
