# Context Tracker MCP Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python MCP server that passively logs Claude Code context usage via hooks, parses transcript JSONL for exact API token counts, and exposes interactive query tools for mid-session context analysis.

**Architecture:** External observer pattern. Claude Code hooks (shell commands) fire a Python script that appends events to JSONL files. A FastMCP server reads those files plus Claude Code's native transcript JSONL to answer queries like "what's eating my context?" No Claude Code source modifications.

**Tech Stack:** Python 3.11+, FastMCP, Pydantic v2, hatchling build system, pytest, MCP Inspector

**Spec:** https://github.com/manavgup/claude-src/issues/1

---

## File Structure

```
mcp-servers/context-tracker/
├── src/context_tracker/
│   ├── __init__.py              # Package init, version
│   ├── server.py                # FastMCP server with 5 query tools + main()
│   ├── hooks.py                 # Hook processor: reads stdin JSON, writes JSONL
│   ├── installer.py             # CLI: install-hooks / uninstall-hooks into settings.json
│   ├── models.py                # Pydantic models for all 10 event types
│   ├── storage.py               # JSONL reader/writer with atomic append
│   └── transcript.py            # Transcript JSONL parser for API usage extraction
├── tests/
│   ├── __init__.py
│   ├── test_models.py           # Model serialization/deserialization
│   ├── test_storage.py          # Atomic append, malformed lines, concurrent writes
│   ├── test_hooks.py            # Hook processor stdin parsing
│   ├── test_transcript.py       # Transcript parsing, missing fields
│   ├── test_installer.py        # Merge safety, idempotent uninstall
│   └── test_server.py           # MCP tool integration tests
├── pyproject.toml
├── Makefile
└── README.md
```

---

### Task 1: Project Scaffold

**Files:**
- Create: `mcp-servers/context-tracker/pyproject.toml`
- Create: `mcp-servers/context-tracker/Makefile`
- Create: `mcp-servers/context-tracker/src/context_tracker/__init__.py`
- Create: `mcp-servers/context-tracker/tests/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "context-tracker"
version = "0.1.0"
description = "MCP server for tracking Claude Code context window usage"
requires-python = ">=3.11"
dependencies = [
    "fastmcp>=2.0.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=1.0.0",
    "pytest-cov>=5.0.0",
    "ruff>=0.5.0",
    "mypy>=1.10.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/context_tracker"]

[project.scripts]
context-tracker = "context_tracker.server:main"
context-tracker-hook = "context_tracker.hooks:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.ruff]
target-version = "py311"
line-length = 88
```

- [ ] **Step 2: Create Makefile**

```makefile
.PHONY: install dev test lint run

install:
	python3 -m pip install -e .

dev:
	python3 -m pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=context_tracker --cov-report=term-missing

lint:
	ruff check src/ tests/
	mypy src/context_tracker

run:
	python3 -m context_tracker.server

hook-install:
	python3 -m context_tracker.installer install

hook-uninstall:
	python3 -m context_tracker.installer uninstall
```

- [ ] **Step 3: Create __init__.py files**

`src/context_tracker/__init__.py`:
```python
"""Context Tracker MCP Server — external observer for Claude Code context usage."""

__version__ = "0.1.0"
```

`tests/__init__.py`:
```python
```

- [ ] **Step 4: Create directory structure and verify**

Run: `mkdir -p mcp-servers/context-tracker/src/context_tracker mcp-servers/context-tracker/tests`

- [ ] **Step 5: Install in dev mode**

Run: `cd mcp-servers/context-tracker && make dev`
Expected: Installs successfully, `context-tracker --help` is available

- [ ] **Step 6: Commit**

```bash
git add mcp-servers/context-tracker/
git commit -m "feat: scaffold context-tracker MCP server project"
```

---

### Task 2: Pydantic Models

**Files:**
- Create: `mcp-servers/context-tracker/src/context_tracker/models.py`
- Create: `mcp-servers/context-tracker/tests/test_models.py`

- [ ] **Step 1: Write the failing test**

`tests/test_models.py`:
```python
import json
from datetime import datetime, timezone

from context_tracker.models import (
    PostToolUseEvent,
    PostToolUseFailureEvent,
    PreCompactEvent,
    PostCompactEvent,
    SessionStartEvent,
    SessionEndEvent,
    UserPromptEvent,
    SubagentStartEvent,
    SubagentStopEvent,
    InstructionsLoadedEvent,
    ApiTurnEvent,
    parse_event,
)


def test_post_tool_use_roundtrip():
    event = PostToolUseEvent(
        session_id="abc-123",
        tool_name="Read",
        input_payload_chars=142,
        output_payload_chars=8420,
        tool_use_id="toolu_01X",
    )
    line = event.to_jsonl()
    parsed = parse_event(line)
    assert isinstance(parsed, PostToolUseEvent)
    assert parsed.tool_name == "Read"
    assert parsed.output_payload_chars == 8420
    assert parsed.event == "post_tool_use"


def test_session_start_roundtrip():
    event = SessionStartEvent(
        session_id="abc-123",
        source="startup",
        model="claude-opus-4-6",
    )
    line = event.to_jsonl()
    parsed = parse_event(line)
    assert isinstance(parsed, SessionStartEvent)
    assert parsed.source == "startup"
    assert parsed.model == "claude-opus-4-6"


def test_api_turn_roundtrip():
    event = ApiTurnEvent(
        session_id="abc-123",
        turn_number=5,
        input_tokens=45200,
        output_tokens=1830,
        cache_read_input_tokens=38000,
        cache_creation_input_tokens=2100,
        model="claude-opus-4-6",
        stop_reason="end_turn",
    )
    line = event.to_jsonl()
    parsed = parse_event(line)
    assert isinstance(parsed, ApiTurnEvent)
    assert parsed.input_tokens == 45200
    assert parsed.cache_read_input_tokens == 38000


def test_parse_event_unknown_type():
    line = json.dumps({"event": "unknown_event", "session_id": "x"})
    result = parse_event(line)
    assert result is None


def test_parse_event_malformed_json():
    result = parse_event("not json at all {{{")
    assert result is None


def test_post_tool_use_failure_roundtrip():
    event = PostToolUseFailureEvent(
        session_id="abc-123",
        tool_name="Bash",
        input_payload_chars=85,
        error_length=320,
        tool_use_id="toolu_02Y",
    )
    line = event.to_jsonl()
    parsed = parse_event(line)
    assert isinstance(parsed, PostToolUseFailureEvent)
    assert parsed.error_length == 320


def test_subagent_stop_roundtrip():
    event = SubagentStopEvent(
        session_id="abc-123",
        agent_id="agent-001",
        agent_type="general-purpose",
        agent_transcript_path="/path/to/transcript.jsonl",
    )
    line = event.to_jsonl()
    parsed = parse_event(line)
    assert isinstance(parsed, SubagentStopEvent)
    assert parsed.agent_transcript_path == "/path/to/transcript.jsonl"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mcp-servers/context-tracker && pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Write models.py**

`src/context_tracker/models.py`:
```python
"""Pydantic models for all context tracker event types."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Literal, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BaseEvent(BaseModel):
    """Base fields shared by all events."""

    session_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_jsonl(self) -> str:
        return self.model_dump_json()


class PostToolUseEvent(BaseEvent):
    event: Literal["post_tool_use"] = "post_tool_use"
    tool_name: str
    input_payload_chars: int
    output_payload_chars: int
    tool_use_id: str


class PostToolUseFailureEvent(BaseEvent):
    event: Literal["post_tool_use_failure"] = "post_tool_use_failure"
    tool_name: str
    input_payload_chars: int
    error_length: int
    tool_use_id: str


class PreCompactEvent(BaseEvent):
    event: Literal["pre_compact"] = "pre_compact"
    trigger: Literal["auto", "manual"]


class PostCompactEvent(BaseEvent):
    event: Literal["post_compact"] = "post_compact"
    trigger: Literal["auto", "manual"]
    compact_summary_length: int


class SessionStartEvent(BaseEvent):
    event: Literal["session_start"] = "session_start"
    source: Literal["startup", "resume", "clear", "compact"]
    model: str


class SessionEndEvent(BaseEvent):
    event: Literal["session_end"] = "session_end"
    reason: str


class UserPromptEvent(BaseEvent):
    event: Literal["user_prompt"] = "user_prompt"
    prompt_length_chars: int


class SubagentStartEvent(BaseEvent):
    event: Literal["subagent_start"] = "subagent_start"
    agent_id: str
    agent_type: str


class SubagentStopEvent(BaseEvent):
    event: Literal["subagent_stop"] = "subagent_stop"
    agent_id: str
    agent_type: str
    agent_transcript_path: str


class InstructionsLoadedEvent(BaseEvent):
    event: Literal["instructions_loaded"] = "instructions_loaded"
    file_path: str
    memory_type: str
    load_reason: str


class ApiTurnEvent(BaseEvent):
    event: Literal["api_turn"] = "api_turn"
    turn_number: int
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    model: str
    stop_reason: str | None = None


EVENT_TYPE_MAP: dict[str, type[BaseEvent]] = {
    "post_tool_use": PostToolUseEvent,
    "post_tool_use_failure": PostToolUseFailureEvent,
    "pre_compact": PreCompactEvent,
    "post_compact": PostCompactEvent,
    "session_start": SessionStartEvent,
    "session_end": SessionEndEvent,
    "user_prompt": UserPromptEvent,
    "subagent_start": SubagentStartEvent,
    "subagent_stop": SubagentStopEvent,
    "instructions_loaded": InstructionsLoadedEvent,
    "api_turn": ApiTurnEvent,
}

TrackerEvent = Union[
    PostToolUseEvent,
    PostToolUseFailureEvent,
    PreCompactEvent,
    PostCompactEvent,
    SessionStartEvent,
    SessionEndEvent,
    UserPromptEvent,
    SubagentStartEvent,
    SubagentStopEvent,
    InstructionsLoadedEvent,
    ApiTurnEvent,
]


def parse_event(line: str) -> TrackerEvent | None:
    """Parse a JSONL line into a typed event. Returns None for unknown or malformed lines."""
    try:
        raw = json.loads(line)
    except json.JSONDecodeError:
        logger.warning("Malformed JSON line: %s", line[:80])
        return None

    event_type = raw.get("event")
    model_class = EVENT_TYPE_MAP.get(event_type)
    if model_class is None:
        logger.debug("Unknown event type: %s", event_type)
        return None

    try:
        return model_class.model_validate(raw)
    except Exception:
        logger.warning("Failed to parse event: %s", line[:80])
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mcp-servers/context-tracker && pytest tests/test_models.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add mcp-servers/context-tracker/src/context_tracker/models.py mcp-servers/context-tracker/tests/test_models.py
git commit -m "feat: add pydantic models for all context tracker event types"
```

---

### Task 3: JSONL Storage

**Files:**
- Create: `mcp-servers/context-tracker/src/context_tracker/storage.py`
- Create: `mcp-servers/context-tracker/tests/test_storage.py`

- [ ] **Step 1: Write the failing test**

`tests/test_storage.py`:
```python
import os
import tempfile
from pathlib import Path

from context_tracker.models import PostToolUseEvent, SessionStartEvent, parse_event
from context_tracker.storage import append_event, read_events, list_sessions


def test_append_and_read_roundtrip(tmp_path):
    trace_dir = tmp_path / "traces"
    event = PostToolUseEvent(
        session_id="sess-1",
        tool_name="Read",
        input_payload_chars=100,
        output_payload_chars=5000,
        tool_use_id="toolu_01",
    )
    append_event(event, trace_dir=trace_dir)

    events = read_events("sess-1", trace_dir=trace_dir)
    assert len(events) == 1
    assert events[0].tool_name == "Read"


def test_multiple_events_same_session(tmp_path):
    trace_dir = tmp_path / "traces"
    for i in range(5):
        event = PostToolUseEvent(
            session_id="sess-1",
            tool_name=f"Tool{i}",
            input_payload_chars=i * 10,
            output_payload_chars=i * 100,
            tool_use_id=f"toolu_{i}",
        )
        append_event(event, trace_dir=trace_dir)

    events = read_events("sess-1", trace_dir=trace_dir)
    assert len(events) == 5


def test_read_nonexistent_session(tmp_path):
    events = read_events("no-such-session", trace_dir=tmp_path)
    assert events == []


def test_malformed_lines_skipped(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir(parents=True)
    filepath = trace_dir / "sess-bad.jsonl"
    filepath.write_text(
        '{"event":"post_tool_use","session_id":"sess-bad","tool_name":"X","input_payload_chars":1,"output_payload_chars":2,"tool_use_id":"t"}\n'
        'not json\n'
        '{"event":"post_tool_use","session_id":"sess-bad","tool_name":"Y","input_payload_chars":3,"output_payload_chars":4,"tool_use_id":"u"}\n'
    )
    events = read_events("sess-bad", trace_dir=trace_dir)
    assert len(events) == 2
    assert events[0].tool_name == "X"
    assert events[1].tool_name == "Y"


def test_list_sessions(tmp_path):
    trace_dir = tmp_path / "traces"
    for sid in ["aaa", "bbb", "ccc"]:
        append_event(
            SessionStartEvent(session_id=sid, source="startup", model="test"),
            trace_dir=trace_dir,
        )
    sessions = list_sessions(trace_dir=trace_dir)
    assert set(sessions) == {"aaa", "bbb", "ccc"}


def test_creates_directory_on_first_write(tmp_path):
    trace_dir = tmp_path / "nonexistent" / "deep" / "traces"
    assert not trace_dir.exists()
    append_event(
        SessionStartEvent(session_id="new", source="startup", model="test"),
        trace_dir=trace_dir,
    )
    assert trace_dir.exists()
    events = read_events("new", trace_dir=trace_dir)
    assert len(events) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mcp-servers/context-tracker && pytest tests/test_storage.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write storage.py**

`src/context_tracker/storage.py`:
```python
"""JSONL storage with atomic append for context trace events."""

from __future__ import annotations

import os
from pathlib import Path

from context_tracker.models import BaseEvent, TrackerEvent, parse_event

DEFAULT_TRACE_DIR = Path.home() / ".claude" / "context-trace"


def _session_path(session_id: str, trace_dir: Path) -> Path:
    return trace_dir / f"{session_id}.jsonl"


def append_event(event: BaseEvent, trace_dir: Path = DEFAULT_TRACE_DIR) -> None:
    """Append a single event as a JSONL line. Uses O_APPEND for atomic writes."""
    trace_dir.mkdir(parents=True, exist_ok=True)
    filepath = _session_path(event.session_id, trace_dir)
    line = event.to_jsonl() + "\n"
    fd = os.open(str(filepath), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)


def read_events(
    session_id: str, trace_dir: Path = DEFAULT_TRACE_DIR
) -> list[TrackerEvent]:
    """Read all events for a session. Skips malformed lines."""
    filepath = _session_path(session_id, trace_dir)
    if not filepath.exists():
        return []

    events: list[TrackerEvent] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parsed = parse_event(line)
            if parsed is not None:
                events.append(parsed)
    return events


def list_sessions(trace_dir: Path = DEFAULT_TRACE_DIR) -> list[str]:
    """List all session IDs with trace files, sorted by modification time (newest first)."""
    if not trace_dir.exists():
        return []
    files = sorted(trace_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [f.stem for f in files]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mcp-servers/context-tracker && pytest tests/test_storage.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add mcp-servers/context-tracker/src/context_tracker/storage.py mcp-servers/context-tracker/tests/test_storage.py
git commit -m "feat: add JSONL storage with atomic append and malformed line handling"
```

---

### Task 4: Hook Processor

**Files:**
- Create: `mcp-servers/context-tracker/src/context_tracker/hooks.py`
- Create: `mcp-servers/context-tracker/tests/test_hooks.py`

- [ ] **Step 1: Write the failing test**

`tests/test_hooks.py`:
```python
import json

from context_tracker.hooks import process_hook_input
from context_tracker.models import (
    PostToolUseEvent,
    PostToolUseFailureEvent,
    SessionStartEvent,
    SessionEndEvent,
    UserPromptEvent,
    PreCompactEvent,
    PostCompactEvent,
    SubagentStartEvent,
    SubagentStopEvent,
    InstructionsLoadedEvent,
)


def test_process_post_tool_use():
    hook_input = {
        "hook_event_name": "PostToolUse",
        "session_id": "sess-1",
        "tool_name": "Read",
        "tool_input": {"file_path": "/some/file.ts"},
        "tool_response": "x" * 5000,
        "tool_use_id": "toolu_01",
        "transcript_path": "/path/to/transcript.jsonl",
        "cwd": "/project",
    }
    event = process_hook_input(json.dumps(hook_input))
    assert isinstance(event, PostToolUseEvent)
    assert event.tool_name == "Read"
    assert event.input_payload_chars == len(json.dumps({"file_path": "/some/file.ts"}))
    assert event.output_payload_chars == 5000


def test_process_post_tool_use_failure():
    hook_input = {
        "hook_event_name": "PostToolUseFailure",
        "session_id": "sess-1",
        "tool_name": "Bash",
        "tool_input": {"command": "ls"},
        "tool_use_id": "toolu_02",
        "error": "Permission denied" * 20,
        "transcript_path": "/path",
        "cwd": "/project",
    }
    event = process_hook_input(json.dumps(hook_input))
    assert isinstance(event, PostToolUseFailureEvent)
    assert event.error_length == len("Permission denied" * 20)


def test_process_session_start():
    hook_input = {
        "hook_event_name": "SessionStart",
        "session_id": "sess-1",
        "source": "startup",
        "model": "claude-opus-4-6",
        "transcript_path": "/path",
        "cwd": "/project",
    }
    event = process_hook_input(json.dumps(hook_input))
    assert isinstance(event, SessionStartEvent)
    assert event.model == "claude-opus-4-6"


def test_process_session_end():
    hook_input = {
        "hook_event_name": "SessionEnd",
        "session_id": "sess-1",
        "reason": "clear",
        "transcript_path": "/path",
        "cwd": "/project",
    }
    event = process_hook_input(json.dumps(hook_input))
    assert isinstance(event, SessionEndEvent)
    assert event.reason == "clear"


def test_process_user_prompt():
    hook_input = {
        "hook_event_name": "UserPromptSubmit",
        "session_id": "sess-1",
        "prompt": "Hello, can you help me fix this bug?",
        "transcript_path": "/path",
        "cwd": "/project",
    }
    event = process_hook_input(json.dumps(hook_input))
    assert isinstance(event, UserPromptEvent)
    assert event.prompt_length_chars == len("Hello, can you help me fix this bug?")


def test_process_pre_compact():
    hook_input = {
        "hook_event_name": "PreCompact",
        "session_id": "sess-1",
        "trigger": "auto",
        "custom_instructions": None,
        "transcript_path": "/path",
        "cwd": "/project",
    }
    event = process_hook_input(json.dumps(hook_input))
    assert isinstance(event, PreCompactEvent)
    assert event.trigger == "auto"


def test_process_post_compact():
    hook_input = {
        "hook_event_name": "PostCompact",
        "session_id": "sess-1",
        "trigger": "manual",
        "compact_summary": "Summary of conversation so far...",
        "transcript_path": "/path",
        "cwd": "/project",
    }
    event = process_hook_input(json.dumps(hook_input))
    assert isinstance(event, PostCompactEvent)
    assert event.compact_summary_length == len("Summary of conversation so far...")


def test_process_subagent_start():
    hook_input = {
        "hook_event_name": "SubagentStart",
        "session_id": "sess-1",
        "agent_id": "agent-001",
        "agent_type": "general-purpose",
        "transcript_path": "/path",
        "cwd": "/project",
    }
    event = process_hook_input(json.dumps(hook_input))
    assert isinstance(event, SubagentStartEvent)
    assert event.agent_type == "general-purpose"


def test_process_instructions_loaded():
    hook_input = {
        "hook_event_name": "InstructionsLoaded",
        "session_id": "sess-1",
        "file_path": "/Users/me/.claude/CLAUDE.md",
        "memory_type": "project",
        "load_reason": "startup",
        "transcript_path": "/path",
        "cwd": "/project",
    }
    event = process_hook_input(json.dumps(hook_input))
    assert isinstance(event, InstructionsLoadedEvent)
    assert event.file_path == "/Users/me/.claude/CLAUDE.md"


def test_process_unknown_hook():
    hook_input = {
        "hook_event_name": "SomeNewHook",
        "session_id": "sess-1",
        "transcript_path": "/path",
        "cwd": "/project",
    }
    event = process_hook_input(json.dumps(hook_input))
    assert event is None


def test_process_malformed_input():
    event = process_hook_input("not json")
    assert event is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mcp-servers/context-tracker && pytest tests/test_hooks.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write hooks.py**

`src/context_tracker/hooks.py`:
```python
"""Hook processor: reads Claude Code HookInput from stdin, writes JSONL events."""

from __future__ import annotations

import json
import logging
import sys

from context_tracker.models import (
    BaseEvent,
    InstructionsLoadedEvent,
    PostCompactEvent,
    PostToolUseEvent,
    PostToolUseFailureEvent,
    PreCompactEvent,
    SessionEndEvent,
    SessionStartEvent,
    SubagentStartEvent,
    SubagentStopEvent,
    UserPromptEvent,
)
from context_tracker.storage import append_event

logger = logging.getLogger(__name__)


def _safe_len(value: object) -> int:
    """Get string length of a value, serializing dicts/lists to JSON first."""
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    if isinstance(value, (dict, list)):
        return len(json.dumps(value))
    return len(str(value))


def process_hook_input(raw_json: str) -> BaseEvent | None:
    """Parse a Claude Code HookInput JSON string into a typed event."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        logger.warning("Malformed hook input: %s", raw_json[:80])
        return None

    hook_event = data.get("hook_event_name")
    session_id = data.get("session_id", "unknown")

    if hook_event == "PostToolUse":
        return PostToolUseEvent(
            session_id=session_id,
            tool_name=data.get("tool_name", "unknown"),
            input_payload_chars=_safe_len(data.get("tool_input")),
            output_payload_chars=_safe_len(data.get("tool_response")),
            tool_use_id=data.get("tool_use_id", ""),
        )

    if hook_event == "PostToolUseFailure":
        return PostToolUseFailureEvent(
            session_id=session_id,
            tool_name=data.get("tool_name", "unknown"),
            input_payload_chars=_safe_len(data.get("tool_input")),
            error_length=_safe_len(data.get("error")),
            tool_use_id=data.get("tool_use_id", ""),
        )

    if hook_event == "SessionStart":
        return SessionStartEvent(
            session_id=session_id,
            source=data.get("source", "startup"),
            model=data.get("model", "unknown"),
        )

    if hook_event == "SessionEnd":
        return SessionEndEvent(
            session_id=session_id,
            reason=data.get("reason", "unknown"),
        )

    if hook_event == "UserPromptSubmit":
        return UserPromptEvent(
            session_id=session_id,
            prompt_length_chars=_safe_len(data.get("prompt")),
        )

    if hook_event == "PreCompact":
        return PreCompactEvent(
            session_id=session_id,
            trigger=data.get("trigger", "auto"),
        )

    if hook_event == "PostCompact":
        return PostCompactEvent(
            session_id=session_id,
            trigger=data.get("trigger", "auto"),
            compact_summary_length=_safe_len(data.get("compact_summary")),
        )

    if hook_event == "SubagentStart":
        return SubagentStartEvent(
            session_id=session_id,
            agent_id=data.get("agent_id", ""),
            agent_type=data.get("agent_type", ""),
        )

    if hook_event == "SubagentStop":
        return SubagentStopEvent(
            session_id=session_id,
            agent_id=data.get("agent_id", ""),
            agent_type=data.get("agent_type", ""),
            agent_transcript_path=data.get("agent_transcript_path", ""),
        )

    if hook_event == "InstructionsLoaded":
        return InstructionsLoadedEvent(
            session_id=session_id,
            file_path=data.get("file_path", ""),
            memory_type=data.get("memory_type", ""),
            load_reason=data.get("load_reason", ""),
        )

    logger.debug("Unhandled hook event: %s", hook_event)
    return None


def main() -> None:
    """Entry point: read HookInput JSON from stdin, write event to JSONL."""
    raw = sys.stdin.read().strip()
    if not raw:
        return

    event = process_hook_input(raw)
    if event is not None:
        append_event(event)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mcp-servers/context-tracker && pytest tests/test_hooks.py -v`
Expected: All 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add mcp-servers/context-tracker/src/context_tracker/hooks.py mcp-servers/context-tracker/tests/test_hooks.py
git commit -m "feat: add hook processor for Claude Code HookInput events"
```

---

### Task 5: Transcript Parser

**Files:**
- Create: `mcp-servers/context-tracker/src/context_tracker/transcript.py`
- Create: `mcp-servers/context-tracker/tests/test_transcript.py`

- [ ] **Step 1: Write the failing test**

`tests/test_transcript.py`:
```python
import json
import tempfile
from pathlib import Path

from context_tracker.transcript import parse_transcript
from context_tracker.models import ApiTurnEvent


def _write_transcript(path: Path, entries: list[dict]) -> None:
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def test_parse_transcript_extracts_usage(tmp_path):
    transcript_path = tmp_path / "session.jsonl"
    _write_transcript(transcript_path, [
        {
            "type": "user",
            "message": {"role": "user", "content": "hello"},
            "sessionId": "sess-1",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-6",
                "content": [{"type": "text", "text": "Hi!"}],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 45000,
                    "output_tokens": 200,
                    "cache_read_input_tokens": 38000,
                    "cache_creation_input_tokens": 2000,
                },
            },
            "sessionId": "sess-1",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-6",
                "content": [{"type": "text", "text": "More"}],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 46000,
                    "output_tokens": 150,
                    "cache_read_input_tokens": 40000,
                    "cache_creation_input_tokens": 1000,
                },
            },
            "sessionId": "sess-1",
        },
    ])

    events = parse_transcript(transcript_path)
    assert len(events) == 2
    assert all(isinstance(e, ApiTurnEvent) for e in events)
    assert events[0].turn_number == 1
    assert events[0].input_tokens == 45000
    assert events[0].cache_read_input_tokens == 38000
    assert events[1].turn_number == 2
    assert events[1].input_tokens == 46000


def test_parse_transcript_skips_non_assistant(tmp_path):
    transcript_path = tmp_path / "session.jsonl"
    _write_transcript(transcript_path, [
        {"type": "user", "message": {"role": "user", "content": "hi"}, "sessionId": "s"},
        {"type": "file-history-snapshot", "messageId": "x", "snapshot": {}},
    ])
    events = parse_transcript(transcript_path)
    assert events == []


def test_parse_transcript_skips_missing_usage(tmp_path):
    transcript_path = tmp_path / "session.jsonl"
    _write_transcript(transcript_path, [
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-6",
                "content": [{"type": "text", "text": "Hi"}],
                "stop_reason": "end_turn",
            },
            "sessionId": "sess-1",
        },
    ])
    events = parse_transcript(transcript_path)
    assert events == []


def test_parse_transcript_deduplicates_by_stop_reason(tmp_path):
    """Claude Code emits multiple assistant entries per API call (streaming chunks).
    Only entries with a non-null stop_reason and usage.output_tokens > 0
    represent completed turns."""
    transcript_path = tmp_path / "session.jsonl"
    _write_transcript(transcript_path, [
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-6",
                "content": [{"type": "text", "text": "chunk1"}],
                "stop_reason": None,
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 10,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 50,
                },
            },
            "sessionId": "sess-1",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-6",
                "content": [{"type": "text", "text": "final"}],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 200,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 50,
                },
            },
            "sessionId": "sess-1",
        },
    ])
    events = parse_transcript(transcript_path)
    assert len(events) == 1
    assert events[0].output_tokens == 200


def test_parse_nonexistent_file(tmp_path):
    events = parse_transcript(tmp_path / "nope.jsonl")
    assert events == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mcp-servers/context-tracker && pytest tests/test_transcript.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write transcript.py**

`src/context_tracker/transcript.py`:
```python
"""Parse Claude Code transcript JSONL files for exact API token usage."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from context_tracker.models import ApiTurnEvent

logger = logging.getLogger(__name__)

# Claude Code uses this model name for synthetic/internal messages
SYNTHETIC_MODEL = "synthetic"


def parse_transcript(transcript_path: Path) -> list[ApiTurnEvent]:
    """Extract API turn events from a Claude Code transcript JSONL file.

    Claude Code emits multiple assistant entries per API call as streaming
    chunks arrive. We only keep entries that have:
    - type == "assistant"
    - a non-null stop_reason (marks a completed API call)
    - a usage object with output_tokens > 0
    """
    if not transcript_path.exists():
        return []

    events: list[ApiTurnEvent] = []
    turn_number = 0

    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if entry.get("type") != "assistant":
                continue

            message = entry.get("message")
            if not isinstance(message, dict):
                continue

            usage = message.get("usage")
            if not isinstance(usage, dict):
                continue

            stop_reason = message.get("stop_reason")
            if stop_reason is None:
                continue

            output_tokens = usage.get("output_tokens", 0)
            if output_tokens == 0:
                continue

            model = message.get("model", "unknown")
            if model == SYNTHETIC_MODEL:
                continue

            session_id = entry.get("sessionId", "unknown")
            turn_number += 1

            events.append(
                ApiTurnEvent(
                    session_id=session_id,
                    turn_number=turn_number,
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=output_tokens,
                    cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
                    cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
                    model=model,
                    stop_reason=stop_reason,
                )
            )

    return events
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mcp-servers/context-tracker && pytest tests/test_transcript.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add mcp-servers/context-tracker/src/context_tracker/transcript.py mcp-servers/context-tracker/tests/test_transcript.py
git commit -m "feat: add transcript parser for exact API token usage extraction"
```

---

### Task 6: Hook Installer

**Files:**
- Create: `mcp-servers/context-tracker/src/context_tracker/installer.py`
- Create: `mcp-servers/context-tracker/tests/test_installer.py`

- [ ] **Step 1: Write the failing test**

`tests/test_installer.py`:
```python
import json
from pathlib import Path

from context_tracker.installer import (
    install_hooks,
    uninstall_hooks,
    CONTEXT_TRACKER_MARKER,
    HOOK_EVENTS_TO_INSTALL,
)


def test_install_into_empty_settings(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("{}")

    install_hooks(settings_path=settings_path, hook_command="python3 -m context_tracker.hooks")
    settings = json.loads(settings_path.read_text())

    assert "hooks" in settings
    for event_name in HOOK_EVENTS_TO_INSTALL:
        assert event_name in settings["hooks"]
        matchers = settings["hooks"][event_name]
        assert len(matchers) == 1
        assert CONTEXT_TRACKER_MARKER in matchers[0]["hooks"][0]["command"]


def test_install_preserves_existing_hooks(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({
        "hooks": {
            "PostToolUse": [
                {"matcher": "Bash", "hooks": [{"type": "command", "command": "echo existing"}]}
            ]
        },
        "someOtherSetting": True,
    }))

    install_hooks(settings_path=settings_path, hook_command="python3 -m context_tracker.hooks")
    settings = json.loads(settings_path.read_text())

    assert settings["someOtherSetting"] is True
    post_tool_matchers = settings["hooks"]["PostToolUse"]
    assert len(post_tool_matchers) == 2
    commands = [m["hooks"][0]["command"] for m in post_tool_matchers]
    assert "echo existing" in commands


def test_install_is_idempotent(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("{}")

    install_hooks(settings_path=settings_path, hook_command="python3 -m context_tracker.hooks")
    install_hooks(settings_path=settings_path, hook_command="python3 -m context_tracker.hooks")

    settings = json.loads(settings_path.read_text())
    for event_name in HOOK_EVENTS_TO_INSTALL:
        context_tracker_entries = [
            m for m in settings["hooks"][event_name]
            if any(CONTEXT_TRACKER_MARKER in h.get("command", "") for h in m.get("hooks", []))
        ]
        assert len(context_tracker_entries) == 1


def test_uninstall_removes_only_owned_hooks(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({
        "hooks": {
            "PostToolUse": [
                {"matcher": "Bash", "hooks": [{"type": "command", "command": "echo user-hook"}]},
                {"hooks": [{"type": "command", "command": f"# {CONTEXT_TRACKER_MARKER}\npython3 -m context_tracker.hooks"}]},
            ]
        }
    }))

    uninstall_hooks(settings_path=settings_path)
    settings = json.loads(settings_path.read_text())

    assert len(settings["hooks"]["PostToolUse"]) == 1
    assert "user-hook" in settings["hooks"]["PostToolUse"][0]["hooks"][0]["command"]


def test_uninstall_from_clean_settings(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("{}")

    uninstall_hooks(settings_path=settings_path)
    settings = json.loads(settings_path.read_text())
    assert settings == {}


def test_install_creates_backup(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text('{"existing": true}')

    install_hooks(settings_path=settings_path, hook_command="python3 -m context_tracker.hooks")
    backup = settings_path.with_suffix(".json.bak")
    assert backup.exists()
    assert json.loads(backup.read_text()) == {"existing": True}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mcp-servers/context-tracker && pytest tests/test_installer.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write installer.py**

`src/context_tracker/installer.py`:
```python
"""Install/uninstall context-tracker hooks into Claude Code settings.json."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

CONTEXT_TRACKER_MARKER = "context-tracker-hook"

DEFAULT_SETTINGS_PATH = Path.home() / ".claude" / "settings.json"

HOOK_EVENTS_TO_INSTALL = [
    "PostToolUse",
    "PostToolUseFailure",
    "PreCompact",
    "PostCompact",
    "SessionStart",
    "SessionEnd",
    "UserPromptSubmit",
    "SubagentStart",
    "SubagentStop",
    "InstructionsLoaded",
]


def _read_settings(settings_path: Path) -> dict:
    if not settings_path.exists():
        return {}
    return json.loads(settings_path.read_text(encoding="utf-8"))


def _write_settings(settings_path: Path, settings: dict) -> None:
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps(settings, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _is_owned_matcher(matcher: dict) -> bool:
    """Check if a hook matcher entry was installed by context-tracker."""
    for hook in matcher.get("hooks", []):
        if CONTEXT_TRACKER_MARKER in hook.get("command", ""):
            return True
    return False


def install_hooks(
    settings_path: Path = DEFAULT_SETTINGS_PATH,
    hook_command: str = "python3 -m context_tracker.hooks",
) -> None:
    """Install context-tracker hooks into settings.json. Merge-safe and idempotent."""
    settings = _read_settings(settings_path)

    # Create backup before modifying
    if settings_path.exists():
        backup_path = settings_path.with_suffix(".json.bak")
        shutil.copy2(settings_path, backup_path)

    hooks = settings.setdefault("hooks", {})

    for event_name in HOOK_EVENTS_TO_INSTALL:
        matchers = hooks.setdefault(event_name, [])

        # Remove any existing context-tracker entries (idempotent reinstall)
        matchers[:] = [m for m in matchers if not _is_owned_matcher(m)]

        # Add our hook
        matchers.append({
            "hooks": [
                {
                    "type": "command",
                    "command": f"# {CONTEXT_TRACKER_MARKER}\n{hook_command}",
                }
            ]
        })

    _write_settings(settings_path, settings)
    print(f"Installed {len(HOOK_EVENTS_TO_INSTALL)} hooks into {settings_path}")


def uninstall_hooks(settings_path: Path = DEFAULT_SETTINGS_PATH) -> None:
    """Remove context-tracker hooks from settings.json. Only removes owned entries."""
    settings = _read_settings(settings_path)
    hooks = settings.get("hooks", {})

    removed = 0
    for event_name in list(hooks.keys()):
        matchers = hooks[event_name]
        original_len = len(matchers)
        matchers[:] = [m for m in matchers if not _is_owned_matcher(m)]
        removed += original_len - len(matchers)

        # Clean up empty arrays
        if not matchers:
            del hooks[event_name]

    # Clean up empty hooks object
    if not hooks and "hooks" in settings:
        del settings["hooks"]

    _write_settings(settings_path, settings)
    print(f"Removed {removed} context-tracker hooks from {settings_path}")


def main() -> None:
    """CLI entry point: context-tracker install-hooks / uninstall-hooks."""
    if len(sys.argv) < 2 or sys.argv[1] not in ("install", "uninstall"):
        print("Usage: python -m context_tracker.installer [install|uninstall]")
        sys.exit(1)

    if sys.argv[1] == "install":
        install_hooks()
    else:
        uninstall_hooks()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mcp-servers/context-tracker && pytest tests/test_installer.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add mcp-servers/context-tracker/src/context_tracker/installer.py mcp-servers/context-tracker/tests/test_installer.py
git commit -m "feat: add merge-safe hook installer for Claude Code settings.json"
```

---

### Task 7: FastMCP Server with Query Tools

**Files:**
- Create: `mcp-servers/context-tracker/src/context_tracker/server.py`
- Create: `mcp-servers/context-tracker/tests/test_server.py`

- [ ] **Step 1: Write the failing test**

`tests/test_server.py`:
```python
import json
from pathlib import Path

import pytest

from context_tracker.models import (
    PostToolUseEvent,
    PostToolUseFailureEvent,
    PreCompactEvent,
    PostCompactEvent,
    SessionStartEvent,
    SessionEndEvent,
    UserPromptEvent,
    ApiTurnEvent,
)
from context_tracker.storage import append_event
from context_tracker.server import (
    get_session_summary,
    get_tool_breakdown,
    get_compaction_history,
    get_context_hogs,
    get_session_history,
)


@pytest.fixture
def populated_session(tmp_path):
    """Create a trace directory with a realistic session."""
    trace_dir = tmp_path / "traces"
    transcript_dir = tmp_path / "transcripts"
    transcript_dir.mkdir(parents=True)

    session_id = "test-session"

    append_event(SessionStartEvent(
        session_id=session_id, source="startup", model="claude-opus-4-6",
    ), trace_dir=trace_dir)

    for i, (name, in_size, out_size) in enumerate([
        ("Read", 50, 8000),
        ("Read", 60, 12000),
        ("Bash", 200, 3000),
        ("Grep", 80, 500),
        ("Read", 45, 6000),
    ]):
        append_event(PostToolUseEvent(
            session_id=session_id,
            tool_name=name,
            input_payload_chars=in_size,
            output_payload_chars=out_size,
            tool_use_id=f"toolu_{i}",
        ), trace_dir=trace_dir)

    append_event(PreCompactEvent(
        session_id=session_id, trigger="auto",
    ), trace_dir=trace_dir)

    append_event(PostCompactEvent(
        session_id=session_id, trigger="auto", compact_summary_length=1500,
    ), trace_dir=trace_dir)

    append_event(UserPromptEvent(
        session_id=session_id, prompt_length_chars=200,
    ), trace_dir=trace_dir)

    # Write a transcript file
    transcript_path = transcript_dir / f"{session_id}.jsonl"
    turns = [
        {"type": "assistant", "sessionId": session_id, "message": {
            "role": "assistant", "model": "claude-opus-4-6", "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "hi"}],
            "usage": {"input_tokens": 30000, "output_tokens": 500,
                       "cache_read_input_tokens": 25000, "cache_creation_input_tokens": 3000},
        }},
        {"type": "assistant", "sessionId": session_id, "message": {
            "role": "assistant", "model": "claude-opus-4-6", "stop_reason": "tool_use",
            "content": [{"type": "text", "text": "let me check"}],
            "usage": {"input_tokens": 45000, "output_tokens": 800,
                       "cache_read_input_tokens": 40000, "cache_creation_input_tokens": 1000},
        }},
    ]
    with open(transcript_path, "w") as f:
        for t in turns:
            f.write(json.dumps(t) + "\n")

    return session_id, trace_dir, transcript_dir


def test_get_session_summary(populated_session):
    session_id, trace_dir, transcript_dir = populated_session
    result = get_session_summary(session_id, trace_dir=trace_dir, transcript_dir=transcript_dir)
    assert result["session_id"] == session_id
    assert result["tool_calls"] == 5
    assert result["compactions"] == 1
    assert result["api_turns"] == 2
    assert result["total_input_tokens"] == 75000
    assert result["total_output_tokens"] == 1300


def test_get_tool_breakdown(populated_session):
    session_id, trace_dir, transcript_dir = populated_session
    result = get_tool_breakdown(session_id, trace_dir=trace_dir)
    assert len(result) == 3  # Read, Bash, Grep
    # Read should be first (highest total output)
    assert result[0]["tool_name"] == "Read"
    assert result[0]["call_count"] == 3
    assert result[0]["total_output_payload_chars"] == 26000


def test_get_compaction_history(populated_session):
    session_id, trace_dir, _ = populated_session
    result = get_compaction_history(session_id, trace_dir=trace_dir)
    assert len(result) == 1
    assert result[0]["trigger"] == "auto"
    assert result[0]["summary_length"] == 1500


def test_get_context_hogs(populated_session):
    session_id, trace_dir, _ = populated_session
    result = get_context_hogs(session_id, top_n=3, trace_dir=trace_dir)
    assert len(result) == 3
    # Largest output first
    assert result[0]["tool_name"] == "Read"
    assert result[0]["output_payload_chars"] == 12000


def test_get_session_history(populated_session):
    session_id, trace_dir, transcript_dir = populated_session
    result = get_session_history(trace_dir=trace_dir, transcript_dir=transcript_dir)
    assert len(result) == 1
    assert result[0]["session_id"] == session_id
    assert result[0]["tool_calls"] == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mcp-servers/context-tracker && pytest tests/test_server.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write server.py**

`src/context_tracker/server.py`:
```python
"""FastMCP server exposing context usage query tools."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from fastmcp import FastMCP

from context_tracker.models import (
    ApiTurnEvent,
    PostCompactEvent,
    PostToolUseEvent,
    PreCompactEvent,
    SessionStartEvent,
    TrackerEvent,
)
from context_tracker.storage import DEFAULT_TRACE_DIR, list_sessions, read_events
from context_tracker.transcript import parse_transcript

mcp = FastMCP(name="context-tracker", version="0.1.0")

DEFAULT_TRANSCRIPT_DIR = Path.home() / ".claude" / "projects"


def _find_transcript(
    session_id: str, transcript_dir: Path = DEFAULT_TRANSCRIPT_DIR
) -> Path | None:
    """Find a transcript JSONL file for a session ID across all project dirs."""
    # Direct match
    direct = transcript_dir / f"{session_id}.jsonl"
    if direct.exists():
        return direct

    # Search in subdirectories (Claude Code stores transcripts under project dirs)
    for jsonl_file in transcript_dir.rglob(f"{session_id}.jsonl"):
        return jsonl_file

    return None


def get_session_summary(
    session_id: str,
    trace_dir: Path = DEFAULT_TRACE_DIR,
    transcript_dir: Path = DEFAULT_TRANSCRIPT_DIR,
) -> dict:
    """Compute summary statistics for a session."""
    events = read_events(session_id, trace_dir=trace_dir)

    tool_calls = [e for e in events if isinstance(e, PostToolUseEvent)]
    compactions = [e for e in events if isinstance(e, (PreCompactEvent, PostCompactEvent))]
    post_compactions = [e for e in events if isinstance(e, PostCompactEvent)]
    starts = [e for e in events if isinstance(e, SessionStartEvent)]

    # Parse transcript for exact token counts
    transcript_path = _find_transcript(session_id, transcript_dir)
    api_turns: list[ApiTurnEvent] = []
    if transcript_path:
        api_turns = parse_transcript(transcript_path)

    total_input = sum(t.input_tokens for t in api_turns)
    total_output = sum(t.output_tokens for t in api_turns)
    total_cache_read = sum(t.cache_read_input_tokens for t in api_turns)
    total_cache_create = sum(t.cache_creation_input_tokens for t in api_turns)

    cache_hit_rate = 0.0
    cache_total = total_cache_read + total_cache_create + total_input
    if cache_total > 0:
        cache_hit_rate = round(total_cache_read / cache_total, 3)

    model = starts[0].model if starts else "unknown"

    # Duration from first to last event timestamp
    timestamps = [e.timestamp for e in events if e.timestamp]
    duration_seconds = None
    if len(timestamps) >= 2:
        from datetime import datetime

        try:
            first = datetime.fromisoformat(timestamps[0])
            last = datetime.fromisoformat(timestamps[-1])
            duration_seconds = round((last - first).total_seconds())
        except ValueError:
            pass

    return {
        "session_id": session_id,
        "model": model,
        "tool_calls": len(tool_calls),
        "compactions": len(post_compactions),
        "total_events": len(events),
        "api_turns": len(api_turns),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_cache_read_tokens": total_cache_read,
        "total_cache_creation_tokens": total_cache_create,
        "cache_hit_rate": cache_hit_rate,
        "duration_seconds": duration_seconds,
    }


def get_tool_breakdown(
    session_id: str,
    trace_dir: Path = DEFAULT_TRACE_DIR,
) -> list[dict]:
    """Rank tools by total output payload size."""
    events = read_events(session_id, trace_dir=trace_dir)
    tool_calls = [e for e in events if isinstance(e, PostToolUseEvent)]

    by_tool: dict[str, dict] = defaultdict(lambda: {
        "call_count": 0,
        "total_input_payload_chars": 0,
        "total_output_payload_chars": 0,
    })

    for tc in tool_calls:
        entry = by_tool[tc.tool_name]
        entry["call_count"] += 1
        entry["total_input_payload_chars"] += tc.input_payload_chars
        entry["total_output_payload_chars"] += tc.output_payload_chars

    result = [
        {"tool_name": name, **stats}
        for name, stats in by_tool.items()
    ]
    result.sort(key=lambda x: x["total_output_payload_chars"], reverse=True)
    return result


def get_compaction_history(
    session_id: str,
    trace_dir: Path = DEFAULT_TRACE_DIR,
) -> list[dict]:
    """Timeline of compaction events."""
    events = read_events(session_id, trace_dir=trace_dir)

    result = []
    for e in events:
        if isinstance(e, PostCompactEvent):
            result.append({
                "timestamp": e.timestamp,
                "trigger": e.trigger,
                "summary_length": e.compact_summary_length,
            })
    return result


def get_context_hogs(
    session_id: str,
    top_n: int = 10,
    trace_dir: Path = DEFAULT_TRACE_DIR,
) -> list[dict]:
    """Top N tool calls by output payload size."""
    events = read_events(session_id, trace_dir=trace_dir)
    tool_calls = [e for e in events if isinstance(e, PostToolUseEvent)]
    tool_calls.sort(key=lambda e: e.output_payload_chars, reverse=True)

    return [
        {
            "tool_name": tc.tool_name,
            "output_payload_chars": tc.output_payload_chars,
            "input_payload_chars": tc.input_payload_chars,
            "tool_use_id": tc.tool_use_id,
            "timestamp": tc.timestamp,
        }
        for tc in tool_calls[:top_n]
    ]


def get_session_history(
    trace_dir: Path = DEFAULT_TRACE_DIR,
    transcript_dir: Path = DEFAULT_TRANSCRIPT_DIR,
) -> list[dict]:
    """List all sessions with summary stats."""
    session_ids = list_sessions(trace_dir=trace_dir)
    return [
        get_session_summary(sid, trace_dir=trace_dir, transcript_dir=transcript_dir)
        for sid in session_ids
    ]


# --- MCP Tool Registrations ---


@mcp.tool(description="Get summary statistics for a Claude Code session including tool calls, compactions, exact API token counts, and cache efficiency.")
def mcp_get_session_summary(session_id: str = "") -> str:
    """If session_id is empty, uses the most recent session."""
    import json

    if not session_id:
        sessions = list_sessions()
        if not sessions:
            return json.dumps({"error": "No sessions found"})
        session_id = sessions[0]
    return json.dumps(get_session_summary(session_id), indent=2)


@mcp.tool(description="Get ranked breakdown of tools by payload size and call frequency for a session.")
def mcp_get_tool_breakdown(session_id: str = "") -> str:
    import json

    if not session_id:
        sessions = list_sessions()
        if not sessions:
            return json.dumps({"error": "No sessions found"})
        session_id = sessions[0]
    return json.dumps(get_tool_breakdown(session_id), indent=2)


@mcp.tool(description="Get timeline of compaction events for a session.")
def mcp_get_compaction_history(session_id: str = "") -> str:
    import json

    if not session_id:
        sessions = list_sessions()
        if not sessions:
            return json.dumps({"error": "No sessions found"})
        session_id = sessions[0]
    return json.dumps(get_compaction_history(session_id), indent=2)


@mcp.tool(description="Get the top N tool calls by output payload size for a session.")
def mcp_get_context_hogs(session_id: str = "", top_n: int = 10) -> str:
    import json

    if not session_id:
        sessions = list_sessions()
        if not sessions:
            return json.dumps({"error": "No sessions found"})
        session_id = sessions[0]
    return json.dumps(get_context_hogs(session_id, top_n=top_n), indent=2)


@mcp.tool(description="List all tracked sessions with summary stats.")
def mcp_get_session_history() -> str:
    import json

    return json.dumps(get_session_history(), indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Context Tracker MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "http"],
        default="stdio",
        help="Transport mode (default: stdio)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9200)
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run()
    else:
        mcp.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mcp-servers/context-tracker && pytest tests/test_server.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add mcp-servers/context-tracker/src/context_tracker/server.py mcp-servers/context-tracker/tests/test_server.py
git commit -m "feat: add FastMCP server with 5 context usage query tools"
```

---

### Task 8: MCP Inspector Testing

**Files:**
- None created. This task validates the server using MCP Inspector.

- [ ] **Step 1: Install MCP Inspector if needed**

Run: `npx @modelcontextprotocol/inspector --help 2>/dev/null || npm install -g @modelcontextprotocol/inspector`
Expected: Inspector is available

- [ ] **Step 2: Start server and test with Inspector**

Run: `cd mcp-servers/context-tracker && npx @modelcontextprotocol/inspector python3 -m context_tracker.server`
Expected: Inspector opens in browser, shows 5 tools listed

- [ ] **Step 3: Test each tool through Inspector UI**

In the Inspector UI:
1. Click `mcp_get_session_history` and invoke it. Expected: returns `[]` or a list of sessions.
2. Click `mcp_get_session_summary` with empty `session_id`. Expected: returns error or most recent session.
3. Verify all 5 tools appear and have descriptions.

- [ ] **Step 4: Register MCP server in Claude Code settings**

Add to `~/.claude/settings.json` under `mcpServers`:
```json
{
  "mcpServers": {
    "context-tracker": {
      "type": "stdio",
      "command": "python3",
      "args": ["-m", "context_tracker.server"],
      "cwd": "<absolute-path-to>/mcp-servers/context-tracker"
    }
  }
}
```

- [ ] **Step 5: Commit**

No files changed. Record test results in a comment on the issue.

---

### Task 9: Install Hooks and Collect Real Traces

**Files:**
- None created. This task validates the full pipeline end-to-end.

- [ ] **Step 1: Install hooks**

Run: `cd mcp-servers/context-tracker && python3 -m context_tracker.installer install`
Expected: "Installed 10 hooks into ~/.claude/settings.json"

- [ ] **Step 2: Verify hooks in settings.json**

Run: `python3 -c "import json; s=json.load(open('$HOME/.claude/settings.json')); print(list(s.get('hooks',{}).keys()))"`
Expected: All 10 hook events listed

- [ ] **Step 3: Run a short Claude Code session to generate traces**

Start a new Claude Code session, perform a few tool calls (Read a file, run a Bash command), then exit.

- [ ] **Step 4: Verify JSONL trace was written**

Run: `ls ~/.claude/context-trace/ && head -5 ~/.claude/context-trace/*.jsonl`
Expected: JSONL file exists with event entries

- [ ] **Step 5: Test the MCP query tools against real data**

In a Claude Code session with the MCP server registered, call `mcp_get_session_summary` and verify it returns real data with token counts.

- [ ] **Step 6: Document findings**

Comment on GitHub issue #1 with:
- Sample JSONL output
- Sample query tool response
- Any data model issues discovered
- Whether the transcript parser correctly extracts token counts

---

### Task 10: Run All Tests

**Files:**
- None created.

- [ ] **Step 1: Run the full test suite**

Run: `cd mcp-servers/context-tracker && pytest tests/ -v --cov=context_tracker --cov-report=term-missing`
Expected: All tests pass, coverage report shows coverage for models, storage, hooks, transcript, installer, server

- [ ] **Step 2: Run linting**

Run: `cd mcp-servers/context-tracker && ruff check src/ tests/`
Expected: No linting errors

- [ ] **Step 3: Commit any fixes**

If any tests or lint issues were found, fix and commit:
```bash
git add -u
git commit -m "fix: address test and lint issues"
```
