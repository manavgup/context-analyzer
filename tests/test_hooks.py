import json

from context_tracker.hooks import process_hook_input
from context_tracker.models import (
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


def test_process_subagent_stop():
    hook_input = {
        "hook_event_name": "SubagentStop",
        "session_id": "sess-1",
        "agent_id": "agent-001",
        "agent_type": "general-purpose",
        "agent_transcript_path": "/path/to/agent.jsonl",
        "transcript_path": "/path",
        "cwd": "/project",
    }
    event = process_hook_input(json.dumps(hook_input))
    assert isinstance(event, SubagentStopEvent)
    assert event.agent_transcript_path == "/path/to/agent.jsonl"


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
