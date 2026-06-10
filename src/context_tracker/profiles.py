"""Installer profiles: the seam for multi-tool hook support.

context-tracker currently installs hooks into Claude Code's global
``~/.claude/settings.json``. The tool-specific knobs — which settings file to
write, what marker comment identifies our entries, which lifecycle events to
register, and how to spell the hook command — are captured here in an
:class:`InstallerProfile` rather than hardcoded in ``installer.py``.

This is Phase 0 of the Codex adapter work (issue #59): pure decoupling, no
behavior change. The only profile that exists today is :data:`CLAUDE_PROFILE`,
which reproduces the previously-hardcoded Claude Code values byte-for-byte.
Phase 1 will add a ``CodexProfile`` targeting Codex's own settings file and
event set, reusing the same install/uninstall machinery without forking it.
"""

from __future__ import annotations

import os
import shlex
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
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


def default_hook_command() -> str:
    """Hook command pinned to the interpreter running the installer.

    The hooks are installed into the *global* ~/.claude/settings.json and fire in
    every Claude Code session, regardless of which project (or virtualenv) is
    active. A bare ``python3`` resolves to whatever interpreter that session
    happens to have on PATH, which usually cannot import context_tracker (it is
    only installed in this project's venv) and the hook dies with
    ModuleNotFoundError. Pinning to ``sys.executable`` — the interpreter the user
    ran ``context-tracker install`` from, which by definition has the package
    installed — makes the hook resolve correctly from any session.

    The interpreter path is quoted so a path containing spaces (e.g. the Windows
    ``C:\\Program Files\\...\\python.exe``) does not break the command when Claude
    Code runs it through a shell. On POSIX we use ``shlex.quote`` (single-quote
    rules); on Windows we wrap in double quotes, since the command may be invoked
    via ``cmd.exe`` where POSIX single-quoting is not honored.
    """
    return f"{_quote_executable(sys.executable)} -m context_tracker.hooks"


def _quote_executable(executable: str) -> str:
    """Quote an interpreter path for safe use inside a shell hook command."""
    if os.name == "nt":
        # cmd.exe does not honor POSIX single-quote rules; use double quotes.
        return f'"{executable}"'
    return shlex.quote(executable)


@dataclass(frozen=True)
class InstallerProfile:
    """Tool-specific configuration for installing context-tracker hooks.

    Bundles everything that varies between the tools we can target (Claude Code
    today, Codex tomorrow):

    - ``settings_path``: the JSON settings file to merge hooks into.
    - ``marker``: the comment string that tags our hook entries so reinstall is
      idempotent and uninstall only removes our own entries.
    - ``event_names``: the lifecycle events to register the hook under.
    - ``command_factory``: a zero-arg callable returning the hook command string.
      A callable (rather than a static template) lets the command be computed at
      install time — e.g. pinned to the current ``sys.executable``.
    """

    settings_path: Path
    marker: str
    event_names: list[str]
    command_factory: Callable[[], str] = field(repr=False)

    def hook_command(self) -> str:
        """Produce the hook command string for this profile."""
        return self.command_factory()


CLAUDE_PROFILE = InstallerProfile(
    settings_path=DEFAULT_SETTINGS_PATH,
    marker=CONTEXT_TRACKER_MARKER,
    event_names=HOOK_EVENTS_TO_INSTALL,
    command_factory=default_hook_command,
)
