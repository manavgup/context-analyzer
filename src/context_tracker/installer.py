"""Install/uninstall context-tracker hooks into Claude Code settings.json."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

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


def _read_settings(settings_path: Path) -> dict[str, Any]:
    if not settings_path.exists():
        return {}
    result = json.loads(settings_path.read_text(encoding="utf-8"))
    return dict(result) if isinstance(result, dict) else {}


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


def _default_hook_command() -> str:
    """Hook command pinned to the interpreter running the installer.

    The hooks are installed into the *global* ~/.claude/settings.json and fire in
    every Claude Code session, regardless of which project (or virtualenv) is
    active. A bare ``python3`` resolves to whatever interpreter that session
    happens to have on PATH, which usually cannot import context_tracker (it is
    only installed in this project's venv) and the hook dies with
    ModuleNotFoundError. Pinning to ``sys.executable`` — the interpreter the user
    ran ``context-tracker install`` from, which by definition has the package
    installed — makes the hook resolve correctly from any session.
    """
    return f"{sys.executable} -m context_tracker.hooks"


def install_hooks(
    settings_path: Path = DEFAULT_SETTINGS_PATH,
    hook_command: str | None = None,
) -> None:
    """Install context-tracker hooks into settings.json. Merge-safe and idempotent."""
    if hook_command is None:
        hook_command = _default_hook_command()
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
        matchers.append(
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": f"# {CONTEXT_TRACKER_MARKER}\n{hook_command}",
                    }
                ]
            }
        )

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
