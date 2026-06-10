"""Install/uninstall context-tracker hooks into Claude Code settings.json."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

from context_tracker.profiles import (
    CLAUDE_PROFILE,
    CONTEXT_TRACKER_MARKER,
    DEFAULT_SETTINGS_PATH,
    HOOK_EVENTS_TO_INSTALL,
    InstallerProfile,
    default_hook_command,
)

# Re-exported for backward compatibility. The canonical definitions now live in
# context_tracker.profiles (the seam for multi-tool support); these names are
# kept here so existing imports of ``from context_tracker.installer import ...``
# keep working unchanged.
__all__ = [
    "CONTEXT_TRACKER_MARKER",
    "DEFAULT_SETTINGS_PATH",
    "HOOK_EVENTS_TO_INSTALL",
    "install_hooks",
    "main",
    "uninstall_hooks",
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


def _is_owned_matcher(matcher: dict, marker: str = CONTEXT_TRACKER_MARKER) -> bool:
    """Check if a hook matcher entry was installed by context-tracker."""
    for hook in matcher.get("hooks", []):
        if marker in hook.get("command", ""):
            return True
    return False


def _default_hook_command() -> str:
    """Hook command pinned to the (quoted) interpreter running the installer.

    Kept as a thin wrapper around :func:`context_tracker.profiles.default_hook_command`
    for backward compatibility. See that function for the rationale.
    """
    return default_hook_command()


def install_hooks(
    settings_path: Path | None = None,
    hook_command: str | None = None,
    profile: InstallerProfile = CLAUDE_PROFILE,
) -> None:
    """Install context-tracker hooks into settings.json. Merge-safe and idempotent.

    The tool-specific knobs (settings file, marker, event set, default command)
    come from ``profile`` (Claude Code by default). The ``settings_path`` and
    ``hook_command`` arguments override the profile's values when provided, so
    existing callers keep working identically.
    """
    if settings_path is None:
        settings_path = profile.settings_path
    if hook_command is None:
        hook_command = profile.hook_command()
    marker = profile.marker
    event_names = profile.event_names

    settings = _read_settings(settings_path)

    # Create backup before modifying
    if settings_path.exists():
        backup_path = settings_path.with_suffix(".json.bak")
        shutil.copy2(settings_path, backup_path)

    hooks = settings.setdefault("hooks", {})

    for event_name in event_names:
        matchers = hooks.setdefault(event_name, [])

        # Remove any existing context-tracker entries (idempotent reinstall)
        matchers[:] = [m for m in matchers if not _is_owned_matcher(m, marker)]

        # Add our hook
        matchers.append(
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": f"# {marker}\n{hook_command}",
                    }
                ]
            }
        )

    _write_settings(settings_path, settings)
    print(f"Installed {len(event_names)} hooks into {settings_path}")


def uninstall_hooks(
    settings_path: Path | None = None,
    profile: InstallerProfile = CLAUDE_PROFILE,
) -> None:
    """Remove context-tracker hooks from settings.json. Only removes owned entries."""
    if settings_path is None:
        settings_path = profile.settings_path
    marker = profile.marker

    settings = _read_settings(settings_path)
    hooks = settings.get("hooks", {})

    removed = 0
    for event_name in list(hooks.keys()):
        matchers = hooks[event_name]
        original_len = len(matchers)
        matchers[:] = [m for m in matchers if not _is_owned_matcher(m, marker)]
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
