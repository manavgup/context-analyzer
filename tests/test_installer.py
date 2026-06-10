import json
import shlex
import sys

from context_tracker.installer import (
    CONTEXT_TRACKER_MARKER,
    HOOK_EVENTS_TO_INSTALL,
    install_hooks,
    uninstall_hooks,
)
from context_tracker.profiles import default_hook_command


def test_default_hook_command_pins_interpreter(tmp_path):
    """The default command must use the absolute interpreter path, not bare python3.

    The hooks run globally in every session; a bare ``python3`` resolves to an
    interpreter that usually cannot import context_tracker, so the hook fails with
    ModuleNotFoundError. Pinning to sys.executable fixes that. Regression guard.
    """
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("{}")

    install_hooks(settings_path=settings_path)
    settings = json.loads(settings_path.read_text())

    command = settings["hooks"]["SessionStart"][0]["hooks"][0]["command"]
    assert sys.executable in command
    assert "-m context_tracker.hooks" in command
    # Must not fall back to a bare interpreter name.
    assert "\npython3 -m" not in command


def test_default_hook_command_quotes_spaced_interpreter(monkeypatch):
    """A spaced interpreter path (normal on Windows) must stay one shell token.

    ``sys.executable`` like ``C:\\Program Files\\...\\python.exe`` contains a
    space; interpolated unquoted into the hook command it splits into two tokens
    and the hook fails. The command must quote the interpreter so a POSIX shell
    parses it back as the single original path. Regression guard for #57.
    """
    spaced = "/a b/python"
    monkeypatch.setattr(sys, "executable", spaced)

    command = default_hook_command()
    # On POSIX the command must be shlex-parseable back to the original argv.
    assert shlex.split(command) == [spaced, "-m", "context_tracker.hooks"]
    # The raw (unquoted) path must not appear verbatim — it had to be quoted.
    assert f"{spaced} -m" not in command


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
    settings_path.write_text(
        json.dumps(
            {
                "hooks": {
                    "PostToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "echo existing"}]}]
                },
                "someOtherSetting": True,
            }
        )
    )

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
            m
            for m in settings["hooks"][event_name]
            if any(CONTEXT_TRACKER_MARKER in h.get("command", "") for h in m.get("hooks", []))
        ]
        assert len(context_tracker_entries) == 1


def test_uninstall_removes_only_owned_hooks(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "hooks": {
                    "PostToolUse": [
                        {"matcher": "Bash", "hooks": [{"type": "command", "command": "echo user-hook"}]},
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": (f"# {CONTEXT_TRACKER_MARKER}\npython3 -m context_tracker.hooks"),
                                }
                            ]
                        },
                    ]
                }
            }
        )
    )

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
