import json

from context_tracker.installer import (
    CONTEXT_TRACKER_MARKER,
    HOOK_EVENTS_TO_INSTALL,
    install_hooks,
    uninstall_hooks,
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
