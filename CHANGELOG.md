# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Hook interpreter is now pinned and quoted.** The hooks installed into the
  global `~/.claude/settings.json` previously ran via a bare `python3`, which
  resolved to whatever interpreter the active session happened to have on
  `PATH` — usually one that cannot import `context_tracker`, so the hook died
  with `ModuleNotFoundError` (#55/#56). The hook command now pins to the
  absolute interpreter that ran `context-tracker install` (`sys.executable`).
- **The interpreter path is now shell-quoted** (#57). An interpreter path
  containing a space (normal on Windows, e.g.
  `C:\Program Files\...\python.exe`) previously broke the hook when Claude Code
  ran it through a shell, because the path split into multiple tokens. The path
  is now quoted (`shlex.quote` on POSIX, double-quoted on Windows).

### Changed

- Installer tool-specific configuration (settings path, marker, event set, hook
  command) is now captured in an `InstallerProfile` (`context_tracker.profiles`)
  instead of being hardcoded in `installer.py`. This is the seam for future
  multi-tool support (e.g. a Codex adapter, #59). Behavior for Claude Code is
  unchanged.

### Upgrading

> [!IMPORTANT]
> If you installed hooks with an earlier version, your
> `~/.claude/settings.json` still contains the old, broken bare-`python3` hook
> command. **Re-run the installer to replace it** with the pinned, quoted
> interpreter:
>
> ```bash
> context-tracker install
> # or, from a checkout:
> make hook-install
> ```
>
> Reinstalling is idempotent and merge-safe: it only replaces context-tracker's
> own hook entries (and writes a `settings.json.bak` backup first), leaving any
> other hooks you have configured untouched.

## [1.0.0]

- Initial public release.
