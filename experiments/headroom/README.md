# Headroom measurement experiment (issue #81)

Scaffolding for independently measuring what `headroom wrap claude` actually does to
a Claude Code session, with context-analyzer as the measurement instrument. Read
`METHODOLOGY.md` before running anything.

> **WARNING — real API spend.** Every pair launches two full non-interactive Claude
> Code sessions. Nothing in this directory runs automatically; executing the
> experiment is a deliberate human decision.

> **WARNING — publication.** Results must be published together with
> `METHODOLOGY.md` and the raw `manifest.jsonl`. A headline number without the
> matched-pair design, exclusion rule, and version pins is exactly the kind of
> unfalsifiable claim this experiment exists to correct.

## Prerequisites

1. **This repo's venv** (for pyyaml, ingestion, and `analyze.py --ingest`):

   ```bash
   make install-dev        # at repo root; requires uv
   ```

2. **Claude Code CLI** — `claude --version` must work and you must be logged in.
   Print mode (`claude -p`) with `--output-format json` is how sessions are launched
   and how session ids are captured.

3. **headroom, pinned.** Install the exact version you intend to publish against and
   record it (run_pair.sh records `headroom --version` in the manifest
   automatically). Use headroom's documented install method, e.g.:

   ```bash
   npm install -g headroom@<PINNED_VERSION>   # confirm against headroom's README
   headroom --version
   ```

   Also confirm the wrap argv form (`headroom wrap claude -- -p ...`) against that
   pinned version and adjust `run_pair.sh` (`HEADROOM_CMD` comment) if it differs.
   Do **not** upgrade headroom mid-experiment.

4. **Transcripts available for ingestion.** context-analyzer ingests sessions from
   `~/.claude/projects/<project>/<session-id>.jsonl` — this happens automatically
   for any Claude Code session. Optional: `make hook-install` for richer hook-event
   data; not required for the core metrics.

## Run order

```bash
cd experiments/headroom

# 1. Sanity-check the harness without spending anything:
./run_pair.sh tasks/01-code-search.yaml 1 --dry-run

# 2. Run all 5 task pairs (pin the model! see METHODOLOGY.md):
for t in tasks/*.yaml; do
  ./run_pair.sh "$t" 1 --model claude-sonnet-4-5
done

# 3. (Recommended) repeat for a second round of pairs:
for t in tasks/*.yaml; do
  ./run_pair.sh "$t" 2 --model claude-sonnet-4-5
done

# 4. Ingest + analyze (use the venv python so --ingest can import context_tracker):
../../.venv/bin/python analyze.py --manifest manifest.jsonl --ingest --out results.md
```

`run_pair.sh` randomizes which arm runs first (recorded in the manifest); pass
`--order PH` / `--order HP` only if you are deliberately testing ordering effects.

Outputs:

- `manifest.jsonl` — one row per arm: session id, arm, order, success, version pins.
  This is the join key between the experiment and the context-analyzer DB. Publish it.
- `results.md` — per-pair and aggregate markdown tables (from `analyze.py`).
- `workspaces/` — throwaway per-arm clones. Safe to delete after analysis; do not
  commit (gitignored along with the manifest and results).

## Expected cost

Rough estimate, assuming Sonnet-class pricing and moderately complex tasks
(~$0.50–$3 per non-interactive session at these task sizes):

- 5 tasks x 2 arms x 1 repetition ≈ 10 sessions ≈ **$5–30**
- Recommended 2 repetitions ≈ 20 sessions ≈ **$10–60**

Opus-class models multiply this ~5x. The test-fix and refactor tasks also run
`make install-dev` per workspace (disk/time, not API cost).

## Contents

| File | Purpose |
| --- | --- |
| `METHODOLOGY.md` | Matched-pair design, metrics, threats to validity. Read first. |
| `tasks/*.yaml` | 5 task definitions (code-search, file-read, test-fix, refactor, issue-triage): prompt + pinned repo setup + objective success check. |
| `run_pair.sh` | Runs one pair (both arms, randomized order), records session ids into `manifest.jsonl`. |
| `analyze.py` | Joins manifest + SQLite DB; emits per-pair and aggregate markdown tables. Tested in `tests/test_headroom_analyze.py` against the real schema. |
