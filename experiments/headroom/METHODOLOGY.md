# Headroom Token-Savings Measurement — Methodology

Issue: [#81](https://github.com/manavgup/context-analyzer/issues/81)

headroom claims 15–95% token savings for coding-agent sessions. Independent reports
disagree (neutral-to-negative results, net cost increases, higher *overall* context use
despite per-turn reductions). This experiment measures what `headroom wrap claude`
actually does to a Claude Code session, using API-reported token counts captured by
context-analyzer as ground truth.

## Design: matched pairs

The unit of measurement is a **task pair**: the same task executed twice —

- **Arm P (plain):** `claude -p "<prompt>"` (non-interactive print mode)
- **Arm H (headroom):** `headroom wrap claude -- -p "<prompt>"` (same prompt, same flags)

Both arms of a pair start from an **identical, freshly created workspace** (same repo,
same pinned commit, same injected state — see `tasks/*.yaml` `setup:` blocks). Nothing
is shared between arms: each arm gets its own clone, its own Claude Code session, and
its own session id.

**n ≥ 5 task pairs** are required (`tasks/` ships 5 definitions of deliberately varied
shape); each task definition can be repeated to raise n further (see Repetition below).

### Why matched pairs

Absolute token counts vary enormously between tasks (a test-fix loop is not comparable
to an issue triage). Pairing on the task removes between-task variance; the estimand is
the **within-pair difference** (headroom − plain) per metric, aggregated across pairs.

## Metrics (per arm)

All token/cost numbers are **API-reported** values ingested by context-analyzer into
SQLite (`~/.context-analyzer/analyzer.db`) — not estimates. Column sources are the
schema in `src/context_tracker/db.py`.

| Metric | Source | What it tests |
| --- | --- | --- |
| Total input tokens | `sessions.total_input_tokens` | headline savings claim |
| Total output tokens | `sessions.total_output_tokens` | does compression change what the model says? |
| Total $ cost | `sessions.total_cost_usd` | the number that actually matters |
| Cache hit rate | `total_cache_read / (total_input_tokens + total_cache_read + total_cache_creation)` | headroom's **CacheAligner** claim — does rewriting the context bust Anthropic prompt caches? A drop here can wipe out raw-token savings, because cache reads are ~10x cheaper than uncached input. |
| Cache creation tokens | `sessions.total_cache_creation` | cache churn — repeated re-writing of cache segments |
| headroom_retrieve round-trips | count of `blocks` rows with `block_type = 'tool_call'` and a label matching the retrieval tool (default pattern `%headroom%retriev%`) | CCR retrievals clawing back savings: every retrieval is an extra API round-trip *plus* re-injected content |
| Peak context | `sessions.peak_context_tokens` | does the wrapped session actually run smaller? |
| API call count | `sessions.total_api_calls` | round-trip inflation |
| Task success | objective `success_check` command from the task YAML, exit code recorded in the manifest | outcome parity — savings on a failed task are worthless |

### Success parity is a gate, not a metric

Token comparisons are only reported for pairs where **both arms** completed the task
(success_check passed). Pairs with divergent outcomes are reported separately as
outcome-parity failures — they are themselves a finding.

## Threats to validity, and mitigations

| Threat | Description | Mitigation |
| --- | --- | --- |
| Session nondeterminism | Same prompt can take different paths (different tool calls, different file reads), producing different token totals unrelated to headroom. | (a) Fixed model per experiment run (pin with `--model`, record in manifest). (b) Objective success checks so divergent-outcome pairs are excluded from the savings estimate. (c) **Repetition**: run each task definition k times (k ≥ 2 recommended) — every repetition is a new pair; report per-pair deltas, not single runs. Paired design means noise shows up as spread in the deltas, which we report. |
| Ordering effects | Running one arm before the other could matter (warm Anthropic prompt caches from a recent similar prompt, background machine load, time-of-day API behavior). | **Randomized arm order per pair** (`run_pair.sh` defaults to `--order random`); the order actually used is recorded in the manifest so it can be checked as a covariate. Fresh workspace per arm removes filesystem-level ordering effects. |
| headroom version drift | headroom moves fast; results are meaningless without a pinned version. | `run_pair.sh` records `headroom --version` and `claude --version` into the manifest for every pair. Install a pinned version before starting (see README) and do not upgrade mid-experiment. |
| Target repo drift | Task difficulty changes if the target repo changes. | All tasks pin an exact commit SHA in their `setup:` block. |
| Prompt-cache contamination across pairs | A previous session may leave Anthropic-side cache entries that benefit whichever arm runs next. | Anthropic prompt caches are prefix-based and TTL-bounded (~5 min default); randomized order plus the natural gap between arms (workspace setup) reduces systematic bias. Optionally insert a fixed ≥5-minute delay between arms. |
| Measurement-tool bias | context-analyzer itself must not favor an arm. | context-analyzer only *reads* transcripts after the fact; it injects nothing into either session. Both arms are measured by the identical ingest path. |
| Experimenter degrees of freedom | Cherry-picking pairs or metrics after seeing results. | Metrics and exclusion rule (success parity) are fixed in this document *before* the experiment runs. Publish the full manifest and all per-pair rows, including excluded pairs. |

## Procedure

1. Complete the prerequisites in `README.md` (pinned headroom, `claude` CLI, this repo's
   venv).
2. For each task file in `tasks/`, for each repetition:
   `./run_pair.sh tasks/<task>.yaml <pair-index>` — this sets up both workspaces, runs
   both arms in randomized order, runs the success check, and appends both session ids
   to `manifest.jsonl`.
3. Ingest + analyze: `python analyze.py --manifest manifest.jsonl` (add `--ingest` to
   ingest the sessions from `~/.claude/projects` transcripts first). This emits the
   per-pair and aggregate markdown tables.
4. Write-up uses those tables verbatim plus dashboard chart exports; publish alongside
   this METHODOLOGY.md and the manifest.

## Analysis

`analyze.py` reports, per metric:

- **Per-pair table:** plain value, headroom value, absolute delta, % delta.
- **Aggregate:** mean of per-pair deltas, mean % delta, and the sign count
  (pairs where headroom was lower / higher / tied). With small n, the sign count is the
  honest summary; no p-values are reported below n = 10 pairs.
- **Outcome parity:** count of pairs where both / only-plain / only-headroom succeeded.

Interpretation guardrails for the write-up:

- Report cost, not just tokens: a session that swaps cheap cache reads for uncached
  input can show "token savings" while costing more.
- Report retrieval round-trips next to input-token savings: per-turn reduction with
  more turns is not a saving.
- Never aggregate across pairs with divergent task outcomes.
