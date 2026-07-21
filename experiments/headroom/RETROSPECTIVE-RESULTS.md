# Retrospective Compression Audit — Ceiling Report (Stage 0 of #94)

**Verdict up front: the kill criterion is MET for the simulated
configuration.** Replaying this machine's real Claude Code workload through
headroom's local compressors (shipped defaults, ML prose model excluded)
yields a savings ceiling of **3.03% of resident token-call volume ($192.06
of the $7,360.53 recorded corpus)** — well below the 10%-of-spend threshold
set in #94. That 3.03% is an upper bound that assumes zero retrieval
clawback, zero prompt-cache damage, and zero answer degradation. One
quantified caveat: the unsimulated ML prose path could in principle add up
to ~10.8 points at an (optimistic) 50% prose ratio — see "Prose
sensitivity" below before treating the verdict as all-in.

## Setup

- **headroom-ai:** `0.32.1` (pinned; installed with `pip install --no-deps
  headroom-ai` — the `litellm` dependency has no macOS wheels and is only
  needed for headroom's proxy/LLM paths, which this offline audit never
  touches). Public API used: `headroom.compress()` with
  `CompressConfig(protect_recent=0, kompress_model="disabled")`; everything
  else at shipped defaults.
- **Active compressors** (from headroom's own routing markers): SmartCrusher
  (JSON/structured), lossless log/search/paths/diff/config compaction,
  tabular, mixed-content. **Not active:** Kompress ML prose compression
  (requires a HuggingFace model download — prose compression was NOT
  simulated), AST code compression (disabled by headroom's own default,
  `enable_code_aware=False`, left at that default), and headroom protects
  file `Read` outputs and error outputs verbatim by default.
- **Tokenizer for ratios:** tiktoken `o200k_base` (neutral). Relative ratios
  are the primary numbers; absolute token units come from the analyzer DB's
  API-usage-derived block sizes, so numerator and denominator share units.
- **Environment:** fully offline (`HEADROOM_OFFLINE=1`), zero API calls,
  zero cost. Headroom's native Magika/ONNX content detector deadlocked on
  this macOS host, so its own escape hatch `HEADROOM_DETECT_BACKEND=python`
  was used. Runtime state confined via `HEADROOM_WORKSPACE_DIR` (no traces
  left in `~/.headroom`).
- **Read-only guarantee:** the analyzer DB is opened with a `mode=ro` sqlite
  URI; transcripts are never modified.

## Corpus — and one forced deviation from the #94 plan

#94 targeted the analyzer DB's 73 recorded sessions ($15,569 at the DB's
fixed Opus rates). **None of those transcripts still exist on disk**: the DB
sessions span 2026-05-07 → 2026-06-13, and Claude Code's transcript
retention has since purged them (oldest surviving transcript: 2026-06-16).
Zero overlap between DB session ids and on-disk transcripts.

So the audit runs on the corpus that *can* be replayed: all 27 full-content
session transcripts currently on disk (2026-06-16 → 2026-07-21, ~140MB
including subagent files), ingested through context-analyzer's own pipeline
(same parser, same fixed Opus rates: input $15/M, output $75/M, cache read
$1.875/M, cache write $18.75/M) into a scratch DB. This keeps every number
internally consistent with how the live DB computes cost. The live DB was
only ever read.

| Corpus stat | Value |
|---|---:|
| Sessions | 27 |
| API calls | 8,502 |
| tool_result items analyzed | 3,582 (2,781 unique contents compressed) |
| Total resident token-call volume | 2.79B tokens |
| Recorded cost (fixed Opus rates) | $7,360.53 |
| Input-side share of recorded cost | $6,301.80 |
| Audit wall time (full corpus) | 2.7s (plus 1.5s scratch-DB ingest) |

## Methodology (residency-weighted — the load-bearing part)

A compressed tool output doesn't save tokens once; it saves them on **every
subsequent API call it stays resident in**. For each tool_result block:

```
saved_token_calls(block) = block_tokens × ratio × residency_calls(block)
residency_calls          = (exit_turn or total_api_calls) − enter_turn
ceiling_% (session)      = Σ saved_token_calls ÷ (input + cache_read + cache_creation)
```

- `block_tokens` and `enter/exit_turn` come from the analyzer DB's blocks
  table (API-call indices; `exit_turn` set on compaction, else resident to
  end of session).
- `ratio` is measured by running the block's **full transcript content**
  (not the DB's 500-char previews) through `headroom.compress()` and
  counting original vs compressed tokens with tiktoken. Ratios < 0
  (inflation) are clamped to 0 — such a rewrite would never be applied.
- Dollarization applies the ceiling % to each session's **input-side cost
  share**, derived from the DB's token columns at the same fixed rates the
  ingest uses. No fresh rate assumptions.
- Join quality: 100% of parent-context tool_result blocks joined cleanly to
  transcript content (0 missing, 0 residency fallbacks, 0 compressor
  errors). 141 collapsed subagent-summary blocks were excluded by
  construction — they are Context-Scope bookkeeping for *subagent-internal*
  context, which is neither resident in the parent context window nor part
  of the recorded session cost (the parent's Task result text is analyzed
  as a normal tool_result). 237 items had empty/no-text content (e.g.,
  image results) and contribute zero.

## Headline

**Ceiling: 3.03% of resident token-call volume = $192.06 of the recorded
$7,360.53 was theoretically compressible** by the compressors actually
simulated (headroom's default local pipeline, ML prose model excluded).

**Kill criterion (#94: ceiling < 10% of spend → publish the finding and
stop): MET for the simulated configuration, at 3.03%** — with one honest
caveat quantified below (prose sensitivity).

For scale: even if the purged $15,569 corpus compressed at the same rate,
the ceiling there would have been ≈ $470 — still below the 10% bar.

A premise check that validates the DB's earlier finding: tool_result blocks
account for **66.0%** of resident token-call volume on this corpus.

## By content type

| Type | Items | Block tokens | One-shot ratio | Saved token-calls | Resident share | Saved share of corpus volume |
|---|---:|---:|---:|---:|---:|---:|
| code | 395 | 1.1M | 11.0% | 49.5M | 15.49% | 1.77% |
| json | 23 | 57.3K | 74.1% | 16.8M | 0.77% | 0.60% |
| other | 1474 | 2.4M | 2.2% | 12.2M | 27.17% | 0.44% |
| log | 21 | 59.7K | 21.1% | 5.5M | 0.94% | 0.20% |
| prose | 1432 | 1.9M | 0.1% | 582.4K | 21.59% | 0.02% |

The shape of the story: headroom's headline compressor (SmartCrusher) is
excellent at big JSON (74% one-shot) — but this workload contains almost no
big JSON (23 items, 0.77% of resident volume). The bulk of resident volume
is mixed shell output, prose, and code, which headroom's simulated pipeline
either protects by policy (Read/Edit/Write outputs, error outputs), cannot
compress without the ML model, or compresses only marginally.

### Prose sensitivity (the honest caveat)

Prose tool_results are 21.6% of resident volume and were effectively
untouched because the ML prose compressor (Kompress) was not simulated. If
Kompress achieved a prose ratio of R on this content, the ceiling would rise
by up to 21.6×R points: R=25% → +5.4 points (total ≈ 8.4%, still under the
bar); R=50% → +10.8 points (total ≈ 13.8%, over the bar). **The Stage 0
verdict therefore applies to headroom's local structural compressors; a
definitive all-in verdict on the ML prose path requires simulating Kompress
(a bounded follow-up: one HuggingFace model download, same harness).** Code
(15.5% of volume) is protected by headroom's own default
(`enable_code_aware=False`), so counting it against headroom would be
counting a feature headroom itself ships turned off.

## By tool

| Tool | Items | Block tokens | One-shot ratio | Saved token-calls | Resident share | Saved share of corpus volume |
|---|---:|---:|---:|---:|---:|---:|
| Bash | 1820 | 2.7M | 6.4% | 63.9M | 30.97% | 2.29% |
| other | 314 | 363.2K | 11.6% | 16.7M | 4.60% | 0.60% |
| Task | 149 | 610.2K | 2.5% | 4.0M | 5.97% | 0.14% |
| Read | 379 | 773.6K | 0.0% | 0 | 11.41% | 0.00% |
| Write | 143 | 580.4K | 0.0% | 0 | 7.54% | 0.00% |
| Edit | 520 | 490.9K | 0.0% | 0 | 5.33% | 0.00% |
| WebFetch | 17 | 18.0K | 0.0% | 0 | 0.13% | 0.00% |
| WebSearch | 3 | 3.4K | 0.0% | 0 | 0.01% | 0.00% |

Read/Write/Edit at exactly 0.0% is headroom's own policy (file contents
protected verbatim so the agent keeps exact bytes to patch), not a failure.

## Top 10 most-compressible sessions

| Date | Session | API calls | Recorded $ | Input-side $ | Ceiling % | Ceiling $ |
|---|---|---:|---:|---:|---:|---:|
| 2026-06-25 | 9c1ab4e0 | 2096 | $1,883.21 | $1,591.98 | 5.01% | $79.82 |
| 2026-06-25 | 3ede3b83 | 1551 | $1,500.39 | $1,310.09 | 3.01% | $39.40 |
| 2026-06-21 | e18b3d4e | 1122 | $1,087.93 | $917.05 | 3.32% | $30.47 |
| 2026-07-21 | a220083b | 641 | $418.72 | $380.20 | 4.06% | $15.44 |
| 2026-06-27 | 3a296ba8 | 1182 | $991.47 | $850.46 | 1.48% | $12.58 |
| 2026-06-25 | aff42566 | 825 | $1,006.62 | $883.05 | 1.01% | $8.92 |
| 2026-07-21 | e683f1b7 | 305 | $197.32 | $161.50 | 1.52% | $2.46 |
| 2026-06-22 | 5fc37d92 | 136 | $67.83 | $47.93 | 3.67% | $1.76 |
| 2026-07-19 | b8079da7 | 153 | $47.47 | $38.54 | 1.12% | $0.43 |
| 2026-06-25 | 7c332ab6 | 253 | $88.80 | $68.65 | 0.40% | $0.27 |

No session exceeds a 5.01% ceiling.

## Headroom routing observed (counts per item)

`router:protected:user_message`: 3345 · `router:excluded:tool`: 1062 ·
`router:tool_result:lossless_log`: 152 · `router:protected:error_output`: 107 ·
`router:tool_result:mixed`: 33 · `router:tool_result:lossless_paths`: 28 ·
`router:tool_result:lossless_search`: 19 · `router:bash:lossless_search`: 11 ·
`router:tool_result:config`: 11 · `router:tool_result:lossless_diff`: 8 ·
`router:tool_result:tabular`: 7 · `router:tool_result:log`: 3 ·
`router:tool_result:smart_crusher`: 2 · `router:tool_result:lossless_config`: 1

4 items compressed >95% — inspected individually: all are large files
embedded in MCP JSON envelopes that SmartCrusher replaced with CCR
retrieval markers (content stashed for on-demand retrieval). Legitimate
under the zero-clawback assumption; in live use the agent would likely
retrieve them, clawing back most of that saving.

## Limitations — this is an upper bound by construction

- **Zero retrieval clawback assumed.** SmartCrusher emits `<<ccr:...>>`
  retrieval markers; every retrieval in live use costs a round-trip that
  claws back savings. Not modeled.
- **Zero prompt-cache damage assumed — mention this prominently.**
  Compression rewrites message content, which breaks prompt-cache prefixes.
  On this corpus the overwhelming majority of resident volume is cache
  reads billed at 0.125× the input rate; rewriting a cached prefix converts
  cheap cache reads into full-price input and cache re-writes. In live use
  this alone can consume a large share — potentially all — of the
  theoretical savings. Not modeled.
- **Zero answer degradation assumed.** Compressed tool outputs may change
  agent behavior (retries, wrong edits). Not modeled.
- **Compression applied from turn 0** of every session — the most
  favorable timing possible.
- **Tokenizer approximation.** Ratios measured with tiktoken o200k_base,
  not Anthropic's tokenizer. Relative ratios are robust to this choice;
  absolute figures inherit the DB's proportional-attribution block sizing.
- **Prose ML compression not simulated** (`kompress_model="disabled"`; the
  ModernBERT model requires a HuggingFace download). Quantified above:
  prose is 21.6% of resident volume, so the unsimulated path could add up
  to 21.6×R points at prose ratio R. AST code compression is off by
  headroom's own default and was left at that default.
- **Native content detector bypassed** (`HEADROOM_DETECT_BACKEND=python`,
  headroom's own escape hatch) because the native Magika/ONNX detector
  deadlocked on this host.
- **Corpus substitution** (described above): the $15,569 DB corpus had no
  surviving transcripts; the audit ran on the 27 replayable sessions
  ($7,360.53), ingested with identical parsing and rates.
- **Single machine, single user's workload.** No generalization claimed.

## Reproduction

```bash
# 1. Build the scratch corpus DB (reads transcripts + live DB only; writes scratch only)
python - <<'PY'
from pathlib import Path
from context_tracker.ingest import ingest_session
import re
uuid_re = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
proj = Path.home()/".claude"/"projects"
for f in sorted(proj.rglob("*.jsonl")):
    if "subagents" not in f.parts and uuid_re.match(f.stem):
        ingest_session(f.stem, db_path=Path("experiments/headroom/scratch/corpus.db"), projects_dir=proj)
PY

# 2. Run the audit (pip install --no-deps headroom-ai==0.32.1 ; pip install tiktoken)
python experiments/headroom/retrospective.py \
    --db experiments/headroom/scratch/corpus.db \
    --out experiments/headroom/scratch/report.md
```

Run date: 2026-07-21. Full corpus wall time: 2.7s.
