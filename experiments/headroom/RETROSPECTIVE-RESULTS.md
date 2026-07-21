# Retrospective Compression Audit — Ceiling Report (Stage 0 of #94)

**Verdict up front: the kill criterion is MET in both configurations.**
Replaying this machine's real Claude Code workload through headroom's local
compressors yields:

| Configuration | Ceiling | Dollars (of $7,360.53) |
|---|---:|---:|
| Shipped defaults (as `pip install` gives you) | **3.03%** | **$192.06** |
| Maximum capability (ML prose model + AST code force-enabled) | **4.25%** | **$267.54** |

Both are far below the 10%-of-spend threshold set in #94, and both are
upper bounds that assume zero retrieval clawback, zero prompt-cache damage,
and zero answer degradation. The prose question that the first pass left
open is now closed empirically: with headroom's own Kompress model running,
real prose tool outputs compressed just **0.9%** — not the hypothetical 50%
that would have pushed the ceiling over the bar.

## Setup

- **headroom-ai:** `0.32.1` (pinned; installed with `pip install --no-deps
  headroom-ai` — the `litellm` dependency has no macOS wheels and is only
  needed for headroom's proxy/LLM paths, which this offline audit never
  touches). Public API used: `headroom.compress()` with
  `CompressConfig(protect_recent=0)`; the defaults profile additionally sets
  `kompress_model="disabled"`, the max profile loads headroom's own model.
- **Two configurations audited.** *Shipped defaults*: SmartCrusher
  (JSON/structured), lossless log/search/paths/diff/config compaction,
  tabular, mixed-content — with Kompress ML prose compression and AST code
  compression left off, exactly as headroom ships them on the library path.
  *Maximum capability*: the same pipeline plus the Kompress prose model
  (`chopratejas/kompress-v2-base`, headroom's pinned revision, ONNX CPU
  backend, loaded in 2.0s from a one-time HuggingFace download) and AST
  code compression force-enabled via
  `ContentRouterConfig(enable_code_aware=True)` — a capability headroom
  itself ships disabled. Kompress's shipped 5s canary / 20s per-call time
  budgets were relaxed for the batch audit so slow items could not silently
  pass through uncompressed. In both configurations headroom protects file
  `Read` outputs and error outputs verbatim by policy.
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

**Shipped defaults: 3.03% ($192.06). Maximum capability: 4.25% ($267.54).**

**Kill criterion (#94: ceiling < 10% of spend → publish the finding and
stop): MET in both configurations.** Turning on everything headroom has —
including features it ships disabled — moves the ceiling by 1.22 points.

For scale: even if the purged $15,569 corpus compressed at the
maximum-capability rate, the ceiling there would have been ≈ $662 — still
well below the 10% bar.

A premise check that validates the DB's earlier finding: tool_result blocks
account for **66.0%** of resident token-call volume on this corpus.

## By content type

Shipped defaults:

| Type | Items | Block tokens | One-shot ratio | Saved token-calls | Resident share | Saved share of corpus volume |
|---|---:|---:|---:|---:|---:|---:|
| code | 395 | 1.1M | 11.0% | 49.5M | 15.49% | 1.77% |
| json | 23 | 57.3K | 74.1% | 16.8M | 0.77% | 0.60% |
| other | 1474 | 2.4M | 2.2% | 12.2M | 27.17% | 0.44% |
| log | 21 | 59.7K | 21.1% | 5.5M | 0.94% | 0.20% |
| prose | 1432 | 1.9M | 0.1% | 582.4K | 21.59% | 0.02% |

Maximum capability (prose model + AST code enabled):

| Type | Items | Block tokens | One-shot ratio | Saved token-calls | Resident share | Saved share of corpus volume |
|---|---:|---:|---:|---:|---:|---:|
| code | 402 | 1.1M | 14.4% | 67.9M | 15.52% | 2.43% |
| other | 1486 | 2.4M | 3.5% | 23.2M | 27.19% | 0.83% |
| json | 23 | 57.3K | 74.1% | 16.8M | 0.77% | 0.60% |
| prose | 1435 | 1.9M | 0.9% | 6.6M | 21.60% | 0.23% |
| log | 21 | 59.7K | 14.6% | 4.4M | 0.94% | 0.16% |

Two empirical findings from the maximum-capability run explain why headroom
ships these features off:

- **AST code compression reverted itself 16 times** — headroom's own
  `code_compressor` logged "Code compression produced invalid syntax for
  python … returning original" on real tool outputs up to 936 tokens. The
  14.4% code ratio is what survives after those self-reverts.
- **Kompress inflated some prose** — headroom's inflation guard reverted 2
  items where the "compressed" output had *more* tokens than the input, and
  across 1,435 real prose items the model's net ratio was 0.9%. Real agent
  prose (test output, explanations, PR text) is information-dense in ways
  benchmark prose apparently is not.

The shape of the story: headroom's headline compressor (SmartCrusher) is
excellent at big JSON (74% one-shot) — but this workload contains almost no
big JSON (23 items, 0.77% of resident volume). The bulk of resident volume
is mixed shell output, prose, and code, which headroom's simulated pipeline
either protects by policy (Read/Edit/Write outputs, error outputs), cannot
compress without the ML model, or compresses only marginally.

### Prose sensitivity — caveat closed empirically

The first pass left prose (21.6% of resident volume) as the attack surface:
hypothetically, a 50% prose ratio would have added +10.8 points and pushed
the total over the kill bar. The maximum-capability run answers it with
headroom's own model instead of a hypothetical: **Kompress achieved 0.9% on
1,435 real prose tool outputs** (+0.21 points of ceiling). The pre-run
sensitivity bound and the post-run empirical value are both reported here
deliberately — that is what closing a stated caveat looks like, and the
distinction matters because Kompress is lossy-by-model (see Limitations):
even its 0.9% assumes the agent never needed the dropped tokens.

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
- **Prose compression is lossy-by-model.** Kompress (ModernBERT,
  `chopratejas/kompress-v2-base`, ONNX CPU backend) is a summarization-style
  token-drop compressor: compressed prose is NOT reconstructible from what
  stays in context, unlike SmartCrusher's reversible CCR path. Every prose
  point in the maximum-capability ceiling assumes the agent never needed
  the dropped tokens. AST code compression was force-enabled for that run;
  headroom ships it off by default, so its contribution (+0.66 points on
  code) counts a capability headroom itself does not enable — and one that
  reverted itself 16 times on invalid output syntax.
- **Wall time:** 2.7s (defaults) / 199.7s (maximum capability, CPU ONNX) for
  the full corpus.
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

# 3. Maximum-capability configuration (adds onnxruntime + a one-time
#    ~HuggingFace download of chopratejas/kompress-v2-base on first run)
python experiments/headroom/retrospective.py \
    --db experiments/headroom/scratch/corpus.db \
    --profile max \
    --out experiments/headroom/scratch/report-max.md
```

Run date: 2026-07-21. Full corpus wall time: 2.7s.
