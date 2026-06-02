# Context Analyzer Dashboard — Design Spec

**Date:** 2026-06-01
**Status:** Draft (v2 — addressing Codex review findings)
**Goal:** Build a production-quality context window analysis dashboard for Claude Code sessions, with staleness detection, session health scoring, and actionable recommendations.

---

## Architecture

### Serving Model: MCP Server + Dashboard Hybrid

Single Python codebase, two entry points:

1. **`context-tracker`** — MCP server mode (existing FastMCP, stdio transport). Claude Code queries context health mid-session via MCP tools.
2. **`context-tracker dashboard`** — Launches a FastAPI server on localhost serving the web dashboard UI and REST API endpoints.

Shared analysis modules are used by both entry points.

### Data Sources

| Source | Location | Contains |
|---|---|---|
| Hook event traces | `~/.claude/context-trace/<session_id>.jsonl` | Tool calls, compactions, session lifecycle, subagent events |
| Transcripts | `~/.claude/projects/<project>/<session_id>.jsonl` | Full conversation messages: user, assistant, tool_use, tool_result with content |
| System prompt templates | Bundled with the package (extracted from Piebald-AI, versioned) | Base system prompt structure and approximate content (labeled as estimates) |
| CLAUDE.md / rules files | Paths from `InstructionsLoaded` hook events, read from disk | User-configurable instructions loaded into context (note: disk content is current version, not necessarily the version loaded at session time) |

### Project Structure (New/Modified Files)

```
src/context_tracker/
├── server.py              # Existing MCP server — add new analysis tools
├── dashboard.py           # NEW: FastAPI app serving dashboard + REST API
├── analysis/
│   ├── __init__.py
│   ├── reconstruction.py  # NEW: Context reconstruction from transcript
│   ├── staleness.py       # NEW: Four-layer staleness detection engine
│   ├── health.py          # NEW: Session health scoring & recommendations
│   └── config.py          # NEW: Configurable thresholds
├── transcript_parser.py   # NEW: Raw transcript parser (extracts full messages with content)
├── models.py              # Existing — extend with analysis models
├── hooks.py               # Existing
├── installer.py           # Existing
├── storage.py             # Existing
└── transcript.py          # Existing (API token extraction only — kept separate)
static/
├── dashboard.html         # NEW: Single-page dashboard
├── dashboard.js           # NEW: Dashboard logic + Chart.js visualizations
└── dashboard.css          # NEW: Dashboard styles
tests/
├── test_transcript_parser.py # NEW
├── test_reconstruction.py    # NEW
├── test_staleness.py         # NEW
├── test_health.py            # NEW
└── ...                       # Existing tests unchanged
```

**Key change:** `transcript.py` (existing) stays as-is for API token extraction. New `transcript_parser.py` is a raw transcript parser that extracts full message content, block structure, and timestamps — the foundation for reconstruction.

---

## Module 0: Raw Transcript Parser (`transcript_parser.py`)

The existing `transcript.py` only extracts API-level token usage. Reconstruction requires a full parser that extracts every message with content.

### Parsed Message Model

```python
@dataclass(frozen=True)
class TranscriptMessage:
    message_id: str              # UUID from transcript entry, or generated
    sequence_index: int          # Global sequential index in transcript
    entry_type: str              # "user", "assistant", "system"
    timestamp: str | None        # From transcript entry (not analysis-time)
    session_id: str

    # Content blocks within this message
    content_blocks: list[ContentBlock]

    # API usage (assistant entries only)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    stop_reason: str | None = None
    model: str | None = None

@dataclass(frozen=True)
class ContentBlock:
    block_type: str              # "text", "tool_use", "tool_result", "thinking"
    content: str                 # Full text content
    size_chars: int              # len(content)
    tool_use_id: str | None      # Links tool_use to its tool_result
    tool_name: str | None        # "Read", "Bash", etc. (tool_use blocks)
    tool_input: dict | None      # Parsed tool input JSON (tool_use blocks)
    is_error: bool = False       # Tool result was an error
```

### Parser Contract

```python
def parse_raw_transcript(transcript_path: Path) -> list[TranscriptMessage]:
    """Parse a Claude Code transcript JSONL into structured messages.

    Extracts ALL message content including:
    - User text prompts
    - Assistant text responses
    - tool_use blocks with input JSON
    - tool_result blocks with full content
    - Thinking blocks
    - System entries (turn_duration, local_command — metadata only)

    Timestamps come from the transcript entry, NOT generated at parse time.
    Malformed lines are skipped with a DataQualityWarning appended to a warnings list.

    Returns messages in transcript order (sequential by sequence_index).
    """
```

### Data Quality

```python
@dataclass
class DataQualityWarning:
    line_number: int
    warning_type: str     # "malformed_json", "missing_field", "unexpected_type"
    description: str
```

Warnings are collected and returned alongside parsed messages, not silently discarded.

---

## Module 1: Context Reconstruction (`analysis/reconstruction.py`)

Rebuilds the context window state at each turn from parsed transcript data.

### Two Levels of Time

Codex correctly identified that "turn" has two meanings. We model both:

```python
@dataclass(frozen=True)
class ApiCall:
    """A single API round-trip: one assistant response (possibly with tool_use)."""
    api_call_index: int          # Sequential within session
    conversation_turn: int       # Which ConversationTurn this belongs to
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    stop_reason: str | None
    timestamp: str | None
    blocks_entered: list[str]    # block_ids that entered context in this API call

@dataclass
class ConversationTurn:
    """One user prompt + all API calls until the next user prompt."""
    turn_number: int             # Starts at 1
    timestamp: str | None        # Timestamp of the user prompt
    user_prompt_text: str        # User's text prompt (for task boundary detection)
    api_calls: list[ApiCall]     # All API round-trips in this turn
    epoch: int                   # Context epoch (increments on compaction)
```

### Block Model (Immutable)

Each item in the context window is an immutable `ContextBlock`. Staleness is tracked separately in per-turn state overlays.

```python
@dataclass(frozen=True)
class ContextBlock:
    block_id: str              # tool_use_id, message_id, or generated
    turn_entered: int          # ConversationTurn number when block entered context
    api_call_entered: int      # ApiCall index when block entered
    epoch_entered: int         # Context epoch when block entered
    block_type: BlockType      # user_prompt, assistant_text, tool_use, tool_result, system
    resource: str | None       # Extracted resource identifier (file path, command, etc.)
    resource_type: str | None  # file, command, pattern, agent
    size_chars: int            # Content character count
    size_tokens_est: int       # Estimated tokens (chars / 4)
    content_hash: str          # SHA256 of content (for dedup / change detection)
    tool_name: str | None      # Read, Bash, Grep, Edit, etc.
    tool_use_id: str | None    # Links tool_use to its tool_result
    parent_block_id: str | None  # tool_result points to its tool_use
    is_error: bool = False     # Tool result was an error
    is_pinned: bool = False    # System prompt, CLAUDE.md — never stale
    timestamp: str | None = None  # From transcript, not analysis-time
```

Content is NOT stored on the block. It is stored separately and loaded lazily:

```python
class ContentStore:
    """Stores full content indexed by block_id. Loaded lazily for drilldown views."""

    def get_content(self, block_id: str) -> str: ...
    def get_preview(self, block_id: str, max_chars: int = 200) -> str: ...
```

This avoids quadratic payload growth when building per-turn snapshots.

### Block Type Enum

```python
class BlockType(str, Enum):
    SYSTEM = "system"
    USER_PROMPT = "user_prompt"
    ASSISTANT_TEXT = "assistant_text"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    COMPACTION_SUMMARY = "compaction_summary"
```

### Per-Turn State (Immutable Overlay)

Staleness scores are NOT stored on blocks. They are computed per turn and stored as overlays:

```python
@dataclass(frozen=True)
class BlockStateAtTurn:
    block_id: str
    staleness_score: float     # 0.0 (fresh) to 1.0 (dead weight)
    staleness_label: str       # active, warm, stale, dead_weight, pinned
    is_superseded: bool        # True if a newer read of the same resource exists
    superseded_by: str | None  # block_id of the newer version
```

### Context Epoch

Compaction creates a new epoch. Blocks from previous epochs are either gone (replaced by a summary) or pinned (system prompt).

```python
@dataclass
class ContextEpoch:
    epoch_number: int          # Starts at 0, increments on each compaction
    started_at_turn: int       # ConversationTurn when this epoch began
    compaction_summary_size: int | None  # From PostCompactEvent.compact_summary_length
    blocks_before_compaction: int        # How many blocks were in context before
```

### Compaction Detection

Use explicit `PostCompactEvent` from hook event traces as the primary signal (already captured by our hooks). Fall back to token-drop heuristic only if hook events are unavailable for a session.

```python
def detect_compactions(
    hook_events: list[TrackerEvent],
    api_calls: list[ApiCall],
) -> list[ContextEpoch]:
    """Detect compaction events. Primary: PostCompactEvent from hooks.
    Fallback: >30% token drop between consecutive API calls (labeled as 'inferred').
    """
```

On compaction:
- Pinned blocks (system, CLAUDE.md) persist into the new epoch
- All other pre-compaction blocks get `turn_exited = compaction_turn`
- An opaque `COMPACTION_SUMMARY` block is created with size from `PostCompactEvent.compact_summary_length`

### Turn Snapshot

```python
@dataclass
class TurnSnapshot:
    turn_number: int
    timestamp: str | None
    epoch: int
    block_ids: list[str]                   # Block IDs in context (NOT full blocks — lazy load)
    block_states: list[BlockStateAtTurn]   # Per-block staleness at this turn
    blocks_entered_ids: list[str]
    blocks_exited_ids: list[str]
    total_tokens_est: int
    input_tokens: int                      # From the LAST API call in this turn
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    compaction_detected: bool
    api_call_count: int                    # How many API round-trips in this turn
```

### Reconstruction Algorithm

1. Parse raw transcript via `parse_raw_transcript()` → list of `TranscriptMessage`
2. Load hook events via `read_events()` → list of `TrackerEvent`
3. Group messages into `ConversationTurn`s:
   - A `user` entry with a text content block starts a new turn
   - A `user` entry with only `tool_result` blocks continues the current turn
4. Within each turn, group API round-trips into `ApiCall`s (each assistant entry with `stop_reason` and `output_tokens > 0`)
5. For each content block, create an immutable `ContextBlock`:
   - Extract `resource` from tool_use input JSON (see Resource Extraction)
   - For `tool_result` blocks, copy `resource` from the paired `tool_use` (matched by `tool_use_id`)
   - Store content in `ContentStore`, not on the block
6. Detect compaction events from hook events (primary) or token drops (fallback)
7. Build `ContextEpoch`s and mark block exits
8. Build per-turn `TurnSnapshot`s with block ID lists (not full blocks)

### Resource Extraction

```python
def extract_resource(tool_name: str, tool_input: dict) -> tuple[str | None, str | None]:
    """Returns (resource, resource_type) from tool_use input.

    File paths: normalized (resolve ../, strip trailing /, expand ~).
    Bash commands: extract primary program name, handling:
      - cd prefixes: 'cd /foo && pytest' → 'pytest'
      - env/wrapper prefixes: 'env FOO=1 pytest' → 'pytest'
      - pipelines: 'cat foo | grep bar' → 'cat' (first command)
      - shell builtins: 'echo', 'cd' → as-is
    """
    EXTRACTORS = {
        "Read":  lambda i: (i.get("file_path"), "file"),
        "Edit":  lambda i: (i.get("file_path"), "file"),
        "Write": lambda i: (i.get("file_path"), "file"),
        "Bash":  lambda i: (_extract_bash_program(i.get("command", "")), "command"),
        "Grep":  lambda i: (f"{i.get('pattern', '')}@{i.get('path', '')}", "pattern"),
        "Glob":  lambda i: (i.get("pattern"), "pattern"),
        "Agent": lambda i: (i.get("prompt", "")[:80], "agent"),
    }
    extractor = EXTRACTORS.get(tool_name)
    if extractor is None:
        return (None, None)
    return extractor(tool_input)
```

---

## Module 2: Staleness Detection (`analysis/staleness.py`)

Four-layer scoring model. Labeled as **experimental heuristic** — requires calibration against real sessions before recommendations are treated as authoritative.

### Superseded Read Detection

Before scoring layers, detect when a block has been superseded by a newer read of the same resource. This is the strongest staleness signal.

```python
def detect_superseded(blocks: list[ContextBlock]) -> dict[str, str]:
    """Returns {old_block_id: new_block_id} for blocks superseded by newer reads.

    A tool_result for resource X is superseded when a later tool_result
    for the same resource exists (the file was re-read, possibly with different content).
    The OLD copy is stale — it was replaced. The NEW copy is the active version.
    """
    resource_to_latest: dict[str, str] = {}  # resource → latest block_id
    superseded: dict[str, str] = {}

    for block in sorted(blocks, key=lambda b: b.turn_entered):
        if block.block_type != BlockType.TOOL_RESULT or block.resource is None:
            continue
        if block.resource in resource_to_latest:
            superseded[resource_to_latest[block.resource]] = block.block_id
        resource_to_latest[block.resource] = block.block_id

    return superseded
```

### Layer 1: Resource Linking

Track `resource → last_turn_used`. Scoped to turns up to `current_turn` only (no future-data leakage).

```python
def resource_factor(
    block: ContextBlock,
    resource_last_used: dict[str, int],  # Built from turns 1..current_turn only
    current_turn: int,
    window: int,
) -> float:
    if block.resource is None:
        return 1.0
    last_used = resource_last_used.get(block.resource)
    if last_used is None:
        return 1.0
    turns_since = current_turn - last_used
    if turns_since <= window:
        return 0.0  # Same resource used recently
    return min(1.0, turns_since / (window * 3))  # Gradual decay
```

Resource matching rules:
- **File paths**: normalize (resolve `../`, strip trailing slashes, expand `~`). Match on full normalized path (not basename — avoids cross-directory collisions).
- **Bash commands**: extract primary program name handling `cd &&` prefixes, `env`/`uv run` wrappers, pipelines. Match by program name.
- **Grep/Glob patterns**: exact string match on the pattern.

### Layer 2: Conversation Reference Scanning

Scan messages AFTER the block was created, within the configured window. Scan ALL message types (user prompts, assistant text, tool_use arguments, tool_result content) — not just assistant text.

```python
def extract_identifiers(block: ContextBlock, content_store: ContentStore) -> set[str]:
    """Extract identifiable tokens from block content."""
    content = content_store.get_content(block.block_id)
    identifiers = set()

    # Full normalized file path (not just basename — avoids collisions)
    if block.resource and block.resource_type == "file":
        identifiers.add(Path(block.resource).name)  # "server.py"
        identifiers.add(block.resource)              # Full path for exact match

    # Function/class/def names — word-boundary aware
    for match in re.findall(r'(?:def|class|function)\s+(\w{3,})', content):
        identifiers.add(match)

    # Filter out common false-positive tokens
    identifiers -= {"self", "None", "True", "False", "return", "import", "from", "the", "and"}

    return identifiers

def reference_factor(
    block: ContextBlock,
    messages_since_block: list[str],  # All message text AFTER block.turn_entered, up to current turn
    scan_window: int,
) -> float:
    """Scan recent messages for identifier references.

    messages_since_block: text from turns (block.turn_entered+1) to current_turn,
    limited to the last scan_window turns. Includes user, assistant, tool_use,
    and tool_result text.
    """
    ids = extract_identifiers(block, content_store)
    if not ids:
        return 0.8

    for text in messages_since_block[-scan_window:]:
        for identifier in ids:
            # Word-boundary matching to reduce false positives
            if re.search(r'\b' + re.escape(identifier) + r'\b', text):
                return 0.0
    return 1.0
```

### Layer 3: Semantic Grouping

Narrower than original spec: only direct parent-child relationships, not entire directories.

```python
def group_factor(block: ContextBlock, active_resources: set[str]) -> float:
    """Discount staleness if a closely-related resource is active.

    'Closely related' means: same file (different read), or files that
    import each other (detected from tool_result content containing
    'from X import' or 'import X' where X matches another active resource's module name).

    Does NOT treat all files in the same directory as related — that's too broad.
    """
    if block.resource is None or block.resource_type != "file":
        return 1.0

    block_module = Path(block.resource).stem  # e.g., "server"
    for res in active_resources:
        res_module = Path(res).stem
        # Same file, different version — handled by superseded detection, not grouping
        if res == block.resource:
            continue
        # Check if block's content imports the active resource's module
        # (This is a lightweight check — not a full dependency graph)
        if block_module in _known_importers.get(res_module, set()):
            return 0.6
    return 1.0
```

### Layer 4: Task Boundary Detection

More conservative detection to avoid false positives on short prompts.

```python
def detect_task_boundaries(turns: list[ConversationTurn], config: StalenessConfig) -> list[int]:
    """Detect task switches. Conservative — requires strong signals.

    Time gap alone is a weak signal (user may have stepped away).
    Short prompts ("yes", "do it", "fix the tests") are excluded from
    keyword overlap analysis as they carry no topical signal.
    """
    boundaries = []
    MIN_PROMPT_LENGTH = 20  # Characters — skip overlap analysis for short confirmations

    for i in range(1, len(turns)):
        prev = turns[i-1].user_prompt_text
        curr = turns[i].user_prompt_text

        # Strong signal: time gap > threshold AND meaningful prompt change
        time_gap = time_gap_minutes(turns[i-1], turns[i])
        if time_gap > config.task_boundary_time_gap:
            if len(curr) >= MIN_PROMPT_LENGTH and len(prev) >= MIN_PROMPT_LENGTH:
                if keyword_overlap(prev, curr) < config.task_boundary_overlap:
                    boundaries.append(turns[i].turn_number)

    return boundaries
```

### Combined Score

**Additive-capped model** (not multiplicative — avoids the zero-factor problem where a single 0 makes an old block permanently fresh).

```python
def compute_staleness(
    block: ContextBlock,
    current_turn: int,
    config: StalenessConfig,
    resource_last_used: dict[str, int],  # Scoped to turns 1..current_turn
    messages_since_block: list[str],     # Messages after block entry, up to current turn
    active_resources: set[str],          # Resources used in last config.resource_window turns
    task_boundaries: list[int],
    superseded_map: dict[str, str],      # {old_block_id: new_block_id}
) -> tuple[float, str]:
    """Returns (score, label). Score is 0.0 (fresh) to 1.0 (dead weight).

    Composition: additive with caps, not multiplicative.
    Each factor contributes 0.0 to 0.25, summing to max 1.0.
    """
    if block.is_pinned:
        return (0.0, "pinned")

    # Superseded blocks are immediately stale — they were replaced
    if block.block_id in superseded_map:
        return (0.9, "dead_weight")

    # Base age decay (0.0 to 0.35)
    age = base_decay(current_turn - block.turn_entered, config.decay_window) * 0.35

    # Resource factor (0.0 to 0.25) — is the same resource still being used?
    res = resource_factor(block, resource_last_used, current_turn, config.resource_window) * 0.25

    # Reference factor (0.0 to 0.25) — is this block's content mentioned?
    ref = reference_factor(block, messages_since_block, config.reference_scan_window) * 0.25

    # Context factors (0.0 to 0.15) — grouping + task boundaries
    ctx = 0.0
    g = group_factor(block, active_resources)
    t = task_factor(block, task_boundaries, current_turn)
    ctx = ((g + t) / 2.0 - 0.5) * 0.15  # Centered: 0 if neutral, positive if stale signals

    score = min(1.0, max(0.0, age + res + ref + ctx))
    return (score, label_staleness(score))

def base_decay(turns_since_entry: int, window: int) -> float:
    if turns_since_entry <= 2:
        return 0.0
    if turns_since_entry <= window:
        return turns_since_entry / window * 0.5
    return 0.5 + min(0.5, (turns_since_entry - window) / (window * 2))
```

### Staleness Labels

```python
def label_staleness(score: float) -> str:
    if score < 0.3: return "active"
    if score < 0.6: return "warm"
    if score < 0.8: return "stale"
    return "dead_weight"
```

### Pinned Blocks

System prompt and CLAUDE.md blocks always return `(0.0, "pinned")`. They don't participate in scoring.

---

## Module 3: Session Health & Recommendations (`analysis/health.py`)

### Context Utilization

Context utilization uses the input token count from the **most recent API call** in the turn (this is the actual context size sent to the API), divided by the model's context window.

```python
MODEL_CONTEXT_WINDOWS = {
    "claude-opus-4-6": 200_000,
    "claude-opus-4-6[1m]": 1_000_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5": 200_000,
    # Add new models as they appear
}

def get_context_window(model: str, config: HealthConfig) -> int:
    """Lookup model context window. Falls back to config default."""
    return MODEL_CONTEXT_WINDOWS.get(model, config.model_context_window)
```

### Health Signals

```python
@dataclass
class HealthSignals:
    turn_number: int
    dead_weight_ratio: float      # Sum of blocks with score >= 0.8 / total context tokens
    context_utilization: float    # Latest API call input_tokens / model context window
    cache_efficiency: float       # cache_read / (cache_read + cache_create + input) this turn
    cache_efficiency_trend: float # Normalized: 0.0 = stable/improving, 1.0 = declining fast
    repeated_reads: dict[str, int]  # resource → UNCHANGED read count in rolling window
    error_rate: float             # PostToolUseFailure / total tool calls in last N turns
    error_rate_spike: float       # max(0, (current_rate / max(session_avg, 0.01)) - 1.0)
    output_inflation: float       # Normalized: 0.0 = normal, 1.0 = 2x+ session avg
    edit_churn: list[str]         # Resources edited 2+ times within window (evidence, not penalty)
    compaction_count: int
    cost_this_turn: float
    cost_cumulative: float
```

Key changes from v1:
- **`dead_weight_ratio`**: only counts blocks with score >= 0.8 (consistent with "dead_weight" label). "Stale" blocks (0.6-0.8) are tracked but not counted as dead weight.
- **`cache_efficiency_trend`**: normalized to 0.0–1.0 using linear regression slope over last N turns, mapped through sigmoid. 0.0 = flat or improving, 1.0 = dropping fast. No unbounded values.
- **`error_rate_spike`**: uses `max(session_avg, 0.01)` to avoid divide-by-zero. Subtracts 1.0 so a normal rate (ratio=1.0) contributes 0 penalty.
- **`repeated_reads`**: uses rolling window (last N turns, not session-wide) and only counts reads where the content hash matches (unchanged rereads = attention loss; changed rereads = legitimate).
- **`edit_churn`**: collected as evidence for attention loss indicators but does NOT feed into urgency score directly (iterative editing is often normal).

### Cost Calculation

```python
# Pricing per million tokens (as of 2026-06-01)
PRICING = {
    "claude-opus-4-6": {
        "input": 15.0, "output": 75.0,
        "cache_read": 1.875, "cache_create": 18.75,
    },
    "claude-sonnet-4-6": {
        "input": 3.0, "output": 15.0,
        "cache_read": 0.375, "cache_create": 3.75,
    },
    # Fallback for unknown models
    "_default": {
        "input": 15.0, "output": 75.0,
        "cache_read": 1.875, "cache_create": 18.75,
    },
}

def compute_turn_cost(api_call: ApiCall, model: str) -> float:
    rates = PRICING.get(model, PRICING["_default"])
    return (
        api_call.input_tokens * rates["input"] / 1_000_000
        + api_call.output_tokens * rates["output"] / 1_000_000
        + api_call.cache_read_tokens * rates["cache_read"] / 1_000_000
        + api_call.cache_creation_tokens * rates["cache_create"] / 1_000_000
    )
```

### Attention Loss Indicators

```python
@dataclass
class AttentionLossSignal:
    signal_type: str   # repeated_read, edit_churn, error_spike, output_inflation
    severity: str      # info, warning, critical
    description: str   # Human-readable explanation
    turn: int
    resource: str | None
    evidence: dict     # Supporting data
```

Detection rules:
- **Repeated unchanged read**: same file Read 3+ times with same content hash in rolling window → warning, 5+ → critical. Changed rereads are not flagged (legitimate post-edit verification).
- **Edit churn**: same file Edited 2+ times within window → info (evidence only, not a health penalty — iterative development is normal)
- **Error spike**: error rate > 2x session average in last 10 turns → warning
- **Output inflation**: rolling avg output tokens > 1.5x session avg → info

### Session Recommendation

```python
@dataclass
class SessionRecommendation:
    urgency_score: float      # 0.0 to 1.0
    recommendation: str       # healthy, degrading, recommend_new_session, urgent
    reasons: list[str]        # Human-readable explanations
    recoverable_tokens: int   # Sum of dead_weight block sizes (score >= 0.8 only)
    recoverable_blocks: int
    top_stale_blocks: list[str]  # Block IDs, top 10 by size × staleness
    confidence: str           # "high" (many turns, calibrated), "low" (few turns, experimental)
```

Scoring formula:

```python
def compute_urgency(signals: HealthSignals, config: HealthConfig) -> float:
    score = (
        signals.dead_weight_ratio * config.weight_dead_weight        # default 0.35
      + signals.context_utilization * config.weight_utilization       # default 0.25
      + signals.cache_efficiency_trend * config.weight_cache          # default 0.15
      + signals.output_inflation * config.weight_output_inflation     # default 0.10
      + min(1.0, len([r for r, c in signals.repeated_reads.items() if c >= 3]) / 5)
        * config.weight_repeated                                     # default 0.10
      + signals.error_rate_spike * config.weight_errors              # default 0.05
    )
    return min(1.0, max(0.0, score))
```

Changes from v1:
- `cache_efficiency_trend` is now pre-normalized (0-1), so no `(1 - trend)` inversion needed
- `compaction_count` removed from urgency — compaction is not inherently bad; a successful compaction may improve health
- `output_inflation` added to urgency (was computed but unused)
- `error_rate_spike` is pre-normalized via the `max(avg, 0.01) - 1.0` formula, so no divide-by-zero
- `edit_churn` feeds attention loss indicators but NOT urgency score

Thresholds (all configurable, labeled as **uncalibrated defaults** requiring tuning against real sessions):

| Score | Recommendation | Description |
|---|---|---|
| < 0.3 | `healthy` | Session is productive, context is mostly active |
| 0.3–0.5 | `degrading` | Dead weight growing, consider clearing after current task |
| 0.5–0.7 | `recommend_new_session` | Significant dead weight, start fresh for best results |
| > 0.7 | `urgent` | Context mostly dead weight or nearly full, start fresh now |

---

## Module 4: Configuration (`analysis/config.py`)

All thresholds configurable. Defaults labeled as **uncalibrated** — to be tuned against real session data.

```python
@dataclass
class StalenessConfig:
    decay_window: int = 10
    resource_window: int = 10
    reference_scan_window: int = 15
    task_boundary_time_gap: int = 10     # Minutes
    task_boundary_overlap: float = 0.2
    min_prompt_length_for_boundary: int = 20  # Chars — skip overlap on short confirmations

@dataclass
class HealthConfig:
    model_context_window: int = 200_000  # Fallback if model not in MODEL_CONTEXT_WINDOWS
    weight_dead_weight: float = 0.35
    weight_utilization: float = 0.25
    weight_cache: float = 0.15
    weight_output_inflation: float = 0.10
    weight_repeated: float = 0.10
    weight_errors: float = 0.05
    threshold_healthy: float = 0.3
    threshold_degrading: float = 0.5
    threshold_recommend_new: float = 0.7
    repeated_read_warning: int = 3
    repeated_read_critical: int = 5
    repeated_read_rolling_window: int = 20  # Turns
    edit_churn_window: int = 5
    error_spike_multiplier: float = 2.0
    output_inflation_multiplier: float = 1.5
    cache_trend_window: int = 10           # Turns for trend calculation
```

Configuration loaded from `~/.claude/context-analyzer.json` if present, otherwise defaults. CLI flag `--config` overrides.

---

## Dashboard UI

### Serving

FastAPI app (`dashboard.py`) serves:
- `GET /` → `static/dashboard.html`
- `GET /static/*` → JS, CSS assets
- `GET /api/sessions` → list all sessions with summary stats (paginated, default 20)
- `GET /api/session/{id}/summary` → session summary + health recommendation
- `GET /api/session/{id}/turns` → turn snapshots with block IDs + staleness states (NOT full content)
- `GET /api/session/{id}/turn/{n}` → single turn: block IDs, states, and content previews
- `GET /api/session/{id}/turn/{n}/blocks` → full block content for drilldown (lazy loaded)
- `GET /api/session/{id}/staleness` → block lifespans, aggregate dead weight over time
- `GET /api/session/{id}/health` → health signals timeline + attention loss indicators
- `GET /api/session/{id}/recommendations` → session recommendation with top stale block IDs

Security: strict localhost binding (`127.0.0.1`), `Origin` header check rejecting non-localhost origins, HTML escaping on all rendered content.

Session identity includes project path to disambiguate session IDs across projects.

### Page Layout (validated via mockups)

**Main page** — five sections, top to bottom:

1. **Header + Health Scorecards** — session identity, 4 key metrics: dead weight %, context used, cache hit rate, tool calls. Health dot (green/yellow/red) based on urgency score.

2. **Turn Scrubber** — play/pause, slider, speed control. Controls all views below.

3. **Sediment Chart** — full-width stacked area chart showing active vs stale vs system context over time. Turn marker (dashed vertical line) tracks scrubber position.

4. **Context Tape + Recommendations** — two-column:
   - Left: block lifespan bars showing entry turn, current status (active/warm/stale/dead_weight), size
   - Right: ranked stale blocks with recoverable token count, attention loss signals

5. **Turn Details** — compact message list for selected turn. Click "Full drilldown" to open modal.

**Turn Drilldown Modal:**

- Summary pills (message count, token delta, cache hit, cost)
- Four stat panels across top: composition, turn growth, staleness at turn, cost
- Single-column message list (full width). Content loaded lazily via `/turn/{n}/blocks` endpoint.
- Stale blocks section below showing dead weight blocks still in context
- Turn navigation (prev/next)

### Tech Stack (Frontend)

- Single HTML file + JS + CSS (no build step, no framework)
- Chart.js for sediment chart, growth rate, cache performance
- Vanilla JS for DOM manipulation, modal, scrubber
- Fetches data from FastAPI REST endpoints
- Content loaded lazily on drilldown open (not pre-fetched for all turns)

### Dependencies (New)

Add to `pyproject.toml`:
```
fastapi>=0.115.0
uvicorn>=0.30.0
```

---

## MCP Server Extensions

New tools added to existing FastMCP server:

1. **`mcp_get_staleness_analysis`** — returns block IDs with staleness scores, aggregate dead weight ratio, top stale block IDs
2. **`mcp_get_session_health`** — returns health signals, attention loss indicators, urgency score with confidence level
3. **`mcp_get_new_session_recommendation`** — returns recommendation (healthy/degrading/recommend_new/urgent) with reasons, recoverable tokens, confidence
4. **`mcp_get_block_lifespans`** — returns block entry/exit turns, staleness labels, sizes

These use the same analysis modules as the dashboard.

---

## System Prompt Handling

- **Base system prompt**: bundled template files (from Piebald-AI extraction, versioned by Claude Code version). Labeled as **approximate** — actual runtime prompt includes interpolated values. Size estimated from first turn's `cache_creation_input_tokens` minus visible message sizes (this is an estimate, not exact — first-turn cache creation includes system prompt + tool definitions + first messages).
- **CLAUDE.md + Skills + Rules**: captured via `InstructionsLoaded` hook (`file_path` field). Read from disk. Note: disk content is the **current** version, not necessarily the version loaded during the session. Content shown in drilldown. Always labeled `pinned`.
- **Tool definitions**: known set, sizes from Piebald-AI data. Dynamic MCP tools may vary — labeled as estimates and versioned.

---

## Implementation Sequence

Codex recommended building bottom-up. Agreed — the sequence:

1. **Raw transcript parser** (`transcript_parser.py`) with tests. Foundation for everything.
2. **Context reconstruction** (`reconstruction.py`) — immutable blocks, content store, turn/epoch modeling, compaction detection.
3. **Deterministic staleness signals** — superseded reads, age decay, resource linking. No heuristics yet.
4. **Per-turn state overlays** (`BlockStateAtTurn`) and lazy content loading.
5. **Heuristic staleness signals** — reference scanning, grouping, task boundaries. Build with fixture-based tests for calibration.
6. **Health scoring** — after underlying metrics are validated against real sessions.
7. **REST API endpoints** and FastAPI dashboard serving.
8. **Dashboard UI** — last, consuming the validated API.

---

## Data Gaps (Known Limitations)

1. **No compaction summary content** — we detect compaction and know the summary length, but the summary text itself is not captured by hooks. Could be addressed by extending the `PostCompact` hook to capture summary content.
2. **System prompt verbatim text** — not in transcript. Approximate content from bundled templates (versioned, labeled as estimates).
3. **True attention tracking** — we approximate staleness via resource/reference heuristics, not actual model attention weights. All staleness scores should be labeled as experimental heuristics.
4. **InstructionsLoaded hook may not fire** — depends on project having CLAUDE.md/rules files.
5. **Token estimation** — per-block tokens estimated from char count (÷4). Exact token counts only available at the API-call level. Per-block estimates may diverge from actual tokenization.
6. **Incomplete transcript writes** — `os.write()` may write fewer bytes than requested. Storage layer needs a write loop.
7. **CLAUDE.md version drift** — files read from disk may differ from the version loaded at session time. Noted in UI as "current version, may differ from session".
8. **Staleness thresholds uncalibrated** — all scoring thresholds are defaults requiring tuning against diverse real sessions before treating recommendations as authoritative.
