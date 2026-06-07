# Error Highlighting: Surface Tool Failures, Retry Patterns, and Self-Corrections

**Issue:** #41
**Date:** 2026-06-07
**Status:** Approved

## Problem

Claude Code sessions contain rich error signal data that is already captured but not surfaced in the dashboard:
- `is_error: true` flag on tool_result blocks (parsed and stored in ContentBlock/ContextBlock)
- `PostToolUseFailureEvent` hook events with tool_name, error_length, timestamp
- `error_rate` and `error_rate_spike` in HealthSignals

Users cannot see where errors happened, whether Claude was struggling with a particular approach, or when Claude acknowledged its own mistakes.

## Design Decisions

- **Three detection layers** with confidence grading: L1 tool failures (HIGH), L2 retry patterns (MEDIUM), L3 self-correction text detection (LOW)
- **Visualization**: Error markers on growth chart, ERR/RETRY/FIX badges in messages pane, error scorecard, error cluster recommendations
- **No new models or DB changes**: All data derived from existing `is_error` flag, `tool_name`, and assistant text content at query time

## Architecture

### Detection Layers

#### L1: Tool Failures (HIGH confidence)

Direct signal: `is_error: true` on `tool_result` blocks. Already captured in `ContentBlock.is_error` and `ContextBlock.is_error`.

Data sources:
- `ContextBlock.is_error` in `block_registry` (from reconstruction)
- `HealthSignals.error_rate` and `error_rate_spike` (already computed in `health.py`)

Per-turn error count: iterate `snap.blocks_entered_ids` for each `TurnSnapshot`, count blocks where `block_registry[bid].is_error == True`.

#### L2: Retry Patterns (MEDIUM confidence)

Same `tool_name` with `is_error: true` appearing 2+ times within a 3-turn sliding window. Indicates Claude is retrying a failing operation.

Detection: For each turn's error blocks, collect `(tool_name, turn_number)` pairs. Slide a 3-turn window, group by tool_name, flag groups with count >= 2.

#### L3: Self-Correction Detection (LOW confidence)

Scan assistant text content for language patterns indicating Claude acknowledged an error. Two confidence tiers:

**High-confidence patterns:**
```python
SELF_CORRECTION_HIGH = [
    r"I (?:made|introduced) (?:an?|the) error",
    r"(?:that|this) (?:was|is) (?:wrong|incorrect|a bug)",
    r"I (?:accidentally|mistakenly)",
    r"(?:let me|I(?:'ll| will)) (?:fix|correct|revert) (?:that|this)",
]
```

**Medium-confidence patterns:**
```python
SELF_CORRECTION_MEDIUM = [
    r"I apologize",
    r"(?:actually|wait),? I (?:need|should) to",
    r"I (?:forgot|missed|overlooked)",
    r"(?:that|the previous) (?:approach|change) (?:didn't|won't) work",
]
```

Detection: For each assistant text block in `block_registry` (where `block_type == ASSISTANT_TEXT`), retrieve content from `content_store`, test against patterns. Record `(turn_number, confidence, matched_pattern, preview)`.

Note: `ContextBlock` does not carry `tool_input` or raw content. Assistant text content must be read from `content_store.get_content(block_id)` or by re-reading the transcript.

## Components

### 1. Backend: `/errors` Endpoint (`dashboard.py`)

`GET /api/session/{session_id}/errors`

Implementation:
1. Reconstruct session (same pattern as `/health`, `/turns`)
2. L1: Iterate `block_registry` for `TOOL_RESULT` blocks with `is_error == True`. Count per turn.
3. L2: Slide 3-turn window over per-turn error lists, group by `tool_name` (look up via `parent_block_id` -> `TOOL_USE` block), flag groups with count >= 2.
4. L3: Iterate `block_registry` for `ASSISTANT_TEXT` blocks, retrieve content from `content_store`, test against self-correction regexes.
5. Cluster detection: group consecutive turns with errors (>= 2 turns = cluster).

Response shape:
```json
{
  "session_id": "93d74aaa...",
  "total_errors": 5,
  "total_tool_results": 94,
  "error_rate": 0.053,
  "per_turn": [
    {"turn": 42, "error_count": 2, "errors": [
      {"block_id": "b123", "tool_name": "Bash", "size_chars": 420}
    ]}
  ],
  "clusters": [
    {"start_turn": 42, "end_turn": 44, "turn_count": 3, "total_errors": 5}
  ],
  "retry_patterns": [
    {"tool_name": "Bash", "window_start_turn": 42, "retry_count": 3}
  ],
  "self_corrections": [
    {"turn": 42, "confidence": "high", "pattern": "I made an error", "preview": "I made an error -- the import path is wrong..."}
  ]
}
```

Note on `tool_name` for error blocks: `TOOL_RESULT` blocks do not carry `tool_name` directly. Must follow `parent_block_id` to the originating `TOOL_USE` block or build a `tool_use_id -> tool_name` map. Same pattern required by #43 tool intelligence -- share the helper.

### 2. Backend: Extend `/turns` Response (`dashboard.py`)

Add `error_count` and `tool_result_count` to each turn in the existing `get_session_turns` response (line 350). Computed from `snap.blocks_entered_ids` against `block_registry`.

### 3. Frontend: Error Scorecard (`dashboard-v3.html`)

Add 6th scorecard (already shown in combined mockup, grid changes to `repeat(6, 1fr)`):

```html
<div class="scorecard">
  <div class="scorecard-label">Tool Errors</div>
  <div class="scorecard-value" id="sc-errors">--</div>
  <div class="scorecard-sub" id="sc-errors-sub">--</div>
</div>
```

`renderErrorScorecard()`: Display total error count (red when > 0), sub-label shows error rate % and cluster count.

### 4. Frontend: Growth Chart Error Markers (`dashboard-v3.html`)

`addErrorAnnotationsToGrowthChart()`:

- **L1 Red dots**: `point` annotations at turns with errors. Radius scales with error count (1 error = r:5, 2+ = r:7). Maps `conv_turn` -> API call index via `turnMap[].last_call`.
- **Cluster bands**: `box` annotations with red tint for runs of 2+ consecutive error turns.
- **L3 Amber triangles**: Point annotations with custom triangle shape at turns where self-corrections were detected.

Called after `rebuildGrowthChartWithStaleness()` and `setBudget()`.

Chart tooltip enhanced: if `errorData` exists for this API call's conv_turn, append "N tool error(s) this turn" to the afterBody tooltip.

Legend addition at bottom-right: red dot = errors, amber triangle = self-corrections.

### 5. Frontend: Messages Pane Badges (`dashboard-v3.html`)

Three badge types in `renderMessagesFromAPI()`:

| Badge | CSS class | When shown | Color |
|-------|-----------|------------|-------|
| `ERR` | `.badge-err` | `msg.is_error == true` (tool_result) | red (#ef4444) |
| `RETRY` | `.badge-retry` | tool_result is part of a retry pattern (from errorData) | amber (#f59e0b) |
| `FIX` | `.badge-fix` | assistant text matches self-correction pattern (from errorData) | dark amber (#d97706) |

Row styling:
- `ERR` rows: red background tint + red left border (`.msg-row.is-error`)
- `FIX` rows: amber background tint + amber left border (`.msg-row.is-self-correct`)
- Badges stack when both apply: `[ERR] [RETRY]`

Badge ordering on message rows (compatible with #40 and #43):
`[ROLE] [ERR] [RETRY or FIX] [TOOL_TYPE from #43] [preview] [IMG from #40] [SIZE]`

Messages pane title: if turn has errors, append error count hint in red.

### 6. Frontend: Error Cluster Recommendations (`dashboard-v3.html`)

`renderErrorClusterBanner()`: Injects error cluster cards and retry pattern cards into the existing recommendations grid (prepended so they appear first).

- Error cluster cards (critical priority): "Error cluster: Turns N-M -- X tool errors across Y turns"
- Retry pattern cards (warning priority): "Retry pattern: ToolName -- N errors within 3 turns"
- Self-correction cards (warning priority): "Self-correction detected -- N patterns found (X high / Y medium confidence)"

Called after `renderRecommendations()` to ensure rec-grid exists.

### 7. Frontend: Modal Error Styling (`dashboard-v3.html`)

In `renderMessageBlock()`:
- Error tool_result blocks: red left border + red background tint + `ERR` badge
- Self-correction assistant text: amber left border + `FIX` badge
- Error content displayed prominently (not truncated to 80 chars in modal view)

### 8. Frontend: State and Fetch (`dashboard-v3.html`)

State variables:
```javascript
let errorData = null;        // from /api/session/{id}/errors
let errorTurnSet = new Set(); // conv_turn numbers with errors (O(1) lookup)
let selfCorrectionTurnSet = new Set(); // turns with self-corrections
```

`fetchErrorData(sessionId)`: async fetch, builds lookup sets, calls `addErrorAnnotationsToGrowthChart()`, `renderErrorScorecard()`, `renderErrorClusterBanner()`.

Wired into `loadSessionFromApi()` and cleared in `switchSession()`.

## Coordinate Space

Growth chart x-axis = 0-based API call index. Error data = 1-based conv_turn. Bridge via `turnMap[].last_call`. Fallback: `conv_turn - 1` when turnMap unavailable.

## Compatibility with Other Issues

| Issue | Touch point | Interaction |
|-------|------------|-------------|
| #40 Images | Messages pane badges | Image badge after preview, error badge before preview. No conflict. |
| #43 Tool Intelligence | Messages pane badges, conv_turn response | Tool-type badge between error badges and preview. No conflict. |

## Files to Modify

| File | Change |
|------|--------|
| `src/context_tracker/dashboard.py` | Add `/errors` endpoint. Extend `/turns` with error counts. |
| `static/dashboard-v3.html` | Error scorecard, chart annotations, ERR/RETRY/FIX badges, error cluster recommendations, modal error styling, state/fetch wiring. |

## Files to Create

None.

## Build Sequence

1. **Backend: `/errors` endpoint** -- L1 per-turn error counts, L2 cluster + retry detection, L3 self-correction scanning
2. **Backend: extend `/turns`** -- add `error_count`, `tool_result_count` per turn
3. **Frontend CSS** -- `.msg-row.is-error`, `.msg-row.is-self-correct`, `.badge-err`, `.badge-retry`, `.badge-fix`
4. **Frontend HTML** -- 6th scorecard `sc-errors`; update grid to `repeat(6, 1fr)`
5. **Frontend JS state** -- `errorData`, `errorTurnSet`, `selfCorrectionTurnSet`
6. **Frontend JS** -- `fetchErrorData()`, wire into `loadSessionFromApi()` and `switchSession()`
7. **Frontend JS** -- `addErrorAnnotationsToGrowthChart()` (red dots, cluster bands, amber triangles)
8. **Frontend JS** -- error badges in `renderMessagesFromAPI()` (ERR, RETRY, FIX)
9. **Frontend JS** -- error count hint in messages pane title
10. **Frontend JS** -- `renderErrorScorecard()`
11. **Frontend JS** -- `renderErrorClusterBanner()`
12. **Frontend JS** -- modal error styling in `renderMessageBlock()`

## Out of Scope

- Persisting error analysis results (computed fresh per request)
- Error categorization by type (syntax error, permission error, etc.) -- future enhancement
- Cross-session error trend analysis
- Automatic error remediation suggestions
