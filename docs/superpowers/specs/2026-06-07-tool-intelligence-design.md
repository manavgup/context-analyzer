# Tool Intelligence: Subagent, MCP, and Skill Visualization

**Issue:** #43  
**Date:** 2026-06-07  
**Status:** Approved

## Problem

Subagents, MCP server tool calls, and skill invocations are significant parts of agentic workflows but are either invisible or misleadingly labeled in the dashboard. The existing composition breakdown lumps all tool I/O together, the "System + Skills" label refers to cached prefix blocks (not skill invocations), and rich subagent data already in the DB is never exposed to the frontend.

## Design Decisions

- **Layout:** Replace the existing composition column (280px left column) with a tabbed interface: Composition | Tools | Agents. No new panels or layout changes.
- **Scope:** All three categories (subagents, MCP, skills) in one feature. Ship incrementally — composition tab first, tools tab second, agents tab third.
- **MCP grouping:** Server-level with expandable per-function detail.
- **Subagent depth:** Full drill-down with mini growth charts per subagent.

## Architecture

### Tool Classification

Classify tool calls by parsing `tool_name` from `TOOL_USE` blocks only. `TOOL_RESULT` blocks do not carry `tool_name` -- they must be classified by following `parent_block_id` to the originating `TOOL_USE` block, or by building a `tool_use_id -> tool_name` lookup map during the classification pass.

| Pattern | Category | Example |
|---------|----------|---------|
| `tool_name.startswith("mcp__")` | MCP | `mcp__serena__find_symbol` |
| `tool_name == "Skill"` | Skill | Skill tool (see note on skill name extraction below) |
| `tool_name == "Agent"` | Agent | Agent tool |
| `tool_name == "Task"` | Task | Task tool (grouped with Agent) |
| Everything else | Builtin | Read, Write, Bash, Edit, Grep, Glob |

**MCP name parsing:** Split `tool_name` on `__` with `maxsplit=2` (not unlimited split), yielding `["mcp", server, function]`. This handles function names containing `__`. If server names contain `__` (unlikely but possible), the first segment after `mcp` is taken as server. Formally: `parts = tool_name.split("__", 2); server = parts[1]; function = parts[2]`.

**Skill name extraction:** `ContextBlock` does not carry `tool_input` -- it only has `tool_name`, `tool_use_id`, `resource`, and `resource_type`. To get the skill name (e.g., "codex"), the `/tool-intelligence` endpoint must re-read the raw transcript `tool_use` block content via `ContentStore` or by re-parsing the transcript entry. Specifically, look up the `TOOL_USE` block's content in `content_store`, parse the JSON `tool_input`, and extract `tool_input["skill"]`. This is the same pattern used by `extract_resource()` in `reconstruction.py`.

**Skill invocations vs skill prefix content:** Skill *invocations* are `tool_use` blocks with `name="Skill"` in the raw transcript. Skill *prefix content* (the skill instruction text that gets injected into the context) is categorized as pinned system content ("System Prefix"). These are distinct: the donut's "Skills" category counts only invocation tokens, not the cached prefix.

No new database fields or schema changes. Classification is computed at query time.

### Data Sources

| Data | Source | Already Exists? |
|------|--------|----------------|
| Tool call counts | `ContextBlock.tool_name` on `TOOL_USE` blocks in `block_registry` | Yes |
| Tool result pairing | `ContextBlock.parent_block_id` on `TOOL_RESULT` -> originating `TOOL_USE` | Yes |
| Tool token cost | `ContextBlock.size_chars` (proportioned to tokens) | Yes |
| Subagent stats | `SubagentRecord` + `SubagentApiCallRecord` in DB | Yes |
| Subagent per-call churn | `SubagentApiCallRecord.{input,output,cache_read,cache_creation}` | Yes |
| Skill name | `content_store` -> parse `tool_input` JSON -> `["skill"]` | Yes (in transcript) |
| MCP server/function | Parsed from `tool_name` string via `split("__", 2)` | Yes (derivable) |

## Components

### 1. Composition Tab (upgraded donut)

Replaces the current 3-category donut (System+Skills / Conversation / Tool I/O) with a 6-category donut:

- **System Prefix** — pinned/cached system blocks (replaces "System + Skills")
- **Conversation** — user prompts + assistant text + thinking
- **Regular Tools** — builtin tool_use + tool_result (Read, Write, Bash, etc.)
- **MCP Tools** — tool_use + tool_result where `tool_name.startswith("mcp__")`
- **Skills** — tool_use + tool_result where `tool_name == "Skill"`
- **Agent Spawns** — tool_use + tool_result where `tool_name in ("Agent", "Task")`

Token proportioning: The existing frontend donut computes composition from `blocks` returned by `/data` (ccscope.reconcile output), not from `block_registry`. The new `/tool-intelligence` backend endpoint replaces this frontend calculation entirely, providing pre-classified token totals. The frontend donut renders directly from the endpoint response -- no client-side block iteration needed. This eliminates the risk of two inconsistent composition models.

The prefix breakdown bar below the donut stays as-is.

### 2. Tools Tab

Three sections, each showing tool rows sorted by token cost descending:

**MCP Servers section:**
- One row per MCP server (grouped by parsed server name)
- Each row shows: server name, total token cost, call count, proportional bar
- Expandable: click to show per-function breakdown (function name + call count)
- Server name extracted from `mcp__<server>__<function>` — display as title case

**Skills section:**
- One row per unique skill name (from `tool_input["skill"]`)
- Shows: skill name, token cost, invocation count

**Regular Tools section:**
- One row per builtin tool (Read, Bash, Edit, etc.)
- Shows: tool name, token cost, call count, proportional bar

**Agent row:** Agent/Task tool calls shown at the bottom of Regular Tools with an `AGENT` badge.

### 3. Agents Tab

Shows per-subagent cards using data from `SubagentRecord` and `SubagentApiCallRecord` (already in DB, already ingested).

Each card contains:
- **Agent type badge** — from `SubagentRecord.agent_type` (e.g., "Explore", "general-purpose", "code-architect")
- **Description** — first 60 chars of `SubagentRecord.description`, truncated with ellipsis
- **Stats row** — Peak: `peak_resident` tokens, Cache: `total_cache_read` tokens, Calls: `total_api_calls`
- **Mini growth chart** — SVG area chart showing context window growth over the subagent's API calls, built from `SubagentApiCallRecord` churn data. X-axis: call index, Y-axis: cumulative resident tokens (input + cache_read + cache_creation per call).

Summary line at top: "N subagents — Total: XK tok"

### 4. Messages Pane Tool Badges

In `renderMessagesFromAPI()`, detect tool category from `msg.tool_name` and add a colored badge:

| Category | Badge | Color |
|----------|-------|-------|
| MCP | `MCP` | pink (#ec4899) |
| Skill | `SKILL` | amber (#f59e0b) |
| Agent/Task | `AGENT` | green (#34d399) |
| Builtin | `TOOL` | indigo (#6366f1) |

Also parse the display name:
- MCP: show `server.function` instead of full `mcp__server__function`
- Skill: show the skill name from tool_input
- Agent: show the agent type or first 40 chars of prompt

### 5. Growth Chart Annotations (optional, low priority)

Add subtle vertical line annotations at turns where subagents were launched, using the existing `chartjs-plugin-annotation` infrastructure. Label: "Agent launched". This is a nice-to-have and can be deferred.

## API Changes

### New endpoint: `GET /api/session/{session_id}/tool-intelligence`

Returns the classified tool breakdown for the composition donut and tools tab:

```json
{
  "composition": {
    "system_prefix_tokens": 27000,
    "conversation_tokens": 31000,
    "regular_tool_tokens": 22000,
    "mcp_tool_tokens": 11000,
    "skill_tokens": 5000,
    "agent_tokens": 4000
  },
  "mcp_servers": [
    {
      "server": "serena",
      "total_tokens": 18200,
      "call_count": 12,
      "functions": [
        {"name": "find_symbol", "count": 5},
        {"name": "get_diagnostics_for_file", "count": 3}
      ]
    }
  ],
  "skills": [
    {"name": "codex", "tokens": 8400, "count": 2}
  ],
  "regular_tools": [
    {"name": "Read", "tokens": 42300, "count": 38}
  ],
  "agents": [
    {"name": "Agent", "tokens": 5800, "count": 3}
  ]
}
```

Implementation: Reconstruct session. First pass: iterate `block_registry` for `TOOL_USE` blocks, classify by `tool_name`, build a `tool_use_id -> category` lookup map. For Skill blocks, read `content_store` to extract the skill name from `tool_input`. Second pass: iterate `TOOL_RESULT` blocks, classify by following `parent_block_id` to the originating `TOOL_USE` block (or via the `tool_use_id` lookup map). Sum `size_chars` per category, proportion against API-reported token totals.

### New endpoint: `GET /api/session/{session_id}/subagents`

Exposes existing `SubagentRecord` + `SubagentApiCallRecord` data from DB:

```json
{
  "count": 3,
  "total_peak_tokens": 182000,
  "subagents": [
    {
      "agent_id": "ae4f152e0565fc686",
      "agent_type": "Explore",
      "description": "Investigate image data in transcripts...",
      "peak_resident": 91000,
      "total_cache_read": 38000,
      "total_api_calls": 27,
      "total_output_tokens": 12400,
      "churn": [
        {"call_index": 0, "input_tokens": 1200, "output_tokens": 340, "cache_read": 0, "cache_creation": 1200},
        {"call_index": 1, "input_tokens": 800, "output_tokens": 520, "cache_read": 1200, "cache_creation": 400}
      ]
    }
  ]
}
```

Implementation: First call `_ensure_ingested(session_id)` (or equivalent) to guarantee the session has been ingested into the DB. The v3 dashboard may load sessions via `/data` (ccscope.reconcile) without triggering DB ingestion, so a pure DB query would return empty. After ensuring ingestion, query `SubagentRecord` + `SubagentApiCallRecord`. Compute `total_output_tokens` from `sum(SubagentApiCallRecord.output_tokens)` since `parse_subagents()` does not currently populate this field in the ingestion path.

### Extend existing: `GET /api/session/{id}/conv_turn/{n}/content`

Add `tool_category` and `tool_display_name` fields to each message in the response. These must be computed server-side because:

1. `tool_result` messages have `tool_use_id` but no `tool_name` -- the backend must look up the originating `tool_use` block to classify results.
2. Skill display names require parsing `tool_input` from the raw transcript entry, which is not currently returned by the endpoint.

For `tool_use` blocks: classify by `tool_name` string, extract display name (MCP: `server.function`, Skill: `tool_input["skill"]`, Agent: first 40 chars of `tool_input["prompt"]`).

For `tool_result` blocks: build a `tool_use_id -> {category, display_name}` map from the preceding `tool_use` blocks in the same turn, then apply to each result.

```json
{
  "type": "tool_use",
  "tool_name": "mcp__serena__find_symbol",
  "tool_category": "mcp",
  "tool_display_name": "serena.find_symbol",
  "content": "..."
}
```

```json
{
  "type": "tool_result",
  "tool_use_id": "toolu_01XYZ",
  "tool_category": "mcp",
  "tool_display_name": "serena.find_symbol",
  "content": "...",
  "is_error": false
}
```

Categories: `"mcp"`, `"skill"`, `"agent"`, `"builtin"`.

## Frontend Changes

### Tab infrastructure

Replace the composition column's static content with a tab bar + three tab content divs. Tab switching uses the same pattern as the existing middle column tabs (Top Growth / Dead Weight).

### State variables

```javascript
let toolIntelData = null;    // from /api/session/{id}/tool-intelligence
let subagentData = null;     // from /api/session/{id}/subagents
```

### Fetch functions

`fetchToolIntelData(sessionId)` — called from `loadSessionFromApi()`, non-blocking async. On success, calls `renderCompositionTab()` and `renderToolsTab()`.

`fetchSubagentData(sessionId)` — called from `loadSessionFromApi()`, non-blocking async. On success, calls `renderAgentsTab()`.

Both cleared in `switchSession()`.

### Render functions

`renderCompositionTab()` — draws the 6-category donut using existing Chart.js doughnut chart pattern, populates legend.

`renderToolsTab()` — builds MCP server rows (with expandable functions), skill rows, regular tool rows. Each row has a proportional bar showing relative token cost.

`renderAgentsTab()` — builds subagent cards with stats and inline SVG mini charts from churn data.

### Messages pane badges

In `renderMessagesFromAPI()`, after extracting `tool_name`, classify and add badge:

```javascript
function classifyTool(toolName) {
  if (!toolName) return { category: 'builtin', badge: 'TOOL', color: '#6366f1' };
  if (toolName.startsWith('mcp__')) return { category: 'mcp', badge: 'MCP', color: '#ec4899' };
  if (toolName === 'Skill') return { category: 'skill', badge: 'SKILL', color: '#f59e0b' };
  if (toolName === 'Agent' || toolName === 'Task') return { category: 'agent', badge: 'AGENT', color: '#34d399' };
  return { category: 'builtin', badge: 'TOOL', color: '#6366f1' };
}
```

## Files to Modify

| File | Change |
|------|--------|
| `src/context_tracker/dashboard.py` | Add `/tool-intelligence` and `/subagents` endpoints. Extend conv_turn content with `tool_category`. |
| `static/dashboard-v3.html` | Replace composition column with tabbed interface. Add CSS for tabs, tool rows, agent cards, mini charts, tool badges. Add JS fetch/render functions. |

## Files to Create

None. All changes go into existing files.

## Build Sequence

1. **Backend: `/tool-intelligence` endpoint** — classify blocks, compute token breakdown
2. **Backend: `/subagents` endpoint** — expose existing DB data
3. **Backend: extend conv_turn** — add `tool_category` and `tool_display_name`
4. **Frontend: tab infrastructure** — replace composition column with tab bar + 3 content divs
5. **Frontend: Composition tab** — 6-category donut + legend
6. **Frontend: Tools tab** — MCP servers (expandable), skills, regular tools
7. **Frontend: Agents tab** — subagent cards with mini growth charts
8. **Frontend: Messages badges** — tool-type colored badges in messages pane
9. **Rename** — "System + Skills" label in growth chart legend to "System Prefix"

## Out of Scope

- Cross-session tool usage analytics (future: requires multi-session aggregation)
- Tool usage timeline overlay on growth chart (deferred as optional enhancement)
- Task tool lifecycle tracking (low value -- tasks are ephemeral)
- MCP server health/error tracking (better suited for issue #41 error highlighting)

## Codex Review Findings (2026-06-07)

Independent review via `/codex` identified 5 P1 and 4 P2 findings. All addressed in this revision:

| # | Severity | Finding | Resolution |
|---|----------|---------|------------|
| 1 | P1 | `ContextBlock.tool_name` only on `TOOL_USE`, not `TOOL_RESULT` | Added parent_block_id pairing logic and tool_use_id lookup map |
| 2 | P1 | `tool_input` not on `ContextBlock` | Changed to read from `content_store` + parse JSON |
| 3 | P1 | conv_turn endpoint drops `tool_input` | Made tool_category/display_name server-computed |
| 4 | P1 | tool_result badges need pairing logic | Added tool_use_id -> category map for results |
| 5 | P1 | `/subagents` fragile for non-ingested sessions | Added `_ensure_ingested()` requirement |
| 6 | P2 | `total_output_tokens` not populated | Changed to compute from churn sum |
| 7 | P2 | Composition donut wrong data path | Clarified backend replaces frontend calculation |
| 8 | P2 | MCP `split("__")` underspecified | Changed to `split("__", 2)` with maxsplit |
| 9 | P2 | Skill invocations vs prefix content | Added explicit distinction section |
