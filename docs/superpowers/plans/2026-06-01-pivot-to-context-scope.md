# Pivot to Context Scope — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pivot the Context Analyzer to produce Context Scope's `blocks.json` + `churn.json` format from real session data (5 sources), serve the Context Scope visualizer, and expose analysis via MCP tools. Individual blocks (not opaque working-set blobs) so each tool result, message, and thinking block is visible and inspectable as you scrub the timeline.

**Architecture:** Our Python backend becomes `ccscope build` — parsing the parent transcript (A), hook events (B), tool-results offload (D), and subagent transcripts (E) into Context Scope's block + churn contracts. The Context Scope HTML visualizer is served as-is. MCP server wraps the same pipeline. The existing `static/dashboard.*` files are removed.

**Tech Stack:** Python 3.11+, FastMCP (existing), FastAPI + Uvicorn (existing), Context Scope HTML/SVG visualizer (adopted)

**Spec:** Context Scope Build Handoff v2 (provided by user)

---

## File Structure (After Pivot)

```
src/context_tracker/
├── server.py                    # MODIFY: update MCP tools to use new pipeline
├── dashboard.py                 # MODIFY: serve context-scope.html, add /blocks.json + /churn.json endpoints
├── ccscope/
│   ├── __init__.py              # CREATE: ccscope package
│   ├── parse_transcript.py      # CREATE: Source A → blocks + churn (the spine)
│   ├── parse_hooks.py           # CREATE: Source B overlay (failures, timing)
│   ├── offload.py               # CREATE: Source D → resident vs spilled resolution
│   ├── subagents.py             # CREATE: Source E → child window blocks + churn
│   ├── reconcile.py             # CREATE: merge A+B+D+E → {blocks, churn, subagents}
│   ├── tokens.py                # CREATE: token estimation (chars/4 default)
│   └── cli.py                   # CREATE: ccscope list / build / open CLI
├── transcript_parser.py         # KEEP: reuse for raw parsing, fed into ccscope/parse_transcript.py
├── analysis/
│   ├── models.py                # KEEP: data models (ContextBlock etc — used internally)
│   ├── staleness.py             # KEEP: feeds into block `ref` field
│   ├── health.py                # KEEP: feeds into MCP health tools
│   ├── config.py                # KEEP: thresholds
│   └── reconstruction.py        # DEPRECATE: replaced by ccscope/parse_transcript.py
├── models.py                    # KEEP: hook event models
├── hooks.py                     # KEEP: hook processor
├── installer.py                 # KEEP: hook installer
├── storage.py                   # KEEP: JSONL storage
└── transcript.py                # KEEP: API token extractor (still useful for quick stats)
static/
├── context-scope.html           # CREATE: copy of Context Scope visualizer
├── blocks.json                  # GENERATED: by ccscope build
└── churn.json                   # GENERATED: by ccscope build
tests/
├── test_parse_transcript_cs.py  # CREATE: Context Scope transcript parser tests
├── test_offload.py              # CREATE
├── test_subagents.py            # CREATE
├── test_reconcile.py            # CREATE
└── ...                          # KEEP: existing tests (may need updates)
```

**Removed:** `static/dashboard.html`, `static/dashboard.js`, `static/dashboard.css` (replaced by context-scope.html)

---

## Output Contracts

### blocks.json — one entry per content block per API call

```jsonc
[
  // Pinned prefix blocks (enter:0, exit:null, cached:true)
  {"id":"sys", "type":"system", "label":"system prompt", "tokens":6300,
   "enter":0, "exit":null, "cached":true, "ref":true,
   "content":"System prompt ~6300 tok. Static, cached."},

  {"id":"skills", "type":"skill", "label":"CLAUDE.md + skills", "tokens":15500,
   "enter":0, "exit":null, "cached":true, "ref":true,
   "content":"CLAUDE.md + skills ~15500 tok. Static, cached."},

  // Per-turn individual blocks (enter:N, exit:null or M)
  {"id":"t14-user", "type":"user", "label":"user prompt", "tokens":28,
   "enter":14, "exit":null, "cached":false, "ref":true,
   "content":"Fix the bug in server.py"},

  {"id":"t14-asst-0", "type":"assistant", "label":"assistant", "tokens":150,
   "enter":14, "exit":null, "cached":false, "ref":true,
   "content":"Let me read the file to understand the issue."},

  {"id":"t14-tc-toolu_01", "type":"tool_call", "label":"Read server.py", "tokens":45,
   "enter":14, "exit":null, "cached":false, "ref":true,
   "content":"{\"file_path\": \"/src/server.py\"}"},

  {"id":"t14-tr-toolu_01", "type":"tool_result", "label":"Read → server.py", "tokens":3200,
   "enter":14, "exit":null, "cached":false, "ref":false,
   "content":"\"\"\"FastMCP server...\"\"\"\n\nfrom __future__..."},

  // After compaction: summary block replaces evicted content
  {"id":"compact-1", "type":"summary", "label":"compaction summary", "tokens":1500,
   "enter":50, "exit":null, "cached":false, "ref":true,
   "content":"(compaction summary — content not captured)"}
]
```

**Key differences from reference data:**
- Individual blocks per message/tool instead of one "working set" blob per API call
- `tokens` is RESIDENT size from `message.usage` (not chars/4 estimate)
- Tool results that were offloaded to disk: `tokens` = preview/truncated size in window, not full file size
- `ref` field indicates whether the block's content is referenced later (staleness signal)
- `exit` is set when compaction evicts the block, or null if still resident

### churn.json — one entry per API call

```jsonc
[
  {"turn":0, "cache_read":0, "cache_creation":30267, "input":2, "output":565},
  {"turn":1, "cache_read":30267, "cache_creation":1031, "input":1, "output":61},
  // ...527 entries total for session 81dc8a2f
]
```

Directly from `message.usage` per completed API call. No estimation.

---

### Task 1: Adopt Context Scope Visualizer

**Files:**
- Create: `static/context-scope.html` (copy from user-provided HTML)
- Remove: `static/dashboard.html`, `static/dashboard.js`, `static/dashboard.css`
- Modify: `src/context_tracker/dashboard.py` (serve context-scope.html, add JSON endpoints)

- [ ] **Step 1: Copy the Context Scope HTML to static/**

Copy the user-provided `context-scope.html` into `static/context-scope.html`. This is the visualizer — we serve it, we don't modify it (except to add fetch-loader for blocks.json/churn.json on load, per the handoff doc §9).

- [ ] **Step 2: Add fetch-loader to the visualizer**

At the top of the `<script>` block in context-scope.html, before `const SAMPLE=[...]`, add:

```javascript
// Fetch real data if available, fall back to built-in SAMPLE
(async function loadData() {
  try {
    const [blocksResp, churnResp] = await Promise.all([
      fetch('./blocks.json').catch(() => null),
      fetch('./churn.json').catch(() => null),
    ]);
    if (blocksResp && blocksResp.ok) {
      const blocks = await blocksResp.json();
      if (Array.isArray(blocks) && blocks.length > 0) {
        trace = blocks;
        sel = null; turn = 0;
        recompute(); renderTour(); renderAll(); setTurn(0);
      }
    }
    if (churnResp && churnResp.ok) {
      const churn = await churnResp.json();
      if (Array.isArray(churn) && churn.length > 0) {
        CHURN = churn;
        drawChurn();
      }
    }
  } catch (e) { console.warn('Failed to load trace data, using built-in sample:', e); }
})();
```

- [ ] **Step 3: Remove old dashboard files**

```bash
rm -f static/dashboard.html static/dashboard.js static/dashboard.css
```

- [ ] **Step 4: Update dashboard.py to serve context-scope.html**

In `dashboard.py`, change the `serve_dashboard()` function to look for `context-scope.html`:

```python
@app.get("/")
def serve_dashboard():
    index = static_dir / "context-scope.html"
    if index.exists():
        return FileResponse(str(index))
    return HTMLResponse("<h1>Context Analyzer</h1><p>Run ccscope build first.</p>")
```

Add endpoints for blocks.json and churn.json:

```python
@app.get("/blocks.json")
def get_blocks_json():
    blocks_path = static_dir / "blocks.json"
    if blocks_path.exists():
        return FileResponse(str(blocks_path), media_type="application/json")
    raise HTTPException(status_code=404, detail="Run ccscope build first")

@app.get("/churn.json")
def get_churn_json():
    churn_path = static_dir / "churn.json"
    if churn_path.exists():
        return FileResponse(str(churn_path), media_type="application/json")
    raise HTTPException(status_code=404, detail="Run ccscope build first")
```

- [ ] **Step 5: Commit**

```bash
git add static/context-scope.html src/context_tracker/dashboard.py
git rm static/dashboard.html static/dashboard.js static/dashboard.css
git commit -m "feat: adopt Context Scope visualizer, remove old dashboard"
```

---

### Task 2: Create ccscope Package + Transcript Parser (Source A → blocks + churn)

This is the spine. Parse the parent transcript JSONL and produce individual blocks + churn series.

**Files:**
- Create: `src/context_tracker/ccscope/__init__.py`
- Create: `src/context_tracker/ccscope/parse_transcript.py`
- Create: `src/context_tracker/ccscope/tokens.py`
- Create: `tests/test_parse_transcript_cs.py`

- [ ] **Step 1: Create ccscope package**

`src/context_tracker/ccscope/__init__.py`:
```python
"""ccscope — Context Scope data pipeline. Produces blocks.json + churn.json from Claude Code session data."""
```

- [ ] **Step 2: Create token estimator**

`src/context_tracker/ccscope/tokens.py`:
```python
"""Token estimation. Default: chars/4. Future: pluggable tokenizer."""

def estimate_tokens(text: str) -> int:
    """Approximate token count from text. ~4 chars/token for English."""
    return max(1, len(text) // 4)
```

- [ ] **Step 3: Write failing tests for transcript parser**

`tests/test_parse_transcript_cs.py` — test that the parser:
- Produces blocks in Context Scope format ({id, type, label, tokens, enter, exit, cached, ref, content})
- Produces churn entries from usage data
- Creates individual blocks per message (not one blob per API call)
- Correctly computes resident tokens from usage deltas
- Handles pinned system prefix
- Generates unique block IDs
- Sets enter/exit turns correctly

- [ ] **Step 4: Write the transcript parser**

`src/context_tracker/ccscope/parse_transcript.py`:

The parser reads the parent transcript and produces:
1. **Pinned prefix blocks**: system prompt + CLAUDE.md/skills estimated from first API call's `cache_creation_input_tokens`
2. **Per-message blocks**: for each user message, assistant message, tool_use, tool_result — a separate block with:
   - `id`: unique (e.g., `t{turn}-{type}-{index}`)
   - `type`: mapped from transcript type to Context Scope type
   - `label`: descriptive (e.g., "Read → server.py", "user prompt", "assistant")
   - `tokens`: RESIDENT tokens. For tool results, computed from usage delta between API calls. For the prefix, from cache_creation on first call.
   - `enter`: the API call number (turn) when this block entered
   - `exit`: null (until compaction), or the turn when evicted
   - `cached`: true for prefix blocks, false for dynamic
   - `ref`: true initially (staleness analysis sets to false later)
   - `content`: truncated to first 500 chars for display (full content available via drilldown)
3. **Churn series**: one entry per completed API call with cache_read, cache_creation, input, output from `message.usage`

**Computing resident tokens per block:**

The key insight: `message.usage` gives us the TOTAL context size per API call, not per block. To get per-block token counts:
- The total resident at API call N = `input_tokens + cache_read_input_tokens + cache_creation_input_tokens`
- The growth from call N-1 to call N = total_N - total_{N-1}
- This growth is the tokens added by the messages/tool results between those calls
- The prefix (system + skills) is constant = first call's cache_creation (before any conversation)

For individual blocks within a turn, distribute the growth proportionally by character count:
```python
growth = total_at_call_N - total_at_call_N_minus_1
blocks_this_call = [list of content blocks in this assistant message + preceding tool results]
total_chars = sum(len(b.content) for b in blocks_this_call)
for b in blocks_this_call:
    b.tokens = round(growth * len(b.content) / max(total_chars, 1))
```

This gives real-token-proportional sizes that sum to the actual context growth.

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_parse_transcript_cs.py -v
```

- [ ] **Step 6: Test against real data**

```python
from context_tracker.ccscope.parse_transcript import parse_transcript_to_blocks
from pathlib import Path

blocks, churn = parse_transcript_to_blocks(
    Path.home() / '.claude/projects/-Users-mg-Downloads-claude-src/81dc8a2f-2bc6-4241-81bb-9dea09f45a68.jsonl'
)
print(f'Blocks: {len(blocks)}, Churn entries: {len(churn)}')
print(f'First block: {blocks[0]["id"]} ({blocks[0]["type"]}) - {blocks[0]["tokens"]} tokens')
print(f'Last block: {blocks[-1]["id"]} ({blocks[-1]["type"]}) - {blocks[-1]["tokens"]} tokens')

# Verify churn totals match ground truth
total_cr = sum(c['cache_read'] for c in churn)
total_in = sum(c['input'] for c in churn)
print(f'Total cache_read: {total_cr:,}, input: {total_in:,}, ratio: {total_cr // max(total_in, 1):,}x')
# Expected: ~94M cache_read, ~452 input, ~208,000x ratio
```

- [ ] **Step 7: Commit**

```bash
git add src/context_tracker/ccscope/ tests/test_parse_transcript_cs.py
git commit -m "feat: ccscope transcript parser producing individual blocks + churn from usage data"
```

---

### Task 3: Tool-Results Offload Resolution (Source D)

**Files:**
- Create: `src/context_tracker/ccscope/offload.py`
- Create: `tests/test_offload.py`

- [ ] **Step 1: Write failing tests**

Test that:
- When a tool_result block's content matches a file in `tool-results/`, the block's `tokens` field reflects the TRUNCATED/preview size, not the full file
- A `spilled_tokens` field is added with the full size
- Blocks without offloaded files are unchanged

- [ ] **Step 2: Write offload resolver**

`src/context_tracker/ccscope/offload.py`:

```python
def resolve_offloads(
    blocks: list[dict],
    tool_results_dir: Path,
) -> list[dict]:
    """Adjust block token counts for tool-result offloads.
    
    When Claude Code spills a large tool output to disk, the in-window
    content is a truncated preview. The full content is in tool-results/.
    
    For each tool_result block, check if a matching file exists in
    tool_results_dir. If so:
    - `tokens` stays as the RESIDENT (preview) size
    - `spilled_tokens` is added with the full file size
    - `content` note updated to indicate offload
    """
```

The matching: tool_result blocks have a tool_use_id. The offload files are named by a content hash/ID. Check if the in-transcript content contains a reference to `tool-results/` or if it's truncated (ends with `...` or has a truncation marker).

- [ ] **Step 3: Test against real data**

28 tool-result files exist for session 81dc8a2f. Verify they match tool_result blocks in the transcript.

- [ ] **Step 4: Commit**

```bash
git add src/context_tracker/ccscope/offload.py tests/test_offload.py
git commit -m "feat: resolve tool-result offloads for resident vs spilled token counts"
```

---

### Task 4: Subagent Parsing (Source E)

**Files:**
- Create: `src/context_tracker/ccscope/subagents.py`
- Create: `tests/test_subagents.py`

- [ ] **Step 1: Write failing tests**

Test that:
- Each subagent transcript produces its own block set + churn
- `.meta.json` is read for agentType and description
- Peak resident is computed from max(input + cache_read + cache_create) across calls
- Subagent churn is separate from parent (never summed into parent totals)
- A collapsed parent-level block is generated for each subagent

- [ ] **Step 2: Write subagent parser**

`src/context_tracker/ccscope/subagents.py`:

```python
def parse_subagents(
    subagents_dir: Path,
) -> list[dict]:
    """Parse subagent transcripts into summary blocks + churn.
    
    Each subagent gets:
    - A collapsed block in the parent timeline (type="tool_result", 
      labeled with agentType + description, sized by peak resident)
    - Its own blocks + churn for nested expansion
    
    Returns list of {
        "agent_id": str,
        "agent_type": str,
        "description": str,
        "peak_resident": int,
        "total_cache_read": int,
        "api_calls": int,
        "parent_block": dict,  # block for parent timeline
        "blocks": list[dict],  # child blocks (for nested view)
        "churn": list[dict],   # child churn
    }
    """
```

- [ ] **Step 3: Test against real data (17 subagents)**

- [ ] **Step 4: Commit**

```bash
git add src/context_tracker/ccscope/subagents.py tests/test_subagents.py
git commit -m "feat: parse subagent transcripts for nested blocks + churn"
```

---

### Task 5: Reconciliation (Merge All Sources)

**Files:**
- Create: `src/context_tracker/ccscope/reconcile.py`
- Create: `tests/test_reconcile.py`

- [ ] **Step 1: Write failing tests**

Test that:
- Blocks from transcript (A) are the spine
- Hook events (B) overlay failure info and timing
- Offload (D) adjusts resident sizes
- Subagents (E) add collapsed blocks to parent timeline
- Output validates against Context Scope's block contract
- Churn totals match transcript usage totals

- [ ] **Step 2: Write reconciler**

`src/context_tracker/ccscope/reconcile.py`:

```python
def reconcile(
    session_id: str,
    transcript_path: Path,
    hook_events_path: Path | None = None,
    tool_results_dir: Path | None = None,
    subagents_dir: Path | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Reconcile all 5 sources into Context Scope format.
    
    Returns:
        blocks: list of block dicts (Context Scope format)
        churn: list of churn dicts (per API call)
        subagents: list of subagent summaries
    """
    # 1. A is the spine
    blocks, churn = parse_transcript_to_blocks(transcript_path)
    
    # 2. Overlay B for failures/timing
    if hook_events_path and hook_events_path.exists():
        blocks = overlay_hook_events(blocks, hook_events_path)
    
    # 3. Resolve D for resident sizes
    if tool_results_dir and tool_results_dir.exists():
        blocks = resolve_offloads(blocks, tool_results_dir)
    
    # 4. Attach E as nested groups
    subagent_data = []
    if subagents_dir and subagents_dir.exists():
        subagent_data = parse_subagents(subagents_dir)
        # Add collapsed blocks to parent timeline
        for sa in subagent_data:
            blocks.append(sa["parent_block"])
    
    # 5. Apply staleness (ref field)
    blocks = apply_staleness(blocks)
    
    return blocks, churn, subagent_data
```

- [ ] **Step 3: Run tests + verify against real data**

- [ ] **Step 4: Commit**

```bash
git add src/context_tracker/ccscope/reconcile.py tests/test_reconcile.py
git commit -m "feat: reconcile all sources into Context Scope blocks + churn format"
```

---

### Task 6: CLI (`ccscope list / build / open`)

**Files:**
- Create: `src/context_tracker/ccscope/cli.py`
- Modify: `pyproject.toml` (add `ccscope` entry point)

- [ ] **Step 1: Write the CLI**

`src/context_tracker/ccscope/cli.py`:

```python
"""ccscope CLI — Context Scope data pipeline.

Usage:
    ccscope list                     # List available sessions
    ccscope build [SESSION]          # Build blocks.json + churn.json
    ccscope open [SESSION]           # Build + serve + open browser
    ccscope build --session-dir DIR  # Build from explicit session dir
"""
```

Commands:
- `ccscope list` — scans `~/.claude/projects/*/` for session JSONL files, prints session IDs with dates and sizes
- `ccscope build [SESSION]` — runs reconcile, writes `blocks.json` + `churn.json` to `static/` (or `--output DIR`)
- `ccscope open [SESSION]` — build + start uvicorn + open browser to localhost

Auto-discovery: given a session ID, find the transcript at `~/.claude/projects/*/{session_id}.jsonl`, the session folder at `~/.claude/projects/*/{session_id}/` (containing `tool-results/` and `subagents/`), and hook events at `~/.claude/context-trace/{session_id}.jsonl`.

- [ ] **Step 2: Add entry point to pyproject.toml**

```toml
[project.scripts]
context-tracker = "context_tracker.server:main"
context-tracker-hook = "context_tracker.hooks:main"
ccscope = "context_tracker.ccscope.cli:main"
```

- [ ] **Step 3: Test the CLI**

```bash
ccscope list
ccscope build 81dc8a2f-2bc6-4241-81bb-9dea09f45a68
ccscope open 81dc8a2f-2bc6-4241-81bb-9dea09f45a68
```

- [ ] **Step 4: Commit**

```bash
git add src/context_tracker/ccscope/cli.py pyproject.toml
git commit -m "feat: add ccscope CLI for list/build/open commands"
```

---

### Task 7: Update MCP Server Tools

**Files:**
- Modify: `src/context_tracker/server.py`

- [ ] **Step 1: Update MCP tools to use ccscope pipeline**

The 4 analysis MCP tools (`mcp_get_staleness_analysis`, `mcp_get_session_health`, `mcp_get_new_session_recommendation`, `mcp_get_block_lifespans`) should use the ccscope reconciler instead of the old reconstruction. The reconciler produces blocks with real token counts and the `ref` field for staleness.

- [ ] **Step 2: Add `mcp_get_cache_churn` tool**

New MCP tool that returns the cache-read churn summary:
```python
@mcp.tool(description="Get cache-read churn analysis: total cache reads, churn ratio, top churn calls.")
def mcp_get_cache_churn(session_id: str = "") -> str:
    # Returns: total_cache_read, total_input, ratio, top_calls
```

This is the metric that actually drove cost in the reference session (208,000x ratio).

- [ ] **Step 3: Run tests**

- [ ] **Step 4: Commit**

```bash
git add src/context_tracker/server.py
git commit -m "feat: update MCP tools to use ccscope pipeline + add cache churn tool"
```

---

### Task 8: Integration Test + Visualizer Smoke Test

**Files:** None new

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

- [ ] **Step 2: Build real session data**

```bash
ccscope build 81dc8a2f-2bc6-4241-81bb-9dea09f45a68
```

Verify `static/blocks.json` and `static/churn.json` exist and are valid.

- [ ] **Step 3: Verify block counts and churn totals**

```python
import json
blocks = json.load(open('static/blocks.json'))
churn = json.load(open('static/churn.json'))

# Blocks should be individual (hundreds, not 527 identical blobs)
print(f'Blocks: {len(blocks)}')
types = {}
for b in blocks:
    types[b['type']] = types.get(b['type'], 0) + 1
print(f'By type: {types}')

# Pinned blocks
pinned = [b for b in blocks if b.get('cached')]
print(f'Pinned: {len(pinned)} ({sum(b["tokens"] for b in pinned)} tokens)')

# Churn should match ground truth
total_cr = sum(c['cache_read'] for c in churn)
total_in = sum(c['input'] for c in churn)
print(f'Churn: {len(churn)} entries, cache_read={total_cr:,}, input={total_in:,}, ratio={total_cr//max(total_in,1):,}x')
# Expected: 314 entries, ~94M cache_read, ~452 input, ~208,000x
```

- [ ] **Step 4: Open in browser and verify visualization**

```bash
ccscope open 81dc8a2f-2bc6-4241-81bb-9dea09f45a68
```

Verify:
- Context growth chart shows monotonic rise to 528K (no false compactions)
- Churn panel shows 94M total cache reads with ~208,000x ratio
- Tape view shows individual blocks (tool_result, assistant, user, etc.)
- Clicking blocks shows content in the inspector
- Scrubbing timeline shows blocks entering turn by turn
- Subagents appear as collapsed blocks in the parent timeline
- Budget line at 1M (opus)

- [ ] **Step 5: Commit any fixes**

```bash
git add -u
git commit -m "fix: integration fixes from visualizer smoke test"
```

---

## Migration Notes

- **Old dashboard tests** (`test_dashboard_api.py`): update to test the new JSON endpoints instead of the old HTML/turns/blocks endpoints
- **Old reconstruction** (`analysis/reconstruction.py`): deprecated but not deleted — the ccscope parser replaces it
- **Staleness engine** (`analysis/staleness.py`): still used, now feeds into the `ref` field on Context Scope blocks
- **Health engine** (`analysis/health.py`): still used via MCP tools
- **The `ref` field**: blocks start with `ref:true`. After staleness analysis, blocks that haven't been referenced in N turns get `ref:false`. The visualizer already styles stale blocks differently (dashed red outline in swimlanes).
