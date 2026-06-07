# Image Support: Surface Images from Context History in Messages Pane

**Issue:** #40
**Date:** 2026-06-07
**Status:** Approved

## Problem

When Claude reads screenshots or images appear in tool results, the transcript contains base64 image blocks that are silently dropped at every layer:

1. `transcript_parser.py` `_parse_content_blocks()` only extracts `text` sub-blocks from `tool_result.content[]`
2. `dashboard.py` `get_call_content()` and `get_conv_turn_content()` duplicate this text-only logic
3. Frontend has zero image rendering capability

Images appear in two locations in transcripts:
- **Tool result content**: `tool_result.content[]` array containing `{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}`
- **User messages**: Top-level `{"type": "image", ...}` blocks when users paste screenshots

## Design Decisions

- **Storage**: No caching or disk extraction. Re-read transcript on demand for each image request. Simple, stateless, no storage footprint.
- **Lazy loading**: Main API returns image metadata only (dimensions, media type, token cost). Full base64 data fetched via separate endpoint when thumbnail scrolls into viewport.
- **Token sizing**: Parse actual image dimensions from PNG/JPEG headers in base64 data. Use Anthropic's formula `(width * height) / 750` for exact token cost.
- **No conflict with #43**: Image badges (`N img`) and tool-type badges (`MCP`/`SKILL`/`AGENT`/`TOOL`) coexist on the same message row. Badge order: ERR -> tool type -> preview -> image badge -> size.

## Architecture

### Data Flow

```
Transcript JSONL (contains base64 image blocks)
  |
  v
GET /conv_turn/{n}/content
  -> returns images[] metadata per message: {index, media_type, width, height, tokens}
  -> NO base64 data in this response
  |
  v
Frontend renders placeholder boxes with IntersectionObserver
  |
  v (placeholder scrolls into viewport)
GET /conv_turn/{n}/image/{msg_index}/{img_index}
  -> re-reads transcript, extracts specific image
  -> returns {"data_uri": "data:image/png;base64,..."}
  |
  v
Thumbnail replaces placeholder -> click opens lightbox
```

### Image Source Types

Image sources in transcripts can be either base64 or URL:
- `{"type": "base64", "media_type": "image/png", "data": "iVBOR..."}` -- inline base64 data
- `{"type": "url", "url": "https://..."}` or `{"type": "url", "url": "data:image/png;base64,..."}` -- external or data URL

The implementation must branch on `source.type`:
- `base64`: extract dimensions from `source.data`, serve as `data:{media_type};base64,{data}`
- `url` with `data:` prefix: parse as data URI, extract base64 portion for dimension parsing
- `url` with `http(s):`: return the URL directly as `{"url": "https://..."}` (no proxying). Dimension extraction not possible without fetching. Use fallback dimensions (1024x1024).

### Image Dimension Extraction

Parse actual dimensions from base64 data headers (only first 256 bytes needed):

```python
import base64, struct

def image_dimensions(b64_data: str, media_type: str) -> tuple[int, int]:
    raw = base64.b64decode(b64_data[:256])
    if media_type == "image/png":
        # PNG: bytes 16-23 = width (4B) + height (4B) big-endian
        w, h = struct.unpack(">II", raw[16:24])
        return w, h
    elif media_type in ("image/jpeg", "image/jpg"):
        # JPEG: scan for SOF0 marker (0xFF 0xC0)
        i = 0
        while i < len(raw) - 8:
            if raw[i] == 0xFF and raw[i+1] == 0xC0:
                h = struct.unpack(">H", raw[i+5:i+7])[0]
                w = struct.unpack(">H", raw[i+7:i+9])[0]
                return w, h
            i += 1
    return 1024, 1024  # fallback for unknown formats

def image_tokens(w: int, h: int) -> int:
    return (w * h) // 750
```

## Components

### 1. Model Change (`analysis/models.py`)

Add `image_count: int = 0` to `ContentBlock`. Since `ContentBlock` is a frozen dataclass, `image_count` must be passed at construction time -- it cannot be set after creation. All existing call sites use kwargs, so the default of 0 applies without changes. New construction sites in `_parse_content_blocks()` must pass `image_count=N` explicitly.

### 2. Parser Fix (`transcript_parser.py`)

In `_parse_content_blocks()`, the `tool_result` branch (lines 86-101) currently iterates content list and only extracts `b.get("text", "")`. Change to:

- Iterate content sub-blocks
- For `type: "text"`: collect as before
- For `type: "image"`: increment `image_count`, skip embedding data
- Set `ContentBlock.image_count` on the resulting block

Also add detection for top-level `image` blocks in user messages (separate elif for `block_type == "image"`).

### 3. Token Sizing Fix (`ccscope/tokens.py`)

**Note:** `char_count_of_block()` works on raw transcript block dicts (not `ContentBlock`). It already iterates `tool_result.content[]` sub-blocks but only sums `sub["text"]`. Guard with `isinstance(sub, dict)` when adding the image branch. Change to detect `type: "image"` sub-blocks and compute chars from actual dimensions:

```python
if sub.get("type") == "image":
    source = sub.get("source", {})
    w, h = image_dimensions(source.get("data", ""), source.get("media_type", "image/png"))
    total += image_tokens(w, h) * 4  # tokens -> chars (x4 ratio)
```

### 4. API: Extend conv_turn Content Response (`dashboard.py`)

**Critical:** `get_conv_turn_content()` does NOT use parsed `ContentBlock` data. It re-reads raw JSONL and flattens transcript entries into message objects. This means parser/model changes alone will not affect the messages pane. The dashboard endpoints must be updated separately with the same image detection logic.

**Shared extraction helper:** Extract a `_extract_content_blocks_with_images(entry)` helper function that both `get_call_content()` and `get_conv_turn_content()` call. This helper must:

- Iterate `tool_result.content[]` sub-blocks
- For `type: "text"`: collect as text content (existing behavior)
- For `type: "image"`: extract metadata (index, media_type, width, height, tokens) from `source`, branch on `source.type` (base64 vs url)
- For top-level `image` blocks in user messages: same extraction
- Return both the text content string AND the `images[]` metadata array

The image endpoint (`/conv_turn/{n}/image/{msg_index}/{img_index}`) MUST reuse the exact same flattening order as `get_conv_turn_content()`. Otherwise `msg_index` will address the wrong message. Both endpoints should share the JSONL-walking code path, not duplicate it.

Response shape (additive -- existing fields unchanged):

```json
{
  "type": "tool_result",
  "tool_use_id": "toolu_01XYZ",
  "content": "Screenshot taken of dashboard...",
  "is_error": false,
  "images": [
    {"index": 0, "media_type": "image/png", "width": 1920, "height": 1080, "tokens": 2765},
    {"index": 1, "media_type": "image/jpeg", "width": 800, "height": 600, "tokens": 640}
  ]
}
```

### 5. API: New Image Endpoint (`dashboard.py`)

`GET /api/session/{session_id}/conv_turn/{conv_turn}/image/{msg_index}/{img_index}`

**`msg_index` definition:** This is the 0-based index into the `data.messages` array returned by `get_conv_turn_content()`. It is NOT the raw transcript entry index. `get_conv_turn_content()` flattens one transcript entry into multiple message rows (e.g., one assistant entry with text + tool_use + tool_use becomes 3 messages). The image endpoint must replicate this exact flattening to address the correct message.

Implementation:
1. Validate session ID via existing `_validate_session_id()`
2. Find transcript via `_find_transcript()`
3. Build turn map via `build_turn_map(transcript_path)` to get `first_call`/`last_call` range
4. Re-read transcript JSONL using the **same walking logic** as `get_conv_turn_content()` -- ideally call a shared helper
5. Walk the flattened messages array to position `msg_index`
6. Within that message, find the `img_index`-th image sub-block
7. Branch on `source.type`:
   - `base64`: extract `source.data` and `source.media_type`, return `{"data_uri": "data:{media_type};base64,{data}"}`
   - `url` with `data:` prefix: return `{"data_uri": source.url}`
   - `url` with `http(s):`: return `{"url": source.url}`

Error cases: 404 for missing session, missing turn, missing message, missing image, or non-image message.

### 6. Frontend: CSS (`dashboard-v3.html`)

Add styles for:
- `.msg-img-badge` -- compact "N img" badge on message rows (right-aligned, indigo)
- `.msg-img-strip` -- flex container for thumbnails in modal
- `.msg-img-thumb` -- 80x60px thumbnail with hover scale effect
- `.msg-img-placeholder` -- dashed border placeholder shown before lazy load
- `.lightbox-overlay` -- fixed fullscreen overlay with dark background
- `.lightbox-img` -- centered image, max 90vw/85vh, object-fit contain
- `.lightbox-close` -- circular close button, top-right
- `.lightbox-meta` -- metadata line at bottom (media type, dimensions, tokens, turn)

### 7. Frontend: JavaScript (`dashboard-v3.html`)

**State:**
- `_imageCache = {}` -- keyed by `"sessionId/convTurn/msgIdx/imgIdx"`, values are data URIs or `"loading"` or `"error"`. Cleared in `switchSession()`.

**Functions:**

`loadConvTurnImage(convTurn, msgIndex, imgIndex)` -- async, fetches from image endpoint, caches result, returns data URI or null.

`openLightbox(src, metaText)` -- creates `.lightbox-overlay` div, appends to body, wires Escape key and click-outside-to-close.

`renderImageStrip(images, convTurn, msgIndex)` -- returns HTML string of `.msg-img-placeholder` divs with data attributes for lazy loading.

`wireImagePlaceholders(containerEl, convTurn)` -- creates `IntersectionObserver` on all `.msg-img-placeholder` elements in the container. When a placeholder enters the viewport (100px margin), fetches the image and replaces the placeholder with an `<img>` thumbnail. Clicking the thumbnail calls `openLightbox()`.

### 8. Frontend: Messages Pane Integration

**List view** (`renderMessagesFromAPI`):
- Check `msg.images` array on each message
- If present and non-empty, append `<span class="msg-img-badge">N img</span>` after the content preview
- Badge position: after preview text, before size bar (right-aligned via `margin-left: auto`)

**Modal view** (`renderMessageBlock`):
- Current `renderMessageBlock(msg)` does not accept a `msgIndex` parameter. Update signature to `renderMessageBlock(msg, convTurn, msgIndex)` to enable image strip rendering.
- After the content `<pre>` block, call `renderImageStrip(msg.images, convTurn, msgIndex)` to append placeholder thumbnails
- In `openConvTurnModal()`, pass `(msg, convTurn, i)` where `i` is the loop index over `data.messages`
- After setting `body.innerHTML`, call `wireImagePlaceholders(body, convTurn)` to activate lazy loading
- Show dimension/token metadata below the strip: "N images -- WxH TYPE (T tok) + ..."

**Session switching**: Clear `_imageCache` in `switchSession()` alongside existing cache clears. Additionally, track a `_imageSessionId` variable set to the current session ID when images are rendered. Before replacing a placeholder with a loaded thumbnail, verify `_imageSessionId` still matches the current session -- otherwise discard the result. This prevents stale async image fetches from a previous session writing into the current session's DOM.

## Compatibility with Other Issues

| Issue | Touch point | Interaction | Conflict? |
|-------|------------|-------------|-----------|
| #41 Error highlighting | Messages pane badges | ERR badge appears before tool badge, image badge after preview | No |
| #42 Regression detection | No shared code | Independent | No |
| #43 Tool intelligence | Messages pane badges, conv_turn response | tool_category + images[] are independent additive fields | No |

Badge order on message rows: `[ROLE] [ERR] [TOOL_TYPE] [preview] [IMG_BADGE] [SIZE]`

## Files to Modify

| File | Change |
|------|--------|
| `src/context_tracker/analysis/models.py` | Add `image_count: int = 0` to `ContentBlock` |
| `src/context_tracker/transcript_parser.py` | Detect image blocks in `_parse_content_blocks()`, set `image_count` |
| `src/context_tracker/ccscope/tokens.py` | Compute token cost from actual image dimensions |
| `src/context_tracker/dashboard.py` | Extend conv_turn endpoints with `images[]` metadata; add image serving endpoint |
| `static/dashboard-v3.html` | CSS for thumbnails/lightbox; JS for lazy loading, caching, rendering |

## Files to Create

None.

## Build Sequence

1. `models.py` -- add `image_count` field
2. `transcript_parser.py` -- detect image blocks, set `image_count`
3. `tokens.py` -- image dimension parsing + token calculation
4. `dashboard.py` -- extract shared `_extract_content_blocks_with_images()` helper
5. `dashboard.py` -- extend conv_turn endpoints using shared helper, return `images[]` metadata
6. `dashboard.py` -- add image serving endpoint using same shared helper for msg_index consistency
6. `dashboard-v3.html` -- CSS for thumbnails, placeholders, lightbox
7. `dashboard-v3.html` -- JS: `loadConvTurnImage`, `openLightbox`, `renderImageStrip`, `wireImagePlaceholders`
8. `dashboard-v3.html` -- integrate into `renderMessagesFromAPI` (badge) and `renderMessageBlock` (strip + lazy load)
9. `dashboard-v3.html` -- clear `_imageCache` in `switchSession()`
10. Manual test with a real session containing screenshots

## Out of Scope

- Disk caching of extracted images (chose re-read-on-demand for simplicity)
- Image search or filtering (future)
- Image diff visualization (e.g., before/after screenshots)
- Video or animated GIF special handling (treated as static images)

## Codex Review Findings (2026-06-07)

Independent review via `/codex` identified 4 P1 and 3 P2 findings. All addressed in this revision:

| # | Severity | Finding | Resolution |
|---|----------|---------|------------|
| 1 | P1 | ContentBlock is frozen -- image_count must be passed at construction | Clarified in spec: pass `image_count=` at construction time |
| 2 | P1 | get_conv_turn_content re-reads raw JSONL, not parsed ContentBlock | Added shared `_extract_content_blocks_with_images()` helper requirement |
| 3 | P1 | msg_index under-specified, flattening order mismatch risk | Defined msg_index as 0-based `data.messages` index, image endpoint must share walking logic |
| 4 | P1 | Only handles base64 sources, not URL-type images | Added source.type branching (base64 / data URL / http URL) |
| 5 | P2 | No top-level image branch in parser | Confirmed, spec already covers this |
| 6 | P2 | char_count_of_block works on raw dicts, needs isinstance guard | Added guard note |
| 7 | P2 | Stale async image fetches after session switch | Added _imageSessionId verification before DOM write |
