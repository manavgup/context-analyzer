"""Extended tests for ccscope/tokens.py — image_dimensions, char_count_of_block."""

import base64
import struct

from context_tracker.ccscope.tokens import (
    char_count_of_block,
    estimate_tokens,
    image_dimensions,
    image_tokens,
)


class TestImageDimensions:
    def test_empty_data(self):
        w, h = image_dimensions("", "image/png")
        assert w == 1024 and h == 1024

    def test_invalid_base64(self):
        w, h = image_dimensions("not-valid-base64!!!", "image/png")
        assert w == 1024 and h == 1024

    def test_valid_png(self):
        """Create a minimal PNG header and verify dimensions are parsed."""
        # PNG signature + IHDR chunk
        png_sig = b"\x89PNG\r\n\x1a\n"
        # IHDR: length=13, type=IHDR, width=800, height=600, ...
        ihdr_data = struct.pack(">II", 800, 600) + b"\x08\x02\x00\x00\x00"
        ihdr_crc = b"\x00\x00\x00\x00"  # fake CRC
        ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + ihdr_crc
        raw = png_sig + ihdr
        b64 = base64.b64encode(raw).decode()

        w, h = image_dimensions(b64, "image/png")
        assert w == 800
        assert h == 600

    def test_png_too_short(self):
        """PNG data too short to contain dimensions."""
        raw = b"\x89PNG\r\n\x1a\n\x00\x00"
        b64 = base64.b64encode(raw).decode()
        w, h = image_dimensions(b64, "image/png")
        assert w == 1024 and h == 1024

    def test_valid_jpeg(self):
        """Create a minimal JPEG with SOF0 marker."""
        # JPEG: SOI marker + some data + SOF0 marker
        soi = b"\xff\xd8"
        # APP0 marker (JFIF)
        app0 = b"\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        # SOF0 marker: length=17, precision=8, height=480, width=640
        sof0 = b"\xff\xc0\x00\x11\x08"
        sof0 += struct.pack(">H", 480)  # height
        sof0 += struct.pack(">H", 640)  # width
        sof0 += b"\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01"
        raw = soi + app0 + sof0
        b64 = base64.b64encode(raw).decode()

        w, h = image_dimensions(b64, "image/jpeg")
        assert w == 640
        assert h == 480

    def test_jpeg_no_sof(self):
        """JPEG without SOF marker falls back."""
        raw = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00"
        b64 = base64.b64encode(raw).decode()
        w, h = image_dimensions(b64, "image/jpeg")
        assert w == 1024 and h == 1024

    def test_unknown_media_type(self):
        """Unknown media type falls back to defaults."""
        w, h = image_dimensions("abc", "image/webp")
        assert w == 1024 and h == 1024


class TestImageTokens:
    def test_standard(self):
        tokens = image_tokens(1024, 1024)
        assert tokens == (1024 * 1024) // 750

    def test_small(self):
        tokens = image_tokens(100, 100)
        assert tokens == 10000 // 750


class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens("") == 0

    def test_short(self):
        assert estimate_tokens("hi") == 1  # min 1

    def test_normal(self):
        text = "a" * 400
        assert estimate_tokens(text) == 100


class TestCharCountOfBlock:
    def test_text_block(self):
        block = {"type": "text", "text": "Hello world"}
        assert char_count_of_block(block) == 11

    def test_thinking_block(self):
        block = {"type": "thinking", "thinking": "Let me think..."}
        assert char_count_of_block(block) == 15

    def test_tool_use_block(self):
        block = {"type": "tool_use", "input": {"file_path": "/src/a.py"}}
        count = char_count_of_block(block)
        assert count > 0

    def test_image_block(self):
        block = {"type": "image", "source": {"data": "", "media_type": "image/png"}}
        count = char_count_of_block(block)
        assert count > 0

    def test_tool_result_string(self):
        block = {"type": "tool_result", "content": "file contents here"}
        assert char_count_of_block(block) == 18

    def test_tool_result_list(self):
        block = {
            "type": "tool_result",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "image", "source": {"data": "", "media_type": "image/png"}},
            ],
        }
        count = char_count_of_block(block)
        assert count > 5  # At least the text part

    def test_tool_result_list_non_dict(self):
        """Non-dict items in tool_result list should be skipped."""
        block = {
            "type": "tool_result",
            "content": ["just a string", {"type": "text", "text": "ok"}],
        }
        count = char_count_of_block(block)
        assert count == 2  # "ok"

    def test_tool_result_empty_list(self):
        block = {"type": "tool_result", "content": []}
        assert char_count_of_block(block) == 0

    def test_tool_result_non_string_non_list(self):
        block = {"type": "tool_result", "content": 12345}
        assert char_count_of_block(block) == 0

    def test_unknown_type(self):
        block = {"type": "unknown_thing", "data": "abc"}
        count = char_count_of_block(block)
        assert count > 0  # Fallback to str(block)
