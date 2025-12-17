"""Tests for ToolMessage OpenAI format support with chunked content.

Issue #166: ToolMessage should accept OpenAI chunked content format:
{"role": "tool", "content": [{"type": "text", "text": "..."}], "tool_call_id": "..."}
"""

import pytest

from mistral_common.protocol.instruct.messages import ToolMessage


class TestToolMessageFromOpenAI:
    """Test ToolMessage.from_openai() with various content formats."""

    def test_accepts_string_content(self):
        """Regression test: ToolMessage should accept string content (existing behavior)."""
        openai_msg = {"role": "tool", "content": "tool response text", "tool_call_id": "call_123"}
        msg = ToolMessage.from_openai(openai_msg)
        assert msg.content == "tool response text"
        assert msg.tool_call_id == "call_123"

    def test_accepts_single_text_chunk(self):
        """Fix for issue #166: ToolMessage should accept OpenAI chunked content format.

        Previously this would fail with:
        ValueError: 1 validation error for ToolMessage
        Input should be a valid string [type=string_type, ...]
        """
        openai_msg = {
            "role": "tool",
            "content": [{"type": "text", "text": "tool response"}],
            "tool_call_id": "call_08fd7550e300441397420db9",
        }
        msg = ToolMessage.from_openai(openai_msg)
        # Content should be normalized to a single string
        assert msg.content == "tool response"
        assert msg.tool_call_id == "call_08fd7550e300441397420db9"

    def test_concatenates_multiple_text_chunks(self):
        """ToolMessage should concatenate multiple text chunks into a single string."""
        openai_msg = {
            "role": "tool",
            "content": [
                {"type": "text", "text": "Part 1"},
                {"type": "text", "text": " Part 2"},
                {"type": "text", "text": " Part 3"},
            ],
            "tool_call_id": "call_123",
        }
        msg = ToolMessage.from_openai(openai_msg)
        assert msg.content == "Part 1 Part 2 Part 3"

    def test_rejects_non_text_chunks(self):
        """ToolMessage should reject non-text chunks in tool content."""
        openai_msg = {
            "role": "tool",
            "content": [
                {"type": "text", "text": "ok"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
            ],
            "tool_call_id": "call_123",
        }
        with pytest.raises(ValueError, match="Unsupported tool content chunk type"):
            ToolMessage.from_openai(openai_msg)


class TestToolMessageRoundTrip:
    """Test ToolMessage round-trip conversion (from_openai -> to_openai)."""

    def test_string_content_roundtrip(self):
        """String content should remain unchanged in round-trip."""
        openai_msg = {"role": "tool", "content": "tool response", "tool_call_id": "call_123"}
        msg = ToolMessage.from_openai(openai_msg)
        out = msg.to_openai()

        assert out["content"] == "tool response"
        assert out["tool_call_id"] == "call_123"
        assert out["role"] == "tool"

    def test_chunked_content_normalizes_to_string_in_roundtrip(self):
        """Chunked content should be normalized to string in output."""
        openai_msg = {"role": "tool", "content": [{"type": "text", "text": "response"}], "tool_call_id": "call_123"}
        msg = ToolMessage.from_openai(openai_msg)
        out = msg.to_openai()

        # Output should be string, not list (normalized during parsing)
        assert isinstance(out["content"], str)
        assert out["content"] == "response"
        assert out["tool_call_id"] == "call_123"
