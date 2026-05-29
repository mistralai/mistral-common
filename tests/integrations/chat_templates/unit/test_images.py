from typing import Any

from mistral_common.integrations.chat_templates.chat_templates import generate_chat_template
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.integrations.chat_templates.helpers import render_template


class TestImageBlockOrdering:
    def test_image_two_blocks_sorted_by_type(self) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v7,
            image_support=True,
            audio_support=False,
            thinking_support=False,
        )

        # Text first, then image — should be sorted so image appears before text
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="}},
                ],
            },
            {"role": "assistant", "content": "An image."},
        ]

        output = render_template(template, messages)
        # [IMG] should appear before "What is this?"
        img_pos = output.find("[IMG]")
        text_pos = output.find("What is this?")
        assert img_pos != -1
        assert text_pos != -1
        assert img_pos < text_pos, "Image token should appear before text in 2-block content"

    def test_v13_image_more_than_two_blocks_preserves_order(self) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v13,
            image_support=True,
            audio_support=False,
            thinking_support=False,
        )

        # Three blocks: text, image, text — NOT sorted (only 2-block content is sorted)
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="}},
                    {"type": "text", "text": "Second"},
                ],
            },
            {"role": "assistant", "content": "Described."},
        ]

        output = render_template(template, messages)
        first_pos = output.find("First")
        img_pos = output.find("[IMG]")
        second_pos = output.find("Second")
        assert first_pos != -1
        assert img_pos != -1
        assert second_pos != -1
        assert first_pos < img_pos < second_pos, "With 3+ blocks and image_support, original order is preserved"
