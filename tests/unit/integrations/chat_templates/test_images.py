from mistral_common.protocol.instruct.chunk import ImageURLChunk, TextChunk
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.integration.chat_recipes import build_user_assistant_request
from tests.utils.chat_templates.configs import TestConfig
from tests.utils.chat_templates.helpers import build_render_args, build_test_template, render_template

_IMAGE_URL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="


class TestImageBlockOrdering:
    def test_image_two_blocks_sorted_by_type(self) -> None:
        template = build_test_template(TestConfig(version=TokenizerVersion.v7, image=True))

        # Text first, then image — should be sorted so image appears before text
        request = build_user_assistant_request(
            user_content=[TextChunk(text="What is this?"), ImageURLChunk(image_url=_IMAGE_URL)],
            assistant_content="An image.",
        )

        output = render_template(template, **build_render_args(request))
        # [IMG] should appear before "What is this?"
        img_pos = output.find("[IMG]")
        text_pos = output.find("What is this?")
        assert img_pos != -1
        assert text_pos != -1
        assert img_pos < text_pos, "Image token should appear before text in 2-block content"

    def test_v13_image_more_than_two_blocks_preserves_order(self) -> None:
        template = build_test_template(TestConfig(version=TokenizerVersion.v13, image=True))

        # Three blocks: text, image, text — NOT sorted (only 2-block content is sorted)
        request = build_user_assistant_request(
            user_content=[
                TextChunk(text="First"),
                ImageURLChunk(image_url=_IMAGE_URL),
                TextChunk(text="Second"),
            ],
            assistant_content="Described.",
        )

        output = render_template(template, **build_render_args(request))
        first_pos = output.find("First")
        img_pos = output.find("[IMG]")
        second_pos = output.find("Second")
        assert first_pos != -1
        assert img_pos != -1
        assert second_pos != -1
        assert first_pos < img_pos < second_pos, "With 3+ blocks and image_support, original order is preserved"
