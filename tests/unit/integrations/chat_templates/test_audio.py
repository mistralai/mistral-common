import pytest

from mistral_common.protocol.instruct.chunk import TextChunk
from mistral_common.protocol.instruct.messages import AssistantMessage, SystemMessage, UserMessage
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.fixtures.audio import get_dummy_audio_url_chunk
from tests.integration.chat_recipes import build_conversation_request
from tests.utils.chat_templates.configs import TestConfig
from tests.utils.chat_templates.helpers import build_render_args, build_test_template, render_template


class TestAudioSystemPrompt:
    @pytest.mark.parametrize(
        "version",
        [TokenizerVersion.v7, TokenizerVersion.v11, TokenizerVersion.v13],
    )
    def test_audio_with_system_prompt_raises(self, version: TokenizerVersion) -> None:
        template = build_test_template(TestConfig(version=version, audio=True))
        request = build_conversation_request(
            messages=[
                SystemMessage(content="You are helpful."),
                UserMessage(content=[TextChunk(text="What is this audio?"), get_dummy_audio_url_chunk()]),
                AssistantMessage(content="An audio clip."),
            ]
        )

        with pytest.raises(
            ValueError, match="Audio chunks are not supported in user message content when system prompt is provided"
        ):
            render_template(template, **build_render_args(request))

    def test_v15_audio_with_system_prompt_succeeds(self) -> None:
        template = build_test_template(TestConfig(version=TokenizerVersion.v15, audio=True))
        request = build_conversation_request(
            messages=[
                SystemMessage(content="You are helpful."),
                UserMessage(content=[TextChunk(text="What is this audio?"), get_dummy_audio_url_chunk()]),
                AssistantMessage(content="An audio clip."),
            ]
        )

        output = render_template(template, **build_render_args(request))
        assert "[AUDIO]" in output
        assert "[SYSTEM_PROMPT]" in output
