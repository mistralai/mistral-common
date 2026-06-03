from pathlib import Path

import pytest

from mistral_common.integrations.chat_templates.chat_templates import generate_chat_template
from mistral_common.protocol.instruct.chunk import TextChunk, ThinkChunk
from mistral_common.protocol.instruct.messages import AssistantMessage, SystemMessage, UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.validator import ValidationMode
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.integrations.chat_templates.fixtures_data import _get_conversations
from tests.integrations.chat_templates.helpers import _get_mistral_tokenizer, encode_mistral_common
from tests.integrations.chat_templates.hf_utils import encode_transformers


class TestPlainThinkingValidation:
    def test_plain_thinking_v11_works(self) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v11,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=True,
            use_special_token_variables=True,
        )
        assert "<think>" in template
        assert "</think>" in template
        assert "[THINK]" not in template


class TestPlainThinkParity:
    @pytest.mark.parametrize(
        ("image", "mode"),
        [
            (False, ValidationMode.test),
            (False, ValidationMode.finetuning),
            (True, ValidationMode.test),
            (True, ValidationMode.finetuning),
        ],
    )
    def test_plain_think_vs_mistral_common_baseline(self, image: bool, mode: ValidationMode, tmp_path: Path) -> None:
        version = TokenizerVersion.v11
        conversations = _get_conversations(version, mode, image, audio=False, think=False)

        mistral_tok = _get_mistral_tokenizer(
            spm=False,
            tokenizer_version=version,
            validation_mode=mode,
            image=image,
            audio=False,
            think=False,
            output_dir=tmp_path,
        )
        plain_think_template = generate_chat_template(
            spm=False,
            tokenizer_version=version,
            image_support=image,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=True,
            use_special_token_variables=True,
        )

        for conversation in conversations:
            plain_think_encoded = encode_transformers(plain_think_template, conversation)
            mistral_encoded = encode_mistral_common(mistral_tok, conversation, spm=False)

            assert plain_think_encoded == mistral_encoded

    @pytest.mark.parametrize("image", [False, True])
    def test_plain_think_vs_mistral_common_with_thinking(self, image: bool, tmp_path: Path) -> None:
        version = TokenizerVersion.v11

        mistral_tok = _get_mistral_tokenizer(
            spm=False,
            tokenizer_version=version,
            validation_mode=ValidationMode.finetuning,
            image=image,
            audio=False,
            think=False,
            output_dir=tmp_path,
        )
        plain_think_template = generate_chat_template(
            spm=False,
            tokenizer_version=version,
            image_support=image,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=True,
            use_special_token_variables=True,
        )

        # Build a conversation with thinking chunks
        think_conversation = ChatCompletionRequest(  # type: ignore[type-var]
            messages=[
                SystemMessage(
                    content=[
                        TextChunk(text="You are a helpful assistant."),
                        ThinkChunk(thinking="System reasoning."),
                        TextChunk(text="Be concise."),
                    ],
                ),
                UserMessage(content=[TextChunk(text="What is 2+2?")]),
                AssistantMessage(
                    content=[
                        ThinkChunk(thinking="Simple arithmetic."),
                        TextChunk(text="4."),
                    ],
                    tool_calls=[],
                ),
                UserMessage(content=[TextChunk(text="Thanks.")]),
                AssistantMessage(content=[TextChunk(text="You're welcome.")]),
            ],
        )

        # Same conversation without thinking chunks
        no_think_conversation = ChatCompletionRequest(  # type: ignore[type-var]
            messages=[
                SystemMessage(content="You are a helpful assistant.Be concise."),
                UserMessage(content=[TextChunk(text="What is 2+2?")]),
                AssistantMessage(content="4.", tool_calls=[]),
                UserMessage(content=[TextChunk(text="Thanks.")]),
                AssistantMessage(content=[TextChunk(text="You're welcome.")]),
            ],
        )

        # Render with plain think template (has <think> tags)
        plain_think_encoded = encode_transformers(plain_think_template, think_conversation)

        # Render baseline without think chunks via mistral-common
        baseline_encoded = encode_mistral_common(mistral_tok, no_think_conversation, spm=False)

        # Verify think tags are present
        assert "<think>System reasoning.</think>" in plain_think_encoded
        assert "<think>Simple arithmetic.</think>" in plain_think_encoded
        assert "[THINK]" not in plain_think_encoded

        # Verify stripping think tags recovers the baseline
        stripped = plain_think_encoded.replace("<think>System reasoning.</think>", "").replace(
            "<think>Simple arithmetic.</think>", ""
        )
        assert stripped == baseline_encoded

    @pytest.mark.parametrize("image", [False, True])
    @pytest.mark.parametrize("mode", [ValidationMode.test, ValidationMode.finetuning])
    def test_plain_think_comprehensive(self, image: bool, mode: ValidationMode, tmp_path: Path) -> None:
        version = TokenizerVersion.v11
        # Use non-thinking conversations (plain think uses same base conversations)
        conversations = _get_conversations(version, mode, image, audio=False, think=False)

        mistral_tok = _get_mistral_tokenizer(
            spm=False,
            tokenizer_version=version,
            validation_mode=mode,
            image=image,
            audio=False,
            think=False,
            output_dir=tmp_path,
        )
        plain_think_template = generate_chat_template(
            spm=False,
            tokenizer_version=version,
            image_support=image,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=True,
            use_special_token_variables=True,
        )

        for conversation in conversations:
            # Plain think template with non-think conversations should match v11 base
            plain_encoded = encode_transformers(plain_think_template, conversation)
            mistral_encoded = encode_mistral_common(mistral_tok, conversation, spm=False)
            assert plain_encoded == mistral_encoded
