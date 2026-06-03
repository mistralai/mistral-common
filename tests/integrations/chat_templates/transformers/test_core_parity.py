from pathlib import Path
from typing import Any

import pytest
from jinja2.exceptions import TemplateError

from mistral_common.integrations.chat_templates.chat_templates import generate_chat_template
from mistral_common.protocol.instruct.chunk import TextChunk
from mistral_common.protocol.instruct.messages import AssistantMessage, UserMessage
from mistral_common.protocol.instruct.validator import ValidationMode
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.integrations.chat_templates.conftest import (
    ALL_TRANSFORMERS_CONFIGS,
    _config_id,
)
from tests.integrations.chat_templates.fixtures_data import _get_conversations
from tests.integrations.chat_templates.helpers import (
    TestConfig,
    _build_spm_path,
    _build_tekken_json,
    encode_mistral_common,
)
from tests.integrations.chat_templates.hf_utils import (
    _build_hf_tokenizer,
    encode_hf_tokens,
    encode_transformers,
    encode_transformers_from_openai,
)


class TestTransformersMistralCommonParity:
    @pytest.mark.parametrize(
        "config",
        ALL_TRANSFORMERS_CONFIGS,
        ids=[_config_id(c) for c in ALL_TRANSFORMERS_CONFIGS],
    )
    @pytest.mark.parametrize("mode", [ValidationMode.test, ValidationMode.finetuning])
    def test_chat_template(self, config: TestConfig, mode: ValidationMode, tmp_path: Path) -> None:
        conversations = _get_conversations(config.version, mode, config.image, config.audio, config.think)

        if config.spm:
            tokenizer_path = _build_spm_path(config, tmp_path)
        else:
            tokenizer_path = _build_tekken_json(config, tmp_path)

        mistral_tokenizer = MistralTokenizer.from_file(str(tokenizer_path), mode=mode)

        chat_template = generate_chat_template(
            spm=config.spm,
            tokenizer_version=config.version,
            image_support=config.image,
            audio_support=config.audio,
            thinking_support=config.think,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

        # Build HF tokenizer for Tekken-based token ID comparison
        hf_tokenizer = None
        if not config.spm:
            hf_tokenizer = _build_hf_tokenizer(tokenizer_path, chat_template)

        if config.version <= TokenizerVersion.v2:
            for conv in conversations:
                for message in conv.messages:
                    # v1/v2 expect string content, not structured chunks.
                    # Mutate here rather than duplicating fixtures per version.
                    if isinstance(message, (UserMessage, AssistantMessage)) and isinstance(message.content, list):
                        assert len(message.content) == 1 and isinstance(message.content[0], TextChunk), (
                            "Only text content is supported for v1 and v2"
                        )
                        message.content = str(message.content[0].text)
        for conversation in conversations:
            for message in conversation.messages:
                if message.role == "tool" and message.name is None:
                    message.name = "tool"

            # Run transformers first since encode_mistral_common may mutate the conversation in-place
            transformers_encoded = encode_transformers(chat_template, conversation, keep_name_for_tools=True)

            # Token ID comparison: apply_chat_template(tokenize=True) vs mistral-common
            # Skipped for image/audio configs because multimodal tokens are generated
            # by encoders (not from text), so token IDs won't match.
            # Skipped for SPM because building an HF tokenizer from SPM is a separate concern.
            # Skipped for V1 because V1 emits control markers (e.g. [INST]) as literal
            # text, but the HF tokenizer treats them as special tokens and encodes
            # them as single IDs, making token-level parity impossible.
            if not config.image and not config.audio and not config.spm and config.version > TokenizerVersion.v1:
                assert hf_tokenizer is not None
                hf_tokens = encode_hf_tokens(hf_tokenizer, conversation.model_copy(deep=True), keep_name_for_tools=True)
                mc_tokens = mistral_tokenizer.encode_chat_completion(conversation.model_copy(deep=True)).tokens
                assert mc_tokens == hf_tokens

            mistral_common_encoded = encode_mistral_common(mistral_tokenizer, conversation, config.spm)
            assert mistral_common_encoded == transformers_encoded

    @pytest.mark.parametrize(
        "config",
        ALL_TRANSFORMERS_CONFIGS,
        ids=[_config_id(c) for c in ALL_TRANSFORMERS_CONFIGS],
    )
    def test_role_error(self, config: TestConfig) -> None:
        chat_template = generate_chat_template(
            spm=config.spm,
            tokenizer_version=config.version,
            image_support=config.image,
            audio_support=config.audio,
            thinking_support=config.think,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

        # Starting with assistant is rejected by the first-message constraint
        invalid_starts_with_assistant = {
            "messages": [
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "Hello"},
            ]
        }

        if config.version >= TokenizerVersion.v7:
            first_msg_match = r"Conversation must start with a user or system message, got assistant\."
        else:
            first_msg_match = r"Conversation must start with a user message, got assistant\."

        with pytest.raises(TemplateError, match=first_msg_match):
            encode_transformers_from_openai(chat_template, invalid_starts_with_assistant)

        # Invalid role after user is caught by the transition table
        invalid_role = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "invalid", "content": "Hello"},
            ]
        }

        with pytest.raises(TemplateError, match=r"Unexpected role 'invalid' after role 'user'"):
            encode_transformers_from_openai(chat_template, invalid_role)

        # Tool after user is rejected by the transition table (tool can only follow assistant or tool)
        if config.version >= TokenizerVersion.v2:
            invalid_tool_after_user = {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "tool", "content": "result", "tool_call_id": "123456789"},
                ]
            }
            with pytest.raises(TemplateError, match=r"Unexpected role 'tool' after role 'user'"):
                encode_transformers_from_openai(chat_template, invalid_tool_after_user)

        # System after assistant is rejected for v7+ (system stays in loop_messages)
        if config.version >= TokenizerVersion.v7:
            invalid_system_after_assistant = {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                    {"role": "system", "content": "New system prompt"},
                    {"role": "user", "content": "World"},
                ]
            }
            with pytest.raises(TemplateError, match=r"Unexpected role 'system' after role 'assistant'"):
                encode_transformers_from_openai(chat_template, invalid_system_after_assistant)

    @pytest.mark.parametrize(
        "config",
        ALL_TRANSFORMERS_CONFIGS,
        ids=[_config_id(c) for c in ALL_TRANSFORMERS_CONFIGS],
    )
    def test_invalid_chunks(
        self,
        config: TestConfig,
        invalid_sp_think: dict[str, Any],
        invalid_sp_random: dict[str, Any],
        invalid_assistant_think: dict[str, Any],
        invalid_assistant_random: dict[str, Any],
        invalid_user_image: dict[str, Any],
        invalid_user_audio: dict[str, Any],
        invalid_user_random: dict[str, Any],
    ) -> None:
        sp_invalids = [invalid_sp_random, invalid_sp_think]
        assistant_invalids = [invalid_assistant_random, invalid_assistant_think]
        user_invalids = [invalid_user_image, invalid_user_audio, invalid_user_random]

        invalid_convs = [invalid_sp_random, invalid_user_random, invalid_assistant_random]
        if not config.think:
            invalid_convs += [invalid_sp_think, invalid_assistant_think]
        if not config.image:
            invalid_convs += [invalid_user_image]
        if not config.audio:
            invalid_convs += [invalid_user_audio]

        chat_template = generate_chat_template(
            spm=config.spm,
            tokenizer_version=config.version,
            image_support=config.image,
            audio_support=config.audio,
            thinking_support=config.think,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )
        # Not using parametrize here because invalid_convs depends on the version/image/audio/think
        # parameters from the outer parametrize. Each sub-case is identifiable via the TemplateError
        # match string which includes the role and allowed chunks.
        for conv in invalid_convs:
            msg_template = "Only {chunks} chunks are supported in {role} message content."
            if conv in sp_invalids:
                chunks = "text and thinking" if config.think and config.version < TokenizerVersion.v15 else "text"
                role = "system"
            elif conv in user_invalids:
                chunks = "text"
                if config.image:
                    chunks += ", image and image_url"
                if config.audio:
                    chunks += ", input_audio and audio_url"
                role = "user"
            elif conv in assistant_invalids:
                chunks = "text and thinking" if config.think else "text"
                role = "assistant"

            err_msg = msg_template.format(chunks=chunks, role=role)
            with pytest.raises(TemplateError, match=err_msg):
                encode_transformers_from_openai(chat_template, conv)
