from pathlib import Path

import pytest

from mistral_common.integrations.chat_templates.template_generator import build_chat_template
from mistral_common.protocol.instruct.chunk import TextChunk, ThinkChunk
from mistral_common.protocol.instruct.messages import AssistantMessage, UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.validator import ValidationMode
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.integrations.chat_templates.helpers import (
    TestConfig,
    build_render_args,
    build_template_config,
    encode_mistral_common,
    get_mistral_tokenizer,
    render_template,
)

_TEKKEN_VERSIONS = [
    TokenizerVersion.v1,
    TokenizerVersion.v2,
    TokenizerVersion.v3,
    TokenizerVersion.v7,
    TokenizerVersion.v11,
    TokenizerVersion.v13,
    TokenizerVersion.v15,
]
_SPM_VERSIONS = [TokenizerVersion.v1, TokenizerVersion.v2, TokenizerVersion.v3, TokenizerVersion.v7]
_CONFIGS = [TestConfig(version=version) for version in _TEKKEN_VERSIONS] + [
    TestConfig(version=version, spm=True) for version in _SPM_VERSIONS
]


def _format_config_id(config: TestConfig) -> str:
    r"""Build a test id such as `v7` or `v7_spm`."""
    return config.version.value + ("_spm" if config.spm else "")


def _assert_template_matches_tokenizer(config: TestConfig, request: ChatCompletionRequest, output_dir: Path) -> str:
    r"""Assert that the generated template renders a request like `encode_chat_completion`.

    Args:
        config: Test configuration selecting the template and the tokenizer.
        request: Finetuning request to render and encode.
        output_dir: Directory for the tokenizer files.

    Returns:
        The rendered template.
    """
    tokenizer = get_mistral_tokenizer(
        spm=config.spm,
        tokenizer_version=config.version,
        validation_mode=ValidationMode.finetuning,
        image=False,
        audio=False,
        think=config.think,
        output_dir=output_dir,
    )
    template = build_chat_template(config=build_template_config(config=config))
    rendered = render_template(template=template, **build_render_args(request=request))
    encoded = encode_mistral_common(mistral_tokenizer=tokenizer, chat_request=request, spm=config.spm)
    assert rendered == encoded
    return rendered


@pytest.mark.parametrize("config", _CONFIGS, ids=[_format_config_id(config) for config in _CONFIGS])
def test_assistant_string_trailing_spaces_match_tokenizer(config: TestConfig, tmp_path: Path) -> None:
    request = ChatCompletionRequest(  # type: ignore[type-var]
        messages=[
            UserMessage(content="Hi"),
            AssistantMessage(content="Hello there  "),
            UserMessage(content="Bye"),
            AssistantMessage(content="Bye\n "),
        ]
    )

    rendered = _assert_template_matches_tokenizer(config=config, request=request, output_dir=tmp_path)

    if config.version == TokenizerVersion.v1:
        assert "Hello there  </s>" in rendered
    else:
        assert "Hello there</s>" in rendered
        assert "Bye\n</s>" in rendered


@pytest.mark.parametrize(
    "config",
    [config for config in _CONFIGS if config.version >= TokenizerVersion.v3],
    ids=[_format_config_id(config) for config in _CONFIGS if config.version >= TokenizerVersion.v3],
)
def test_assistant_text_chunks_trailing_spaces_match_tokenizer(config: TestConfig, tmp_path: Path) -> None:
    request = ChatCompletionRequest(  # type: ignore[type-var]
        messages=[
            UserMessage(content="Hi"),
            AssistantMessage(content=[TextChunk(text="Hello "), TextChunk(text="there  ")]),
            UserMessage(content="Bye"),
            AssistantMessage(content=[TextChunk(text="Bye ")]),
        ]
    )

    rendered = _assert_template_matches_tokenizer(config=config, request=request, output_dir=tmp_path)

    assert "there</s>" in rendered
    assert "Bye</s>" in rendered


@pytest.mark.parametrize("version", [TokenizerVersion.v13, TokenizerVersion.v15])
def test_assistant_thinking_chunks_keep_trailing_spaces(version: TokenizerVersion, tmp_path: Path) -> None:
    # The tokenizer only strips text-only content, so a list that keeps a thinking chunk is rendered as is.
    request = ChatCompletionRequest(  # type: ignore[type-var]
        messages=[
            UserMessage(content="Hi"),
            AssistantMessage(content=[ThinkChunk(thinking="Hmm "), TextChunk(text="Hello there ")]),
            UserMessage(content="Bye"),
            AssistantMessage(content="Bye "),
        ]
    )

    rendered = _assert_template_matches_tokenizer(
        config=TestConfig(version=version, think=True), request=request, output_dir=tmp_path
    )

    assert "Hello there </s>" in rendered
    assert "Bye</s>" in rendered
