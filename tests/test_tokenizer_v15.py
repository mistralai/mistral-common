r"""Pipeline tests for the v15 instruct tokenizer (encode_chat_completion).

These tests exercise the full normalizer + validator + tokenizer pipeline
and complement the direct encode_instruct unit tests in
tests/unit/tokens/tokenizers/instruct/test_v15.py.
"""

import pytest

from mistral_common.exceptions import InvalidRequestException
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    ChatMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.normalize import get_normalizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest, ReasoningEffort
from mistral_common.protocol.instruct.tool_calls import Function, FunctionCall, Tool, ToolCall
from mistral_common.protocol.instruct.validator import ValidationMode, get_validator
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV15
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.model_settings_builder import EnumBuilder, ModelSettingsBuilder
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.test_tekken import get_special_tokens, quick_vocab
from tests.utils import decode_keep


def _build_v15_tekkenizer(model_settings_builder: ModelSettingsBuilder | None) -> Tekkenizer:
    r"""Build a v15 Tekkenizer with the given model settings builder."""
    return Tekkenizer(
        quick_vocab([b"a", b"b", b"c", b"f", b"de"]),
        special_tokens=get_special_tokens(TokenizerVersion.v15, add_think=True),
        pattern=r".+",
        vocab_size=256 + 100,
        num_special_tokens=100,
        version=TokenizerVersion.v15,
        model_settings_builder=model_settings_builder,
    )


def get_v15_mistral_tokenizer(
    model_settings_builder: ModelSettingsBuilder | None,
) -> MistralTokenizer:
    r"""Build a MistralTokenizer wrapping a v15 instruct tokenizer."""
    tekkenizer = _build_v15_tekkenizer(model_settings_builder)
    request_normalizer = get_normalizer(TokenizerVersion.v15, tekkenizer.model_settings_builder)
    validator = get_validator(TokenizerVersion.v15, mode=ValidationMode.test)
    return MistralTokenizer(
        InstructTokenizerV15(tekkenizer),
        validator=validator,
        request_normalizer=request_normalizer,
    )


def _build_model_settings_builder(
    allowed_reasoning_effort: tuple[str, ...] | None,
) -> ModelSettingsBuilder:
    r"""Build a ModelSettingsBuilder from allowed reasoning effort values.

    When `allowed_reasoning_effort` is `None`, returns `ModelSettingsBuilder.none()`
    (all fields ignored).
    """
    if allowed_reasoning_effort is None:
        return ModelSettingsBuilder.none()
    return ModelSettingsBuilder(
        reasoning_effort=EnumBuilder[ReasoningEffort](
            values=[ReasoningEffort(v) for v in allowed_reasoning_effort],
            accepts_none=True,
            default=ReasoningEffort(allowed_reasoning_effort[0]) if allowed_reasoning_effort else None,
        )
    )


@pytest.fixture
def available_tools() -> list[Tool]:
    return [
        Tool(
            function=Function(
                name="math_interpreter",
                description="Get the value of an arithmetic expression.",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression.",
                        }
                    },
                },
            )
        )
    ]


@pytest.fixture
def messages() -> list[ChatMessage]:
    return [
        SystemMessage(content="S"),
        UserMessage(content="U1"),
        AssistantMessage(
            content="A1",
            tool_calls=[
                ToolCall(id="123456789", function=FunctionCall(name="F1", arguments="{}")),
                ToolCall(id="999999999", function=FunctionCall(name="F2", arguments="{}")),
            ],
        ),
        ToolMessage(content="R1", tool_call_id="123456789"),
        ToolMessage(content="R2", tool_call_id="999999999"),
        AssistantMessage(content="A2"),
        UserMessage(content="U2"),
    ]


@pytest.mark.parametrize(
    ("reasoning_effort", "allowed_reasoning_effort", "raises", "match"),
    [
        (None, ("none", "high"), None, None),
        ("none", ("none", "high"), None, None),
        ("high", ("none", "high"), None, None),
        ("high", ("none",), InvalidRequestException, "should be one of"),
        ("none", ("none",), None, None),
        ("none", (), InvalidRequestException, "not supported for this model"),
    ],
)
def test_forbidden_reasoning_effort_raises(
    available_tools: list[Tool],
    messages: list[ChatMessage],
    reasoning_effort: str | None,
    allowed_reasoning_effort: tuple[str, ...] | None,
    raises: type[Exception] | None,
    match: str | None,
) -> None:
    builder = _build_model_settings_builder(allowed_reasoning_effort)
    tokenizer_v15 = get_v15_mistral_tokenizer(builder)
    request = ChatCompletionRequest(
        messages=messages,
        tools=available_tools,
        reasoning_effort=reasoning_effort,  # type: ignore[arg-type]
    )
    if raises is not None:
        assert match is not None
        with pytest.raises(raises, match=match):
            tokenizer_v15.encode_chat_completion(request)
    else:
        assert match is None
        tokenizer_v15.encode_chat_completion(request)


@pytest.mark.parametrize(("reasoning_effort", "allowed_reasoning_effort"), [(None, ()), ("none", None)])
def test_encode_ignore_one_model_settings(
    messages: list[ChatMessage],
    reasoning_effort: str | None,
    allowed_reasoning_effort: tuple[str, ...] | None,
) -> None:
    builder = _build_model_settings_builder(allowed_reasoning_effort)
    tokenizer_v15 = get_v15_mistral_tokenizer(builder)
    request = ChatCompletionRequest(messages=messages, reasoning_effort=reasoning_effort)  # type: ignore[arg-type]
    tokenized = tokenizer_v15.encode_chat_completion(request)
    text = decode_keep(tokenizer_v15, tokenized)
    assert "[MODEL_SETTINGS]" not in text


@pytest.mark.parametrize("reasoning_effort", [None, *list(ReasoningEffort)])
def test_end_to_end_with_default(messages: list[ChatMessage], reasoning_effort: ReasoningEffort | None) -> None:
    """When the builder has a default, None reasoning_effort resolves to the default."""
    builder = _build_model_settings_builder(("none", "high"))
    tokenizer_v15 = get_v15_mistral_tokenizer(builder)
    request = ChatCompletionRequest(messages=messages, reasoning_effort=reasoning_effort)
    tokenized = tokenizer_v15.encode_chat_completion(request)
    text = decode_keep(tokenizer_v15, tokenized)
    if reasoning_effort == ReasoningEffort.high:
        assert '[MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS]' in text
    else:
        # Both None (default applied) and explicit "none" resolve to the default "none"
        assert '[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]' in text


@pytest.mark.parametrize("reasoning_effort", [None, *list(ReasoningEffort)])
def test_end_to_end_no_default(messages: list[ChatMessage], reasoning_effort: ReasoningEffort | None) -> None:
    """When the builder has no default, None reasoning_effort means no model settings."""
    builder = ModelSettingsBuilder(
        reasoning_effort=EnumBuilder[ReasoningEffort](values=list(ReasoningEffort), accepts_none=True, default=None)
    )
    tokenizer_v15 = get_v15_mistral_tokenizer(builder)
    request = ChatCompletionRequest(messages=messages, reasoning_effort=reasoning_effort)
    tokenized = tokenizer_v15.encode_chat_completion(request)
    text = decode_keep(tokenizer_v15, tokenized)
    if reasoning_effort == ReasoningEffort.high:
        assert '[MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS]' in text
    elif reasoning_effort == ReasoningEffort.none:
        assert '[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]' in text
    else:
        # None with no default -> no model settings encoded
        assert "[MODEL_SETTINGS]" not in text
