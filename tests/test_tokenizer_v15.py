import pytest

from mistral_common.exceptions import InvalidRequestException, TokenizerException
from mistral_common.protocol.instruct.chunk import ThinkChunk
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    ChatMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.normalize import get_normalizer
from mistral_common.protocol.instruct.request import (
    ChatCompletionRequest,
    InstructRequest,
    ModelSettings,
    ReasoningEffort,
)
from mistral_common.protocol.instruct.tool_calls import Function, FunctionCall, Tool, ToolCall
from mistral_common.protocol.instruct.validator import ValidationMode, get_validator
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV15
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.model_settings_builder import EnumBuilder, ModelSettingsBuilder
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.test_tekken import get_special_tokens, quick_vocab

EXPECTED_TEXT_V15: str = (
    r"<s>[SYSTEM_PROMPT]S[/SYSTEM_PROMPT]"
    r'[AVAILABLE_TOOLS][{"type": "function", "function": {"name": "math_interpreter",'
    r' "description": "Get the value of an arithmetic expression.",'
    r' "parameters": {"type": "object", "properties": {"expression":'
    r' {"type": "string", "description": "Math expression."}}}}}]'
    r'[/AVAILABLE_TOOLS][MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS]'
    r"[INST]U1[/INST]A1"
    r"[TOOL_CALLS]F1[ARGS]{}[TOOL_CALLS]F2[ARGS]{}</s>"
    r"[TOOL_RESULTS]R1[/TOOL_RESULTS]"
    r"[TOOL_RESULTS]R2[/TOOL_RESULTS]A2</s>"
    r"[INST]U2[/INST]"
)

EXPECTED_TEXT_V15_NO_TOOLS: str = (
    r"<s>[SYSTEM_PROMPT]S[/SYSTEM_PROMPT]"
    r'[MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS]'
    r"[INST]U1[/INST]A1"
    r"[TOOL_CALLS]F1[ARGS]{}[TOOL_CALLS]F2[ARGS]{}</s>"
    r"[TOOL_RESULTS]R1[/TOOL_RESULTS]"
    r"[TOOL_RESULTS]R2[/TOOL_RESULTS]A2</s>"
    r"[INST]U2[/INST]"
)


def _build_v15_tekkenizer(model_settings_builder: ModelSettingsBuilder | None) -> Tekkenizer:
    r"""Build a v15 Tekkenizer with the given model settings builder.

    Args:
        model_settings_builder: The model settings builder, or None to create
            a tekkenizer without model settings support.
    """
    return Tekkenizer(
        quick_vocab([b"a", b"b", b"c", b"f", b"de"]),
        special_tokens=get_special_tokens(TokenizerVersion.v15, add_think=True),
        pattern=r".+",
        vocab_size=256 + 100,
        num_special_tokens=100,
        version=TokenizerVersion.v15,
        model_settings_builder=model_settings_builder,
    )


def get_v15_tekkenizer(
    model_settings_builder: ModelSettingsBuilder | None,
) -> InstructTokenizerV15:
    """Build an InstructTokenizerV15 with the given model settings builder."""
    return InstructTokenizerV15(_build_v15_tekkenizer(model_settings_builder))


def get_v15_mistral_tokenizer(
    model_settings_builder: ModelSettingsBuilder | None,
) -> MistralTokenizer:
    """Build a MistralTokenizer wrapping a v15 instruct tokenizer."""
    tekkenizer = _build_v15_tekkenizer(model_settings_builder)
    request_normalizer = get_normalizer(TokenizerVersion.v15, tekkenizer.model_settings_builder)
    validator = get_validator(TokenizerVersion.v15, mode=ValidationMode.test)
    return MistralTokenizer(
        InstructTokenizerV15(tekkenizer),
        validator=validator,
        request_normalizer=request_normalizer,
    )


@pytest.fixture(scope="session")
def v15_tekkenizer() -> InstructTokenizerV15:
    return get_v15_tekkenizer(_build_model_settings_builder(tuple(ReasoningEffort)))


@pytest.fixture(scope="session")
def v15_tekkenizer_no_reasoning() -> InstructTokenizerV15:
    return get_v15_tekkenizer(_build_model_settings_builder(None))


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


def test_tools_and_reasoning_effort(
    v15_tekkenizer: InstructTokenizerV15, available_tools: list[Tool], messages: list[ChatMessage]
) -> None:
    request = InstructRequest(
        messages=messages,
        available_tools=available_tools,
        settings=ModelSettings(reasoning_effort=ReasoningEffort.high),
    )
    tokenized = v15_tekkenizer.encode_instruct(request)
    assert tokenized.text == EXPECTED_TEXT_V15, tokenized.text


def test_no_tools_and_reasoning_effort(v15_tekkenizer: InstructTokenizerV15, messages: list[ChatMessage]) -> None:
    request: InstructRequest = InstructRequest(
        messages=messages, available_tools=None, settings=ModelSettings(reasoning_effort=ReasoningEffort.none)
    )
    tokenized = v15_tekkenizer.encode_instruct(request)
    expected_text_no_tools = EXPECTED_TEXT_V15_NO_TOOLS.replace("high", "none")
    assert tokenized.text == expected_text_no_tools, tokenized.text


def test_no_settings_does_not_encode_model_settings(
    v15_tekkenizer_no_reasoning: InstructTokenizerV15, messages: list[ChatMessage]
) -> None:
    request: InstructRequest = InstructRequest(messages=messages, available_tools=None, settings=ModelSettings.none())
    tokenized = v15_tekkenizer_no_reasoning.encode_instruct(request)
    assert "[MODEL_SETTINGS]" not in (tokenized.text or "")


def test_system_think_chunk_raises_v15(v15_tekkenizer: InstructTokenizerV15) -> None:
    messages = [SystemMessage(content=[ThinkChunk(thinking="Hi")])]
    request: InstructRequest = InstructRequest(
        messages=messages, settings=ModelSettings(reasoning_effort=ReasoningEffort.high)
    )
    with pytest.raises(TokenizerException, match="ThinkChunk in system message is not supported for this model"):
        v15_tekkenizer.encode_instruct(request)


def _build_model_settings_builder(
    allowed_reasoning_effort: tuple[str, ...] | None,
) -> ModelSettingsBuilder:
    """Build a ModelSettingsBuilder from allowed reasoning effort values.

    When ``allowed_reasoning_effort`` is ``None``, returns ``ModelSettingsBuilder.none()``
    (all fields ignored). This matches the behavior of ``Tekkenizer.from_file`` when no
    ``model_settings_builder`` key is present in the JSON.
    """
    if allowed_reasoning_effort is None:
        return ModelSettingsBuilder.none()
    if not allowed_reasoning_effort:
        return ModelSettingsBuilder(
            reasoning_effort=EnumBuilder[ReasoningEffort](values=[], accepts_none=True, default=None)
        )
    return ModelSettingsBuilder(
        reasoning_effort=EnumBuilder[ReasoningEffort](
            values=[ReasoningEffort(v) for v in allowed_reasoning_effort],
            accepts_none=True,
            default=ReasoningEffort(allowed_reasoning_effort[0]) if allowed_reasoning_effort else None,
        )
    )


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
    assert "[MODEL_SETTINGS]" not in (tokenized.text or "")


@pytest.mark.parametrize("reasoning_effort", [None, *list(ReasoningEffort)])
def test_end_to_end_with_default(messages: list[ChatMessage], reasoning_effort: ReasoningEffort | None) -> None:
    """When the builder has a default, None reasoning_effort resolves to the default."""
    builder = _build_model_settings_builder(("none", "high"))
    tokenizer_v15 = get_v15_mistral_tokenizer(builder)
    request = ChatCompletionRequest(messages=messages, reasoning_effort=reasoning_effort)
    tokenized = tokenizer_v15.encode_chat_completion(request)
    text = tokenized.text or ""
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
    text = tokenized.text or ""
    if reasoning_effort == ReasoningEffort.high:
        assert '[MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS]' in text
    elif reasoning_effort == ReasoningEffort.none:
        assert '[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]' in text
    else:
        # None with no default -> no model settings encoded
        assert "[MODEL_SETTINGS]" not in text
