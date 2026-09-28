import pytest

from mistral_common.exceptions import TokenizerException
from mistral_common.protocol.instruct.chunk import ThinkChunk
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    ChatMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.request import (
    InstructRequest,
    ModelSettings,
    ReasoningEffort,
)
from mistral_common.protocol.instruct.tool_calls import Function, FunctionCall, Tool, ToolCall
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV15
from mistral_common.tokens.tokenizers.model_settings_builder import EnumBuilder, ModelSettingsBuilder
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.test_tekken import get_special_tokens, quick_vocab
from tests.utils import decode_keep

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


def _build_model_settings_builder(
    allowed_reasoning_effort: tuple[str, ...] | None,
) -> ModelSettingsBuilder:
    """Build a ModelSettingsBuilder from allowed reasoning effort values.

    When `allowed_reasoning_effort` is `None`, returns `ModelSettingsBuilder.none()`
    (all fields ignored). This matches the behavior of `Tekkenizer.from_file` when no
    `model_settings_builder` key is present in the JSON.
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
    text = decode_keep(v15_tekkenizer, tokenized)
    assert text == EXPECTED_TEXT_V15, text


def test_no_tools_and_reasoning_effort(v15_tekkenizer: InstructTokenizerV15, messages: list[ChatMessage]) -> None:
    request: InstructRequest = InstructRequest(
        messages=messages, available_tools=None, settings=ModelSettings(reasoning_effort=ReasoningEffort.none)
    )
    tokenized = v15_tekkenizer.encode_instruct(request)
    expected_text_no_tools = EXPECTED_TEXT_V15_NO_TOOLS.replace("high", "none")
    text = decode_keep(v15_tekkenizer, tokenized)
    assert text == expected_text_no_tools, text


def test_no_settings_does_not_encode_model_settings(
    v15_tekkenizer_no_reasoning: InstructTokenizerV15, messages: list[ChatMessage]
) -> None:
    request: InstructRequest = InstructRequest(messages=messages, available_tools=None, settings=ModelSettings.none())
    tokenized = v15_tekkenizer_no_reasoning.encode_instruct(request)
    text = decode_keep(v15_tekkenizer_no_reasoning, tokenized)
    assert "[MODEL_SETTINGS]" not in text


def test_system_think_chunk_raises_v15(v15_tekkenizer: InstructTokenizerV15) -> None:
    messages = [SystemMessage(content=[ThinkChunk(thinking="Hi")])]
    request: InstructRequest = InstructRequest(
        messages=messages, settings=ModelSettings(reasoning_effort=ReasoningEffort.high)
    )
    with pytest.raises(TokenizerException, match="ThinkChunk in system message is not supported for this model"):
        v15_tekkenizer.encode_instruct(request)
