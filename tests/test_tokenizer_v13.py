import pytest

from mistral_common.exceptions import InvalidAssistantMessageException, TokenizerException
from mistral_common.protocol.instruct.chunk import (
    TextChunk,
    ThinkChunk,
)
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.normalize import InstructRequestNormalizerV13
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import Function, FunctionCall, Tool, ToolCall
from mistral_common.protocol.instruct.validator import MistralRequestValidatorV13
from mistral_common.tokens.tokenizers.base import InstructTokenizer, Tokenized, TokenizerVersion
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV13
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.tekken import SpecialTokenPolicy, Tekkenizer
from tests.test_tekken import _quick_vocab, get_special_tokens


@pytest.fixture(scope="session")
def v13_tekkenizer() -> InstructTokenizerV13:
    special_tokens = get_special_tokens(TokenizerVersion.v13, add_think=False)
    tokenizer = Tekkenizer(
        _quick_vocab([b"a", b"b", b"c", b"f", b"de"]),
        special_tokens=special_tokens,
        pattern=r".+",  # single token, whole string
        vocab_size=256 + 100,
        num_special_tokens=100,
        version=TokenizerVersion.v13,
    )
    return InstructTokenizerV13(tokenizer)


@pytest.fixture(scope="session")
def v13_tekkenizer_think() -> InstructTokenizerV13:
    special_tokens = get_special_tokens(TokenizerVersion.v13, add_think=True)
    tokenizer = Tekkenizer(
        _quick_vocab([b"a", b"b", b"c", b"f", b"de"]),
        special_tokens=special_tokens,
        pattern=r".+",  # single token, whole string
        vocab_size=256 + 100,
        num_special_tokens=100,
        version=TokenizerVersion.v13,
    )
    return InstructTokenizerV13(tokenizer)


EXPECTED_TEXT_V13: str = (
    r"<s>[SYSTEM_PROMPT]S1[THINK]TS[/THINK]S2[/SYSTEM_PROMPT][AVAILABLE_TOOLS][{"
    r'"type": "function", "function": {"name": "math_interpreter", '
    r'"description": "Get the value of an arithmetic expression.", '
    r'"parameters": {"type": "object", "properties": {'
    r'"expression": {"type": "string", "description": '
    r'"Math expression."}}}}}][/AVAILABLE_TOOLS][INST]U1[/INST]A1'
    r"[TOOL_CALLS]F1[ARGS]{}[TOOL_CALLS]F2[ARGS]{}</s>"
    r"[TOOL_RESULTS]R1[/TOOL_RESULTS][TOOL_RESULTS]R2"
    r"[/TOOL_RESULTS]A2[THINK]T1[/THINK]</s>[INST]U2[/INST]"
)


EXPECTED_TEXT_V13_FROM_WRONG_ORDER: str = (
    r"<s>[SYSTEM_PROMPT]S[/SYSTEM_PROMPT][AVAILABLE_TOOLS][{"
    r'"type": "function", "function": {"name": "math_interpreter", '
    r'"description": "Get the value of an arithmetic expression.", '
    r'"parameters": {"type": "object", "properties": {'
    r'"expression": {"type": "string", "description": '
    r'"Math expression."}}}}}][/AVAILABLE_TOOLS][INST]U1[/INST]A1'
    r"[TOOL_CALLS]F1[ARGS]{}[TOOL_CALLS]F2[ARGS]{}</s>"
    r"[TOOL_RESULTS]R1[/TOOL_RESULTS][TOOL_RESULTS]R2"
    r"[/TOOL_RESULTS]A2</s>[INST]U2[/INST]"
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
def messages() -> list[BaseMessage]:
    return [
        SystemMessage(content=[TextChunk(text="S1"), ThinkChunk(thinking="TS"), TextChunk(text="S2")]),
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
        AssistantMessage(content=[TextChunk(text="A2"), ThinkChunk(thinking="T1")]),
        UserMessage(content="U2"),
    ]


@pytest.fixture
def messages_wrong_order_results() -> list[BaseMessage]:
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
        ToolMessage(
            content="R2", tool_call_id="999999999"
        ),  # wrong order of results but passes validation and is later normalized
        ToolMessage(content="R1", tool_call_id="123456789"),
        AssistantMessage(content="A2"),
        UserMessage(content="U2"),
    ]


def test_end_to_end_v13(
    v13_tekkenizer_think: InstructTokenizer,
    available_tools: list[Tool],
    messages: list[BaseMessage],
) -> None:
    """
    Tests normalization (including reordering) and validation
    """
    request_normalizer = InstructRequestNormalizerV13.normalizer()
    validator = MistralRequestValidatorV13()
    mistral_tokenizer_v13 = MistralTokenizer(
        instruct_tokenizer=v13_tekkenizer_think, validator=validator, request_normalizer=request_normalizer
    )
    chat_completion_request: ChatCompletionRequest = ChatCompletionRequest(
        messages=messages,
        tools=available_tools,
    )

    assert isinstance(mistral_tokenizer_v13, MistralTokenizer), type(mistral_tokenizer_v13)
    # This does validation, normalization and encoding
    tokenized_v13 = mistral_tokenizer_v13.encode_chat_completion(chat_completion_request)
    assert isinstance(tokenized_v13, Tokenized)
    assert tokenized_v13.text == EXPECTED_TEXT_V13, tokenized_v13.text


def test_end_to_end_v13_wrong_order(
    v13_tekkenizer: InstructTokenizer,
    available_tools: list[Tool],
    messages_wrong_order_results: list[BaseMessage],
) -> None:
    """
    Tests normalization (including reordering) and validation
    """
    request_normalizer = InstructRequestNormalizerV13.normalizer()
    validator = MistralRequestValidatorV13()
    mistral_tokenizer_v13 = MistralTokenizer(
        instruct_tokenizer=v13_tekkenizer, validator=validator, request_normalizer=request_normalizer
    )
    chat_completion_request: ChatCompletionRequest = ChatCompletionRequest(
        messages=messages_wrong_order_results,
        tools=available_tools,
    )

    assert isinstance(mistral_tokenizer_v13, MistralTokenizer), type(mistral_tokenizer_v13)
    # This does validation, normalization and encoding
    tokenized_v13 = mistral_tokenizer_v13.encode_chat_completion(chat_completion_request)
    assert isinstance(tokenized_v13, Tokenized)
    assert tokenized_v13.text == EXPECTED_TEXT_V13_FROM_WRONG_ORDER, tokenized_v13.text


def test_encode_tool_message(v13_tekkenizer: InstructTokenizerV13) -> None:
    tool_message = ToolMessage(content="R1", tool_call_id="123456789")
    assert isinstance(v13_tekkenizer, InstructTokenizerV13)
    encoded = v13_tekkenizer.encode_tool_message(tool_message, is_before_last_user_message=False)
    assert encoded == [7, 182, 149, 8]


def test_encode_think_chunk(v13_tekkenizer_think: InstructTokenizerV13) -> None:
    assert isinstance(v13_tekkenizer_think, InstructTokenizerV13)
    think_chunk = ThinkChunk(
        thinking="T1",
    )
    encoded = v13_tekkenizer_think.encode_think(think_chunk)
    assert v13_tekkenizer_think.decode(encoded, special_token_policy=SpecialTokenPolicy.KEEP) == "[THINK]T1[/THINK]"

    think_chunk = ThinkChunk(
        thinking="T1",
        closed=False,
    )
    encoded = v13_tekkenizer_think.encode_think(think_chunk)
    assert v13_tekkenizer_think.decode(encoded, special_token_policy=SpecialTokenPolicy.KEEP) == "[THINK]T1"


@pytest.mark.parametrize(
    "message, expected",
    [
        (
            AssistantMessage(content="A1"),
            "A1",
        ),
        (
            AssistantMessage(content="A1", prefix=True),
            "A1",
        ),
        (
            AssistantMessage(content=[TextChunk(text="A1")]),
            "A1",
        ),
        (
            AssistantMessage(content=[TextChunk(text="A1"), ThinkChunk(thinking="T1")]),
            "A1[THINK]T1[/THINK]",
        ),
        (
            AssistantMessage(
                content=[TextChunk(text="A1"), ThinkChunk(thinking="R1", closed=False)],
                tool_calls=[ToolCall(id="123456789", function=FunctionCall(name="F1", arguments="{'a': 1}"))],
            ),
            "A1[THINK]R1[TOOL_CALLS]F1[ARGS]\"{'a': 1}\"",
        ),
    ],
)
@pytest.mark.parametrize("continue_final_message", [True, False])
def test_tokenize_assistant_message(
    v13_tekkenizer_think: InstructTokenizerV13, message: AssistantMessage, expected: str, continue_final_message: bool
) -> None:
    if not continue_final_message:
        tokens = v13_tekkenizer_think.encode_assistant_message(
            message, is_before_last_user_message=False, continue_message=continue_final_message
        )
        if not message.prefix:
            expected += "</s>"
    else:
        if message.prefix:
            with pytest.raises(
                InvalidAssistantMessageException,
                match="`continue_message` is only supported for assistant messages that have `prefix=False`.",
            ):
                v13_tekkenizer_think.encode_assistant_message(
                    message, is_before_last_user_message=False, continue_message=continue_final_message
                )
            return
        tokens = v13_tekkenizer_think.encode_assistant_message(
            message, is_before_last_user_message=False, continue_message=continue_final_message
        )
    assert v13_tekkenizer_think.decode(tokens, special_token_policy=SpecialTokenPolicy.KEEP) == expected


def test_tokenize_assistant_message_error(v13_tekkenizer: InstructTokenizerV13) -> None:
    with pytest.raises(TokenizerException, match=r"Invalid assistant message"):
        v13_tekkenizer.encode_assistant_message(
            AssistantMessage(content="", tool_calls=[]), is_before_last_user_message=False, continue_message=False
        )

    with pytest.raises(
        InvalidAssistantMessageException,
        match="`continue_message` is only supported for assistant messages that have `prefix=False`.",
    ):
        v13_tekkenizer.encode_assistant_message(
            AssistantMessage(content="z", tool_calls=[], prefix=True),
            is_before_last_user_message=False,
            continue_message=True,
        )


@pytest.mark.parametrize(
    "message, expected",
    [
        (
            SystemMessage(content="S1"),
            "[SYSTEM_PROMPT]S1[/SYSTEM_PROMPT]",
        ),
        (
            SystemMessage(content=[TextChunk(text="S1"), ThinkChunk(thinking="TS"), TextChunk(text="S2")]),
            "[SYSTEM_PROMPT]S1[THINK]TS[/THINK]S2[/SYSTEM_PROMPT]",
        ),
        (
            SystemMessage(
                content=[
                    TextChunk(text="S1"),
                    TextChunk(text="S3"),
                    ThinkChunk(thinking="TS", closed=True),
                    ThinkChunk(thinking="TS", closed=True),
                    TextChunk(text="S2"),
                ]
            ),
            "[SYSTEM_PROMPT]S1S3[THINK]TS[/THINK][THINK]TS[/THINK]S2[/SYSTEM_PROMPT]",
        ),
        (
            SystemMessage(
                content=[
                    TextChunk(text="S1"),
                    TextChunk(text="S3"),
                    ThinkChunk(thinking="TS", closed=False),
                ]
            ),
            "[SYSTEM_PROMPT]S1S3[THINK]TS[/SYSTEM_PROMPT]",
        ),
    ],
)
def test_encode_system_message(
    v13_tekkenizer_think: InstructTokenizerV13, message: SystemMessage, expected: str
) -> None:
    encoded = v13_tekkenizer_think.encode_system_message(message)
    assert v13_tekkenizer_think.decode(encoded, special_token_policy=SpecialTokenPolicy.KEEP) == expected
