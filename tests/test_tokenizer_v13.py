import pytest
from PIL import Image

from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    BaseMessage,
    ImageChunk,
    SystemMessage,
    TextChunk,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.normalize import InstructRequestNormalizerV13
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import Function, FunctionCall, Tool, ToolCall
from mistral_common.protocol.instruct.validator import MistralRequestValidatorV13
from mistral_common.tokens.tokenizers.base import InstructTokenizer, Tokenized, TokenizerVersion
from mistral_common.tokens.tokenizers.image import ImageConfig, ImageEncoder, SpecialImageIDs
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV13
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.test_tekken import _quick_vocab, get_special_tokens


@pytest.fixture(scope="session")
def v13_tekkenizer() -> InstructTokenizerV13:
    special_tokens = get_special_tokens(TokenizerVersion.v13)
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
def v13_tekkenizer_with_image_encoder() -> InstructTokenizerV13:
    special_tokens = get_special_tokens(TokenizerVersion.v13)
    tokenizer = Tekkenizer(
        _quick_vocab([b"a", b"b", b"c", b"f", b"de"]),
        special_tokens=special_tokens,
        pattern=r".+",  # single token, whole string
        vocab_size=256 + 100,
        num_special_tokens=100,
        version=TokenizerVersion.v13,
        image_config=ImageConfig(
            image_patch_size=2,
            max_image_size=4,
        ),
    )
    encoder = ImageEncoder(
        image_config=ImageConfig(
            image_patch_size=2,
            max_image_size=4,
        ),
        special_ids=SpecialImageIDs(
            img=tokenizer.get_control_token("[IMG]"),
            img_end=tokenizer.get_control_token("[IMG_END]"),
            img_break=tokenizer.get_control_token("[IMG_BREAK]"),
        ),
    )
    return InstructTokenizerV13(tokenizer, encoder)


EXPECTED_TEXT_V13: str = (
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

EXPECTED_TEXT_V13_ASSISTANT_AND_TOOLS_WITH_IMAGES: str = (
    r'<s>[SYSTEM_PROMPT]S[/SYSTEM_PROMPT][AVAILABLE_TOOLS][{"type": "function", "function": {"name": "math_interpreter"'
    r', "description": "Get the value of an arithmetic expression.", "parameters": {"type": "object", "properties": '
    r'{"expression": {"type": "string", "description": "Math expression."}}}}}][/AVAILABLE_TOOLS][INST]U1[/INST][IMG]'
    r"[IMG][IMG_BREAK][IMG][IMG][IMG_END]A1[TOOL_CALLS]F1[ARGS]{}[TOOL_CALLS]F2[ARGS]{}</s>[TOOL_RESULTS]R1"
    r"[/TOOL_RESULTS][TOOL_RESULTS][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]R2[/TOOL_RESULTS]A2</s>[INST]U2[/INST]"
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


@pytest.fixture
def messages_with_images() -> list[BaseMessage]:
    return [
        SystemMessage(content="S"),
        UserMessage(content="U1"),
        AssistantMessage(
            content=[
                TextChunk(text="A1"),
                ImageChunk(image=Image.new("RGB", (4, 4), "red")),
            ],
            tool_calls=[
                ToolCall(id="123456789", function=FunctionCall(name="F1", arguments="{}")),
                ToolCall(id="999999999", function=FunctionCall(name="F2", arguments="{}")),
            ],
        ),
        ToolMessage(content="R1", tool_call_id="123456789"),
        ToolMessage(
            content=[ImageChunk(image=Image.new("RGB", (4, 4), "red")), TextChunk(text="R2")], tool_call_id="999999999"
        ),
        AssistantMessage(content="A2"),
        UserMessage(content="U2"),
    ]


def test_end_to_end_v13(
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
    assert tokenized_v13.text == EXPECTED_TEXT_V13, tokenized_v13.text


def test_end_to_end_v13_assistant_and_tools_with_images(
    v13_tekkenizer_with_image_encoder: InstructTokenizer,
    available_tools: list[Tool],
    messages_with_images: list[BaseMessage],
) -> None:
    """
    Tests normalization (including reordering) and validation
    """
    request_normalizer = InstructRequestNormalizerV13.normalizer()
    validator = MistralRequestValidatorV13()
    mistral_tokenizer_v13 = MistralTokenizer(
        instruct_tokenizer=v13_tekkenizer_with_image_encoder, validator=validator, request_normalizer=request_normalizer
    )
    chat_completion_request: ChatCompletionRequest = ChatCompletionRequest(
        messages=messages_with_images,
        tools=available_tools,
    )

    assert isinstance(mistral_tokenizer_v13, MistralTokenizer), type(mistral_tokenizer_v13)
    # This does validation, normalization and encoding
    tokenized_v13 = mistral_tokenizer_v13.encode_chat_completion(chat_completion_request)
    assert isinstance(tokenized_v13, Tokenized)
    assert tokenized_v13.text == EXPECTED_TEXT_V13_ASSISTANT_AND_TOOLS_WITH_IMAGES, tokenized_v13.text


def test_encode_tool_message(v13_tekkenizer: InstructTokenizer) -> None:
    tool_message = ToolMessage(content=[TextChunk(text="R1")], tool_call_id="123456789")
    assert isinstance(v13_tekkenizer, InstructTokenizerV13)
    encoded = v13_tekkenizer.encode_tool_message(tool_message, is_before_last_user_message=False)
    assert encoded[0] == [7, 182, 149, 8]
