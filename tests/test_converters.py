import copy
import io
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from openai.types.audio.transcription_create_params import TranscriptionCreateParamsBase as OpenAITranscriptionRequest
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam as OpenAIAssistantMessage,
)
from openai.types.chat.chat_completion_content_part_image_param import (
    ChatCompletionContentPartImageParam as OpenAIImageChunk,
)
from openai.types.chat.chat_completion_content_part_input_audio_param import (
    ChatCompletionContentPartInputAudioParam as OpenAIInputAudioChunk,
)
from openai.types.chat.chat_completion_content_part_text_param import (
    ChatCompletionContentPartTextParam as OpenAITextChunk,
)
from openai.types.chat.chat_completion_message_tool_call_param import (
    ChatCompletionMessageToolCallParam as OpenAIToolCall,
)
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam as OpenAISystemMessage,
)
from openai.types.chat.chat_completion_tool_message_param import ChatCompletionToolMessageParam as OpenAIToolMessage
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam as OpenAITool
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam as OpenAIUserMessage
from PIL import Image
from pydantic_extra_types.language_code import LanguageAlpha2

from mistral_common.exceptions import InvalidAssistantMessageException
from mistral_common.protocol.instruct.chunk import (
    AudioChunk,
    AudioURL,
    AudioURLChunk,
    ImageChunk,
    ImageURL,
    ImageURLChunk,
    TextChunk,
    ThinkChunk,
)
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    ChatMessage,
    ReasoningFieldFormat,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.request import (
    ChatCompletionRequest,
    InstructRequest,
    ModelSettings,
    ReasoningEffort,
)
from mistral_common.protocol.instruct.tool_calls import (
    Function,
    FunctionCall,
    FunctionName,
    NamedToolChoice,
    Tool,
    ToolCall,
    ToolChoiceEnum,
)
from mistral_common.protocol.speech.request import SpeechRequest
from mistral_common.protocol.transcription.request import TranscriptionRequest
from mistral_common.tokens.tokenizers.audio import Audio

from .test_tokenizer_v7_audio_tts import _make_fake_audio

CURRENT_FILE_PATH = Path(__file__).resolve()
ROOT_PATH = CURRENT_FILE_PATH.parents[1]
LOGO_PATH = ROOT_PATH / "docs" / "assets" / "logo_favicon.png"
AUDIO_SAMPLE_URL = "https://freetestdata.com/wp-content/uploads/2021/09/Free_Test_Data_100KB_MP3.mp3"


def _get_audio_chunk() -> AudioChunk:
    import soundfile as sf

    sample_rate = 44100  # Sample rate in Hz
    duration = 3  # Duration in seconds
    frequency = 440  # Frequency of the sine wave in Hz

    # Time array
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)

    # Write to in-memory buffer
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format="WAV")

    buffer.seek(0)
    data, sr = sf.read(buffer)

    audio = Audio(audio_array=data, sampling_rate=sr, format="wav")

    return AudioChunk.from_audio(audio)


DUMMY_AUDIO_CHUNK = _get_audio_chunk()
assert isinstance(DUMMY_AUDIO_CHUNK.input_audio, str)
DUMMY_AUDIO_URL_CHUNK_BASE64 = AudioURLChunk(audio_url=AudioURL(url=DUMMY_AUDIO_CHUNK.input_audio))
DUMMY_AUDIO_URL_CHUNK_BASE64_STR = AudioURLChunk(audio_url=DUMMY_AUDIO_CHUNK.input_audio)
DUMMY_AUDIO_URL_CHUNK_BASE64_PREFIX = AudioURLChunk(
    audio_url=AudioURL(url=f"data:audio/wav;base64,{DUMMY_AUDIO_CHUNK.input_audio}")
)
DUMMY_AUDIO_URL_CHUNK_URL = AudioURLChunk(audio_url=AudioURL(url=AUDIO_SAMPLE_URL))


def test_convert_image_chunk() -> None:
    image = Image.open(LOGO_PATH.as_posix())
    chunk = ImageChunk(image=image)

    openai_image = chunk.to_openai()
    assert openai_image["type"] == "image_url"
    assert isinstance(openai_image["image_url"], dict)
    assert openai_image["image_url"]["url"].startswith("data:image/png;base64,")

    assert isinstance(ImageChunk.from_openai(openai_image), ImageChunk)

    typeddict_openai = OpenAIImageChunk(**openai_image)  # type: ignore[typeddict-item]

    assert isinstance(ImageChunk.from_openai(typeddict_openai), ImageChunk)


def test_convert_image_chunk_from_openai_does_not_mutate_input() -> None:
    image = Image.open(LOGO_PATH.as_posix())
    original_chunk = ImageChunk(image=image)
    openai_chunk = original_chunk.to_openai()
    original_url = openai_chunk["image_url"]["url"]

    ImageChunk.from_openai(openai_chunk)

    assert openai_chunk["image_url"]["url"] == original_url


def test_convert_text_chunk() -> None:
    chunk = TextChunk(text="Hello")
    text_openai = chunk.to_openai()

    assert text_openai == {"type": "text", "text": "Hello"}

    assert TextChunk.from_openai(text_openai) == chunk

    typeddict_openai = OpenAITextChunk(**chunk.to_openai())  # type: ignore[typeddict-item]
    assert TextChunk.from_openai(typeddict_openai) == chunk


def test_convert_input_audio_chunk() -> None:
    chunk = DUMMY_AUDIO_CHUNK
    openai_dict = chunk.to_openai()

    # Verify OpenAI-compliant shape
    assert openai_dict["type"] == "input_audio"
    assert isinstance(openai_dict["input_audio"], dict)
    assert "data" in openai_dict["input_audio"]
    assert "format" in openai_dict["input_audio"]
    assert openai_dict["input_audio"]["format"] in ("wav", "mp3", "flac", "ogg")

    # Roundtrip
    assert AudioChunk.from_openai(openai_dict) == chunk

    typeddict_openai = OpenAIInputAudioChunk(**openai_dict)  # type: ignore[typeddict-item]
    assert AudioChunk.from_openai(typeddict_openai) == chunk


@pytest.mark.parametrize(
    ["openai_image_url_chunk", "image_url_chunk"],
    [
        (
            OpenAIImageChunk(
                type="image_url",
                image_url={
                    "url": "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg",
                    "detail": "auto",
                },
            ),
            ImageURLChunk(
                image_url=ImageURL(
                    url="https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg",
                    detail="auto",
                )
            ),
        ),
        (
            OpenAIImageChunk(
                type="image_url",
                image_url={
                    "url": "data:image/png;base64,iVBORw0",
                },
            ),
            ImageURLChunk(
                image_url="data:image/png;base64,iVBORw0",
            ),
        ),
    ],
)
def test_convert_image_url_chunk(openai_image_url_chunk: dict, image_url_chunk: ImageURLChunk) -> None:
    assert image_url_chunk.to_openai() == openai_image_url_chunk
    if not isinstance(image_url_chunk.image_url, ImageURL):
        image_url_from_openai = ImageURLChunk.from_openai(openai_image_url_chunk)
        assert isinstance(image_url_from_openai.image_url, ImageURL)
        assert image_url_from_openai.image_url.url == image_url_chunk.image_url
        assert image_url_from_openai.type == image_url_chunk.type
    else:
        assert ImageURLChunk.from_openai(openai_image_url_chunk) == image_url_chunk

    typeddict_openai = OpenAIImageChunk(**openai_image_url_chunk)  # type: ignore[typeddict-item]
    if not isinstance(image_url_chunk.image_url, ImageURL):
        image_url_chunk.image_url = ImageURL(url=image_url_chunk.image_url, detail=None)

    assert ImageURLChunk.from_openai(typeddict_openai) == image_url_chunk


@pytest.mark.parametrize(
    ["vllm_audio_url_chunk", "audio_url_chunk"],
    [
        (
            {
                "type": "audio_url",
                "audio_url": {"url": AUDIO_SAMPLE_URL},
            },
            DUMMY_AUDIO_URL_CHUNK_URL,
        ),
        (
            {
                "type": "audio_url",
                "audio_url": {"url": DUMMY_AUDIO_CHUNK.input_audio},
            },
            DUMMY_AUDIO_URL_CHUNK_BASE64,
        ),
        (
            {
                "type": "audio_url",
                "audio_url": {"url": DUMMY_AUDIO_CHUNK.input_audio},
            },
            DUMMY_AUDIO_URL_CHUNK_BASE64_STR,
        ),
        (
            {
                "type": "audio_url",
                "audio_url": {"url": f"data:audio/wav;base64,{DUMMY_AUDIO_CHUNK.input_audio}"},
            },
            DUMMY_AUDIO_URL_CHUNK_BASE64_PREFIX,
        ),
    ],
)
def test_convert_audio_url_chunk(vllm_audio_url_chunk: dict, audio_url_chunk: AudioURLChunk) -> None:
    assert audio_url_chunk.to_openai() == vllm_audio_url_chunk
    if not isinstance(audio_url_chunk.audio_url, AudioURL):
        audio_url_from_openai = AudioURLChunk.from_openai(vllm_audio_url_chunk)
        assert isinstance(audio_url_from_openai.audio_url, AudioURL)
        assert audio_url_from_openai.audio_url.url == audio_url_chunk.audio_url
        assert audio_url_from_openai.type == audio_url_chunk.type
    else:
        assert AudioURLChunk.from_openai(vllm_audio_url_chunk) == audio_url_chunk


def test_convert_function_from_openai_missing_parameters_and_description_and_unk_args() -> None:
    openai_function: dict[str, Any] = {"name": "do_nothing", "unk_field": "1"}
    assert Function.from_openai(openai_function) == Function(name="do_nothing", description="", parameters={})


def test_convert_tool() -> None:
    tool = Tool(
        function=Function(
            name="get_current_weather",
            description="Get the current weather",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the user's location.",
                    },
                },
                "required": ["location", "format"],
            },
            strict=True,
        )
    )

    tool_openai = tool.to_openai()
    assert tool_openai == (
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use. Infer this from the user's location.",
                        },
                    },
                    "required": ["location", "format"],
                },
                "strict": True,
            },
        }
    )
    assert Tool.from_openai(tool.to_openai()) == tool

    typeddict_openai = OpenAITool(**tool.to_openai())  # type: ignore[typeddict-item]
    assert Tool.from_openai(typeddict_openai) == tool


def test_convert_tool_from_openai_missing_parameters_description_and_unknown_field() -> None:
    openai_tool: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "do_nothing",
            "unknown_field": "should be ignored",
        },
    }
    original_openai_tool = copy.deepcopy(openai_tool)
    tool = Tool.from_openai(openai_tool)

    assert tool == Tool(function=Function(name="do_nothing", description="", parameters={}))
    assert openai_tool == original_openai_tool


def test_convert_tool_call() -> None:
    tool_call = ToolCall(
        id="VvvODy9mT",
        function=FunctionCall(
            name="get_current_weather",
            arguments='{"location": "Paris, France", "format": "celsius"}',
        ),
    )
    tool_call_openai = tool_call.to_openai()

    assert tool_call_openai == (
        {
            "id": "VvvODy9mT",
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "arguments": '{"location": "Paris, France", "format": "celsius"}',
            },
        }
    )
    assert ToolCall.from_openai(tool_call.to_openai()) == tool_call

    typeddict_openai = OpenAIToolCall(**tool_call.to_openai())  # type: ignore[typeddict-item]
    assert ToolCall.from_openai(typeddict_openai) == tool_call


def test_tool_call_from_openai_ignores_index() -> None:
    openai_tool_call = {
        "id": "call_123",
        "index": 0,
        "type": "function",
        "function": {"name": "foo", "arguments": "{}"},
    }
    assert ToolCall.from_openai(openai_tool_call) == ToolCall(
        id="call_123", function=FunctionCall(name="foo", arguments="{}")
    )


def test_convert_think_chunk() -> None:
    chunk = ThinkChunk(thinking="Hello", closed=False)
    text_openai = chunk.to_openai()

    assert ThinkChunk.from_openai(text_openai) == chunk
    assert text_openai == {"type": "thinking", "thinking": "Hello", "closed": False}


@pytest.mark.parametrize(
    ["openai_message", "message"],
    [
        ({"role": "user", "content": "Hello"}, UserMessage(content="Hello")),
        (
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            UserMessage(content=[TextChunk(text="Hello")]),
        ),
        (
            OpenAIUserMessage(
                role="user",
                content=[
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg",
                            "detail": "auto",
                        },
                    },
                ],
            ),
            UserMessage(
                content=[
                    TextChunk(text="Describe this image"),
                    ImageURLChunk(
                        image_url=ImageURL(
                            url="https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg",
                            detail="auto",
                        )
                    ),
                ]
            ),
        ),
        (OpenAIAssistantMessage(role="assistant", content="Hi"), AssistantMessage(content="Hi")),
        (
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Hello", "closed": True},
                    {"type": "thinking", "thinking": "Hello", "closed": False},
                    {"type": "text", "text": "Hi"},
                ],
            },
            AssistantMessage(
                content=[
                    ThinkChunk(thinking="Hello", closed=True),
                    ThinkChunk(thinking="Hello", closed=False),
                    TextChunk(text="Hi"),
                ]
            ),
        ),
        (
            OpenAIAssistantMessage(
                role="assistant",
                content="Hi",
                tool_calls=[
                    {
                        "id": "VvvODy9mT",
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": '{"location": "Paris, France", "format": "celsius"}',
                        },
                    }
                ],
            ),
            AssistantMessage(
                content="Hi",
                tool_calls=[
                    ToolCall(
                        id="VvvODy9mT",
                        function=FunctionCall(
                            name="get_current_weather",
                            arguments='{"location": "Paris, France", "format": "celsius"}',
                        ),
                    )
                ],
            ),
        ),
        (
            OpenAIToolMessage(role="tool", content="22", tool_call_id="VvvODy9mT"),
            ToolMessage(tool_call_id="VvvODy9mT", content="22"),
        ),
        (
            OpenAIToolMessage(
                role="tool",
                content=[{"type": "text", "text": "22"}, {"type": "text", "text": "23"}],
                tool_call_id="VvvODy9mT",
            ),
            ToolMessage(tool_call_id="VvvODy9mT", content=[TextChunk(text="22"), TextChunk(text="23")]),
        ),
        (
            OpenAISystemMessage(role="system", content="You are a helpful assistant."),
            SystemMessage(content="You are a helpful assistant."),
        ),
        (
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant."},
                    {"type": "thinking", "thinking": "Hello", "closed": False},
                ],
            },
            SystemMessage(
                content=[TextChunk(text="You are a helpful assistant."), ThinkChunk(thinking="Hello", closed=False)]
            ),
        ),
    ],
)
def test_convert_openai_message_to_message_and_back(openai_message: dict, message: ChatMessage) -> None:
    assert type(message).from_openai(openai_message) == message
    assert message.to_openai() == openai_message


@pytest.mark.parametrize(
    ["openai_message", "expected"],
    [
        (
            {"role": "assistant", "content": "Hi", "reasoning": "Let me think..."},
            AssistantMessage(content=[ThinkChunk(thinking="Let me think...", closed=True), TextChunk(text="Hi")]),
        ),
        (
            {
                "role": "assistant",
                "content": None,
                "reasoning": "Thinking aloud",
                "reasoning_content": "Thinking aloud",
            },
            AssistantMessage(content=[ThinkChunk(thinking="Thinking aloud", closed=True)]),
        ),
        (
            {"role": "assistant", "reasoning": "Thinking aloud"},
            AssistantMessage(content=[ThinkChunk(thinking="Thinking aloud", closed=True)]),
        ),
        (
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello"}],
                "reasoning": "Deep thought",
            },
            AssistantMessage(content=[ThinkChunk(thinking="Deep thought", closed=True), TextChunk(text="Hello")]),
        ),
        (
            {"role": "assistant", "content": "Hi", "reasoning_content": "Only reasoning"},
            AssistantMessage(content=[ThinkChunk(thinking="Only reasoning", closed=True), TextChunk(text="Hi")]),
        ),
    ],
)
def test_from_openai_reasoning_in_assistant_message(openai_message: dict[str, Any], expected: AssistantMessage) -> None:
    assert AssistantMessage.from_openai(openai_message) == expected


def test_from_openai_reasoning_differ_reasoning_content_in_assistant_message() -> None:
    openai_message = {"role": "assistant", "content": "Hi", "reasoning": "Primary", "reasoning_content": "Fallback"}
    with pytest.raises(ValueError, match=r"`reasoning_content` and `reasoning` should be equal"):
        AssistantMessage.from_openai(openai_message)


@pytest.mark.parametrize(
    "openai_message",
    [
        {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "hmm", "closed": True}, {"type": "text", "text": "Hi"}],
            "reasoning": "also thinking",
        },
        {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "hmm", "closed": True}],
            "reasoning_content": "also thinking",
        },
        {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "hmm", "closed": True}],
            "reasoning": "also thinking",
            "reasoning_content": "also thinking",
        },
    ],
)
def test_from_openai_thinking_chunks_and_reasoning_raises(openai_message: dict[str, Any]) -> None:
    with pytest.raises(InvalidAssistantMessageException):
        AssistantMessage.from_openai(openai_message)


def test_non_leading_think_chunks_construction_ok() -> None:
    """Non-leading ThinkChunks are allowed at construction time."""
    msg = AssistantMessage(
        content=[
            ThinkChunk(thinking="First", closed=True),
            TextChunk(text="Reply"),
            ThinkChunk(thinking="Third", closed=False),
        ]
    )
    assert msg.content is not None


@pytest.mark.parametrize(
    "content",
    [
        [ThinkChunk(thinking="First", closed=True), TextChunk(text="Reply"), ThinkChunk(thinking="Third")],
        [TextChunk(text="Reply"), ThinkChunk(thinking="After", closed=True)],
        [TextChunk(text="A"), TextChunk(text="B"), ThinkChunk(thinking="End", closed=True)],
    ],
)
def test_non_leading_think_chunks_to_openai_raises(content: list[TextChunk | ThinkChunk]) -> None:
    """to_openai raises when ThinkChunks are not leading."""
    msg = AssistantMessage(content=content)
    with pytest.raises(InvalidAssistantMessageException, match="ThinkChunks must be leading"):
        msg.to_openai()


@pytest.mark.parametrize(
    ["message", "convert_thinking_format", "expected"],
    [
        # thinking: chunks stay inline
        (
            AssistantMessage(content=[ThinkChunk(thinking="Deep thought", closed=True), TextChunk(text="Answer")]),
            ReasoningFieldFormat.thinking_chunks,
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Deep thought", "closed": True},
                    {"type": "text", "text": "Answer"},
                ],
            },
        ),
        # thinking: multiple leading ThinkChunks stay as-is (no aggregation)
        (
            AssistantMessage(
                content=[
                    ThinkChunk(thinking="First", closed=True),
                    ThinkChunk(thinking="Second", closed=False),
                    TextChunk(text="Reply"),
                ]
            ),
            ReasoningFieldFormat.thinking_chunks,
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "First", "closed": True},
                    {"type": "thinking", "thinking": "Second", "closed": False},
                    {"type": "text", "text": "Reply"},
                ],
            },
        ),
        # reasoning: single leading ThinkChunk extracted as flat string
        (
            AssistantMessage(content=[ThinkChunk(thinking="Let me think", closed=True), TextChunk(text="Done")]),
            ReasoningFieldFormat.reasoning,
            {"role": "assistant", "reasoning": "Let me think", "content": "Done"},
        ),
        # reasoning_content: single leading ThinkChunk extracted as flat string
        (
            AssistantMessage(content=[ThinkChunk(thinking="Pondering", closed=True), TextChunk(text="Result")]),
            ReasoningFieldFormat.reasoning_content,
            {"role": "assistant", "reasoning_content": "Pondering", "content": "Result"},
        ),
        # reasoning: multiple leading ThinkChunks concatenated with newline
        (
            AssistantMessage(
                content=[
                    ThinkChunk(thinking="Part 1", closed=True),
                    ThinkChunk(thinking="Part 2", closed=True),
                    TextChunk(text="Final"),
                ]
            ),
            ReasoningFieldFormat.reasoning,
            {"role": "assistant", "reasoning": "Part 1\nPart 2", "content": "Final"},
        ),
        # thinking: ThinkChunk only, no remaining content
        (
            AssistantMessage(content=[ThinkChunk(thinking="Just thinking", closed=True)]),
            ReasoningFieldFormat.thinking_chunks,
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "Just thinking", "closed": True}],
            },
        ),
        # reasoning: ThinkChunk only, no remaining content
        (
            AssistantMessage(content=[ThinkChunk(thinking="Only reasoning", closed=True)]),
            ReasoningFieldFormat.reasoning,
            {"role": "assistant", "reasoning": "Only reasoning"},
        ),
        # reasoning: leading ThinkChunk with remaining list content (multiple chunks)
        (
            AssistantMessage(
                content=[
                    ThinkChunk(thinking="Think", closed=True),
                    TextChunk(text="A"),
                    TextChunk(text="B"),
                ]
            ),
            ReasoningFieldFormat.reasoning,
            {
                "role": "assistant",
                "reasoning": "Think",
                "content": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}],
            },
        ),
        # String content unchanged regardless of convert_thinking_format
        (
            AssistantMessage(content="Simple text"),
            ReasoningFieldFormat.reasoning,
            {"role": "assistant", "content": "Simple text"},
        ),
        # None content unchanged
        (
            AssistantMessage(content=None),
            ReasoningFieldFormat.thinking_chunks,
            {"role": "assistant"},
        ),
    ],
)
def test_assistant_message_to_openai_convert_thinking_format(
    message: AssistantMessage,
    convert_thinking_format: ReasoningFieldFormat,
    expected: dict[str, Any],
) -> None:
    assert message.to_openai(reasoning_field_format=convert_thinking_format) == expected


def test_assistant_message_to_openai_none_warns_with_think_chunks() -> None:
    message = AssistantMessage(content=[ThinkChunk(thinking="Hmm", closed=True), TextChunk(text="Answer")])
    with pytest.warns(FutureWarning, match=r"convert_thinking_format.*defaults to 'thinking_chunks'"):
        result = message.to_openai()
    assert result == {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "Hmm", "closed": True},
            {"type": "text", "text": "Answer"},
        ],
    }


def test_assistant_message_to_openai_none_no_warning_without_think_chunks() -> None:
    message = AssistantMessage(content="Plain text")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = message.to_openai()
    assert result == {"role": "assistant", "content": "Plain text"}


def test_assistant_message_to_openai_none_no_warning_with_none_content() -> None:
    message = AssistantMessage(content=None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = message.to_openai()
    assert result == {"role": "assistant"}


@pytest.mark.parametrize(
    "request_cls",
    [ChatCompletionRequest, InstructRequest],
)
def test_request_to_openai_forwards_reasoning_field_format(
    request_cls: type[ChatCompletionRequest | InstructRequest],
) -> None:
    messages: list[ChatMessage] = [
        UserMessage(content="Hi"),
        AssistantMessage(content=[ThinkChunk(thinking="Let me think", closed=True), TextChunk(text="Done")]),
    ]
    request: ChatCompletionRequest | InstructRequest
    if request_cls == ChatCompletionRequest:
        request = ChatCompletionRequest(messages=messages)
    else:
        request = InstructRequest(messages=messages)

    openai_request = request.to_openai(reasoning_field_format=ReasoningFieldFormat.reasoning)
    assistant_msg = [m for m in openai_request["messages"] if m["role"] == "assistant"][0]
    assert assistant_msg == {"role": "assistant", "reasoning": "Let me think", "content": "Done"}


@pytest.mark.parametrize(
    "reasoning_effort",
    [None, ReasoningEffort.none, ReasoningEffort.high],
)
@pytest.mark.parametrize(
    ["request_cls"],
    [
        (ChatCompletionRequest,),
        (InstructRequest,),
    ],
)
@pytest.mark.parametrize(
    ["openai_messages", "messages", "openai_tools", "tools"],
    [
        (
            [
                OpenAISystemMessage({"role": "system", "content": "You are a helpful assistant."}),
                OpenAIUserMessage({"role": "user", "content": "What's the weather like in Paris?"}),
                OpenAIAssistantMessage(
                    {
                        "role": "assistant",
                        "content": "Let me think...",
                        "tool_calls": [
                            {
                                "id": "VvvODy9mT",
                                "type": "function",
                                "function": {
                                    "name": "get_current_weather",
                                    "arguments": '{"location": "Paris, France", "format": "celsius"}',
                                },
                            }
                        ],
                    }
                ),
                OpenAIToolMessage({"role": "tool", "content": "22", "tool_call_id": "VvvODy9mT"}),
            ],
            [
                SystemMessage(content="You are a helpful assistant."),
                UserMessage(content="What's the weather like in Paris?"),
                AssistantMessage(
                    content="Let me think...",
                    tool_calls=[
                        ToolCall(
                            id="VvvODy9mT",
                            function=FunctionCall(
                                name="get_current_weather",
                                arguments='{"location": "Paris, France", "format": "celsius"}',
                            ),
                        )
                    ],
                ),
                ToolMessage(tool_call_id="VvvODy9mT", name="get_current_weather", content="22"),
            ],
            [
                OpenAITool(
                    {
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "description": "Get the current weather",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {
                                        "type": "string",
                                        "description": "The city and state, e.g. San Francisco, CA",
                                    },
                                    "format": {
                                        "type": "string",
                                        "enum": ["celsius", "fahrenheit"],
                                        "description": "The temperature unit to use. Infer this from the user's location.",  # noqa: E501
                                    },
                                },
                                "required": ["location", "format"],
                            },
                            "strict": False,
                        },
                    }
                )
            ],
            [
                Tool(
                    function=Function(
                        name="get_current_weather",
                        description="Get the current weather",
                        parameters={
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA",
                                },
                                "format": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                    "description": "The temperature unit to use. Infer this from the user's location.",
                                },
                            },
                            "required": ["location", "format"],
                        },
                    )
                )
            ],
        ),
        (
            [
                OpenAISystemMessage({"role": "system", "content": "You are a helpful assistant."}),
                OpenAIUserMessage({"role": "user", "content": "What's the weather like in Paris?"}),
                OpenAIAssistantMessage(
                    {
                        "role": "assistant",
                        "content": "How should I know?",
                    }
                ),
            ],
            [
                SystemMessage(content="You are a helpful assistant."),
                UserMessage(content="What's the weather like in Paris?"),
                AssistantMessage(
                    content="How should I know?",
                ),
            ],
            None,
            None,
        ),
        (
            [
                OpenAIUserMessage({"role": "user", "content": "What's the weather like in Paris?"}),
                OpenAIAssistantMessage(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "VvvODy9mT",
                                "type": "function",
                                "function": {
                                    "name": "get_current_weather",
                                    "arguments": '{"location": "Paris, France", "format": "celsius"}',
                                },
                            }
                        ],
                    }
                ),
                OpenAIToolMessage({"role": "tool", "content": "22", "tool_call_id": "VvvODy9mT"}),
            ],
            [
                UserMessage(content="What's the weather like in Paris?"),
                AssistantMessage(
                    tool_calls=[
                        ToolCall(
                            id="VvvODy9mT",
                            function=FunctionCall(
                                name="get_current_weather",
                                arguments='{"location": "Paris, France", "format": "celsius"}',
                            ),
                        )
                    ]
                ),
                ToolMessage(tool_call_id="VvvODy9mT", name="get_current_weather", content="22"),
            ],
            [
                OpenAITool(
                    {
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "description": "Get the current weather",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {
                                        "type": "string",
                                        "description": "The city and state, e.g. San Francisco, CA",
                                    },
                                    "format": {
                                        "type": "string",
                                        "enum": ["celsius", "fahrenheit"],
                                        "description": "The temperature unit to use. Infer this from the user's location.",  # noqa: E501
                                    },
                                },
                                "required": ["location", "format"],
                            },
                            "strict": False,
                        },
                    }
                )
            ],
            [
                Tool(
                    function=Function(
                        name="get_current_weather",
                        description="Get the current weather",
                        parameters={
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA",
                                },
                                "format": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                    "description": "The temperature unit to use. Infer this from the user's location.",
                                },
                            },
                            "required": ["location", "format"],
                        },
                    )
                )
            ],
        ),
        (
            [
                OpenAIUserMessage({"role": "user", "content": "Listen to this"}),
                OpenAIAssistantMessage(
                    {
                        "role": "assistant",
                        "content": "Pass the URL please.",
                    }
                ),
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Here it is !"},
                        {
                            "type": "audio_url",
                            "audio_url": {
                                "url": AUDIO_SAMPLE_URL,
                            },
                        },
                        {"type": "text", "text": "What do you think also of these ones?"},
                        {
                            "type": "audio_url",
                            "audio_url": {
                                "url": DUMMY_AUDIO_URL_CHUNK_URL.audio_url.url,
                            },
                        },
                        {
                            "type": "audio_url",
                            "audio_url": {
                                "url": DUMMY_AUDIO_URL_CHUNK_BASE64.audio_url.url,
                            },
                        },
                        {
                            "type": "audio_url",
                            "audio_url": {
                                "url": DUMMY_AUDIO_URL_CHUNK_BASE64_PREFIX.audio_url.url,
                            },
                        },
                    ],
                },
            ],
            [
                UserMessage(content="Listen to this"),
                AssistantMessage(content="Pass the URL please."),
                UserMessage(
                    content=[
                        TextChunk(text="Here it is !"),
                        AudioURLChunk(audio_url=AudioURL(url=AUDIO_SAMPLE_URL)),
                        TextChunk(text="What do you think also of these ones?"),
                        DUMMY_AUDIO_URL_CHUNK_URL,
                        DUMMY_AUDIO_URL_CHUNK_BASE64,
                        DUMMY_AUDIO_URL_CHUNK_BASE64_PREFIX,
                    ]
                ),
            ],
            None,
            None,
        ),
    ],
)
def test_convert_requests(
    openai_messages: list[dict[str, Any]],
    messages: list[ChatMessage],
    openai_tools: list[dict[str, Any]] | None,
    tools: list[Tool] | None,
    request_cls: type[ChatCompletionRequest | InstructRequest],
    reasoning_effort: ReasoningEffort | None,
) -> None:
    request: ChatCompletionRequest | InstructRequest
    if request_cls == ChatCompletionRequest:
        request = ChatCompletionRequest(
            messages=messages,
            tools=tools,
            reasoning_effort=reasoning_effort,
        )
    else:
        request = InstructRequest(
            messages=messages,
            available_tools=tools,
            settings=ModelSettings(reasoning_effort=reasoning_effort),
        )

    openai_request = request.to_openai(stream=True)

    assert openai_request["messages"] == openai_messages
    if tools is not None:
        assert openai_request["tools"] == openai_tools
    else:
        assert "tools" not in openai_request

    if reasoning_effort is not None:
        assert openai_request["reasoning_effort"] == reasoning_effort.value
    else:
        assert "reasoning_effort" not in openai_request

    if isinstance(request, ChatCompletionRequest):
        assert openai_request["temperature"] == 0.7

    stream = openai_request.pop("stream")
    assert stream is True

    reconstructed_request: ChatCompletionRequest | InstructRequest = type(request).from_openai(**openai_request)

    for i, reconstructed_message in enumerate(reconstructed_request.messages):
        if isinstance(reconstructed_message, (SystemMessage, UserMessage, AssistantMessage)):
            assert reconstructed_message == messages[i]
        elif isinstance(reconstructed_message, ToolMessage):
            assert reconstructed_message.model_dump(exclude={"name"}) == messages[i].model_dump(exclude={"name"})

    if tools is not None:
        reconstructed_tools = (
            reconstructed_request.tools
            if isinstance(reconstructed_request, ChatCompletionRequest)
            else reconstructed_request.available_tools
        )
        assert isinstance(tools, list)
        assert isinstance(reconstructed_tools, list)

        # Not using zip below because of mypy not recognizing reconstructed_tools as a list of Tools.
        assert len(tools) == len(reconstructed_tools)
        for i in range(len(tools)):
            assert reconstructed_tools[i] == tools[i]

    if isinstance(reconstructed_request, ChatCompletionRequest):
        assert reconstructed_request.reasoning_effort == reasoning_effort
    else:
        assert reconstructed_request.settings.reasoning_effort == reasoning_effort


@pytest.mark.parametrize(
    ["audio", "language", "stream"],
    [
        (DUMMY_AUDIO_CHUNK, None, False),
        (DUMMY_AUDIO_CHUNK, "en", False),
        (DUMMY_AUDIO_CHUNK, "en", True),
    ],
)
def test_convert_transcription(audio: AudioChunk, language: LanguageAlpha2 | None, stream: bool) -> None:
    def check_equality(a: TranscriptionRequest, b: TranscriptionRequest) -> bool:
        if a.audio != b.audio:
            return False
        if a.id != b.id:
            return False
        if a.model != b.model:
            return False
        if a.language != b.language:
            return False
        if a.strict_audio_validation != b.strict_audio_validation:
            return False
        if a.temperature != b.temperature:
            return False
        if a.top_p != b.top_p:
            return False
        if a.max_tokens != b.max_tokens:
            return False
        if a.random_seed != b.random_seed:
            return False

        return True

    seed: int = 43
    request = TranscriptionRequest(
        audio=audio.input_audio, language=language, model="model", random_seed=seed, target_streaming_delay_ms=None
    )
    openai_request = request.to_openai(stream=stream)

    assert check_equality(request, TranscriptionRequest.from_openai(openai_request))

    openai_transcription = OpenAITranscriptionRequest(**openai_request)  # type: ignore

    from_oai = TranscriptionRequest.from_openai(openai_transcription)
    assert isinstance(from_oai, TranscriptionRequest)

    assert check_equality(request, from_oai)


def _audio_to_wav_bytes(audio: Audio) -> bytes:
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, audio.audio_array, audio.sampling_rate, format="wav")
    return buffer.getvalue()


def test_convert_transcription_str_buffer_name() -> None:
    """Verify that the BytesIO buffer has a .name when audio is a base64 string."""
    audio = _make_fake_audio(0.5)
    b64 = audio.to_base64("wav")

    request = TranscriptionRequest(audio=b64, model="model", language=None, target_streaming_delay_ms=None)
    openai_request = request.to_openai()

    buffer = openai_request["file"]
    assert isinstance(buffer, io.BytesIO)
    assert hasattr(buffer, "name")
    assert buffer.name == "audio.wav"


def test_convert_transcription_bytes_buffer_name() -> None:
    """Verify that the BytesIO buffer has a .name when audio is raw bytes."""
    audio = _make_fake_audio(0.5)
    raw_bytes = _audio_to_wav_bytes(audio)

    request = TranscriptionRequest(audio=raw_bytes, model="model", language=None, target_streaming_delay_ms=None)
    openai_request = request.to_openai()

    buffer = openai_request["file"]
    assert isinstance(buffer, io.BytesIO)
    assert hasattr(buffer, "name")
    assert buffer.name == "audio.wav"


def test_convert_transcription_bytes_invalid_format() -> None:
    """Verify that invalid audio bytes raise a ValueError."""
    request = TranscriptionRequest(
        audio=b"not valid audio data", model="model", language=None, target_streaming_delay_ms=None
    )
    with pytest.raises(ValueError, match="Failed to detect audio format"):
        request.to_openai()


@pytest.mark.parametrize("fmt", ["wav", "flac"])
def test_audio_chunk_to_openai_format_detection(fmt: str) -> None:
    audio = _make_fake_audio(0.5)
    b64 = audio.to_base64(fmt)
    chunk = AudioChunk(input_audio=b64)
    result = chunk.to_openai()

    assert result["input_audio"]["format"] == fmt
    assert result["input_audio"]["data"] == b64
    assert AudioChunk.from_openai(result).input_audio == b64


@pytest.mark.parametrize("fmt", ["wav", "flac"])
def test_transcription_to_openai_format_detection(fmt: str) -> None:
    audio = _make_fake_audio(0.5)
    b64 = audio.to_base64(fmt)
    request = TranscriptionRequest(audio=b64, model="model", language=None, target_streaming_delay_ms=None)
    openai_request = request.to_openai()

    assert openai_request["file"].name == f"audio.{fmt}"

    recovered = Audio.from_bytes(openai_request["file"].getvalue())
    assert np.allclose(recovered.audio_array, audio.audio_array, atol=1e-3)


@pytest.mark.parametrize("fmt", ["wav", "flac"])
def test_transcription_to_openai_bytes_format_detection(fmt: str) -> None:
    import soundfile as sf

    audio = _make_fake_audio(0.5)
    buffer = io.BytesIO()
    sf.write(buffer, audio.audio_array, audio.sampling_rate, format=fmt)
    raw_bytes = buffer.getvalue()

    request = TranscriptionRequest(audio=raw_bytes, model="model", language=None, target_streaming_delay_ms=None)
    openai_request = request.to_openai()

    assert openai_request["file"].name == f"audio.{fmt}"


def test_convert_speech_request_from_openai() -> None:
    audio = _make_fake_audio(0.5)
    raw_bytes = _audio_to_wav_bytes(audio)
    openai_dict: dict[str, Any] = {
        "input": "Hello world",
        "model": "tts-1",
        "voice": "female",
        "ref_audio": io.BytesIO(raw_bytes),
        "instructions": "Speak slowly",  # OAI-only field, should be ignored
    }
    request = SpeechRequest.from_openai(openai_dict)

    assert request.input == "Hello world"
    assert request.model == "tts-1"
    assert request.voice == "female"
    assert isinstance(request.ref_audio, str)
    decoded_audio = Audio.from_base64(request.ref_audio)
    assert np.allclose(decoded_audio.audio_array, audio.audio_array, atol=1e-3)

    # Voice as dict with "id" (OAI format) should be normalized to string
    openai_dict_voice_obj: dict[str, Any] = {
        "input": "Hello",
        "voice": {"id": "custom-voice-123"},
    }
    request_voice = SpeechRequest.from_openai(openai_dict_voice_obj)
    assert request_voice.voice == "custom-voice-123"


def test_convert_speech_request_round_trip() -> None:
    audio = _make_fake_audio(0.5)
    original = SpeechRequest(
        input="Round trip test",
        ref_audio=audio.to_base64("wav"),
        voice="female",
        model="tts-1",
    )

    openai_dict = original.to_openai()
    assert isinstance(openai_dict["ref_audio"], io.BytesIO)

    restored = SpeechRequest.from_openai(openai_dict)

    assert restored.input == original.input
    assert restored.voice == original.voice
    assert restored.model == original.model
    assert isinstance(restored.ref_audio, str)
    assert isinstance(original.ref_audio, str)
    original_audio = Audio.from_base64(original.ref_audio)
    restored_audio = Audio.from_base64(restored.ref_audio)
    assert np.allclose(restored_audio.audio_array, original_audio.audio_array, atol=1e-3)

    # Voice-only request (no ref_audio) should not crash
    voice_only = SpeechRequest(input="Hello", voice="female")
    voice_only_dict = voice_only.to_openai()
    assert "ref_audio" not in voice_only_dict
    assert voice_only_dict["input"] == "Hello"
    assert voice_only_dict["voice"] == "female"


class TestToolChoice:
    @pytest.mark.parametrize(
        ["tool_choice", "expected_openai", "expected_reconstructed"],
        [
            (ToolChoiceEnum.auto, "auto", ToolChoiceEnum.auto.value),
            (ToolChoiceEnum.none, "none", ToolChoiceEnum.none.value),
            (ToolChoiceEnum.required, "required", ToolChoiceEnum.required.value),
            (ToolChoiceEnum.any, "required", ToolChoiceEnum.required.value),
            (
                NamedToolChoice(function=FunctionName(name="get_weather")),
                {"type": "function", "function": {"name": "get_weather"}},
                NamedToolChoice(function=FunctionName(name="get_weather")),
            ),
        ],
    )
    def test_tool_choice_round_trip(
        self,
        tool_choice: ToolChoiceEnum | NamedToolChoice,
        expected_openai: str | dict[str, Any],
        expected_reconstructed: str | NamedToolChoice,
    ) -> None:
        request = ChatCompletionRequest(messages=[UserMessage(content="Hello")], tool_choice=tool_choice)
        openai_request = request.to_openai()
        assert openai_request["tool_choice"] == expected_openai

        reconstructed = ChatCompletionRequest.from_openai(**openai_request)
        assert reconstructed.tool_choice == expected_reconstructed


@pytest.mark.parametrize(
    ["from_openai_call", "expected"],
    [
        # Messages with extra fields
        (
            lambda: UserMessage.from_openai({"role": "user", "content": "Hello", "name": "user1"}),
            UserMessage(content="Hello"),
        ),
        (
            lambda: UserMessage.from_openai(
                {"role": "user", "content": [{"type": "text", "text": "Hello"}], "name": "user1"}
            ),
            UserMessage(content=[TextChunk(text="Hello")]),
        ),
        (
            lambda: SystemMessage.from_openai({"role": "system", "content": "Be helpful", "name": "sys"}),
            SystemMessage(content="Be helpful"),
        ),
        (
            lambda: ToolMessage.from_openai(
                {"role": "tool", "content": "42", "tool_call_id": "c1", "extra": "ignored"}
            ),
            ToolMessage(content="42", tool_call_id="c1"),
        ),
        (
            lambda: AssistantMessage.from_openai(
                {"role": "assistant", "content": "Hi", "refusal": None, "audio": None}
            ),
            AssistantMessage(content="Hi"),
        ),
        # ToolCall with index
        (
            lambda: ToolCall.from_openai(
                {"id": "c1", "index": 0, "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ),
            ToolCall(id="c1", function=FunctionCall(name="f", arguments="{}")),
        ),
        # Tool with extra field
        (
            lambda: Tool.from_openai(
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "",
                        "parameters": {"type": "object"},
                        "strict": False,
                    },
                    "extra_openai_field": True,
                }
            ),
            Tool(function=Function(name="get_weather", description="", parameters={"type": "object"})),
        ),
        # Chunks with extra fields
        (
            lambda: TextChunk.from_openai({"type": "text", "text": "Hello", "annotations": []}),
            TextChunk(text="Hello"),
        ),
        (
            lambda: ThinkChunk.from_openai({"type": "thinking", "thinking": "hmm", "closed": True, "extra": 1}),
            ThinkChunk(thinking="hmm", closed=True),
        ),
        (
            lambda: AudioURLChunk.from_openai(
                {"type": "audio_url", "audio_url": {"url": AUDIO_SAMPLE_URL}, "extra": True}
            ),
            AudioURLChunk(audio_url=AudioURL(url=AUDIO_SAMPLE_URL)),
        ),
        (
            lambda: AudioChunk.from_openai({**DUMMY_AUDIO_CHUNK.to_openai(), "extra": True}),
            DUMMY_AUDIO_CHUNK,
        ),
        # Requests with unsupported OpenAI / unknown fields
        (
            lambda: ChatCompletionRequest.from_openai(
                messages=[{"role": "user", "content": "Hello"}],
                temperature=0.5,
                stream=False,
                n=2,
                logprobs=True,
                frequency_penalty=0.1,
                unknown_field="value",
            ),
            ChatCompletionRequest(messages=[UserMessage(content="Hello")], temperature=0.5),
        ),
        (
            lambda: InstructRequest.from_openai(
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
                n=5,
                logprobs=False,
            ),
            InstructRequest(messages=[UserMessage(content="Hello")]),
        ),
    ],
)
def test_from_openai_drops_extra_fields(from_openai_call: Any, expected: Any) -> None:
    assert from_openai_call() == expected


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: UserMessage(content="Hello", name="user1"),  # type: ignore[call-arg]
        lambda: SystemMessage(content="Be helpful", name="sys"),  # type: ignore[call-arg]
        lambda: TextChunk(text="Hello", extra="bad"),  # type: ignore[call-arg]
    ],
)
def test_direct_construction_still_strict(constructor: Any) -> None:
    with pytest.raises(Exception):
        constructor()
