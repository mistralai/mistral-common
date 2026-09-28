import base64
import io
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from openai.types.audio.transcription_create_params import TranscriptionCreateParamsBase as OpenAITranscriptionRequest
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam as OpenAIAssistantMessage,
)
from openai.types.chat.chat_completion_content_part_input_audio_param import (
    ChatCompletionContentPartInputAudioParam as OpenAIInputAudioChunk,
)
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam as OpenAIUserMessage
from pydantic_extra_types.language_code import LanguageAlpha2

from mistral_common.protocol.instruct.chunk import (
    AudioChunk,
    AudioURL,
    AudioURLChunk,
    ImageURL,
    ImageURLChunk,
    TextChunk,
)
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    ChatMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.request import (
    ChatCompletionRequest,
    ReasoningEffort,
)
from mistral_common.protocol.instruct.tool_calls import (
    Tool,
)
from mistral_common.protocol.speech.request import SpeechRequest
from mistral_common.protocol.transcription.request import TranscriptionRequest
from mistral_common.tokens.tokenizers.audio import Audio

from .test_tokenizer_v7_audio_tts import _make_fake_audio

AUDIO_SAMPLE_URL = "https://freetestdata.com/wp-content/uploads/2021/09/Free_Test_Data_100KB_MP3.mp3"


def _get_audio_chunk() -> AudioChunk:
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


@pytest.mark.parametrize(
    ["openai_message", "message"],
    [
        pytest.param(
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
            id="user-image-url-detail-order",
        ),
    ],
)
def test_convert_openai_message_to_message_and_back(openai_message: dict, message: ChatMessage) -> None:
    assert type(message).from_openai(openai_message) == message
    assert message.to_openai() == openai_message


@pytest.mark.parametrize(
    "reasoning_effort",
    [None, ReasoningEffort.none, ReasoningEffort.high],
)
@pytest.mark.parametrize(
    ["openai_messages", "messages", "openai_tools", "tools"],
    [
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
    reasoning_effort: ReasoningEffort | None,
) -> None:
    request = ChatCompletionRequest(
        messages=messages,
        tools=tools,
        reasoning_effort=reasoning_effort,
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

    assert openai_request["temperature"] == 0.7

    stream = openai_request.pop("stream")
    assert stream is True

    reconstructed_request = ChatCompletionRequest.from_openai(**openai_request)

    for i, reconstructed_message in enumerate(reconstructed_request.messages):
        if isinstance(reconstructed_message, (SystemMessage, UserMessage, AssistantMessage)):
            assert reconstructed_message == messages[i]
        elif isinstance(reconstructed_message, ToolMessage):
            assert reconstructed_message.model_dump(exclude={"name"}) == messages[i].model_dump(exclude={"name"})

    if tools is not None:
        reconstructed_tools = reconstructed_request.tools
        assert isinstance(tools, list)
        assert isinstance(reconstructed_tools, list)

        # Not using zip below because of mypy not recognizing reconstructed_tools as a list of Tools.
        assert len(tools) == len(reconstructed_tools)
        for i in range(len(tools)):
            assert reconstructed_tools[i] == tools[i]

    assert reconstructed_request.reasoning_effort == reasoning_effort


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
def test_audio_chunk_to_openai_raw_bytes_format_detection(fmt: str) -> None:
    audio = _make_fake_audio(0.5)
    buffer = io.BytesIO()
    sf.write(buffer, audio.audio_array, audio.sampling_rate, format=fmt)
    raw_bytes = buffer.getvalue()

    result = AudioChunk(input_audio=raw_bytes).to_openai()

    assert result["input_audio"]["format"] == fmt
    assert result["input_audio"]["data"] == base64.b64encode(raw_bytes).decode("utf-8")
    assert AudioChunk.from_openai(result).input_audio == result["input_audio"]["data"]


@pytest.mark.parametrize("fmt", ["wav", "flac"])
def test_audio_chunk_to_openai_strips_base64_data_url_prefix(fmt: str) -> None:
    audio = _make_fake_audio(0.5)
    b64 = audio.to_base64(fmt)
    chunk = AudioChunk(input_audio=f"data:audio/{fmt};base64,{b64}")

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

    buffer = openai_request["file"]
    assert isinstance(buffer, io.BytesIO)
    assert buffer.name == f"audio.{fmt}"

    recovered = Audio.from_bytes(buffer.getvalue())
    assert np.allclose(recovered.audio_array, audio.audio_array, atol=1e-3)


@pytest.mark.parametrize("fmt", ["wav", "flac"])
def test_transcription_to_openai_bytes_format_detection(fmt: str) -> None:
    audio = _make_fake_audio(0.5)
    buf = io.BytesIO()
    sf.write(buf, audio.audio_array, audio.sampling_rate, format=fmt)
    raw_bytes = buf.getvalue()

    request = TranscriptionRequest(audio=raw_bytes, model="model", language=None, target_streaming_delay_ms=None)
    openai_request = request.to_openai()

    buffer = openai_request["file"]
    assert isinstance(buffer, io.BytesIO)
    assert buffer.name == f"audio.{fmt}"


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


@pytest.mark.parametrize("fmt", ["wav", "flac"])
def test_speech_to_openai_base64_ref_audio_filename(fmt: str) -> None:
    audio = _make_fake_audio(0.5)
    request = SpeechRequest(input="Hello world", ref_audio=audio.to_base64(fmt))

    buffer = request.to_openai()["ref_audio"]

    assert isinstance(buffer, io.BytesIO)
    assert buffer.name == f"audio.{fmt}"

    recovered = Audio.from_bytes(buffer.getvalue())
    assert np.allclose(recovered.audio_array, audio.audio_array, atol=1e-3)


@pytest.mark.parametrize("fmt", ["wav", "flac"])
def test_speech_to_openai_bytes_ref_audio_filename(fmt: str) -> None:
    audio = _make_fake_audio(0.5)
    source = io.BytesIO()
    sf.write(source, audio.audio_array, audio.sampling_rate, format=fmt)
    request = SpeechRequest(input="Hello world", ref_audio=source.getvalue())

    buffer = request.to_openai()["ref_audio"]

    assert isinstance(buffer, io.BytesIO)
    assert buffer.name == f"audio.{fmt}"


def test_speech_to_openai_bytes_invalid_format() -> None:
    """Verify that invalid reference audio bytes raise a ValueError."""
    request = SpeechRequest(input="Hello world", ref_audio=b"not valid audio data")

    with pytest.raises(ValueError, match="Failed to detect audio format"):
        request.to_openai()


@pytest.mark.parametrize(
    ["voice", "with_ref_audio", "random_seed"],
    [
        ("female", True, None),
        ("female", False, 1234),
        (None, True, 0),
    ],
)
def test_convert_speech_request_round_trip(voice: str | None, with_ref_audio: bool, random_seed: int | None) -> None:
    audio = _make_fake_audio(0.5)
    original = SpeechRequest(
        input="Round trip test",
        ref_audio=audio.to_base64("wav") if with_ref_audio else None,
        voice=voice,
        model="tts-1",
        random_seed=random_seed,
    )

    openai_dict = original.to_openai()

    # OpenAI uses "seed"; mistral-common uses "random_seed" (parity with TranscriptionRequest).
    assert "random_seed" not in openai_dict
    assert openai_dict["seed"] == random_seed

    if with_ref_audio:
        assert isinstance(openai_dict["ref_audio"], io.BytesIO)
    else:
        assert "ref_audio" not in openai_dict

    restored = SpeechRequest.from_openai(openai_dict)
    # ref_audio is re-encoded through the OpenAI buffer, so it is not byte-identical; compare it separately.
    assert original.model_dump(exclude={"ref_audio"}) == restored.model_dump(exclude={"ref_audio"})

    if with_ref_audio:
        assert isinstance(original.ref_audio, str)
        assert isinstance(restored.ref_audio, str)
        original_audio = Audio.from_base64(original.ref_audio)
        restored_audio = Audio.from_base64(restored.ref_audio)
        assert np.allclose(restored_audio.audio_array, original_audio.audio_array, atol=1e-3)
    else:
        assert restored.ref_audio is None
