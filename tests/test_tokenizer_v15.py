import base64
from collections.abc import Callable
from io import BytesIO

import pytest
from PIL import Image

from mistral_common.exceptions import InvalidRequestException, TokenizerException
from mistral_common.protocol.instruct.chunk import (
    AudioChunk,
    AudioURLChunk,
    ImageURLChunk,
    TextChunk,
    ThinkChunk,
)
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
from mistral_common.tokens.tokenizers.audio import AudioConfig, AudioEncoder, AudioSpectrogramConfig, SpecialAudioIDs
from mistral_common.tokens.tokenizers.base import SpecialTokens, TokenizerVersion
from mistral_common.tokens.tokenizers.image import ImageConfig, ImageEncoder, SpecialImageIDs
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV15
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.model_settings_builder import EnumBuilder, ModelSettingsBuilder
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.fixtures.audio import get_dummy_audio_chunk, get_dummy_audio_url_chunk
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

EXPECTED_TEXT_TOOL_AUDIO: str = (
    r"<s>"
    r'[AVAILABLE_TOOLS][{"type": "function", "function": {"name": "fn",'
    r' "description": "test", "parameters": {}}}]'
    r'[/AVAILABLE_TOOLS][MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]'
    r"[INST]Use the tool[/INST]"
    r"[TOOL_CALLS]fn[ARGS]{}</s>"
    r"[TOOL_RESULTS]result[BEGIN_AUDIO][AUDIO][AUDIO][/TOOL_RESULTS]"
)

EXPECTED_TEXT_TOOL_IMAGE: str = (
    r"<s>"
    r'[AVAILABLE_TOOLS][{"type": "function", "function": {"name": "fn",'
    r' "description": "test", "parameters": {}}}]'
    r'[/AVAILABLE_TOOLS][MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]'
    r"[INST]Use the tool[/INST]"
    r"[TOOL_CALLS]fn[ARGS]{}</s>"
    r"[TOOL_RESULTS]result[IMG][IMG_END][/TOOL_RESULTS]"
)

EXPECTED_TEXT_SYSTEM_AUDIO: str = (
    r"<s>[SYSTEM_PROMPT]System with content[BEGIN_AUDIO][AUDIO][AUDIO][/SYSTEM_PROMPT]"
    r'[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]'
    r"[INST]Hello[/INST]"
)

EXPECTED_TEXT_USER_AUDIO: str = (
    r"<s>"
    r'[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]'
    r"[INST]Here is content[BEGIN_AUDIO][AUDIO][AUDIO][/INST]"
)

EXPECTED_TEXT_USER_IMAGE: str = (
    r"<s>"
    r'[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]'
    r"[INST][IMG][IMG_END]Here is content[/INST]"
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


def _get_dummy_image_url_chunk() -> ImageURLChunk:
    r"""Build a small base64-encoded image URL chunk for testing."""
    img = Image.new("RGB", (4, 4), "red")
    buf = BytesIO()
    img.save(buf, "PNG")
    data_url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
    return ImageURLChunk(image_url=data_url)


def get_v15_mistral_tokenizer_with_audio() -> MistralTokenizer:
    r"""Build a V15 MistralTokenizer with audio encoder."""
    builder = _build_model_settings_builder(tuple(ReasoningEffort))
    tekkenizer = Tekkenizer(
        quick_vocab([b"a", b"b", b"c", b"f", b"de"]),
        special_tokens=get_special_tokens(TokenizerVersion.v15, add_audio=True),
        pattern=r".+",
        vocab_size=256 + 100,
        num_special_tokens=100,
        version=TokenizerVersion.v15,
        model_settings_builder=builder,
    )
    audio_config = AudioConfig(
        sampling_rate=24_000,
        frame_rate=12.5,
        encoding_config=AudioSpectrogramConfig(
            num_mel_bins=128,
            hop_length=160,
            window_size=400,
        ),
    )
    special_audio_ids = SpecialAudioIDs(
        audio=tekkenizer.get_special_token(SpecialTokens.audio.value),
        begin_audio=tekkenizer.get_special_token(SpecialTokens.begin_audio.value),
        streaming_pad=None,
        text_to_audio=None,
        audio_to_text=None,
    )
    audio_encoder = AudioEncoder(audio_config, special_audio_ids)
    instruct_tokenizer = InstructTokenizerV15(tekkenizer, audio_encoder=audio_encoder)
    request_normalizer = get_normalizer(TokenizerVersion.v15, tekkenizer.model_settings_builder)
    validator = get_validator(TokenizerVersion.v15, mode=ValidationMode.test)
    return MistralTokenizer(
        instruct_tokenizer=instruct_tokenizer,
        validator=validator,
        request_normalizer=request_normalizer,
    )


def get_v15_mistral_tokenizer_with_image() -> MistralTokenizer:
    r"""Build a V15 MistralTokenizer with image encoder."""
    builder = _build_model_settings_builder(tuple(ReasoningEffort))
    tekkenizer = Tekkenizer(
        quick_vocab([b"a", b"b", b"c", b"f", b"de"]),
        special_tokens=get_special_tokens(TokenizerVersion.v15, add_think=True),
        pattern=r".+",
        vocab_size=256 + 100,
        num_special_tokens=100,
        version=TokenizerVersion.v15,
        model_settings_builder=builder,
    )
    image_config = ImageConfig(image_patch_size=16, max_image_size=1024)
    special_image_ids = SpecialImageIDs(
        img=tekkenizer.get_special_token(SpecialTokens.img.value),
        img_break=tekkenizer.get_special_token(SpecialTokens.img_break.value),
        img_end=tekkenizer.get_special_token(SpecialTokens.img_end.value),
    )
    image_encoder = ImageEncoder(image_config, special_image_ids)
    instruct_tokenizer = InstructTokenizerV15(tekkenizer, image_encoder=image_encoder)
    request_normalizer = get_normalizer(TokenizerVersion.v15, tekkenizer.model_settings_builder)
    validator = get_validator(TokenizerVersion.v15, mode=ValidationMode.test)
    return MistralTokenizer(
        instruct_tokenizer=instruct_tokenizer,
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


@pytest.fixture(scope="session")
def audio_chunk() -> AudioChunk:
    return get_dummy_audio_chunk()


# Multimodal content chunks and their corresponding tokenizer factories for parametrized tests.
_TOOL_MULTIMODAL_PARAMS = [
    pytest.param(
        get_dummy_audio_chunk(), 1, 0, get_v15_mistral_tokenizer_with_audio, EXPECTED_TEXT_TOOL_AUDIO, id="audio"
    ),
    pytest.param(
        get_dummy_audio_url_chunk(),
        1,
        0,
        get_v15_mistral_tokenizer_with_audio,
        EXPECTED_TEXT_TOOL_AUDIO,
        id="audio_url",
    ),
    pytest.param(
        _get_dummy_image_url_chunk(),
        0,
        1,
        get_v15_mistral_tokenizer_with_image,
        EXPECTED_TEXT_TOOL_IMAGE,
        id="image_url",
    ),
]
_SYSTEM_MULTIMODAL_PARAMS = [
    pytest.param(
        get_dummy_audio_chunk(), 1, 0, get_v15_mistral_tokenizer_with_audio, EXPECTED_TEXT_SYSTEM_AUDIO, id="audio"
    ),
]
_USER_MULTIMODAL_PARAMS = [
    pytest.param(
        get_dummy_audio_chunk(), 1, 0, get_v15_mistral_tokenizer_with_audio, EXPECTED_TEXT_USER_AUDIO, id="audio"
    ),
    pytest.param(
        get_dummy_audio_url_chunk(),
        1,
        0,
        get_v15_mistral_tokenizer_with_audio,
        EXPECTED_TEXT_USER_AUDIO,
        id="audio_url",
    ),
    pytest.param(
        _get_dummy_image_url_chunk(),
        0,
        1,
        get_v15_mistral_tokenizer_with_image,
        EXPECTED_TEXT_USER_IMAGE,
        id="image_url",
    ),
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


def test_encode_chat_completion_continue_final_message() -> None:
    builder = _build_model_settings_builder(("none", "high"))
    tokenizer_v15 = get_v15_mistral_tokenizer(builder)
    request: ChatCompletionRequest = ChatCompletionRequest(
        messages=[UserMessage(content="a"), AssistantMessage(content="b")],
        continue_final_message=True,
    )
    encoded = tokenizer_v15.encode_chat_completion(request)

    eos_id = tokenizer_v15.instruct_tokenizer.tokenizer.eos_id
    assert encoded.tokens[-1] != eos_id


@pytest.mark.parametrize(
    ("content_chunk", "expected_audios", "expected_images", "tokenizer_factory", "expected_text"),
    _TOOL_MULTIMODAL_PARAMS,
)
def test_encode_chat_completion_with_multimodal_tool(
    content_chunk: AudioChunk | AudioURLChunk | ImageURLChunk,
    expected_audios: int,
    expected_images: int,
    tokenizer_factory: Callable[[], MistralTokenizer],
    expected_text: str,
) -> None:
    mistral_tokenizer = tokenizer_factory()
    chat_request = ChatCompletionRequest(  # type: ignore[type-var]
        messages=[
            UserMessage(content="Use the tool"),
            AssistantMessage(tool_calls=[ToolCall(id="test12345", function=FunctionCall(name="fn", arguments="{}"))]),
            ToolMessage(
                content=[TextChunk(text="result"), content_chunk],
                tool_call_id="test12345",
            ),
        ],
        tools=[Tool(function=Function(name="fn", description="test", parameters={}))],
    )
    encoded = mistral_tokenizer.encode_chat_completion(chat_request)
    assert encoded.text == expected_text, encoded.text
    assert len(encoded.audios) == expected_audios
    assert len(encoded.images) == expected_images


@pytest.mark.parametrize(
    ("content_chunk", "expected_audios", "expected_images", "tokenizer_factory", "expected_text"),
    _SYSTEM_MULTIMODAL_PARAMS,
)
def test_encode_chat_completion_with_multimodal_system(
    content_chunk: AudioChunk,
    expected_audios: int,
    expected_images: int,
    tokenizer_factory: Callable[[], MistralTokenizer],
    expected_text: str,
) -> None:
    mistral_tokenizer = tokenizer_factory()
    chat_request = ChatCompletionRequest(  # type: ignore[type-var]
        messages=[
            SystemMessage(content=[TextChunk(text="System with content"), content_chunk]),
            UserMessage(content="Hello"),
        ],
    )
    encoded = mistral_tokenizer.encode_chat_completion(chat_request)
    assert encoded.text == expected_text, encoded.text
    assert len(encoded.audios) == expected_audios
    assert len(encoded.images) == expected_images


@pytest.mark.parametrize(
    ("content_chunk", "expected_audios", "expected_images", "tokenizer_factory", "expected_text"),
    _USER_MULTIMODAL_PARAMS,
)
def test_encode_chat_completion_with_multimodal_user(
    content_chunk: AudioChunk | AudioURLChunk | ImageURLChunk,
    expected_audios: int,
    expected_images: int,
    tokenizer_factory: Callable[[], MistralTokenizer],
    expected_text: str,
) -> None:
    mistral_tokenizer = tokenizer_factory()
    chat_request = ChatCompletionRequest(
        messages=[
            UserMessage(content=[TextChunk(text="Here is content"), content_chunk]),
        ],
    )
    encoded = mistral_tokenizer.encode_chat_completion(chat_request)
    assert encoded.text == expected_text, encoded.text
    assert len(encoded.audios) == expected_audios
    assert len(encoded.images) == expected_images
