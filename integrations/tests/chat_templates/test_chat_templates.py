import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from jinja2.exceptions import TemplateError
from PIL import Image
from transformers.utils.chat_template_utils import render_jinja_template  # type: ignore[import-not-found]

from integrations.chat_templates.chat_templates import generate_chat_template_dynamic
from mistral_common.audio import Audio
from mistral_common.protocol.instruct.chunk import (
    AudioChunk,
    AudioURL,
    AudioURLChunk,
    ImageChunk,
    ImageURLChunk,
    RawAudio,
    TextChunk,
    ThinkChunk,
)
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.normalize import get_normalizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest, InstructRequest, ReasoningEffort
from mistral_common.protocol.instruct.tool_calls import Function, FunctionCall, Tool, ToolCall
from mistral_common.protocol.instruct.validator import ValidationMode, get_validator
from mistral_common.tokens.tokenizers.audio import (
    AudioConfig,
    AudioEncoder,
    AudioSpectrogramConfig,
    SpecialAudioIDs,
)
from mistral_common.tokens.tokenizers.base import InstructTokenizer, Tokenizer, TokenizerVersion
from mistral_common.tokens.tokenizers.image import ImageConfig, ImageEncoder, SpecialImageIDs
from mistral_common.tokens.tokenizers.instruct import (
    InstructTokenizerV1,
    InstructTokenizerV2,
    InstructTokenizerV3,
    InstructTokenizerV7,
    InstructTokenizerV11,
    InstructTokenizerV13,
    InstructTokenizerV15,
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.model_settings_builder import EnumBuilder, ModelSettingsBuilder
from mistral_common.tokens.tokenizers.sentencepiece import SentencePieceTokenizer
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.test_tekken import get_special_tokens

ROOT_DIR = Path(__file__).parent.parent.parent
TEST_DIR = ROOT_DIR / "tests"


mistral_tokenizer = MistralTokenizer.from_hf_hub("mistralai/Magistral-Small-2509")

_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/7/78/Red_Square_%282x2_Pixel%29.png"


def _create_dummy_image() -> Image.Image:
    """Create a simple dummy 2x2 red square image for testing."""
    # Create a 2x2 red image (same as the one that was being downloaded)
    img = Image.new("RGB", (2, 2), color="red")
    return img


# Create the dummy image once for module-level use
_IMAGE = _create_dummy_image()


@pytest.fixture(autouse=True, scope="module")
def mock_download_image():
    """Mock the download_image function to return a dummy image for all tests."""
    with patch("mistral_common.image.download_image") as mock_download:
        mock_download.return_value = _IMAGE
        # Also mock it in other modules where it might be imported
        with patch("mistral_common.tokens.tokenizers.image.download_image") as mock_download2:
            mock_download2.return_value = _IMAGE
            yield


def _sin_wave(sampling_rate: int, duration: float) -> np.ndarray:
    return np.sin(np.ones([int(duration * sampling_rate)]))


def _sample_audio() -> Audio:
    sampling_rate = 44100
    original_array = _sin_wave(sampling_rate, 1)

    audio = Audio(
        audio_array=original_array,
        sampling_rate=sampling_rate,
        format="wav",
    )
    return audio


_AUDIO_URL = _sample_audio().to_base64("wav")
_AUDIO = RawAudio(data=_AUDIO_URL, format="wav")

SPM_SPECIAL_WHITESPACE = "▁"
SPM_WHITESPACE = "▁"


def sample_audio_url_chunk_base64_prefix(sample_audio: Audio) -> AudioURLChunk:
    return AudioURLChunk(audio_url=AudioURL(url=f"data:audio/wav;base64,{sample_audio.to_base64('wav')}"))


def encode_mistral_common(mistral_tokenizer: MistralTokenizer, chat_request: ChatCompletionRequest, spm: bool) -> str:
    mistral_encoded = str(mistral_tokenizer.encode_chat_completion(chat_request).text)
    # remove image tokens except one per image to make the expected output
    mistral_encoded = mistral_encoded.replace("[IMG]", "").replace("[IMG_BREAK]", "").replace("[IMG_END]", "[IMG]")
    # remove audio tokens except one per audio to make the expected output
    mistral_encoded = mistral_encoded.replace("[AUDIO]", "").replace("[BEGIN_AUDIO]", "[AUDIO]")
    # handle spm
    if spm:
        mistral_encoded = (
            mistral_encoded.replace(SPM_SPECIAL_WHITESPACE, " ").replace(SPM_WHITESPACE, " ").replace("<0x0A>", "\n")
        )
    return mistral_encoded


def encode_instruct_mistral_common(
    mistral_tokenizer: InstructTokenizer, chat_request: InstructRequest, spm: bool
) -> str:
    mistral_encoded = str(mistral_tokenizer.encode_instruct(chat_request).text)
    # remove image tokens except one per image to make the expected output
    mistral_encoded = mistral_encoded.replace("[IMG]", "").replace("[IMG_BREAK]", "").replace("[IMG_END]", "[IMG]")
    # remove audio tokens except one per audio to make the expected output
    mistral_encoded = mistral_encoded.replace("[AUDIO]", "").replace("[BEGIN_AUDIO]", "[AUDIO]")
    if spm:
        mistral_encoded = (
            mistral_encoded.replace(SPM_SPECIAL_WHITESPACE, " ").replace(SPM_WHITESPACE, " ").replace("<0x0A>", "\n")
        )
    return mistral_encoded


def _get_image_encoder(tokenizer: Tokenizer) -> ImageEncoder:
    image_config = ImageConfig(image_patch_size=2, max_image_size=10, spatial_merge_size=1)
    image_encoder = ImageEncoder(
        image_config=image_config,
        special_ids=SpecialImageIDs(
            img=tokenizer.get_special_token("[IMG]"),
            img_break=tokenizer.get_special_token("[IMG_BREAK]"),
            img_end=tokenizer.get_special_token("[IMG_END]"),
        ),
    )
    return image_encoder


def _get_audio_encoder() -> AudioEncoder:
    audio_config = audio_config = AudioConfig(
        sampling_rate=24_000,
        frame_rate=12.5,
        encoding_config=AudioSpectrogramConfig(
            num_mel_bins=128,
            window_size=400,
            hop_length=160,
        ),
    )
    audio_encoder = AudioEncoder(
        audio_config=audio_config, special_ids=SpecialAudioIDs(audio=24, begin_audio=25, streaming_pad=26)
    )
    return audio_encoder


def _maybe_skip(
    spm: bool,
    version: TokenizerVersion,
    image: bool,
    audio: bool,
    think: bool,
) -> None:
    if spm and (version >= TokenizerVersion.v11 or audio):
        pytest.skip("SPM tokenizer is not supported for tokenizer versions v11 and above or audio")
    elif version < TokenizerVersion.v7 and audio:
        pytest.skip("Audio is not supported for tokenizer version v1, v2, v3")
    elif version < TokenizerVersion.v13 and think:
        pytest.skip("Think is not supported for tokenizer version v1, v2, v3, v7, v11")
    elif version < TokenizerVersion.v3 and image:
        pytest.skip("Image is not supported for tokenizer version v1, v2")
    elif image and audio:
        pytest.skip("Image and audio are mutually exclusive")
    elif audio and think:
        pytest.skip("Audio and think are mutually exclusive")


def _get_mistral_tekkenizer(
    tokenizer_version: TokenizerVersion, validation_mode: ValidationMode, image: bool, audio: bool, think: bool
) -> MistralTokenizer:
    special_tokens = get_special_tokens(tokenizer_version=tokenizer_version, add_audio=audio, add_think=think)
    with open((MistralTokenizer._data_path() / "tekken_240911.json"), "r", encoding="utf-8") as f:
        json_tekkenizer = json.load(f)
    vocab = json_tekkenizer["vocab"]
    vocab_size = json_tekkenizer["config"]["default_vocab_size"]
    pattern = json_tekkenizer["config"]["pattern"]
    model_settings_builder = (
        ModelSettingsBuilder(
            reasoning_effort=EnumBuilder(
                accepts_none=True, default=ReasoningEffort.none, values=[ReasoningEffort.none, ReasoningEffort.high]
            )
        )
        if tokenizer_version.supports_model_settings
        else None
    )
    tokenizer = Tekkenizer(
        vocab,
        special_tokens,
        pattern=pattern,
        vocab_size=vocab_size,
        num_special_tokens=100,
        version=tokenizer_version,
        model_settings_builder=model_settings_builder,
    )

    if audio:
        audio_encoder = _get_audio_encoder()

    if image:
        image_encoder = _get_image_encoder(tokenizer)

    if tokenizer_version == TokenizerVersion.v1:
        instruct_cls = InstructTokenizerV1
    elif tokenizer_version == TokenizerVersion.v2:
        instruct_cls = InstructTokenizerV2
    elif tokenizer_version == TokenizerVersion.v3:
        instruct_cls = InstructTokenizerV3
    elif tokenizer_version == TokenizerVersion.v7:
        instruct_cls = InstructTokenizerV7
    elif tokenizer_version == TokenizerVersion.v11:
        instruct_cls = InstructTokenizerV11
    elif tokenizer_version == TokenizerVersion.v13:
        instruct_cls = InstructTokenizerV13
    elif tokenizer_version == TokenizerVersion.v15:
        instruct_cls = InstructTokenizerV15
    else:
        raise ValueError(f"Unknown tokenizer version: {tokenizer_version}")

    instruct_tokenizer: InstructTokenizer = instruct_cls(
        tokenizer=tokenizer,
        image_encoder=image_encoder if image else None,
        audio_encoder=audio_encoder if audio else None,
    )
    model_settings_builder = tokenizer.model_settings_builder if isinstance(tokenizer, Tekkenizer) else None
    mistral_tokenizer = MistralTokenizer(
        instruct_tokenizer,
        validator=get_validator(mode=validation_mode, version=tokenizer_version),
        request_normalizer=get_normalizer(version=tokenizer_version, model_settings_builder=model_settings_builder),
    )

    return mistral_tokenizer


def _get_mistral_sentenpiece(
    tokenizer_version: TokenizerVersion, validation_mode: ValidationMode, image: bool
) -> MistralTokenizer:
    tokenizer = SentencePieceTokenizer(
        MistralTokenizer._data_path() / "mistral_instruct_tokenizer_241114.model.v7m1",
        tokenizer_version,
    )

    if image:
        image_encoder = _get_image_encoder(tokenizer)

    if tokenizer_version == TokenizerVersion.v1:
        instruct_cls = InstructTokenizerV1
    elif tokenizer_version == TokenizerVersion.v2:
        instruct_cls = InstructTokenizerV2
    elif tokenizer_version == TokenizerVersion.v3:
        instruct_cls = InstructTokenizerV3
    elif tokenizer_version == TokenizerVersion.v7:
        instruct_cls = InstructTokenizerV7
    else:
        raise ValueError(f"Unknown tokenizer version: {tokenizer_version}")

    instruct_tokenizer: InstructTokenizer = instruct_cls(
        tokenizer=tokenizer,
        image_encoder=image_encoder if image else None,
    )
    mistral_tokenizer = MistralTokenizer(
        instruct_tokenizer,
        validator=get_validator(mode=validation_mode, version=tokenizer_version),
        request_normalizer=get_normalizer(version=tokenizer_version),
    )

    return mistral_tokenizer


def _get_mistral_tokenizer(
    spm: bool,
    tokenizer_version: TokenizerVersion,
    validation_mode: ValidationMode,
    image: bool,
    audio: bool,
    think: bool,
) -> MistralTokenizer:
    if spm:
        return _get_mistral_sentenpiece(tokenizer_version, validation_mode, image)
    else:
        return _get_mistral_tekkenizer(tokenizer_version, validation_mode, image, audio, think)


def encode_transformers(
    chat_template: str, chat_request: ChatCompletionRequest | dict[str, Any], keep_name_for_tools: bool = False
) -> str:
    if isinstance(chat_request, ChatCompletionRequest):
        openai_request = chat_request.to_openai()
        if keep_name_for_tools:
            for openai_message, chat_message in zip(openai_request["messages"], chat_request.messages):
                if chat_message.role == "tool":
                    openai_message["name"] = chat_message.name
    else:
        openai_request = chat_request
    for tool in openai_request.get("tools", []):
        tool["function"].pop("strict", False)

    # Extract reasoning_effort from the request if present
    reasoning_effort = openai_request.get("reasoning_effort")

    # Prepare kwargs for template rendering
    template_kwargs = {}
    if reasoning_effort is not None:
        template_kwargs["reasoning_effort"] = reasoning_effort

    encoded = render_jinja_template(
        [openai_request["messages"]],
        tools=openai_request.get("tools", None),
        chat_template=chat_template,
        bos_token="<s>",
        eos_token="</s>",
        **template_kwargs,
    )[0][0]
    assert isinstance(encoded, str), type(encoded)
    return encoded


REQUEST_ONE_TURN_TEST = ChatCompletionRequest(
    messages=[
        UserMessage(content="User says hello"),
    ]
)

REQUEST_ONE_TURN_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(content="User says hello"),
        AssistantMessage(content="Assistant says hi"),
    ]
)

REQUEST_ONE_TURN_WITH_SYSTEM_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="User says hello"),
    ]
)

REQUEST_ONE_TURN_WITH_SYSTEM_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="User says hello"),
        AssistantMessage(content="Assistant says hi"),
    ]
)

REQUEST_MULTI_TURN_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(content="User says hello"),
        AssistantMessage(content="Assistant says hi"),
        UserMessage(content="User says how are you ?"),
    ]
)

REQUEST_MULTI_TURN_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(content="User says hello"),
        AssistantMessage(content="Assistant says hi"),
        UserMessage(content="User says how are you ?"),
        AssistantMessage(content="Assistant says hi"),
    ]
)

REQUEST_MULTI_TURN_WITH_SYSTEM_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="User says hello"),
        AssistantMessage(content="Assistant says hi"),
        UserMessage(content="User says how are you ?"),
    ]
)

REQUEST_MULTI_TURN_WITH_SYSTEM_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="User says hello"),
        AssistantMessage(content="Assistant says hi"),
        UserMessage(content="User says how are you ?"),
        AssistantMessage(content="Assistant says hi"),
    ]
)

REQUEST_MULTI_TURN_WITH_TOOLS_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="User says hello"),
        AssistantMessage(content="Assistant says hi"),
        UserMessage(content="User says how are you ?"),
    ],
    tools=[
        Tool(
            function=Function(
                name="tool1",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                            "required": ["location"],
                        }
                    },
                },
            )
        ),
        Tool(function=Function(name="tool2", parameters={})),
    ],
)

REQUEST_MULTI_TURN_WITH_TOOLS_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="User says hello"),
        AssistantMessage(content="Assistant says hi"),
        UserMessage(content="User says how are you ?"),
        AssistantMessage(content="Assistant says hi"),
    ],
    tools=[
        Tool(
            function=Function(
                name="tool1",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                            "required": ["location"],
                        }
                    },
                },
            )
        ),
        Tool(function=Function(name="tool2", parameters={})),
    ],
)

REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="User says hello"),
        AssistantMessage(
            content=None,
            tool_calls=[
                ToolCall(
                    id="123456789",
                    function=FunctionCall(
                        name="tool1",
                        arguments={  # type: ignore[arg-type]
                            "location": "San Francisco, CA",
                        },
                    ),
                ),
            ],
        ),
        ToolMessage(content="32", tool_call_id="123456789"),
        AssistantMessage(content="Whether is 32 degrees in San Francisco, CA"),
        UserMessage(content=[TextChunk(text="User says thanks.")]),
    ],
    tools=[
        Tool(
            function=Function(
                name="tool1",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                            "required": ["location"],
                        }
                    },
                },
            )
        ),
        Tool(function=Function(name="tool2", parameters={})),
    ],
)

REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST_2 = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="User says hello"),
        AssistantMessage(content="Whether is 32 degrees in San Francisco, CA"),
        UserMessage(content=[TextChunk(text="User says thanks.")]),
        AssistantMessage(content=[TextChunk(text="Assistant says you're welcome.")]),
        UserMessage(content="bye"),
        AssistantMessage(
            content=None,
            tool_calls=[
                ToolCall(
                    id="123456789",
                    function=FunctionCall(
                        name="tool1",
                        arguments={  # type: ignore[arg-type]
                            "location": "San Francisco, CA",
                        },
                    ),
                ),
                ToolCall(
                    id="023456789",
                    function=FunctionCall(
                        name="tool2",
                        arguments={},  # type: ignore[arg-type]
                    ),
                ),
            ],
        ),
        ToolMessage(content="32", tool_call_id="123456789"),
    ],
    tools=[
        Tool(
            function=Function(
                name="tool1",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                            "required": ["location"],
                        }
                    },
                },
            )
        ),
        Tool(function=Function(name="tool2", parameters={})),
    ],
)

REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="User says hello"),
        AssistantMessage(
            content=None,
            tool_calls=[
                ToolCall(
                    id="123456789",
                    function=FunctionCall(
                        name="tool1",
                        arguments={  # type: ignore[arg-type]
                            "location": "San Francisco, CA",
                        },
                    ),
                ),
            ],
        ),
        ToolMessage(content="32", tool_call_id="123456789"),
        AssistantMessage(content="Whether is 32 degrees in San Francisco, CA"),
        UserMessage(content=[TextChunk(text="User says thanks.")]),
        AssistantMessage(content=[TextChunk(text="Assistant says you're welcome.")]),
    ],
    tools=[
        Tool(
            function=Function(
                name="tool1",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                            "required": ["location"],
                        }
                    },
                },
            )
        ),
        Tool(function=Function(name="tool2", parameters={})),
    ],
)

REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN_2 = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="User says hello"),
        AssistantMessage(content="Whether is 32 degrees in San Francisco, CA"),
        UserMessage(content=[TextChunk(text="User says thanks.")]),
        AssistantMessage(content=[TextChunk(text="Assistant says you're welcome.")]),
        UserMessage(content="bye"),
        AssistantMessage(
            content=None,
            tool_calls=[
                ToolCall(
                    id="123456789",
                    function=FunctionCall(
                        name="tool1",
                        arguments={  # type: ignore[arg-type]
                            "location": "San Francisco, CA",
                        },
                    ),
                ),
                ToolCall(
                    id="023456789",
                    function=FunctionCall(
                        name="tool2",
                        arguments={},  # type: ignore[arg-type]
                    ),
                ),
            ],
        ),
        ToolMessage(content="32", tool_call_id="123456789"),
        ToolMessage(content="aya", tool_call_id="023456789"),
        AssistantMessage(content="wow 32", tool_calls=[]),
    ],
    tools=[
        Tool(
            function=Function(
                name="tool1",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                            "required": ["location"],
                        }
                    },
                },
            )
        ),
        Tool(function=Function(name="tool2", parameters={})),
    ],
)

REQUEST_MULTI_TURN_WITH_CONTENT_AND_TOOLS_CALLS_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="User says hello"),
        AssistantMessage(content="Whether is 32 degrees in San Francisco, CA"),
        UserMessage(content=[TextChunk(text="User says thanks.")]),
        AssistantMessage(content=[TextChunk(text="Assistant says you're welcome.")]),
        UserMessage(content="bye"),
        AssistantMessage(
            content="Assistant says hi, let me fetch the weather for you.",
            tool_calls=[
                ToolCall(
                    id="123456789",
                    function=FunctionCall(
                        name="tool1",
                        arguments={  # type: ignore[arg-type]
                            "location": "San Francisco, CA",
                        },
                    ),
                ),
                ToolCall(
                    id="023456789",
                    function=FunctionCall(
                        name="tool2",
                        arguments={},  # type: ignore[arg-type]
                    ),
                ),
            ],
        ),
        ToolMessage(content="32", tool_call_id="123456789"),
    ],
    tools=[
        Tool(
            function=Function(
                name="tool1",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                            "required": ["location"],
                        }
                    },
                },
            )
        ),
        Tool(function=Function(name="tool2", parameters={})),
    ],
)

REQUEST_MULTI_TURN_WITH_CONTENT_AND_TOOLS_CALLS_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="User says hello"),
        AssistantMessage(content="Whether is 32 degrees in San Francisco, CA"),
        UserMessage(content=[TextChunk(text="User says thanks.")]),
        AssistantMessage(content=[TextChunk(text="Assistant says you're welcome.")]),
        UserMessage(content="bye"),
        AssistantMessage(
            content="Assistant says hi, let me fetch the weather for you.",
            tool_calls=[
                ToolCall(
                    id="123456789",
                    function=FunctionCall(
                        name="tool1",
                        arguments={  # type: ignore[arg-type]
                            "location": "San Francisco, CA",
                        },
                    ),
                ),
                ToolCall(
                    id="023456789",
                    function=FunctionCall(
                        name="tool2",
                        arguments={},  # type: ignore[arg-type]
                    ),
                ),
            ],
        ),
        ToolMessage(content="32", tool_call_id="123456789"),
        ToolMessage(content="aya", tool_call_id="023456789"),
        AssistantMessage(content="wow 32", tool_calls=[]),
    ],
    tools=[
        Tool(
            function=Function(
                name="tool1",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                            "required": ["location"],
                        }
                    },
                },
            )
        ),
        Tool(function=Function(name="tool2", parameters={})),
    ],
)

REQUEST_MULTI_TURN_IMAGE_URL_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(
            content=[
                TextChunk(text="User asks what is this image ?"),
                ImageURLChunk(image_url=_IMAGE_URL),
                ImageURLChunk(image_url=_IMAGE_URL),
            ]
        ),
        AssistantMessage(content="Assistant answers It is a red square."),
        UserMessage(content=[TextChunk(text="User says thanks.")]),
    ]
)

REQUEST_MULTI_TURN_IMAGE_URL_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(
            content=[
                TextChunk(text="User asks what is this image ?"),
                ImageURLChunk(image_url=_IMAGE_URL),
                ImageURLChunk(image_url=_IMAGE_URL),
            ]
        ),
        AssistantMessage(content="Assistant answers It is a red square."),
        UserMessage(content=[TextChunk(text="User says thanks.")]),
        AssistantMessage(content=[TextChunk(text="Assistant says you're welcome.")]),
    ]
)

REQUEST_MULTI_TURN_IMAGE_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(
            content=[
                TextChunk(text="User asks what is this image ?"),
                ImageChunk(image=_IMAGE),
            ]
        ),
        AssistantMessage(content="Assistant answers It is a red square."),
        UserMessage(content=[TextChunk(text="User says thanks.")]),
    ]
)

REQUEST_MULTI_TURN_IMAGE_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(
            content=[
                TextChunk(text="User asks what is this image ?"),
                ImageChunk(image=_IMAGE),
            ]
        ),
        AssistantMessage(content="Assistant answers It is a red square."),
        UserMessage(content=[TextChunk(text="User says thanks.")]),
        AssistantMessage(content=[TextChunk(text="Assistant says you're welcome.")]),
    ]
)

REQUEST_MULTI_TURN_AUDIO_URL_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(
            content=[
                TextChunk(text="Users asks what is this audio ?"),
                AudioURLChunk(audio_url=_AUDIO_URL),
                AudioURLChunk(audio_url=_AUDIO_URL),
            ]
        ),
        AssistantMessage(content="Assistant answers it is a music."),
        UserMessage(content=[TextChunk(text="User says thanks.")]),
    ]
)

REQUEST_MULTI_TURN_AUDIO_URL_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(
            content=[
                TextChunk(text="Users asks what is this audio ?"),
                AudioURLChunk(audio_url=_AUDIO_URL),
            ]
        ),
        AssistantMessage(content="Assistant answers it is a music."),
        UserMessage(content=[TextChunk(text="User says thanks.")]),
        AssistantMessage(content=[TextChunk(text="Assistant says you're welcome.")]),
    ]
)

REQUEST_MULTI_TURN_AUDIO_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(
            content=[
                TextChunk(text="Users asks what is this audio ?"),
                AudioChunk(input_audio=_AUDIO),
            ]
        ),
        AssistantMessage(content="Assistant answers it is a music."),
        UserMessage(content=[TextChunk(text="User says thanks.")]),
    ]
)

REQUEST_MULTI_TURN_AUDIO_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(
            content=[
                TextChunk(text="Users asks what is this audio ?"),
                AudioChunk(input_audio=_AUDIO),
                AudioChunk(input_audio=_AUDIO),
            ]
        ),
        AssistantMessage(content="Assistant answers it is a music."),
        UserMessage(content=[TextChunk(text="User says thanks.")]),
        AssistantMessage(content=[TextChunk(text="Assistant says you're welcome.")]),
    ]
)

REQUEST_MULTI_TURN_IMAGE_AND_THINKING_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(
            content=[
                TextChunk(text="You are a helpful assistant that can think."),
                ThinkChunk(thinking="You need to think here."),
                TextChunk(text="Here you need to answer."),
            ],
        ),
        UserMessage(
            content=[
                TextChunk(text="User asks what is this image ?"),
                ImageURLChunk(image_url=_IMAGE_URL),
            ]
        ),
        AssistantMessage(
            content=[
                TextChunk(text="Assistant says wow I need to think."),
                ThinkChunk(thinking="Assistant thinks it's a red square."),
                TextChunk(text="Assistant says it is a red square."),
            ],
            tool_calls=[],
        ),
        UserMessage(content=[TextChunk(text="User says thanks.")]),
    ],
)

REQUEST_MULTI_TURN_THINKING_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(
            content=[
                TextChunk(text="You are a helpful assistant that can think."),
                ThinkChunk(thinking="You need to think here."),
                TextChunk(text="Here you need to answer."),
            ],
        ),
        UserMessage(
            content=[
                TextChunk(text="User asks what is this image ?"),
            ]
        ),
        AssistantMessage(
            content=[
                TextChunk(text="Assistant says wow I need to think."),
                ThinkChunk(thinking="Assistant thinks it's a red square."),
                TextChunk(text="Assistant says it is a red square."),
            ],
            tool_calls=[],
        ),
        UserMessage(content=[TextChunk(text="User says thanks.")]),
        AssistantMessage(content=[TextChunk(text="Assistant says you're welcome.")]),
    ],
)

REQUEST_MULTI_TURN_THINKING_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(
            content=[
                TextChunk(text="You are a helpful assistant that can think."),
                ThinkChunk(thinking="You need to think here."),
                TextChunk(text="Here you need to answer."),
            ],
        ),
        UserMessage(
            content=[
                TextChunk(text="User asks what is this image ?"),
            ]
        ),
        AssistantMessage(
            content=[
                TextChunk(text="Assistant says wow I need to think."),
                ThinkChunk(thinking="Assistant thinks it's a red square."),
                TextChunk(text="Assistant says it is a red square."),
            ],
            tool_calls=[],
        ),
        UserMessage(content=[TextChunk(text="User says thanks.")]),
    ],
)

REQUEST_MULTI_TURN_IMAGE_AND_THINKING_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(
            content=[
                TextChunk(text="You are a helpful assistant that can think."),
                ThinkChunk(thinking="You need to think here."),
                TextChunk(text="Here you need to answer."),
            ],
        ),
        UserMessage(
            content=[
                TextChunk(text="User asks what is this image ?"),
                ImageURLChunk(image_url=_IMAGE_URL),
            ]
        ),
        AssistantMessage(
            content=[
                TextChunk(text="Assistant says wow I need to think."),
                ThinkChunk(thinking="Assistant thinks it's a red square."),
                TextChunk(text="Assistant says it is a red square."),
            ],
            tool_calls=[],
        ),
        UserMessage(content=[TextChunk(text="User says thanks.")]),
        AssistantMessage(content=[TextChunk(text="Assistant says you're welcome.")]),
    ],
)


# -- Message aggregation test fixtures --

REQUEST_CONSECUTIVE_USERS_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(content="Hello"),
        UserMessage(content="World"),
    ]
)

REQUEST_CONSECUTIVE_USERS_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(content="Hello"),
        UserMessage(content="World"),
        AssistantMessage(content="Hi there"),
    ]
)

REQUEST_CONSECUTIVE_USERS_WITH_SYSTEM_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are helpful."),
        UserMessage(content="Hello"),
        UserMessage(content="World"),
        AssistantMessage(content="Hi there"),
    ]
)

REQUEST_CONSECUTIVE_ASSISTANTS_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(content="Hello"),
        AssistantMessage(content="Hi"),
        AssistantMessage(content="How can I help?"),
        UserMessage(content="Thanks"),
        AssistantMessage(content="You're welcome"),
    ]
)

REQUEST_MULTIPLE_SYSTEMS_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="System prompt 1."),
        SystemMessage(content="System prompt 2."),
        UserMessage(content="Hello"),
        AssistantMessage(content="Hi"),
    ]
)

REQUEST_CONSECUTIVE_USERS_IMAGE_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(content="What is this?"),
        UserMessage(
            content=[
                ImageChunk(image=_IMAGE),
                TextChunk(text="Describe it"),
            ]
        ),
        AssistantMessage(content="It's an image."),
    ]
)

# -- Multi-chunk aggregation test fixtures --

REQUEST_CONSECUTIVE_USERS_TEXT_CHUNKS_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(content="First as string"),
        UserMessage(content=[TextChunk(text="Second as chunk")]),
        UserMessage(content=[TextChunk(text="Third part A"), TextChunk(text="Third part B")]),
        AssistantMessage(content="Response"),
    ]
)

REQUEST_CONSECUTIVE_USERS_MULTI_IMAGE_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(content=[TextChunk(text="Describe this"), ImageChunk(image=_IMAGE), TextChunk(text="What color?")]),
        UserMessage(content=[TextChunk(text="Also this"), ImageChunk(image=_IMAGE), TextChunk(text="What shape?")]),
        AssistantMessage(content="Both are red squares."),
    ]
)

REQUEST_CONSECUTIVE_USERS_AUDIO_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(
            content=[
                TextChunk(text="Listen to this"),
                AudioURLChunk(audio_url=_AUDIO_URL),
                TextChunk(text="What language?"),
            ]
        ),
        UserMessage(
            content=[
                TextChunk(text="And this"),
                AudioURLChunk(audio_url=_AUDIO_URL),
                TextChunk(text="Transcribe it"),
            ]
        ),
        AssistantMessage(content="Both are in English."),
    ]
)

REQUEST_CONSECUTIVE_ASSISTANTS_THINK_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(content="Solve this problem"),
        AssistantMessage(
            content=[
                TextChunk(text="Hmm."),
                ThinkChunk(thinking="Let me think..."),
                TextChunk(text="I need more context."),
            ]
        ),
        AssistantMessage(
            content=[
                TextChunk(text="OK."),
                ThinkChunk(thinking="Now I understand."),
                TextChunk(text="The answer is 42."),
            ]
        ),
        UserMessage(content="Thanks"),
        AssistantMessage(content="You're welcome"),
    ]
)

REQUEST_CONSECUTIVE_ASSISTANTS_TOOL_CALLS_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="What's the weather?"),
        AssistantMessage(content="Let me check."),
        AssistantMessage(
            content="Fetching data.",
            tool_calls=[
                ToolCall(
                    id="123456789",
                    function=FunctionCall(name="tool1", arguments={"location": "Paris"}),  # type: ignore[arg-type]
                ),
                ToolCall(
                    id="023456789",
                    function=FunctionCall(name="tool1", arguments={"location": "London"}),  # type: ignore[arg-type]
                ),
            ],
        ),
        ToolMessage(content="22", tool_call_id="123456789"),
        ToolMessage(content="15", tool_call_id="023456789"),
        AssistantMessage(content="Paris: 22, London: 15"),
        UserMessage(content="Thanks"),
        AssistantMessage(content="Welcome"),
    ],
    tools=[
        Tool(
            function=Function(
                name="tool1",
                parameters={"type": "object", "properties": {"location": {"type": "string"}}},
            )
        ),
    ],
)

REQUEST_SYSTEM_TEXT_CHUNKS_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content=[TextChunk(text="You are helpful."), TextChunk(text="Be concise.")]),
        UserMessage(content="Hello"),
        AssistantMessage(content="Hi"),
    ]
)

REQUEST_CONSECUTIVE_SYSTEMS_THINK_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content=[TextChunk(text="Rule A"), TextChunk(text="Rule B"), ThinkChunk(thinking="Think 1")]),
        SystemMessage(content=[ThinkChunk(thinking="Think 2"), TextChunk(text="Rule C"), TextChunk(text="Rule D")]),
        UserMessage(content="Hello"),
        AssistantMessage(content="Hi"),
    ]
)


def _get_conversations(
    tokenizer_version: TokenizerVersion,
    validation_mode: ValidationMode,
    image: bool,
    audio: bool,
    think: bool,
) -> list[ChatCompletionRequest]:
    conversations: list[ChatCompletionRequest] = (
        [REQUEST_ONE_TURN_TEST, REQUEST_MULTI_TURN_TEST, REQUEST_MULTI_TURN_WITH_SYSTEM_TEST]
        if validation_mode == ValidationMode.test
        else [REQUEST_ONE_TURN_TRAIN, REQUEST_MULTI_TURN_TRAIN, REQUEST_MULTI_TURN_WITH_SYSTEM_TRAIN]
    )

    if tokenizer_version > TokenizerVersion.v1:
        if validation_mode == ValidationMode.test:
            conversations.extend(
                [
                    REQUEST_MULTI_TURN_WITH_TOOLS_TEST,
                    REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST,
                    REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST_2,
                ]
            )
        else:
            conversations.extend(
                [
                    REQUEST_MULTI_TURN_WITH_TOOLS_TRAIN,
                    REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN,
                    REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN_2,
                ]
            )
    if tokenizer_version > TokenizerVersion.v7:
        conversations.append(
            REQUEST_MULTI_TURN_WITH_CONTENT_AND_TOOLS_CALLS_TEST
            if validation_mode == ValidationMode.test
            else REQUEST_MULTI_TURN_WITH_CONTENT_AND_TOOLS_CALLS_TRAIN
        )

    if image:
        if validation_mode == ValidationMode.test:
            conversations.extend([REQUEST_MULTI_TURN_IMAGE_URL_TEST, REQUEST_MULTI_TURN_IMAGE_TEST])
        else:
            conversations.extend([REQUEST_MULTI_TURN_IMAGE_URL_TRAIN, REQUEST_MULTI_TURN_IMAGE_TRAIN])

    if audio:
        if validation_mode == ValidationMode.test:
            conversations.extend([REQUEST_MULTI_TURN_AUDIO_URL_TEST, REQUEST_MULTI_TURN_AUDIO_TEST])

        else:
            conversations.extend([REQUEST_MULTI_TURN_AUDIO_URL_TRAIN, REQUEST_MULTI_TURN_AUDIO_TRAIN])

    if think:
        if validation_mode == ValidationMode.test:
            conversations.extend([REQUEST_MULTI_TURN_THINKING_TEST])
        else:
            conversations.extend([REQUEST_MULTI_TURN_THINKING_TRAIN])

    if image and think:
        if validation_mode == ValidationMode.test:
            conversations.extend([REQUEST_MULTI_TURN_IMAGE_AND_THINKING_TEST])
        else:
            conversations.extend([REQUEST_MULTI_TURN_IMAGE_AND_THINKING_TRAIN])

    # Message aggregation test fixtures (finetuning only since last msg must be assistant)
    if validation_mode == ValidationMode.finetuning:
        # String-only aggregation (all versions)
        conversations.extend(
            [
                REQUEST_CONSECUTIVE_USERS_TRAIN,
                REQUEST_CONSECUTIVE_USERS_WITH_SYSTEM_TRAIN,
                REQUEST_CONSECUTIVE_ASSISTANTS_TRAIN,
                REQUEST_MULTIPLE_SYSTEMS_TRAIN,
            ]
        )
        # Multi-chunk aggregation (v3+ since v1/v2 only support string or single-TextChunk content)
        if tokenizer_version >= TokenizerVersion.v3:
            conversations.extend(
                [
                    REQUEST_CONSECUTIVE_USERS_TEXT_CHUNKS_TRAIN,
                    REQUEST_SYSTEM_TEXT_CHUNKS_TRAIN,
                ]
            )
        if image:
            conversations.extend(
                [
                    REQUEST_CONSECUTIVE_USERS_IMAGE_TRAIN,
                    REQUEST_CONSECUTIVE_USERS_MULTI_IMAGE_TRAIN,
                ]
            )
        if audio:
            conversations.append(REQUEST_CONSECUTIVE_USERS_AUDIO_TRAIN)
        if think:
            conversations.extend(
                [
                    REQUEST_CONSECUTIVE_ASSISTANTS_THINK_TRAIN,
                    REQUEST_CONSECUTIVE_SYSTEMS_THINK_TRAIN,
                ]
            )
    else:
        conversations.append(REQUEST_CONSECUTIVE_USERS_TEST)

    # v7+ only: consecutive assistants with tool calls (requires content+tool_calls in same message)
    if tokenizer_version >= TokenizerVersion.v7 and validation_mode == ValidationMode.finetuning:
        conversations.append(REQUEST_CONSECUTIVE_ASSISTANTS_TOOL_CALLS_TRAIN)

    conversations = [c.model_copy(deep=True) for c in conversations]

    if think and tokenizer_version >= TokenizerVersion.v15:
        for conv in conversations:
            for message in conv.messages:
                if isinstance(message, SystemMessage) and isinstance(message.content, list):
                    message.content = [
                        TextChunk(text="\n".join([c.text for c in message.content if isinstance(c, TextChunk)]))
                    ]

    return conversations


@pytest.mark.parametrize(
    ("spm", "version", "image", "audio", "think"),
    [
        (False, TokenizerVersion.v1, False, False, False),
        (True, TokenizerVersion.v1, False, False, False),
        (False, TokenizerVersion.v2, False, False, False),
        (True, TokenizerVersion.v2, False, False, False),
        (False, TokenizerVersion.v3, False, False, False),
        (True, TokenizerVersion.v3, False, False, False),
        (False, TokenizerVersion.v3, True, False, False),
        (True, TokenizerVersion.v3, True, False, False),
        (False, TokenizerVersion.v7, False, False, False),
        (True, TokenizerVersion.v7, False, False, False),
        (False, TokenizerVersion.v7, True, False, False),
        (True, TokenizerVersion.v7, True, False, False),
        (False, TokenizerVersion.v7, False, True, False),
        (False, TokenizerVersion.v11, False, False, False),
        (False, TokenizerVersion.v11, True, False, False),
        (False, TokenizerVersion.v11, False, True, False),
        (False, TokenizerVersion.v13, False, False, False),
        (False, TokenizerVersion.v13, True, False, False),
        (False, TokenizerVersion.v13, False, True, False),
        (False, TokenizerVersion.v13, True, False, True),
        (False, TokenizerVersion.v15, False, False, False),
        (False, TokenizerVersion.v15, True, False, False),
        (False, TokenizerVersion.v15, False, True, False),
        (False, TokenizerVersion.v15, True, False, True),
    ],
)
@pytest.mark.parametrize("mode", [ValidationMode.test, ValidationMode.finetuning])
def test_chat_template(
    spm: bool,
    version: TokenizerVersion,
    mode: ValidationMode,
    image: bool,
    audio: bool,
    think: bool,
) -> None:
    conversations = _get_conversations(version, mode, image, audio, think)

    mistral_tokenizer = _get_mistral_tokenizer(
        spm=spm, tokenizer_version=version, validation_mode=mode, image=image, audio=audio, think=think
    )
    chat_template = generate_chat_template_dynamic(spm, version, image, audio, think)
    if version <= TokenizerVersion.v2:
        for conv in conversations:
            for message in conv.messages:
                if isinstance(message, (UserMessage, AssistantMessage)) and isinstance(message.content, list):
                    assert len(message.content) == 1 and isinstance(message.content[0], TextChunk), (
                        "Only text content is supported for v1 and v2"
                    )
                    message.content = str(message.content[0].text)
    for conversation in conversations:
        if version == TokenizerVersion.v2:
            for message in conversation.messages:
                if isinstance(message, ToolMessage):
                    message.name = "tool"
        # Run transformers first since encode_mistral_common may mutate the conversation in-place
        transformers_encoded = encode_transformers(
            chat_template, conversation, keep_name_for_tools=version == TokenizerVersion.v2
        )
        mistral_common_encoded = encode_mistral_common(mistral_tokenizer, conversation, spm)

        assert mistral_common_encoded == transformers_encoded


@pytest.mark.parametrize(
    ("spm", "version", "image", "audio", "think"),
    [
        (False, TokenizerVersion.v1, False, False, False),
        (True, TokenizerVersion.v1, False, False, False),
        (False, TokenizerVersion.v2, False, False, False),
        (True, TokenizerVersion.v2, False, False, False),
        (False, TokenizerVersion.v3, False, False, False),
        (True, TokenizerVersion.v3, False, False, False),
        (False, TokenizerVersion.v3, True, False, False),
        (True, TokenizerVersion.v3, True, False, False),
        (False, TokenizerVersion.v7, False, False, False),
        (True, TokenizerVersion.v7, False, False, False),
        (False, TokenizerVersion.v7, True, False, False),
        (True, TokenizerVersion.v7, True, False, False),
        (False, TokenizerVersion.v7, False, True, False),
        (False, TokenizerVersion.v11, False, False, False),
        (False, TokenizerVersion.v11, True, False, False),
        (False, TokenizerVersion.v11, False, True, False),
        (False, TokenizerVersion.v13, False, False, False),
        (False, TokenizerVersion.v13, True, False, False),
        (False, TokenizerVersion.v13, False, True, False),
        (False, TokenizerVersion.v13, True, False, True),
        (False, TokenizerVersion.v15, False, False, False),
        (False, TokenizerVersion.v15, True, False, False),
        (False, TokenizerVersion.v15, False, True, False),
        (False, TokenizerVersion.v15, True, False, True),
    ],
)
def test_role_error(
    spm: bool,
    version: TokenizerVersion,
    image: bool,
    audio: bool,
    think: bool,
) -> None:
    chat_template = generate_chat_template_dynamic(spm, version, image, audio, think)

    # Consecutive user messages should be aggregated (not raise an error)
    VALID_CONSECUTIVE_USERS = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "World"},
            {"role": "assistant", "content": "Hi"},
        ]
    }
    # This should not raise
    encode_transformers(chat_template, VALID_CONSECUTIVE_USERS)

    # But actual alternation violations should still be caught
    INVALID_ALTERNATE_CONVERSATION = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "assistant", "content": "Help?"},
            {"role": "assistant", "content": "More"},
            {"role": "user", "content": "Thanks"},
            # After aggregation: user, assistant, user — this is valid
        ]
    }
    # This should be valid since consecutive assistants get aggregated
    encode_transformers(chat_template, INVALID_ALTERNATE_CONVERSATION)

    INVALID_ROLE = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "invalid", "content": "Hello"},
        ]
    }

    if version >= TokenizerVersion.v7:
        role_match = r"Only user, assistant, system and tool roles are supported, got invalid\."
    elif version > TokenizerVersion.v1:
        role_match = r"Only user, assistant and tool roles are supported, got invalid\."
    else:
        role_match = r"Only user and assistant roles are supported, got invalid\."

    with pytest.raises(TemplateError, match=role_match):
        encode_transformers(chat_template, INVALID_ROLE)


@pytest.mark.parametrize(
    ("spm", "version", "image", "audio", "think"),
    [
        (False, TokenizerVersion.v1, False, False, False),
        (True, TokenizerVersion.v1, False, False, False),
        (False, TokenizerVersion.v2, False, False, False),
        (True, TokenizerVersion.v2, False, False, False),
        (False, TokenizerVersion.v3, False, False, False),
        (True, TokenizerVersion.v3, False, False, False),
        (False, TokenizerVersion.v3, True, False, False),
        (True, TokenizerVersion.v3, True, False, False),
        (False, TokenizerVersion.v7, False, False, False),
        (True, TokenizerVersion.v7, False, False, False),
        (False, TokenizerVersion.v7, True, False, False),
        (True, TokenizerVersion.v7, True, False, False),
        (False, TokenizerVersion.v7, False, True, False),
        (False, TokenizerVersion.v11, False, False, False),
        (False, TokenizerVersion.v11, True, False, False),
        (False, TokenizerVersion.v11, False, True, False),
        (False, TokenizerVersion.v13, False, False, False),
        (False, TokenizerVersion.v13, True, False, False),
        (False, TokenizerVersion.v13, False, True, False),
        (False, TokenizerVersion.v13, True, False, True),
        (False, TokenizerVersion.v15, False, False, False),
        (False, TokenizerVersion.v15, True, False, False),
        (False, TokenizerVersion.v15, False, True, False),
        (False, TokenizerVersion.v15, True, False, True),
    ],
)
def test_invalid_chunks(
    spm: bool,
    version: TokenizerVersion,
    image: bool,
    audio: bool,
    think: bool,
) -> None:
    INVALID_SP_THINK = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "think", "thinking": "Hello"},
                ],
            }
        ]
    }

    INVALID_SP_RANDOM = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "random", "random": "Hello"},
                ],
            }
        ]
    }

    INVALID_ASSISTANT_THINK = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "think", "thinking": "Hello"},
                ],
            },
        ]
    }

    INVALID_ASSISTANT_RANDOM = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "random", "random": "Hello"},
                ],
            },
        ]
    }

    INVALID_USER_IMAGE = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "image", "image_url": "Hello"},
                ],
            }
        ]
    }

    INVALID_USER_AUDIO = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "audio", "audio_url": "Hello"},
                ],
            }
        ]
    }

    INVALID_USER_RANDOM = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "random", "random": "Hello"},
                ],
            }
        ]
    }

    SP_INVALIDS = [INVALID_SP_RANDOM, INVALID_SP_THINK]
    ASSISTANT_INVALIDS = [INVALID_ASSISTANT_RANDOM, INVALID_ASSISTANT_THINK]
    USER_INVALIDS = [INVALID_USER_IMAGE, INVALID_USER_AUDIO, INVALID_USER_RANDOM]

    invalid_convs = [INVALID_SP_RANDOM, INVALID_USER_RANDOM, INVALID_ASSISTANT_RANDOM]
    if not think:
        invalid_convs += [INVALID_SP_THINK, INVALID_ASSISTANT_THINK]
    if not image:
        invalid_convs += [INVALID_USER_IMAGE]
    if not audio:
        invalid_convs += [INVALID_USER_AUDIO]

    chat_template = generate_chat_template_dynamic(spm, version, image, audio, think)
    for conv in invalid_convs:
        msg_template = "Only {chunks} chunks are supported in {role} message content."
        if conv in SP_INVALIDS:
            chunks = "text and thinking" if think and version < TokenizerVersion.v15 else "text"
            role = "system"
        elif conv in USER_INVALIDS:
            chunks = "text"
            if image:
                chunks += ", image and image_url"
            if audio:
                chunks += ", input_audio and audio_url"
            role = "user"
        elif conv in ASSISTANT_INVALIDS:
            chunks = "text and thinking" if think else "text"
            role = "assistant"

        err_msg = msg_template.format(chunks=chunks, role=role)
        with pytest.raises(TemplateError, match=err_msg):
            encode_transformers(chat_template, conv)


@pytest.mark.parametrize(
    ("spm", "version", "image", "audio", "think"),
    [
        (False, TokenizerVersion.v3, False, False, False),
        (True, TokenizerVersion.v3, False, False, False),
        (False, TokenizerVersion.v3, True, False, False),
        (True, TokenizerVersion.v3, True, False, False),
        (False, TokenizerVersion.v7, False, False, False),
        (True, TokenizerVersion.v7, False, False, False),
        (False, TokenizerVersion.v7, True, False, False),
        (True, TokenizerVersion.v7, True, False, False),
        (False, TokenizerVersion.v7, False, True, False),
        (False, TokenizerVersion.v11, False, False, False),
        (False, TokenizerVersion.v11, True, False, False),
        (False, TokenizerVersion.v11, False, True, False),
    ],
)
def test_tool_call_errors(
    spm: bool,
    version: TokenizerVersion,
    image: bool,
    audio: bool,
    think: bool,
) -> None:
    invalid_id_conv = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "function": {"name": "func", "arguments": "{}"}}],
            },
        ]
    }

    chat_template = generate_chat_template_dynamic(spm, version, image, audio, think)
    with pytest.raises(TemplateError, match="Tool call must have an id of 9 characters or numbers."):
        encode_transformers(chat_template, invalid_id_conv)


@pytest.mark.parametrize(
    ("spm", "version", "image", "audio", "think"),
    [
        (False, TokenizerVersion.v2, False, False, False),
        (True, TokenizerVersion.v2, False, False, False),
        (False, TokenizerVersion.v3, False, False, False),
        (True, TokenizerVersion.v3, False, False, False),
        (False, TokenizerVersion.v3, True, False, False),
        (True, TokenizerVersion.v3, True, False, False),
    ],
)
def test_invalid_assistant(
    spm: bool,
    version: TokenizerVersion,
    image: bool,
    audio: bool,
    think: bool,
) -> None:
    invalid_message_conv = {
        "messages": [
            {
                "role": "assistant",
                "content": "hey",
                "tool_calls": [{"id": "123456789", "function": {"name": "func", "arguments": "{}"}}],
            },
        ]
    }

    chat_template = generate_chat_template_dynamic(spm, version, image, audio, think)
    with pytest.raises(TemplateError, match="Assistant message cannot have both content and tool calls."):
        encode_transformers(chat_template, invalid_message_conv)


class TestDefaultSystemPrompt:
    """Tests for the default_system_prompt parameter in generate_chat_template_dynamic.

    There are two different system prompt handling methods:
    1. Legacy style (v1-v3): System message injected into user messages
    2. Modern style (v7+): Uses [SYSTEM_PROMPT]...[/SYSTEM_PROMPT] tokens

    Tests cover representative versions for each method.
    """

    DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."

    CONV_NO_SYSTEM = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    }

    CONV_WITH_SYSTEM = {
        "messages": [
            {"role": "system", "content": "You are a custom assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    }

    def test_legacy_style_default_system_prompt_used(self) -> None:
        chat_template = generate_chat_template_dynamic(
            spm=False,
            tokenizer_version=TokenizerVersion.v3,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=self.DEFAULT_SYSTEM_PROMPT,
        )

        output = encode_transformers(chat_template, self.CONV_NO_SYSTEM)

        assert self.DEFAULT_SYSTEM_PROMPT in output
        assert f"[INST]{self.DEFAULT_SYSTEM_PROMPT}\n\nHello[/INST]" in output

    def test_legacy_style_default_system_prompt_ignored_when_provided(self) -> None:
        chat_template = generate_chat_template_dynamic(
            spm=False,
            tokenizer_version=TokenizerVersion.v3,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=self.DEFAULT_SYSTEM_PROMPT,
        )

        output = encode_transformers(chat_template, self.CONV_WITH_SYSTEM)

        assert self.DEFAULT_SYSTEM_PROMPT not in output
        assert "You are a custom assistant." in output

    def test_legacy_style_no_default_system_prompt(self) -> None:
        chat_template = generate_chat_template_dynamic(
            spm=False,
            tokenizer_version=TokenizerVersion.v3,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
        )

        output = encode_transformers(chat_template, self.CONV_NO_SYSTEM)

        assert "[INST]Hello[/INST]" in output

    def test_modern_style_default_system_prompt_used(self) -> None:
        chat_template = generate_chat_template_dynamic(
            spm=False,
            tokenizer_version=TokenizerVersion.v7,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=self.DEFAULT_SYSTEM_PROMPT,
        )

        output = encode_transformers(chat_template, self.CONV_NO_SYSTEM)

        assert f"[SYSTEM_PROMPT]{self.DEFAULT_SYSTEM_PROMPT}[/SYSTEM_PROMPT]" in output

    def test_modern_style_default_system_prompt_ignored_when_provided(self) -> None:
        chat_template = generate_chat_template_dynamic(
            spm=False,
            tokenizer_version=TokenizerVersion.v7,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=self.DEFAULT_SYSTEM_PROMPT,
        )

        output = encode_transformers(chat_template, self.CONV_WITH_SYSTEM)

        assert self.DEFAULT_SYSTEM_PROMPT not in output
        assert "[SYSTEM_PROMPT]You are a custom assistant.[/SYSTEM_PROMPT]" in output

    def test_modern_style_no_default_system_prompt(self) -> None:
        chat_template = generate_chat_template_dynamic(
            spm=False,
            tokenizer_version=TokenizerVersion.v7,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
        )

        output = encode_transformers(chat_template, self.CONV_NO_SYSTEM)

        assert "[SYSTEM_PROMPT]" not in output
        assert "[/SYSTEM_PROMPT]" not in output

    def test_legacy_spm_style_default_system_prompt(self) -> None:
        """Test default system prompt with SPM tokenizer (v3 SPM style)."""
        chat_template = generate_chat_template_dynamic(
            spm=True,
            tokenizer_version=TokenizerVersion.v3,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=self.DEFAULT_SYSTEM_PROMPT,
        )

        output = encode_transformers(chat_template, self.CONV_NO_SYSTEM)

        assert self.DEFAULT_SYSTEM_PROMPT in output
        assert f"[INST] {self.DEFAULT_SYSTEM_PROMPT}\n\nHello[/INST]" in output

    def test_modern_spm_style_default_system_prompt(self) -> None:
        chat_template = generate_chat_template_dynamic(
            spm=True,
            tokenizer_version=TokenizerVersion.v7,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=self.DEFAULT_SYSTEM_PROMPT,
        )

        output = encode_transformers(chat_template, self.CONV_NO_SYSTEM)

        assert f"[SYSTEM_PROMPT]{self.DEFAULT_SYSTEM_PROMPT}[/SYSTEM_PROMPT]" in output

    def test_v11_style_default_system_prompt(self) -> None:
        chat_template = generate_chat_template_dynamic(
            spm=False,
            tokenizer_version=TokenizerVersion.v11,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=self.DEFAULT_SYSTEM_PROMPT,
        )

        output = encode_transformers(chat_template, self.CONV_NO_SYSTEM)

        assert f"[SYSTEM_PROMPT]{self.DEFAULT_SYSTEM_PROMPT}[/SYSTEM_PROMPT]" in output

    def test_v13_style_default_system_prompt(self) -> None:
        chat_template = generate_chat_template_dynamic(
            spm=False,
            tokenizer_version=TokenizerVersion.v13,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=self.DEFAULT_SYSTEM_PROMPT,
        )

        output = encode_transformers(chat_template, self.CONV_NO_SYSTEM)

        assert f"[SYSTEM_PROMPT]{self.DEFAULT_SYSTEM_PROMPT}[/SYSTEM_PROMPT]" in output

    def test_special_characters_in_system_prompt(self) -> None:
        special_prompt = "You're a helpful assistant! Use \"quotes\" and 'apostrophes'."
        chat_template = generate_chat_template_dynamic(
            spm=False,
            tokenizer_version=TokenizerVersion.v7,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=special_prompt,
        )

        output = encode_transformers(chat_template, self.CONV_NO_SYSTEM)

        assert special_prompt in output


@pytest.mark.parametrize(
    ("spm", "version", "image", "audio", "think"),
    [
        (False, TokenizerVersion.v15, False, False, False),
        (False, TokenizerVersion.v15, True, False, False),
        (False, TokenizerVersion.v15, False, False, True),
        (False, TokenizerVersion.v15, True, False, True),
    ],
)
def test_reasoning_effort_validation(
    spm: bool,
    version: TokenizerVersion,
    image: bool,
    audio: bool,
    think: bool,
) -> None:
    """Test that reasoning_effort must be either 'none' or 'high' for v15 templates."""
    chat_template = generate_chat_template_dynamic(spm, version, image, audio, think)

    # Test valid reasoning_effort values — v15 always emits [MODEL_SETTINGS]
    # (None/undefined defaults to 'none' in the template)
    valid_conversations = [
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
            "reasoning_effort": "none",
        },
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
            "reasoning_effort": "high",
        },
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ]
            # No reasoning_effort — template defaults to 'none'
        },
    ]

    for conv in valid_conversations:
        result = encode_transformers(chat_template, conv)  # type: ignore
        assert result is not None
        assert "[MODEL_SETTINGS]" in result

    # Test invalid reasoning_effort values
    invalid_conversations = [
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
            "reasoning_effort": "low",
        },
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
            "reasoning_effort": "medium",
        },
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
            "reasoning_effort": "invalid_value",
        },
    ]

    for conv in invalid_conversations:
        with pytest.raises(TemplateError, match='reasoning_effort must be either "none" or "high"'):
            encode_transformers(chat_template, conv)
