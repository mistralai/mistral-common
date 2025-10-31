import json

import numpy as np
import pytest
from transformers.utils.chat_template_utils import render_jinja_template

from chat_templates.chat_templates import get_chat_template
from mistral_common.audio import Audio
from mistral_common.image import download_image
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
from mistral_common.protocol.instruct.messages import AssistantMessage, SystemMessage, ToolMessage, UserMessage
from mistral_common.protocol.instruct.normalize import get_normalizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest, InstructRequest
from mistral_common.protocol.instruct.tool_calls import Function, FunctionCall, Tool, ToolCall
from mistral_common.protocol.instruct.validator import ValidationMode, get_validator
from mistral_common.tokens.tokenizers.audio import AudioConfig, AudioEncoder, AudioSpectrogramConfig, SpecialAudioIDs
from mistral_common.tokens.tokenizers.base import InstructTokenizer, TokenizerVersion
from mistral_common.tokens.tokenizers.image import ImageConfig, ImageEncoder, SpecialImageIDs
from mistral_common.tokens.tokenizers.instruct import (
    InstructTokenizerV1,
    InstructTokenizerV2,
    InstructTokenizerV3,
    InstructTokenizerV7,
    InstructTokenizerV11,
    InstructTokenizerV13,
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.test_tekken import get_special_tokens

mistral_tokenizer = MistralTokenizer.from_hf_hub("mistralai/Magistral-Small-2509")

_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/7/78/Red_Square_%282x2_Pixel%29.png"
_IMAGE = download_image(_IMAGE_URL)


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


def sample_audio_url_chunk_base64_prefix(sample_audio: Audio) -> AudioURLChunk:
    return AudioURLChunk(audio_url=AudioURL(url=f"data:audio/wav;base64,{sample_audio.to_base64('wav')}"))


def encode_mistral_common(mistral_tokenizer: MistralTokenizer, chat_request: ChatCompletionRequest) -> str:
    mistral_encoded = str(mistral_tokenizer.encode_chat_completion(chat_request).text)
    # remove image tokens except one per image to make the expected output
    mistral_encoded = mistral_encoded.replace("[IMG]", "").replace("[IMG_BREAK]", "").replace("[IMG_END]", "[IMG]")
    # remove audio tokens except one per audio to make the expected output
    mistral_encoded = mistral_encoded.replace("[AUDIO]", "").replace("[BEGIN_AUDIO]", "[AUDIO]")
    return mistral_encoded


def encode_instruct_mistral_common(mistral_tokenizer: InstructTokenizer, chat_request: InstructRequest) -> str:
    mistral_encoded = str(mistral_tokenizer.encode_instruct(chat_request).text)
    # remove image tokens except one per image to make the expected output
    mistral_encoded = mistral_encoded.replace("[IMG]", "").replace("[IMG_BREAK]", "").replace("[IMG_END]", "[IMG]")
    # remove audio tokens except one per audio to make the expected output
    mistral_encoded = mistral_encoded.replace("[AUDIO]", "").replace("[BEGIN_AUDIO]", "[AUDIO]")
    return mistral_encoded


def _get_image_encoder() -> ImageEncoder:
    image_config = ImageConfig(image_patch_size=2, max_image_size=10, spatial_merge_size=1)
    image_encoder = ImageEncoder(
        image_config=image_config, special_ids=SpecialImageIDs(img=10, img_break=12, img_end=13)
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
    audio_encoder = AudioEncoder(audio_config=audio_config, special_ids=SpecialAudioIDs(audio=24, begin_audio=25))
    return audio_encoder


def _get_mistral_tekkenizer(
    tokenizer_version: TokenizerVersion, validation_mode: ValidationMode, image: bool, audio: bool, think: bool
) -> MistralTokenizer:
    special_tokens = get_special_tokens(tokenizer_version=tokenizer_version, add_audio=audio, add_think=think)
    with open((MistralTokenizer._data_path() / "tekken_240911.json"), "r", encoding="utf-8") as f:
        json_tekkenizer = json.load(f)
    vocab = json_tekkenizer["vocab"]
    vocab_size = json_tekkenizer["config"]["default_vocab_size"]
    pattern = json_tekkenizer["config"]["pattern"]
    tokenizer = Tekkenizer(
        vocab,
        special_tokens,
        pattern=pattern,
        vocab_size=vocab_size,
        num_special_tokens=100,
        version=tokenizer_version,
    )

    if audio:
        audio_encoder = _get_audio_encoder()

    if image:
        image_encoder = _get_image_encoder()

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
    else:
        raise ValueError(f"Unknown tokenizer version: {tokenizer_version}")

    instruct_tokenizer: InstructTokenizer = instruct_cls(
        tokenizer=tokenizer,
        image_encoder=image_encoder if image else None,
        audio_encoder=audio_encoder if audio else None,
    )
    mistral_tokenizer = MistralTokenizer(
        instruct_tokenizer,
        validator=get_validator(mode=validation_mode, version=tokenizer_version),
        request_normalizer=get_normalizer(version=tokenizer_version),
    )

    return mistral_tokenizer


def encode_transformers(chat_template: str, chat_request: ChatCompletionRequest) -> str:
    openai_request = chat_request.to_openai()
    return render_jinja_template(
        [openai_request["messages"]],
        tools=openai_request.get("tools", None),  # type: ignore[arg-type]
        chat_template=chat_template,
        bos_token="<s>",
        eos_token="</s>",
    )[0][0]


REQUEST_ONE_TURN_TEST = ChatCompletionRequest(
    messages=[
        UserMessage(content="Hi"),
    ]
)

REQUEST_ONE_TURN_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(content="Hi"),
        AssistantMessage(content="yo"),
    ]
)

REQUEST_ONE_TURN_WITH_SYSTEM_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="hello"),
        UserMessage(content="Hi"),
    ]
)

REQUEST_ONE_TURN_WITH_SYSTEM_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="hello"),
        UserMessage(content="Hi"),
        AssistantMessage(content="yo"),
    ]
)

REQUEST_MULTI_TURN_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(content="Hi"),
        AssistantMessage(content="yo"),
        UserMessage(content="da"),
    ]
)

REQUEST_MULTI_TURN_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(content="Hi"),
        AssistantMessage(content="yo"),
        UserMessage(content="da"),
        AssistantMessage(content="yo"),
    ]
)

REQUEST_MULTI_TURN_WITH_SYSTEM_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="hello"),
        UserMessage(content="Hi"),
        AssistantMessage(content="yo"),
        UserMessage(content="da"),
    ]
)

REQUEST_MULTI_TURN_WITH_SYSTEM_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="hello"),
        UserMessage(content="Hi"),
        AssistantMessage(content="yo"),
        UserMessage(content="da"),
        AssistantMessage(content="yo"),
    ]
)

REQUEST_MULTI_TURN_WITH_TOOLS_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="hello"),
        UserMessage(content="Hi"),
        AssistantMessage(content="yo"),
        UserMessage(content="da"),
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
        SystemMessage(content="hello"),
        UserMessage(content="Hi"),
        AssistantMessage(content="yo"),
        UserMessage(content="da"),
        AssistantMessage(content="yo"),
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
        SystemMessage(content="hello"),
        UserMessage(content="Hi"),
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
        UserMessage(content="thanks"),
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
        SystemMessage(content="hello"),
        UserMessage(content="Hi"),
        AssistantMessage(content="Whether is 32 degrees in San Francisco, CA"),
        UserMessage(content="thanks"),
        AssistantMessage(content="You're welcome!"),
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
        SystemMessage(content="hello"),
        UserMessage(content="Hi"),
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
        UserMessage(content="thanks"),
        AssistantMessage(content="You're welcome!"),
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
        SystemMessage(content="hello"),
        UserMessage(content="Hi"),
        AssistantMessage(content="Whether is 32 degrees in San Francisco, CA"),
        UserMessage(content="thanks"),
        AssistantMessage(content="You're welcome!"),
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

REQUEST_MULTI_TURN_IMAGE_URL_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="hello"),
        UserMessage(
            content=[
                TextChunk(text="what is this image"),
                ImageURLChunk(image_url=_IMAGE_URL),
            ]
        ),
        AssistantMessage(content="It is a red square"),
        UserMessage(content="thanks"),
    ]
)

REQUEST_MULTI_TURN_IMAGE_URL_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="hello"),
        UserMessage(
            content=[
                TextChunk(text="what is this image"),
                ImageURLChunk(image_url=_IMAGE_URL),
            ]
        ),
        AssistantMessage(content="It is a red square"),
        UserMessage(content="thanks"),
        AssistantMessage(content="You're welcome!"),
    ]
)

REQUEST_MULTI_TURN_IMAGE_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="hello"),
        UserMessage(
            content=[
                TextChunk(text="what is this image"),
                ImageChunk(image=_IMAGE),
            ]
        ),
        AssistantMessage(content="It is a red square"),
        UserMessage(content="thanks"),
    ]
)

REQUEST_MULTI_TURN_IMAGE_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="hello"),
        UserMessage(
            content=[
                TextChunk(text="what is this image"),
                ImageChunk(image=_IMAGE),
            ]
        ),
        AssistantMessage(content="It is a red square"),
        UserMessage(content="thanks"),
        AssistantMessage(content="You're welcome!"),
    ]
)

REQUEST_MULTI_TURN_AUDIO_URL_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(
            content=[
                TextChunk(text="what is this audio"),
                AudioURLChunk(audio_url=_AUDIO_URL),
                AudioURLChunk(audio_url=_AUDIO_URL),
            ]
        ),
        AssistantMessage(content="It is a music"),
        UserMessage(content="thanks"),
    ]
)

REQUEST_MULTI_TURN_AUDIO_URL_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(
            content=[
                TextChunk(text="what is this audio"),
                AudioURLChunk(audio_url=_AUDIO_URL),
            ]
        ),
        AssistantMessage(content="It is a music"),
        UserMessage(content="thanks"),
        AssistantMessage(content="You're welcome!"),
    ]
)

REQUEST_MULTI_TURN_AUDIO_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(
            content=[
                TextChunk(text="what is this audio"),
                AudioChunk(input_audio=_AUDIO),
            ]
        ),
        AssistantMessage(content="It is a music"),
        UserMessage(content="thanks"),
    ]
)

REQUEST_MULTI_TURN_AUDIO_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(
            content=[
                TextChunk(text="what is this audio"),
                AudioChunk(input_audio=_AUDIO),
                AudioChunk(input_audio=_AUDIO),
            ]
        ),
        AssistantMessage(content="It is a music"),
        UserMessage(content="thanks"),
        AssistantMessage(content="You're welcome!"),
    ]
)

REQUEST_MULTI_TURN_IMAGE_AND_THINKING_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(
            content=[
                TextChunk(text="You are"),
                ThinkChunk(thinking="a"),
                TextChunk(text="system prompt"),
            ],
        ),
        UserMessage(
            content=[
                TextChunk(text="what is this image"),
                ImageURLChunk(image_url=_IMAGE_URL),
            ]
        ),
        AssistantMessage(
            content=[
                TextChunk(text="assist"),
                ThinkChunk(thinking="thinking assist"),
                TextChunk(text="assist again"),
            ],
            tool_calls=[],
        ),
        UserMessage(content="thanks"),
    ],
)

REQUEST_MULTI_TURN_THINKING_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(
            content=[
                TextChunk(text="You are"),
                ThinkChunk(thinking="a"),
                TextChunk(text="system prompt"),
            ],
        ),
        UserMessage(
            content=[
                TextChunk(text="what is this image"),
            ]
        ),
        AssistantMessage(
            content=[
                TextChunk(text="assist"),
                ThinkChunk(thinking="thinking assist"),
                TextChunk(text="assist again"),
            ],
            tool_calls=[],
        ),
        UserMessage(content="thanks"),
        AssistantMessage(content="You're welcome!"),
    ],
)

REQUEST_MULTI_TURN_THINKING_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(
            content=[
                TextChunk(text="You are"),
                ThinkChunk(thinking="a"),
                TextChunk(text="system prompt"),
            ],
        ),
        UserMessage(
            content=[
                TextChunk(text="what is this image"),
            ]
        ),
        AssistantMessage(
            content=[
                TextChunk(text="assist"),
                ThinkChunk(thinking="thinking assist"),
                TextChunk(text="assist again"),
            ],
            tool_calls=[],
        ),
        UserMessage(content="thanks"),
    ],
)

REQUEST_MULTI_TURN_IMAGE_AND_THINKING_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(
            content=[
                TextChunk(text="You are"),
                ThinkChunk(thinking="a"),
                TextChunk(text="system prompt"),
            ],
        ),
        UserMessage(
            content=[
                TextChunk(text="what is this image"),
                ImageURLChunk(image_url=_IMAGE_URL),
            ]
        ),
        AssistantMessage(
            content=[
                TextChunk(text="assist"),
                ThinkChunk(thinking="thinking assist"),
                TextChunk(text="assist again"),
            ],
            tool_calls=[],
        ),
        UserMessage(content="thanks"),
        AssistantMessage(content="You're welcome!"),
    ],
)


@pytest.mark.parametrize(
    "version,mode,image,audio,think,conversations",
    [
        (
            TokenizerVersion.v1,
            ValidationMode.test,
            False,
            False,
            False,
            [REQUEST_ONE_TURN_TEST, REQUEST_MULTI_TURN_TEST, REQUEST_MULTI_TURN_WITH_SYSTEM_TEST],
        ),
        (
            TokenizerVersion.v1,
            ValidationMode.finetuning,
            False,
            False,
            False,
            [REQUEST_ONE_TURN_TRAIN, REQUEST_MULTI_TURN_TRAIN, REQUEST_MULTI_TURN_WITH_SYSTEM_TRAIN],
        ),
        (
            TokenizerVersion.v2,
            ValidationMode.test,
            False,
            False,
            False,
            [
                REQUEST_ONE_TURN_TEST,
                REQUEST_MULTI_TURN_TEST,
                REQUEST_MULTI_TURN_WITH_SYSTEM_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST_2,
            ],
        ),
        (
            TokenizerVersion.v2,
            ValidationMode.finetuning,
            False,
            False,
            False,
            [
                REQUEST_ONE_TURN_TRAIN,
                REQUEST_MULTI_TURN_TRAIN,
                REQUEST_MULTI_TURN_WITH_SYSTEM_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN_2,
            ],
        ),
        (
            TokenizerVersion.v3,
            ValidationMode.test,
            False,
            False,
            False,
            [
                REQUEST_ONE_TURN_TEST,
                REQUEST_MULTI_TURN_TEST,
                REQUEST_MULTI_TURN_WITH_SYSTEM_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST_2,
            ],
        ),
        (
            TokenizerVersion.v3,
            ValidationMode.finetuning,
            False,
            False,
            False,
            [
                REQUEST_ONE_TURN_TRAIN,
                REQUEST_MULTI_TURN_TRAIN,
                REQUEST_MULTI_TURN_WITH_SYSTEM_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN_2,
            ],
        ),
        (
            TokenizerVersion.v3,
            ValidationMode.test,
            True,
            False,
            False,
            [
                REQUEST_ONE_TURN_TEST,
                REQUEST_MULTI_TURN_TEST,
                REQUEST_MULTI_TURN_WITH_SYSTEM_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST_2,
                REQUEST_MULTI_TURN_IMAGE_URL_TEST,
                REQUEST_MULTI_TURN_IMAGE_TEST,
            ],
        ),
        (
            TokenizerVersion.v3,
            ValidationMode.finetuning,
            True,
            False,
            False,
            [
                REQUEST_ONE_TURN_TRAIN,
                REQUEST_MULTI_TURN_TRAIN,
                REQUEST_MULTI_TURN_WITH_SYSTEM_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN_2,
                REQUEST_MULTI_TURN_IMAGE_URL_TRAIN,
                REQUEST_MULTI_TURN_IMAGE_TRAIN,
            ],
        ),
        (
            TokenizerVersion.v7,
            ValidationMode.test,
            False,
            False,
            False,
            [
                REQUEST_ONE_TURN_TEST,
                REQUEST_MULTI_TURN_TEST,
                REQUEST_MULTI_TURN_WITH_SYSTEM_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST_2,
            ],
        ),
        (
            TokenizerVersion.v7,
            ValidationMode.finetuning,
            False,
            False,
            False,
            [
                REQUEST_ONE_TURN_TRAIN,
                REQUEST_MULTI_TURN_TRAIN,
                REQUEST_MULTI_TURN_WITH_SYSTEM_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN_2,
            ],
        ),
        (
            TokenizerVersion.v7,
            ValidationMode.test,
            True,
            False,
            False,
            [
                REQUEST_ONE_TURN_TEST,
                REQUEST_MULTI_TURN_TEST,
                REQUEST_MULTI_TURN_WITH_SYSTEM_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST_2,
                REQUEST_MULTI_TURN_IMAGE_URL_TEST,
                REQUEST_MULTI_TURN_IMAGE_TEST,
            ],
        ),
        (
            TokenizerVersion.v7,
            ValidationMode.finetuning,
            True,
            False,
            False,
            [
                REQUEST_ONE_TURN_TRAIN,
                REQUEST_MULTI_TURN_TRAIN,
                REQUEST_MULTI_TURN_WITH_SYSTEM_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN_2,
                REQUEST_MULTI_TURN_IMAGE_URL_TRAIN,
                REQUEST_MULTI_TURN_IMAGE_TRAIN,
            ],
        ),
        (
            TokenizerVersion.v7,
            ValidationMode.test,
            False,
            True,
            False,
            [
                REQUEST_ONE_TURN_TEST,
                REQUEST_MULTI_TURN_TEST,
                REQUEST_MULTI_TURN_WITH_SYSTEM_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST_2,
                REQUEST_MULTI_TURN_AUDIO_URL_TEST,
                REQUEST_MULTI_TURN_AUDIO_TEST,
            ],
        ),
        (
            TokenizerVersion.v7,
            ValidationMode.finetuning,
            False,
            True,
            False,
            [
                REQUEST_ONE_TURN_TRAIN,
                REQUEST_MULTI_TURN_TRAIN,
                REQUEST_MULTI_TURN_WITH_SYSTEM_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN_2,
                REQUEST_MULTI_TURN_AUDIO_URL_TRAIN,
                REQUEST_MULTI_TURN_AUDIO_TRAIN,
            ],
        ),
        (
            TokenizerVersion.v11,
            ValidationMode.test,
            False,
            False,
            False,
            [
                REQUEST_ONE_TURN_TEST,
                REQUEST_MULTI_TURN_TEST,
                REQUEST_MULTI_TURN_WITH_SYSTEM_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST_2,
            ],
        ),
        (
            TokenizerVersion.v11,
            ValidationMode.finetuning,
            False,
            False,
            False,
            [
                REQUEST_ONE_TURN_TRAIN,
                REQUEST_MULTI_TURN_TRAIN,
                REQUEST_MULTI_TURN_WITH_SYSTEM_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN_2,
            ],
        ),
        (
            TokenizerVersion.v11,
            ValidationMode.test,
            True,
            False,
            False,
            [
                REQUEST_ONE_TURN_TEST,
                REQUEST_MULTI_TURN_TEST,
                REQUEST_MULTI_TURN_WITH_SYSTEM_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST_2,
                REQUEST_MULTI_TURN_IMAGE_URL_TEST,
                REQUEST_MULTI_TURN_IMAGE_TEST,
            ],
        ),
        (
            TokenizerVersion.v11,
            ValidationMode.finetuning,
            True,
            False,
            False,
            [
                REQUEST_ONE_TURN_TRAIN,
                REQUEST_MULTI_TURN_TRAIN,
                REQUEST_MULTI_TURN_WITH_SYSTEM_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN_2,
                REQUEST_MULTI_TURN_IMAGE_URL_TRAIN,
                REQUEST_MULTI_TURN_IMAGE_TRAIN,
            ],
        ),
        (
            TokenizerVersion.v11,
            ValidationMode.test,
            False,
            True,
            False,
            [
                REQUEST_ONE_TURN_TEST,
                REQUEST_MULTI_TURN_TEST,
                REQUEST_MULTI_TURN_WITH_SYSTEM_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST_2,
                REQUEST_MULTI_TURN_AUDIO_URL_TEST,
                REQUEST_MULTI_TURN_AUDIO_TEST,
            ],
        ),
        (
            TokenizerVersion.v11,
            ValidationMode.finetuning,
            False,
            True,
            False,
            [
                REQUEST_ONE_TURN_TRAIN,
                REQUEST_MULTI_TURN_TRAIN,
                REQUEST_MULTI_TURN_WITH_SYSTEM_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN_2,
                REQUEST_MULTI_TURN_AUDIO_URL_TRAIN,
                REQUEST_MULTI_TURN_AUDIO_TRAIN,
            ],
        ),
        (
            TokenizerVersion.v13,
            ValidationMode.test,
            False,
            False,
            False,
            [
                REQUEST_ONE_TURN_TEST,
                REQUEST_MULTI_TURN_TEST,
                REQUEST_MULTI_TURN_WITH_SYSTEM_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST_2,
            ],
        ),
        (
            TokenizerVersion.v13,
            ValidationMode.finetuning,
            False,
            False,
            False,
            [
                REQUEST_ONE_TURN_TRAIN,
                REQUEST_MULTI_TURN_TRAIN,
                REQUEST_MULTI_TURN_WITH_SYSTEM_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN_2,
            ],
        ),
        (
            TokenizerVersion.v13,
            ValidationMode.test,
            True,
            False,
            False,
            [
                REQUEST_ONE_TURN_TEST,
                REQUEST_MULTI_TURN_TEST,
                REQUEST_MULTI_TURN_WITH_SYSTEM_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST_2,
                REQUEST_MULTI_TURN_IMAGE_URL_TEST,
                REQUEST_MULTI_TURN_IMAGE_TEST,
            ],
        ),
        (
            TokenizerVersion.v13,
            ValidationMode.finetuning,
            True,
            False,
            False,
            [
                REQUEST_ONE_TURN_TRAIN,
                REQUEST_MULTI_TURN_TRAIN,
                REQUEST_MULTI_TURN_WITH_SYSTEM_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN_2,
                REQUEST_MULTI_TURN_IMAGE_URL_TRAIN,
                REQUEST_MULTI_TURN_IMAGE_TRAIN,
            ],
        ),
        (
            TokenizerVersion.v13,
            ValidationMode.test,
            False,
            True,
            False,
            [
                REQUEST_ONE_TURN_TEST,
                REQUEST_MULTI_TURN_TEST,
                REQUEST_MULTI_TURN_WITH_SYSTEM_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST_2,
                REQUEST_MULTI_TURN_AUDIO_URL_TEST,
                REQUEST_MULTI_TURN_AUDIO_TEST,
            ],
        ),
        (
            TokenizerVersion.v13,
            ValidationMode.finetuning,
            False,
            True,
            False,
            [
                REQUEST_ONE_TURN_TRAIN,
                REQUEST_MULTI_TURN_TRAIN,
                REQUEST_MULTI_TURN_WITH_SYSTEM_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN_2,
                REQUEST_MULTI_TURN_AUDIO_URL_TRAIN,
                REQUEST_MULTI_TURN_AUDIO_TRAIN,
            ],
        ),
        (
            TokenizerVersion.v13,
            ValidationMode.test,
            False,
            False,
            True,
            [
                REQUEST_ONE_TURN_TEST,
                REQUEST_MULTI_TURN_TEST,
                REQUEST_MULTI_TURN_WITH_SYSTEM_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST_2,
                REQUEST_MULTI_TURN_THINKING_TEST
            ],
        ),
        (
            TokenizerVersion.v13,
            ValidationMode.finetuning,
            False,
            False,
            True,
            [
                REQUEST_ONE_TURN_TRAIN,
                REQUEST_MULTI_TURN_TRAIN,
                REQUEST_MULTI_TURN_WITH_SYSTEM_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN_2,
                REQUEST_MULTI_TURN_THINKING_TRAIN
            ],
        ),
        (
            TokenizerVersion.v13,
            ValidationMode.test,
            True,
            False,
            True,
            [
                REQUEST_ONE_TURN_TEST,
                REQUEST_MULTI_TURN_TEST,
                REQUEST_MULTI_TURN_WITH_SYSTEM_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST_2,
                REQUEST_MULTI_TURN_IMAGE_URL_TEST,
                REQUEST_MULTI_TURN_IMAGE_TEST,
                REQUEST_MULTI_TURN_IMAGE_AND_THINKING_TEST
            ],
        ),
        (
            TokenizerVersion.v13,
            ValidationMode.finetuning,
            True,
            False,
            True,
            [
                REQUEST_ONE_TURN_TRAIN,
                REQUEST_MULTI_TURN_TRAIN,
                REQUEST_MULTI_TURN_WITH_SYSTEM_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN,
                REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN_2,
                REQUEST_MULTI_TURN_IMAGE_URL_TRAIN,
                REQUEST_MULTI_TURN_IMAGE_TRAIN,
                REQUEST_MULTI_TURN_IMAGE_AND_THINKING_TRAIN
            ],
        ),
    ],
)
def test_chat_template(
    version: TokenizerVersion,
    mode: ValidationMode,
    image: bool,
    audio: bool,
    think: bool,
    conversations: list[ChatCompletionRequest],
) -> None:
    mistral_tokenizer = _get_mistral_tekkenizer(
        tokenizer_version=version, validation_mode=mode, image=image, audio=audio, think=think
    )
    chat_template = get_chat_template(version, image, audio, think)
    for conversation in conversations:
        if version == TokenizerVersion.v2:
            for message in conversation.messages:
                if isinstance(message, ToolMessage):
                    message.name = "tool"
        if version in [TokenizerVersion.v1, TokenizerVersion.v2] and isinstance(
            conversation.messages[0], SystemMessage
        ):
            instruct_conversation = InstructRequest(
                messages=conversation.messages[1:],
                system_prompt=conversation.messages[0].content,  # type: ignore[arg-type]
                available_tools=conversation.tools,
            )

            mistral_common_encoded = encode_instruct_mistral_common(
                mistral_tokenizer.instruct_tokenizer, instruct_conversation
            )
        else:
            mistral_common_encoded = encode_mistral_common(mistral_tokenizer, conversation)
        transformers_encoded = encode_transformers(chat_template, conversation)
        assert mistral_common_encoded == transformers_encoded
