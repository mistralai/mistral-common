import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from jinja2.exceptions import TemplateError
from transformers.utils.chat_template_utils import render_jinja_template

from integrations.chat_templates.chat_templates import get_chat_template
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
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.normalize import get_normalizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest, InstructRequest
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
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.sentencepiece import SentencePieceTokenizer
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.test_tekken import get_special_tokens

ROOT_DIR = Path(__file__).parent.parent.parent
TEST_DIR = ROOT_DIR / "tests"


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
    audio_encoder = AudioEncoder(audio_config=audio_config, special_ids=SpecialAudioIDs(audio=24, begin_audio=25))
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
    return render_jinja_template(
        [openai_request["messages"]],
        tools=openai_request.get("tools", None),  # type: ignore[arg-type]
        chat_template=chat_template,
        bos_token="<s>",
        eos_token="</s>",
    )[0][0]


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
    chat_template = get_chat_template(spm, version, image, audio, think)
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
        if version in [TokenizerVersion.v1, TokenizerVersion.v2] and isinstance(
            conversation.messages[0], SystemMessage
        ):
            instruct_conversation = InstructRequest(
                messages=conversation.messages[1:],
                system_prompt=conversation.messages[0].content,  # type: ignore[arg-type]
                available_tools=conversation.tools,
            )

            mistral_common_encoded = encode_instruct_mistral_common(
                mistral_tokenizer.instruct_tokenizer, instruct_conversation, spm
            )
        else:
            mistral_common_encoded = encode_mistral_common(mistral_tokenizer, conversation, spm)
        transformers_encoded = encode_transformers(
            chat_template, conversation, keep_name_for_tools=version == TokenizerVersion.v2
        )

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
    ],
)
def test_role_error(
    spm: bool,
    version: TokenizerVersion,
    image: bool,
    audio: bool,
    think: bool,
) -> None:
    chat_template = get_chat_template(spm, version, image, audio, think)

    INVALID_ALTERNATE_CONVERSATION = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "Hello"},
        ]
    }

    INVALID_ROLE = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "invalid", "content": "Hello"},
        ]
    }

    alternate_match = (
        (
            r"After the optional system message, conversation roles must alternate user and assistant roles except for "
            r"tool calls and results."
        )
        if version > TokenizerVersion.v1
        else (r"After the optional system message, conversation roles must alternate user and assistant.")
    )

    role_match = (
        r"Only user, assistant and tool roles are supported, got invalid."
        if version > TokenizerVersion.v1
        else r"Only user and assistant roles are supported, got invalid."
    )

    with pytest.raises(expected_exception=TemplateError, match=alternate_match):
        encode_transformers(chat_template, INVALID_ALTERNATE_CONVERSATION)

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

    chat_template = get_chat_template(spm, version, image, audio, think)
    for conv in invalid_convs:
        msg_template = "Only {chunks} chunks are supported in {role} message content."
        if conv in SP_INVALIDS:
            chunks = "text and thinking" if think else "text"
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

    chat_template = get_chat_template(spm, version, image, audio, think)
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

    chat_template = get_chat_template(spm, version, image, audio, think)
    with pytest.raises(TemplateError, match="Assistant message cannot have both content and tool calls."):
        encode_transformers(chat_template, invalid_message_conv)
