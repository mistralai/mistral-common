r"""Shared instruct request builders and canonical conversations.

This module is the single source of truth for standard `ChatCompletionRequest`
inputs used across the test suite. It provides:

- Deterministic dummy multimodal inputs (`DUMMY_IMAGE`, `DUMMY_AUDIO`, ...) with fixed
  content and seeds, so requests are reproducible without being serialized.
- Reusable tool fixtures (`math_interpreter_tools`).
- The `REQUEST_*` conversation constants and `get_conversations` selector
  (folded from the former `tests/integrations/chat_templates/fixtures_data.py`),
  consumed by both the core tokenizer tests and the chat-template parity tests.
- A curated `registry_instruct_requests` mapping used by the golden registry.
"""

import base64
import json
from io import BytesIO

import numpy as np
from PIL import Image

from mistral_common.protocol.instruct.chunk import (
    AudioChunk,
    AudioURLChunk,
    ContentChunk,
    ImageChunk,
    ImageURLChunk,
    TextChunk,
    ThinkChunk,
)
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
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
from mistral_common.protocol.instruct.tool_calls import Function, FunctionCall, Tool, ToolCall
from mistral_common.protocol.instruct.validator import ValidationMode
from mistral_common.tokens.tokenizers.audio import Audio
from mistral_common.tokens.tokenizers.base import TokenizerVersion


def _create_dummy_image() -> Image.Image:
    r"""Create a simple dummy 2x2 red square image.

    Returns:
        A 2x2 RGB red `PIL.Image`.
    """
    return Image.new("RGB", (2, 2), color="red")


def _sin_wave(sampling_rate: int, duration: float) -> np.ndarray:
    r"""Generate a deterministic sine-wave array.

    Args:
        sampling_rate: Samples per second.
        duration: Duration in seconds.

    Returns:
        A 1-D float array of `int(duration * sampling_rate)` samples.
    """
    return np.sin(np.ones([int(duration * sampling_rate)]))


def _sample_audio() -> Audio:
    r"""Create a deterministic sample `Audio` instance.

    Returns:
        A 1-second, 44.1 kHz WAV `Audio` built from `_sin_wave`.
    """
    sampling_rate = 44100
    original_array = _sin_wave(sampling_rate, 1)
    return Audio(
        audio_array=original_array,
        sampling_rate=sampling_rate,
        format="wav",
    )


DUMMY_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/7/78/Red_Square_%282x2_Pixel%29.png"
DUMMY_IMAGE = _create_dummy_image()
DUMMY_AUDIO_URL = _sample_audio().to_base64("wav")
DUMMY_AUDIO = DUMMY_AUDIO_URL


def dummy_base64_image_url_chunk() -> ImageURLChunk:
    r"""Return an `ImageURLChunk` backed by an inline base64 PNG.

    Uses a 4x4 red PNG so no network access is required to decode it.

    Returns:
        An `ImageURLChunk` whose URL is a base64 data URI.
    """
    img = Image.new("RGB", (4, 4), "red")
    buf = BytesIO()
    img.save(buf, "PNG")
    data_url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
    return ImageURLChunk(image_url=data_url)


def dummy_audio_chunk() -> AudioChunk:
    r"""Return an `AudioChunk` with a deterministic silent 16 kHz clip.

    Returns:
        An `AudioChunk` of 1600 zero samples at 16 kHz, base64-encoded as WAV.
    """
    audio = Audio(audio_array=np.zeros(1600), sampling_rate=16000, format="wav")
    return AudioChunk(input_audio=audio.to_base64("wav"))


def dummy_audio_url_chunk() -> AudioURLChunk:
    r"""Return an `AudioURLChunk` mirroring `dummy_audio_chunk`.

    Returns:
        An `AudioURLChunk` whose URL is the base64 payload of `dummy_audio_chunk`.
    """
    chunk = dummy_audio_chunk()
    return AudioURLChunk(audio_url=str(chunk.input_audio))


def math_interpreter_tools() -> list[Tool]:
    r"""Return the shared ``math_interpreter`` single-tool list.

    Returns:
        A one-element list with the ``math_interpreter`` tool.
    """
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


_TOOLS = [
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
]

# -- Request fixtures --

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
    tools=_TOOLS,
)

REQUEST_MULTI_TURN_WITH_TOOLS_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="User says hello"),
        AssistantMessage(content="Assistant says hi"),
        UserMessage(content="User says how are you ?"),
        AssistantMessage(content="Assistant says hi"),
    ],
    tools=_TOOLS,
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
    tools=_TOOLS,
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
    tools=_TOOLS,
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
    tools=_TOOLS,
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
    tools=_TOOLS,
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
    tools=_TOOLS,
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
    tools=_TOOLS,
)

REQUEST_TOOL_THEN_USER_TEST = ChatCompletionRequest(  # type: ignore[type-var]
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
        UserMessage(content="What does that mean?"),
    ],
    tools=_TOOLS,
)

REQUEST_TOOL_THEN_USER_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
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
        UserMessage(content="What does that mean?"),
        AssistantMessage(content="The temperature is 32 degrees in San Francisco."),
    ],
    tools=_TOOLS,
)

REQUEST_TOOL_THEN_USER_FULL_LOOP_TEST = ChatCompletionRequest(  # type: ignore[type-var]
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
        ToolMessage(content="sunny", tool_call_id="023456789"),
        UserMessage(content="Now what about Tokyo?"),
        AssistantMessage(
            content=None,
            tool_calls=[
                ToolCall(
                    id="234567890",
                    function=FunctionCall(
                        name="tool1",
                        arguments={  # type: ignore[arg-type]
                            "location": "Tokyo, JP",
                        },
                    ),
                ),
            ],
        ),
        ToolMessage(content="28", tool_call_id="234567890"),
        AssistantMessage(content="San Francisco is 32 and sunny, Tokyo is 28."),
        UserMessage(content="Thanks!"),
    ],
    tools=_TOOLS,
)

REQUEST_TOOL_THEN_USER_FULL_LOOP_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
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
        ToolMessage(content="sunny", tool_call_id="023456789"),
        UserMessage(content="Now what about Tokyo?"),
        AssistantMessage(
            content=None,
            tool_calls=[
                ToolCall(
                    id="234567890",
                    function=FunctionCall(
                        name="tool1",
                        arguments={  # type: ignore[arg-type]
                            "location": "Tokyo, JP",
                        },
                    ),
                ),
            ],
        ),
        ToolMessage(content="28", tool_call_id="234567890"),
        AssistantMessage(content="San Francisco is 32 and sunny, Tokyo is 28."),
    ],
    tools=_TOOLS,
)

REQUEST_TOOL_THEN_USER_WITH_CONTENT_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="User says hello"),
        AssistantMessage(
            content="Let me check the weather for you.",
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
        ToolMessage(content="sunny", tool_call_id="023456789"),
        UserMessage(content="What does that mean?"),
    ],
    tools=_TOOLS,
)

REQUEST_TOOL_THEN_USER_WITH_CONTENT_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="User says hello"),
        AssistantMessage(
            content="Let me check the weather for you.",
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
        ToolMessage(content="sunny", tool_call_id="023456789"),
        UserMessage(content="What does that mean?"),
        AssistantMessage(content="It is 32 degrees and sunny in San Francisco."),
    ],
    tools=_TOOLS,
)

REQUEST_MULTI_TURN_IMAGE_URL_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(
            content=[
                TextChunk(text="User asks what is this image ?"),
                ImageURLChunk(image_url=DUMMY_IMAGE_URL),
                ImageURLChunk(image_url=DUMMY_IMAGE_URL),
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
                ImageURLChunk(image_url=DUMMY_IMAGE_URL),
                ImageURLChunk(image_url=DUMMY_IMAGE_URL),
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
                ImageChunk(image=DUMMY_IMAGE),
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
                ImageChunk(image=DUMMY_IMAGE),
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
                AudioURLChunk(audio_url=DUMMY_AUDIO_URL),
                AudioURLChunk(audio_url=DUMMY_AUDIO_URL),
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
                AudioURLChunk(audio_url=DUMMY_AUDIO_URL),
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
                AudioChunk(input_audio=DUMMY_AUDIO),
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
                AudioChunk(input_audio=DUMMY_AUDIO),
                AudioChunk(input_audio=DUMMY_AUDIO),
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
                ImageURLChunk(image_url=DUMMY_IMAGE_URL),
            ]
        ),
        AssistantMessage(
            content=[
                ThinkChunk(thinking="Assistant thinks it's a red square."),
                TextChunk(text="Assistant says wow I need to think."),
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
                ThinkChunk(thinking="Assistant thinks it's a red square."),
                TextChunk(text="Assistant says wow I need to think."),
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
                ThinkChunk(thinking="Assistant thinks it's a red square."),
                TextChunk(text="Assistant says wow I need to think."),
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
                ImageURLChunk(image_url=DUMMY_IMAGE_URL),
            ]
        ),
        AssistantMessage(
            content=[
                ThinkChunk(thinking="Assistant thinks it's a red square."),
                TextChunk(text="Assistant says wow I need to think."),
                TextChunk(text="Assistant says it is a red square."),
            ],
            tool_calls=[],
        ),
        UserMessage(content=[TextChunk(text="User says thanks.")]),
        AssistantMessage(content=[TextChunk(text="Assistant says you're welcome.")]),
    ],
)


# -- Message aggregation test fixtures --

REQUEST_CONSECUTIVE_USERS_TEST = ChatCompletionRequest(
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
                ImageChunk(image=DUMMY_IMAGE),
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

REQUEST_CONSECUTIVE_ASSISTANTS_TEXT_CHUNKS_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(content="Hello"),
        AssistantMessage(content="First as string"),
        AssistantMessage(content=[TextChunk(text="Second as chunk")]),
        AssistantMessage(content=[TextChunk(text="Third part A"), TextChunk(text="Third part B")]),
        UserMessage(content="Thanks"),
        AssistantMessage(content="Response"),
    ]
)

REQUEST_CONSECUTIVE_USERS_MULTI_IMAGE_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(
            content=[TextChunk(text="Describe this"), ImageChunk(image=DUMMY_IMAGE), TextChunk(text="What color?")]
        ),
        UserMessage(
            content=[TextChunk(text="Also this"), ImageChunk(image=DUMMY_IMAGE), TextChunk(text="What shape?")]
        ),
        AssistantMessage(content="Both are red squares."),
    ]
)

REQUEST_CONSECUTIVE_USERS_AUDIO_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(
            content=[
                TextChunk(text="Listen to this"),
                AudioURLChunk(audio_url=DUMMY_AUDIO_URL),
                TextChunk(text="What language?"),
            ]
        ),
        UserMessage(
            content=[
                TextChunk(text="And this"),
                AudioURLChunk(audio_url=DUMMY_AUDIO_URL),
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
                ThinkChunk(thinking="Let me think..."),
                TextChunk(text="Hmm."),
                TextChunk(text="I need more context."),
            ]
        ),
        AssistantMessage(
            content=[
                ThinkChunk(thinking="Now I understand."),
                TextChunk(text="OK."),
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

REQUEST_MID_CONV_SYSTEM_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(content="Hello"),
        SystemMessage(content="New instruction."),
        AssistantMessage(content="Got it"),
    ]
)

REQUEST_MID_CONV_SYSTEM_WITH_CONSECUTIVE_USERS_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="Be helpful."),
        UserMessage(content="Hello"),
        UserMessage(content="World"),
        SystemMessage(content="Now be concise."),
        AssistantMessage(content="Got it"),
    ]
)

# -- Multimodal content in non-user messages (v15+) --

REQUEST_TOOL_IMAGE_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(content="What is in this image?"),
        AssistantMessage(
            content=None,
            tool_calls=[
                ToolCall(
                    id="tl1mg2345",
                    function=FunctionCall(
                        name="tool1",
                        arguments={"location": "San Francisco, CA"},  # type: ignore[arg-type]
                    ),
                ),
            ],
        ),
        ToolMessage(
            content=[
                TextChunk(text="Here is the result."),
                ImageURLChunk(image_url=DUMMY_IMAGE_URL),
            ],
            tool_call_id="tl1mg2345",
        ),
        AssistantMessage(content="The tool returned an image of a red square."),
    ],
    tools=_TOOLS,
)

REQUEST_TOOL_AUDIO_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(content="What does this sound like?"),
        AssistantMessage(
            content=None,
            tool_calls=[
                ToolCall(
                    id="tl1mg2345",
                    function=FunctionCall(
                        name="tool1",
                        arguments={"location": "San Francisco, CA"},  # type: ignore[arg-type]
                    ),
                ),
            ],
        ),
        ToolMessage(
            content=[
                TextChunk(text="Here is the audio result."),
                AudioChunk(input_audio=DUMMY_AUDIO),
            ],
            tool_call_id="tl1mg2345",
        ),
        AssistantMessage(content="The tool returned audio of a sine wave."),
    ],
    tools=_TOOLS,
)

REQUEST_SYSTEM_AUDIO_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(
            content=[
                TextChunk(text="You are an audio assistant."),
                AudioChunk(input_audio=DUMMY_AUDIO),
            ],
        ),
        UserMessage(content="What was that sound?"),
        AssistantMessage(content="That was a sine wave tone."),
    ],
)


def get_conversations(
    tokenizer_version: TokenizerVersion,
    validation_mode: ValidationMode,
    image: bool,
    audio: bool,
    think: bool,
) -> list[ChatCompletionRequest]:
    r"""Build a list of test conversations for the given configuration.

    Conversations are selected based on the tokenizer version (controls tool
    call and aggregation scenarios), validation mode (test vs finetuning), and
    modality flags (image, audio, think).

    Args:
        tokenizer_version: Determines which tool/aggregation scenarios are included.
        validation_mode: Test mode selects inference-style requests, finetuning
            mode selects training-style requests with additional aggregation fixtures.
        image: Whether to include image-related conversations.
        audio: Whether to include audio-related conversations.
        think: Whether to include thinking-related conversations.

    Returns:
        Deep-copied list of `ChatCompletionRequest` instances for the given
        configuration.
    """
    conversations: list[ChatCompletionRequest] = (
        [
            REQUEST_ONE_TURN_TEST,
            REQUEST_ONE_TURN_WITH_SYSTEM_TEST,
            REQUEST_MULTI_TURN_TEST,
            REQUEST_MULTI_TURN_WITH_SYSTEM_TEST,
        ]
        if validation_mode == ValidationMode.test
        else [
            REQUEST_ONE_TURN_TRAIN,
            REQUEST_ONE_TURN_WITH_SYSTEM_TRAIN,
            REQUEST_MULTI_TURN_TRAIN,
            REQUEST_MULTI_TURN_WITH_SYSTEM_TRAIN,
        ]
    )

    if tokenizer_version > TokenizerVersion.v1:
        if validation_mode == ValidationMode.test:
            conversations.extend(
                [
                    REQUEST_MULTI_TURN_WITH_TOOLS_TEST,
                    REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST,
                    REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST_2,
                    REQUEST_TOOL_THEN_USER_TEST,
                    REQUEST_TOOL_THEN_USER_FULL_LOOP_TEST,
                ]
            )
        else:
            conversations.extend(
                [
                    REQUEST_MULTI_TURN_WITH_TOOLS_TRAIN,
                    REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN,
                    REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN_2,
                    REQUEST_TOOL_THEN_USER_TRAIN,
                    REQUEST_TOOL_THEN_USER_FULL_LOOP_TRAIN,
                ]
            )
    if tokenizer_version > TokenizerVersion.v7:
        if validation_mode == ValidationMode.test:
            conversations.extend(
                [
                    REQUEST_MULTI_TURN_WITH_CONTENT_AND_TOOLS_CALLS_TEST,
                    REQUEST_TOOL_THEN_USER_WITH_CONTENT_TEST,
                ]
            )
        else:
            conversations.extend(
                [
                    REQUEST_MULTI_TURN_WITH_CONTENT_AND_TOOLS_CALLS_TRAIN,
                    REQUEST_TOOL_THEN_USER_WITH_CONTENT_TRAIN,
                ]
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
        conversations.extend(
            [
                REQUEST_CONSECUTIVE_USERS_TRAIN,
                REQUEST_CONSECUTIVE_USERS_WITH_SYSTEM_TRAIN,
                REQUEST_CONSECUTIVE_ASSISTANTS_TRAIN,
                REQUEST_MULTIPLE_SYSTEMS_TRAIN,
            ]
        )
        if tokenizer_version >= TokenizerVersion.v3:
            conversations.extend(
                [
                    REQUEST_CONSECUTIVE_USERS_TEXT_CHUNKS_TRAIN,
                    REQUEST_CONSECUTIVE_ASSISTANTS_TEXT_CHUNKS_TRAIN,
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

    # v7+ only: mid-conversation system messages and combined aggregation scenarios
    if tokenizer_version >= TokenizerVersion.v7 and validation_mode == ValidationMode.finetuning:
        conversations.extend(
            [
                REQUEST_MID_CONV_SYSTEM_TRAIN,
                REQUEST_MID_CONV_SYSTEM_WITH_CONSECUTIVE_USERS_TRAIN,
                REQUEST_CONSECUTIVE_ASSISTANTS_TOOL_CALLS_TRAIN,
            ]
        )

    # v15+ only: multimodal content in non-user messages (finetuning only)
    if tokenizer_version >= TokenizerVersion.v15 and validation_mode == ValidationMode.finetuning:
        if image:
            conversations.extend(
                [
                    REQUEST_TOOL_IMAGE_TRAIN,
                ]
            )
        if audio:
            conversations.append(REQUEST_SYSTEM_AUDIO_TRAIN)
            conversations.append(REQUEST_TOOL_AUDIO_TRAIN)

    conversations = [c.model_copy(deep=True) for c in conversations]

    if think and tokenizer_version >= TokenizerVersion.v15:
        for conv in conversations:
            for message in conv.messages:
                if isinstance(message, SystemMessage) and isinstance(message.content, list):
                    message.content = [
                        TextChunk(text="\n".join([c.text for c in message.content if isinstance(c, TextChunk)]))
                    ]

    return conversations


# -- Golden registry request set --

_REGISTRY_IMAGE = Image.new("RGB", (64, 64), color="red")


def registry_instruct_requests() -> dict[str, ChatCompletionRequest]:
    r"""Return the curated text-only requests encoded by the golden registry.

    These inference-style (test-mode) requests are valid across the shipped
    tokenizer versions and give the golden registry stable, real-vocab token
    ids and decoded text to assert against.

    Returns:
        Ordered mapping from request name to `ChatCompletionRequest`.
    """
    return {
        "single_turn": REQUEST_ONE_TURN_TEST.model_copy(deep=True),
        "single_turn_system": REQUEST_ONE_TURN_WITH_SYSTEM_TEST.model_copy(deep=True),
        "multi_turn": REQUEST_MULTI_TURN_TEST.model_copy(deep=True),
        "multi_turn_system": REQUEST_MULTI_TURN_WITH_SYSTEM_TEST.model_copy(deep=True),
        "multi_turn_tools": REQUEST_MULTI_TURN_WITH_TOOLS_TEST.model_copy(deep=True),
        "tool_calls": REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST.model_copy(deep=True),
    }


def registry_image_request() -> ChatCompletionRequest:
    r"""Return the single-image request whose processed array the registry stores.

    Uses an in-memory 64x64 red `ImageChunk` so no network access is required.

    Returns:
        A one-turn `ChatCompletionRequest` carrying a single image chunk.
    """
    return ChatCompletionRequest(
        messages=[
            UserMessage(
                content=[
                    TextChunk(text="What is this image ?"),
                    ImageChunk(image=_REGISTRY_IMAGE),
                ]
            )
        ]
    )


# -- Small structural builders for unit-level tokenizer tests --


def simple_tool(name: str = "tool1", description: str = "1") -> Tool:
    r"""Build a minimal parameterless `Tool`.

    Args:
        name: The function name.
        description: The function description.

    Returns:
        A `Tool` with no parameters.
    """
    return Tool(function=Function(name=name, description=description, parameters={}))


def abcd_messages(turns: int = 2) -> list[UserMessage | AssistantMessage]:
    r"""Build the canonical alternating ``a``/``b``/``c``/``d`` message list.

    One turn is a user message followed by an assistant message, using the
    single-character contents the low-level tokenizer tests assert on.

    Args:
        turns: Number of user/assistant turns, from 1 to 2.

    Returns:
        The alternating message list.

    Raises:
        ValueError: If `turns` is not 1 or 2.

    Examples:
        >>> [m.content for m in abcd_messages(turns=1)]
        ['a', 'b']
    """
    if turns not in (1, 2):
        raise ValueError(f"turns must be 1 or 2, got {turns}")
    messages: list[UserMessage | AssistantMessage] = [UserMessage(content="a"), AssistantMessage(content="b")]
    if turns == 2:
        messages += [UserMessage(content="c"), AssistantMessage(content="d")]
    return messages


def abcd_trailing_user_messages() -> list[UserMessage | AssistantMessage]:
    r"""Build the canonical ``a``/``b``/``c`` messages ending in a user turn.

    Used to exercise the invalid case of continuing the final message when the last
    message is not from the assistant.

    Returns:
        The three-message list ``[UserMessage("a"), AssistantMessage("b"), UserMessage("c")]``.
    """
    return [UserMessage(content="a"), AssistantMessage(content="b"), UserMessage(content="c")]


def single_user_message(content: str | list[ContentChunk]) -> list[UserMessage]:
    r"""Build a one-element message list wrapping arbitrary content in a lone user turn.

    Args:
        content: Text or content chunks for the sole user message.

    Returns:
        A one-element list containing a `UserMessage` with the given content.

    Examples:
        >>> [m.content for m in single_user_message(content="a")]
        ['a']
    """
    return [UserMessage(content=content)]


def text_image_user_message(text: str = "a") -> UserMessage:
    r"""Build a user message mixing a text chunk with a red 4x4 image chunk.

    Args:
        text: The text chunk's content.

    Returns:
        A `UserMessage` combining a `TextChunk` and an `ImageChunk`.
    """
    return UserMessage(content=[TextChunk(text=text), ImageChunk(image=Image.new("RGB", (4, 4), "red"))])


def tool_response_messages(
    tool_content: str | list[ContentChunk], *, tool_call_id: str | None = None
) -> list[UserMessage | AssistantMessage | ToolMessage]:
    r"""Build the shared user/assistant-tool-call/tool-result conversation.

    Covers the tool-response format tests across tokenizer versions that match tool
    calls by name only (``tool_call_id=None``) and versions that require an explicit id
    on both the call and its result.

    Args:
        tool_content: Content of the tool result message, in the shapes exercised by the
            tool-response format tests (plain text, JSON string, or content chunks).
        tool_call_id: The shared id threaded through the call and its result, or `None`
            to omit ids entirely.

    Returns:
        The three-message ``[UserMessage("a"), AssistantMessage(tool_calls=...), ToolMessage]``
        conversation.
    """
    if tool_call_id is not None:
        tool_call = ToolCall(id=tool_call_id, function=FunctionCall(name="b", arguments="{}"))
        tool_message = ToolMessage(name="b", content=tool_content, tool_call_id=tool_call_id)
    else:
        tool_call = ToolCall(function=FunctionCall(name="b", arguments="{}"))
        tool_message = ToolMessage(name="b", content=tool_content)
    return [
        UserMessage(content="a"),
        AssistantMessage(content=None, tool_calls=[tool_call]),
        tool_message,
    ]


def tool_multiple_calls_messages() -> list[UserMessage | AssistantMessage | ToolMessage]:
    r"""Build a two-round conversation with two parallel tool calls per round.

    Returns:
        The seven-message conversation used to test multiple simultaneous tool calls
        across two exchanges.
    """
    return [
        UserMessage(content="a"),
        AssistantMessage(
            tool_calls=[
                ToolCall(id="0", function=FunctionCall(name="b", arguments="{}")),
                ToolCall(id="1", function=FunctionCall(name="q", arguments="{}")),
            ]
        ),
        ToolMessage(name="b", content="d", tool_call_id="0"),
        ToolMessage(name="q", content="d", tool_call_id="1"),
        AssistantMessage(content="e"),
        UserMessage(content="f"),
        AssistantMessage(
            tool_calls=[
                ToolCall(id="2", function=FunctionCall(name="b", arguments="{}")),
                ToolCall(id="3", function=FunctionCall(name="q", arguments="{}")),
            ]
        ),
        ToolMessage(name="b", content="d", tool_call_id="2"),
        ToolMessage(name="q", content="d", tool_call_id="3"),
    ]


def system_and_user_chunks_request() -> ChatCompletionRequest:
    r"""Build a system-prompt conversation whose user turn carries two text chunks.

    Shared by the v3 multimodal tests that assert multimodal and text-only tokenizers
    agree on plain-text conversations.

    Returns:
        A `ChatCompletionRequest` with a system message, a two-chunk user message, an
        assistant reply, and a final user message.
    """
    return ChatCompletionRequest(
        messages=[
            SystemMessage(content="You are an AI assistant"),
            UserMessage(content=[TextChunk(text="aaa"), TextChunk(text="bbb")]),
            AssistantMessage(content="aaa"),
            UserMessage(content="goodbye"),
        ]
    )


def _hello_goodbye_user_messages(*, include_chunked_reply: bool) -> list[UserMessage | AssistantMessage]:
    r"""Build the alternating ``hello``/``aaa``/``goodbye`` turns shared by v3 mm alignment requests.

    Args:
        include_chunked_reply: Whether to insert a two-chunk user turn between ``hello``
            and ``aaa``, the only difference between `text_alignment_requests` and
            `text_requests`.

    Returns:
        The message list for the first request of `text_alignment_requests`/`text_requests`.
    """
    messages: list[UserMessage | AssistantMessage] = [UserMessage(content="hello")]
    if include_chunked_reply:
        messages.append(UserMessage(content=[TextChunk(text="bbb"), TextChunk(text="ccc")]))
    messages += [AssistantMessage(content="aaa"), UserMessage(content="goodbye")]
    return messages


def _text_alignment_trailer_requests() -> list[ChatCompletionRequest]:
    r"""Build the three requests shared verbatim by `text_alignment_requests` and `text_requests`.

    Returns:
        A single short user turn, an empty text chunk, and `system_and_user_chunks_request`.
    """
    return [
        ChatCompletionRequest(messages=single_user_message(content="hello")),
        ChatCompletionRequest(messages=single_user_message(content=[TextChunk(text="")])),
        system_and_user_chunks_request(),
    ]


def text_alignment_requests() -> list[ChatCompletionRequest]:
    r"""Build the text-only conversations exercised by the v3 mm/text-only agreement test.

    Each conversation must tokenize identically whether encoded by an image-capable v3
    tokenizer or a text-only v3 tokenizer.

    Returns:
        Four requests: a chunked-reply conversation, a single short user turn, an empty
        text chunk, and `system_and_user_chunks_request`.
    """
    return [
        ChatCompletionRequest(messages=_hello_goodbye_user_messages(include_chunked_reply=True)),
        *_text_alignment_trailer_requests(),
    ]


def text_requests() -> list[ChatCompletionRequest]:
    r"""Build `text_alignment_requests` without its two-chunk user turn.

    Paired with `image_alignment_requests` to exercise the mm normalizer on text-only
    conversations.

    Returns:
        The same requests as `text_alignment_requests`, minus the two-chunk user turn.
    """
    return [
        ChatCompletionRequest(messages=_hello_goodbye_user_messages(include_chunked_reply=False)),
        *_text_alignment_trailer_requests(),
    ]


def image_alignment_requests() -> list[ChatCompletionRequest]:
    r"""Build the multimodal conversations exercising image/text chunk ordering for v3.

    Covers a single leading image, image-after-text, image-before-text, multiple images
    across turns, and a longer text/image interleaving, all sharing one 4x4 red image.

    Returns:
        Five requests spanning the multimodal ordering cases used by the v3 mm normalizer
        and image-tokenization-integration tests.
    """
    img = Image.new("RGB", (4, 4), "red")
    return [
        ChatCompletionRequest(
            messages=[
                UserMessage(content=[TextChunk(text="a"), ImageChunk(image=img)]),
            ],
        ),
        ChatCompletionRequest(
            messages=[
                SystemMessage(content="A B"),
                UserMessage(content=[TextChunk(text="C"), ImageChunk(image=img)]),
            ]
        ),
        ChatCompletionRequest(
            messages=[
                SystemMessage(content="A B"),
                UserMessage(content=[ImageChunk(image=img), TextChunk(text="C")]),
            ]
        ),
        ChatCompletionRequest(
            messages=[
                SystemMessage(content="A B"),
                UserMessage(
                    content=[
                        ImageChunk(image=img),
                        ImageChunk(image=img),
                        TextChunk(text="C"),
                    ]
                ),
                AssistantMessage(content="D"),
                UserMessage(
                    content=[
                        ImageChunk(image=img),
                        TextChunk(text="E"),
                        ImageChunk(image=img),
                    ]
                ),
            ]
        ),
        ChatCompletionRequest(
            messages=[
                UserMessage(
                    content=[
                        TextChunk(text="A"),
                        ImageChunk(image=img),
                        TextChunk(text="B"),
                        TextChunk(text="C"),
                        ImageChunk(image=img),
                        TextChunk(text="D"),
                        TextChunk(text="E"),
                    ]
                ),
            ]
        ),
    ]


def tool_call_messages(
    *,
    think: bool = False,
    swap_tool_results: bool = False,
) -> list[SystemMessage | UserMessage | AssistantMessage | ToolMessage]:
    r"""Build the shared system/user/tool-call/tool-result conversation.

    The base shape is ``S`` / ``U1`` / ``A1`` with two tool calls / ``R1`` / ``R2`` /
    ``A2`` / ``U2``, shared by the v13 and v15 instruct tokenizer tests.

    Args:
        think: Whether the system and second assistant messages carry think chunks.
        swap_tool_results: Whether to emit ``R2`` before ``R1``, which is valid but
            gets reordered by normalization.

    Returns:
        The conversation message list.
    """
    system: SystemMessage = (
        SystemMessage(content=[TextChunk(text="S1"), ThinkChunk(thinking="TS"), TextChunk(text="S2")])
        if think
        else SystemMessage(content="S")
    )
    final_assistant: AssistantMessage = (
        AssistantMessage(content=[ThinkChunk(thinking="T1"), TextChunk(text="A2")])
        if think
        else AssistantMessage(content="A2")
    )
    results = [
        ToolMessage(content="R1", tool_call_id="123456789"),
        ToolMessage(content="R2", tool_call_id="999999999"),
    ]
    if swap_tool_results:
        results.reverse()
    return [
        system,
        UserMessage(content="U1"),
        AssistantMessage(
            content="A1",
            tool_calls=[
                ToolCall(id="123456789", function=FunctionCall(name="F1", arguments="{}")),
                ToolCall(id="999999999", function=FunctionCall(name="F2", arguments="{}")),
            ],
        ),
        *results,
        final_assistant,
        UserMessage(content="U2"),
    ]


def multimodal_tool_request(content_chunk: ContentChunk) -> ChatCompletionRequest:
    r"""Build a tool-result request whose tool message carries a multimodal chunk.

    Args:
        content_chunk: The audio or image chunk appended to the tool result.

    Returns:
        A `ChatCompletionRequest` exercising multimodal tool results.
    """
    return ChatCompletionRequest(
        messages=[
            UserMessage(content="Use the tool"),
            AssistantMessage(tool_calls=[ToolCall(id="test12345", function=FunctionCall(name="fn", arguments="{}"))]),
            ToolMessage(content=[TextChunk(text="result"), content_chunk], tool_call_id="test12345"),
        ],
        tools=[Tool(function=Function(name="fn", description="test", parameters={}))],
    )


def multimodal_system_request(content_chunk: ContentChunk) -> ChatCompletionRequest:
    r"""Build a request whose system message carries a multimodal chunk.

    Args:
        content_chunk: The audio or image chunk appended to the system message.

    Returns:
        A `ChatCompletionRequest` exercising multimodal system content.
    """
    return ChatCompletionRequest(
        messages=[
            SystemMessage(content=[TextChunk(text="System with content"), content_chunk]),
            UserMessage(content="Hello"),
        ],
    )


def multimodal_user_request(content_chunk: ContentChunk) -> ChatCompletionRequest:
    r"""Build a request whose user message carries a multimodal chunk.

    Args:
        content_chunk: The audio or image chunk appended to the user message.

    Returns:
        A `ChatCompletionRequest` exercising multimodal user content.
    """
    return ChatCompletionRequest(
        messages=[UserMessage(content=[TextChunk(text="Here is content"), content_chunk])],
    )


def tool_call_chat_request(*, think: bool = False, swap_tool_results: bool = False) -> ChatCompletionRequest:
    r"""Build the shared tool-call conversation as a `ChatCompletionRequest`.

    Args:
        think: Whether the system and second assistant messages carry think chunks.
        swap_tool_results: Whether to emit ``R2`` before ``R1``.

    Returns:
        A `ChatCompletionRequest` with the `math_interpreter` tool available.
    """
    return ChatCompletionRequest(
        messages=tool_call_messages(think=think, swap_tool_results=swap_tool_results),
        tools=math_interpreter_tools(),
    )


def abcd_multi_turn_request() -> InstructRequest:
    r"""Build the canonical abcd conversation as a raw `InstructRequest`.

    Returns:
        An `InstructRequest` with no system prompt, wrapping `abcd_messages`.
    """
    return InstructRequest(messages=abcd_messages())


def abcd_system_single_turn_request() -> InstructRequest:
    r"""Build a single-turn `InstructRequest` carrying a system prompt.

    Returns:
        An `InstructRequest` with `system_prompt="SYSTEM"` and a single user turn.
    """
    return InstructRequest(messages=[UserMessage(content="a")], system_prompt="SYSTEM")


def abcd_system_multi_turn_request() -> InstructRequest:
    r"""Build the canonical abcd conversation as a raw `InstructRequest` with a system prompt.

    Returns:
        An `InstructRequest` with `system_prompt="SYSTEM"`, wrapping `abcd_messages`.
    """
    return InstructRequest(messages=abcd_messages(), system_prompt="SYSTEM")


def abcd_system_multi_turn_continue_request() -> InstructRequest:
    r"""Build the canonical abcd conversation as a continued raw `InstructRequest`.

    Returns:
        An `InstructRequest` with `system_prompt="SYSTEM"` and `continue_final_message=True`.
    """
    return InstructRequest(messages=abcd_messages(), system_prompt="SYSTEM", continue_final_message=True)


def abcd_single_turn_continue_request() -> ChatCompletionRequest:
    r"""Build a single-turn abcd conversation as a continued `ChatCompletionRequest`.

    Returns:
        A `ChatCompletionRequest` with no system prompt, a single abcd turn, and
        `continue_final_message=True`.
    """
    return ChatCompletionRequest(messages=abcd_messages(turns=1), continue_final_message=True)


def abcd_multi_turn_tools_request() -> InstructRequest:
    r"""Build the canonical abcd conversation as a raw `InstructRequest` with two tools.

    Returns:
        An `InstructRequest` wrapping `abcd_messages` with `tool1` and `tool2` available.
    """
    return InstructRequest(
        messages=abcd_messages(), available_tools=[simple_tool(), simple_tool(name="tool2", description="2")]
    )


def assistant_prefix_tool_call_request() -> InstructRequest:
    r"""Build a single-message prefix `InstructRequest` carrying only a tool call.

    Returns:
        An `InstructRequest` with one `prefix=True` assistant message calling `test_fn`.
    """
    return InstructRequest(
        messages=[
            AssistantMessage(
                content=None,
                tool_calls=[ToolCall(function=FunctionCall(name="test_fn", arguments="{}"))],
                prefix=True,
            )
        ]
    )


def v11_continue_final_message_request() -> InstructRequest:
    r"""Build a single-message continued `InstructRequest` for v11.

    Returns:
        An `InstructRequest` with one assistant message and `continue_final_message=True`.
    """
    return InstructRequest(messages=[AssistantMessage(content='"blabla"')], continue_final_message=True)


def v11_plain_text_think_request() -> ChatCompletionRequest:
    r"""Build a request whose assistant message spells out thinking as literal text tags.

    v11 has no `[THINK]`/`[/THINK]` special tokens, so a plain-text thinking mode wraps the
    reasoning in literal `<think>`/`</think>` text instead; v11 assigns the tags no special
    meaning and encodes them like any other text.

    Returns:
        A `ChatCompletionRequest` whose assistant message content is plain text carrying
        literal think tags.
    """
    return ChatCompletionRequest(
        messages=[
            UserMessage(content="Solve 2+2."),
            AssistantMessage(content="<think>2+2 is 4.</think>The answer is 4.", prefix=True),
        ]
    )


def tool_call_instruct_request(
    *, reasoning_effort: ReasoningEffort | None = None, with_tools: bool = True
) -> InstructRequest:
    r"""Build the shared tool-call conversation as an `InstructRequest`.

    Args:
        reasoning_effort: Reasoning effort encoded in the model settings, or `None` for
            settings that encode nothing.
        with_tools: Whether the `math_interpreter` tool is made available.

    Returns:
        An `InstructRequest` carrying the shared conversation.
    """
    settings = (
        ModelSettings(reasoning_effort=reasoning_effort) if reasoning_effort is not None else ModelSettings.none()
    )
    return InstructRequest(
        messages=tool_call_messages(),
        available_tools=math_interpreter_tools() if with_tools else None,
        settings=settings,
    )


def _string_pair_tool() -> Tool:
    r"""Build the shared single-tool ``t`` accepting two string parameters.

    Returns:
        A `Tool` named ``t`` with ``g``/``h`` string properties.
    """
    return Tool(
        function=Function(
            name="t",
            parameters={
                "type": "object",
                "properties": {
                    "g": {"type": "string"},
                    "h": {"type": "string"},
                },
            },
        )
    )


def _tool_call_string_pair() -> ToolCall:
    r"""Build a `ToolCall` invoking `_string_pair_tool` with `{"g": "h"}` arguments.

    Returns:
        A `ToolCall` for the ``t`` tool with `{"g": "h"}` arguments.
    """
    return ToolCall(function=FunctionCall(name="t", arguments=json.dumps({"g": "h"}, ensure_ascii=False)))


def image_user_assistant_tool_result_request() -> InstructRequest:
    r"""Build an image user turn followed by an assistant reply and a tool result.

    Returns:
        An `InstructRequest` with `text_image_user_message`, an assistant reply, and a
        tool result message.
    """
    return InstructRequest(
        messages=[
            text_image_user_message(),
            AssistantMessage(content="b"),
            ToolMessage(tool_call_id="b", content="f"),
        ]
    )


def image_user_assistant_continue_final_message_request() -> InstructRequest:
    r"""Build a continued exchange starting with an image user turn.

    Returns:
        An `InstructRequest` with `text_image_user_message`, an assistant reply, and
        `continue_final_message=True`.
    """
    return InstructRequest(
        messages=[
            text_image_user_message(),
            AssistantMessage(content="b"),
        ],
        continue_final_message=True,
    )


def system_user_tool_call_request() -> InstructRequest:
    r"""Build a system/user turn followed by an assistant tool call.

    Returns:
        An `InstructRequest` with `_string_pair_tool` available and a single assistant
        tool call.
    """
    return InstructRequest(
        available_tools=[_string_pair_tool()],
        messages=[
            SystemMessage(content="a"),
            UserMessage(content="a"),
            AssistantMessage(content="b", tool_calls=[_tool_call_string_pair()]),
        ],
    )


def system_two_users_tool_call_result_request() -> InstructRequest:
    r"""Build a system/two-user turn with an assistant tool call and its result.

    Returns:
        An `InstructRequest` with `_string_pair_tool` available, two consecutive user
        turns, an assistant tool call, and its tool result.
    """
    return InstructRequest(
        available_tools=[_string_pair_tool()],
        messages=[
            SystemMessage(content="a"),
            UserMessage(content="a"),
            UserMessage(content="c"),
            AssistantMessage(content="b", tool_calls=[_tool_call_string_pair()]),
            ToolMessage(content="b", tool_call_id="1234"),
        ],
    )


def system_image_tool_result_chat_request() -> ChatCompletionRequest:
    r"""Build a chat completion request mixing a system tool and an image user turn.

    Returns:
        A `ChatCompletionRequest` with `_string_pair_tool`, an image user message, an
        assistant reply, and a tool result message.
    """
    return ChatCompletionRequest(
        tools=[_string_pair_tool()],
        messages=[
            SystemMessage(content="a"),
            text_image_user_message(),
            AssistantMessage(content="b"),
            ToolMessage(tool_call_id="123456789", content="f"),
        ],
    )


def single_turn_tool_request() -> InstructRequest:
    r"""Build a single-turn `InstructRequest` with one available tool.

    Returns:
        An `InstructRequest` wrapping `single_user_message` with `simple_tool` available.
    """
    return InstructRequest(messages=single_user_message(content="a"), available_tools=[simple_tool()])


def abcd_system_tools_multi_turn_request() -> InstructRequest:
    r"""Build the canonical abcd conversation as a raw `InstructRequest` with a tool and system prompt.

    Returns:
        An `InstructRequest` wrapping `abcd_messages` with `simple_tool` available and
        `system_prompt="SYSTEM"`.
    """
    return InstructRequest(messages=abcd_messages(), available_tools=[simple_tool()], system_prompt="SYSTEM")


def tool_response_plain_request() -> InstructRequest:
    r"""Build a tool-response `InstructRequest` with a plain-text tool result.

    Returns:
        An `InstructRequest` wrapping `tool_response_messages` with no tool call id.
    """
    return InstructRequest(messages=tool_response_messages("d"))


def tool_response_json_request() -> InstructRequest:
    r"""Build a tool-response `InstructRequest` with a JSON-string tool result.

    Returns:
        An `InstructRequest` wrapping `tool_response_messages` with no tool call id.
    """
    return InstructRequest(messages=tool_response_messages('{"a": 1}'))


def tool_response_chunks_request() -> InstructRequest:
    r"""Build a tool-response `InstructRequest` with a chunked tool result.

    Returns:
        An `InstructRequest` wrapping `tool_response_messages` with no tool call id.
    """
    return InstructRequest(messages=tool_response_messages([TextChunk(text="d"), TextChunk(text='{"a": 1}')]))


def tool_message_multiple_shots_without_history_request() -> InstructRequest:
    r"""Build a two-round, id-less tool-call `InstructRequest` with no shared history.

    Returns:
        An `InstructRequest` with two independent user/assistant-tool-call/tool-result
        rounds, none of the tool calls or results carrying an explicit id.
    """
    return InstructRequest(
        messages=[
            UserMessage(content="a"),
            AssistantMessage(tool_calls=[ToolCall(function=FunctionCall(name="b", arguments="{}"))]),
            ToolMessage(name="b", content="d"),
            AssistantMessage(content="e"),
            UserMessage(content="f"),
            AssistantMessage(tool_calls=[ToolCall(function=FunctionCall(name="b", arguments="{}"))]),
            ToolMessage(name="b", content="d"),
        ],
    )


def tool_message_plain_request() -> InstructRequest:
    r"""Build a tool-response `InstructRequest` with a plain-text result and an explicit id.

    Returns:
        An `InstructRequest` wrapping `tool_response_messages` with `tool_call_id="123456789"`.
    """
    return InstructRequest(messages=tool_response_messages("d", tool_call_id="123456789"))


def tool_message_json_request() -> InstructRequest:
    r"""Build a tool-response `InstructRequest` with a JSON-string result and an explicit id.

    Returns:
        An `InstructRequest` wrapping `tool_response_messages` with `tool_call_id="123456789"`.
    """
    return InstructRequest(messages=tool_response_messages('{"a": 1}', tool_call_id="123456789"))


def tool_message_chunks_request() -> InstructRequest:
    r"""Build a tool-response `InstructRequest` with a chunked result and an explicit id.

    Returns:
        An `InstructRequest` wrapping `tool_response_messages` with `tool_call_id="123456789"`.
    """
    return InstructRequest(
        messages=tool_response_messages([TextChunk(text="d"), TextChunk(text='{"a": 1}')], tool_call_id="123456789")
    )


def tool_call_null_id_request() -> InstructRequest:
    r"""Build a single tool-call `InstructRequest` whose call id is the literal string ``"null"``.

    Returns:
        An `InstructRequest` with one assistant tool call carrying `id="null"`.
    """
    return InstructRequest(
        messages=[
            UserMessage(content="a"),
            AssistantMessage(tool_calls=[ToolCall(id="null", function=FunctionCall(name="b", arguments="{}"))]),
        ],
    )


def tool_call_no_id_request() -> InstructRequest:
    r"""Build a single tool-call `InstructRequest` whose call carries no id.

    Returns:
        An `InstructRequest` with one assistant tool call carrying no explicit id.
    """
    return InstructRequest(
        messages=[
            UserMessage(content="a"),
            AssistantMessage(tool_calls=[ToolCall(function=FunctionCall(name="b", arguments="{}"))]),
        ],
    )


def tool_message_multiple_shots_with_history_request() -> InstructRequest:
    r"""Build a two-round tool-call `InstructRequest` sharing history via explicit ids.

    Returns:
        An `InstructRequest` with two user/assistant-tool-call/tool-result rounds, each
        tool call and its result sharing an explicit id (``"0"`` then ``"1"``).
    """
    return InstructRequest(
        messages=[
            UserMessage(content="a"),
            AssistantMessage(tool_calls=[ToolCall(id="0", function=FunctionCall(name="b", arguments="{}"))]),
            ToolMessage(name="b", content="d", tool_call_id="0"),
            AssistantMessage(content="e"),
            UserMessage(content="f"),
            AssistantMessage(tool_calls=[ToolCall(id="1", function=FunctionCall(name="b", arguments="{}"))]),
            ToolMessage(name="b", content="d", tool_call_id="1"),
        ],
    )


def tool_multiple_calls_request() -> InstructRequest:
    r"""Build the shared two-round, two-parallel-tool-call conversation as an `InstructRequest`.

    Returns:
        An `InstructRequest` wrapping `tool_multiple_calls_messages`.
    """
    return InstructRequest(messages=tool_multiple_calls_messages())


def v7_truncation_keep_sys_and_last_message_request() -> InstructRequest:
    r"""Build a truncation `InstructRequest` where both system messages and the last message survive.

    Returns:
        An `InstructRequest` with two system messages and a long final user message,
        truncated to 15 tokens.
    """
    return InstructRequest(
        messages=[
            SystemMessage(content="a"),
            UserMessage(content="c"),
            UserMessage(content="c"),
            SystemMessage(content="a"),
            UserMessage(content="bbbbbbb"),
        ],
        truncate_at_max_tokens=15,
    )


def v7_truncation_full_convo_request() -> InstructRequest:
    r"""Build a truncation `InstructRequest` spanning a full system/user/assistant conversation.

    Returns:
        An `InstructRequest` alternating system, user, and assistant messages, truncated
        to 15 tokens.
    """
    return InstructRequest(
        messages=[
            SystemMessage(content="a"),
            UserMessage(content="c"),
            AssistantMessage(content="c"),
            UserMessage(content="a"),
            AssistantMessage(content="a"),
            SystemMessage(content="b"),
            UserMessage(content="a"),
        ],
        truncate_at_max_tokens=15,
    )


def v7_assistant_tool_call_and_content_request() -> InstructRequest:
    r"""Build an `InstructRequest` whose assistant message carries both content and tool calls.

    Returns:
        An `InstructRequest` with two available tools and one assistant message combining
        text content with two tool calls.
    """
    return InstructRequest(
        available_tools=[
            Tool(function=Function(name="t1", parameters={})),
            Tool(function=Function(name="t2", parameters={})),
        ],
        messages=[
            UserMessage(content="a"),
            AssistantMessage(
                content="b1b2",
                tool_calls=[
                    ToolCall(id="000000000", function=FunctionCall(name="t1", arguments="{}")),
                    ToolCall(id="111111111", function=FunctionCall(name="t2", arguments="{}")),
                ],
            ),
        ],
    )


def v13_system_user_audio_request(audio_chunk: AudioChunk | AudioURLChunk) -> ChatCompletionRequest:
    r"""Build a request with a plain-text system message and an audio-only user message.

    Args:
        audio_chunk: The audio chunk carried by the user message.

    Returns:
        A `ChatCompletionRequest` with a system greeting and a user message containing
        only the given audio chunk.
    """
    return ChatCompletionRequest(
        messages=[
            SystemMessage(content="hello"),
            UserMessage(content=[audio_chunk]),
        ],
    )
