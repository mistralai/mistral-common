r"""Test data constants for chat template tests."""

from mistral_common.protocol.instruct.chunk import (
    AudioChunk,
    AudioURLChunk,
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
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import Function, FunctionCall, Tool, ToolCall
from mistral_common.protocol.instruct.validator import ValidationMode
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.integrations.chat_templates.helpers import _AUDIO, _AUDIO_URL, _IMAGE, _IMAGE_URL

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
                ImageURLChunk(image_url=_IMAGE_URL),
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

REQUEST_SYSTEM_AUDIO_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(
            content=[
                TextChunk(text="You are a transcription assistant. Listen to this context:"),
                AudioURLChunk(audio_url=_AUDIO_URL),
            ]
        ),
        UserMessage(content="Summarize what you heard"),
        AssistantMessage(content="The audio contains a brief conversation."),
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


def _get_conversations(
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
            if tokenizer_version >= TokenizerVersion.v15:
                conversations.append(REQUEST_SYSTEM_AUDIO_TRAIN)
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

    conversations = [c.model_copy(deep=True) for c in conversations]

    if think and tokenizer_version >= TokenizerVersion.v15:
        for conv in conversations:
            for message in conv.messages:
                if isinstance(message, SystemMessage) and isinstance(message.content, list):
                    message.content = [
                        TextChunk(text="\n".join([c.text for c in message.content if isinstance(c, TextChunk)]))
                    ]

    return conversations
