import json
import pickle
from collections.abc import Callable
from typing import assert_type

import pytest
from PIL import Image

from mistral_common.exceptions import InvalidRequestException
from mistral_common.protocol.instruct.chunk import (
    AudioChunk,
    ChunkTypes,
    ImageChunk,
    ImageURLChunk,
    TextChunk,
    ThinkChunk,
)
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    ChatMessage,
    FinetuningAssistantMessage,
    FinetuningMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.normalize import (
    InstructRequestNormalizer,
    InstructRequestNormalizerV7,
    InstructRequestNormalizerV13,
    InstructRequestNormalizerV15,
    get_normalizer,
)
from mistral_common.protocol.instruct.request import (
    ChatCompletionRequest,
    InstructRequest,
    ModelSettings,
    ReasoningEffort,
)
from mistral_common.protocol.instruct.tool_calls import Function, FunctionCall, Tool, ToolCall
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from mistral_common.tokens.tokenizers.model_settings_builder import ModelSettingsBuilder, ReasoningEffortEnumBuilder

REQUEST: ChatCompletionRequest = ChatCompletionRequest(
    messages=[
        SystemMessage(content="Be brief."),
        UserMessage(content="Weather in Paris?"),
        AssistantMessage(
            tool_calls=[
                ToolCall(id="abcdefghi", function=FunctionCall(name="get_weather", arguments='{"city": "Paris"}'))
            ]
        ),
        ToolMessage(tool_call_id="abcdefghi", content="Sunny"),
    ],
    tools=[Tool(function=Function(name="get_weather", description="Get the weather", parameters={}))],
)


class TestNormalizerPickle:
    @pytest.mark.parametrize("version", list(TokenizerVersion))
    def test_normalizer_and_instruct_request_pickle(self, version: TokenizerVersion) -> None:
        builder = ModelSettingsBuilder() if version >= TokenizerVersion.v15 else None
        normalizer = get_normalizer(version, builder)
        instruct_request = normalizer.from_chat_completion_request(REQUEST)

        restored_request = pickle.loads(pickle.dumps(instruct_request))
        restored_normalizer = pickle.loads(pickle.dumps(normalizer))

        assert restored_request == instruct_request
        assert restored_normalizer.from_chat_completion_request(REQUEST) == instruct_request


# Valid JSON that `json.loads` rejects without a JSONDecodeError.
_UNLOADABLE_JSON = [
    pytest.param('{"a":' + "1" * 5000 + "}", id="oversized-int"),  # ValueError: over the int digit limit
    pytest.param("[" * 100_000 + "]" * 100_000, id="deep-nesting"),  # RecursionError
]


class TestUnloadableJsonPassthrough:
    @pytest.mark.parametrize(argnames="payload", argvalues=_UNLOADABLE_JSON)
    @pytest.mark.parametrize(
        argnames="version", argvalues=[TokenizerVersion.v3, TokenizerVersion.v7, TokenizerVersion.v13]
    )
    def test_keeps_unloadable_json_unchanged(self, version: TokenizerVersion, payload: str) -> None:
        r"""Pass tool-call arguments and tool content that json.loads rejects through unchanged."""
        request = ChatCompletionRequest[ChatMessage](
            messages=[
                UserMessage(content="a"),
                AssistantMessage(
                    tool_calls=[ToolCall(id="123456789", function=FunctionCall(name="b", arguments=payload))]
                ),
                ToolMessage(content=payload, tool_call_id="123456789"),
            ]
        )

        _, assistant, tool = get_normalizer(version=version).from_chat_completion_request(request).messages

        assert isinstance(assistant, AssistantMessage) and assistant.tool_calls is not None
        assert assistant.tool_calls[0].function.arguments == payload
        assert isinstance(tool, ToolMessage)
        assert tool.content == payload


def mock_chat_completion(messages: list[ChatMessage]) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test",
        messages=messages,
        top_p=1.0,
        temperature=0.7,
    )


class TestBaseNormalization:
    @pytest.fixture(autouse=True)
    def normalizer(self) -> InstructRequestNormalizer:
        return InstructRequestNormalizer.normalizer()

    def test_user_system_user(self, normalizer: InstructRequestNormalizer) -> None:
        chat_completion_request = mock_chat_completion(
            messages=[
                UserMessage(content="U"),
                SystemMessage(content="S"),
                UserMessage(content="U"),
            ]
        )

        parsed_request = normalizer.from_chat_completion_request(chat_completion_request)
        assert parsed_request == InstructRequest[ChatMessage, Tool](
            messages=[UserMessage(content="U"), UserMessage(content="U")],
            system_prompt="S",
        )

    def test_multiple_system(self, normalizer: InstructRequestNormalizer) -> None:
        chat_completion_request = mock_chat_completion(
            messages=[
                SystemMessage(content="S"),
                SystemMessage(content="S"),
                SystemMessage(content="S"),
            ]
        )

        parsed_request = normalizer.from_chat_completion_request(chat_completion_request)
        assert parsed_request == InstructRequest[ChatMessage, Tool](
            messages=[UserMessage(content="")],
            system_prompt="S\n\nS\n\nS",
        )

    def test_single_system(self, normalizer: InstructRequestNormalizer) -> None:
        chat_completion_request = mock_chat_completion(
            messages=[
                SystemMessage(content="S"),
            ]
        )

        parsed_request = normalizer.from_chat_completion_request(chat_completion_request)
        assert parsed_request == InstructRequest[ChatMessage, Tool](
            messages=[UserMessage(content="")],
            system_prompt="S",
        )

    def test_system_assistant_user(self, normalizer: InstructRequestNormalizer) -> None:
        chat_completion_request = mock_chat_completion(
            messages=[
                SystemMessage(content="S"),
                AssistantMessage(content="A"),
                UserMessage(content="U"),
            ]
        )

        parsed_request = normalizer.from_chat_completion_request(chat_completion_request)
        assert parsed_request == InstructRequest[ChatMessage, Tool](
            messages=[UserMessage(content=""), AssistantMessage(content="A"), UserMessage(content="U")],
            system_prompt="S",
        )

    def test_assistant_content_with_tool_calls(self, normalizer: InstructRequestNormalizer) -> None:
        chat_completion_request = mock_chat_completion(
            messages=[
                AssistantMessage(
                    content="A",
                    tool_calls=[ToolCall(function=FunctionCall(name="tool1", arguments='{"input": "11"}'))],
                )
            ]
        )
        with pytest.raises(ValueError, match="Tool calls and content cannot be used together in the same message"):
            normalizer.from_chat_completion_request(chat_completion_request)

    def test_assistant_system_user_adds_user(self, normalizer: InstructRequestNormalizer) -> None:
        chat_completion_request = mock_chat_completion(
            messages=[
                AssistantMessage(content="A"),
                SystemMessage(content="S"),
                UserMessage(content="U"),
            ]
        )

        parsed_request = normalizer.from_chat_completion_request(chat_completion_request)
        assert parsed_request == InstructRequest[ChatMessage, Tool](
            messages=[UserMessage(content=""), AssistantMessage(content="A"), UserMessage(content="U")],
            system_prompt="S",
        )

    def test_message_aggregation_system_then_user(self, normalizer: InstructRequestNormalizer) -> None:
        parsed = normalizer.from_chat_completion_request(
            mock_chat_completion(
                [
                    SystemMessage(content="s"),
                    SystemMessage(content="s"),
                    SystemMessage(content="s"),
                    UserMessage(content="u"),
                ]
            )
        )
        assert parsed == InstructRequest[ChatMessage, Tool](
            messages=[UserMessage(content="u")], system_prompt="s\n\ns\n\ns"
        )

    def test_message_aggregation_system_then_users(self, normalizer: InstructRequestNormalizer) -> None:
        parsed = normalizer.from_chat_completion_request(
            mock_chat_completion(
                [
                    SystemMessage(content="s"),
                    SystemMessage(content="s"),
                    SystemMessage(content="s"),
                    UserMessage(content="u"),
                    UserMessage(content="u"),
                ]
            )
        )
        assert parsed == InstructRequest[ChatMessage, Tool](
            messages=[UserMessage(content="u\n\nu")], system_prompt="s\n\ns\n\ns"
        )

    def test_message_aggregation_mixed_with_middle_system(self, normalizer: InstructRequestNormalizer) -> None:
        parsed = normalizer.from_chat_completion_request(
            mock_chat_completion(
                [
                    SystemMessage(content="s"),
                    SystemMessage(content="s"),
                    SystemMessage(content="s"),
                    UserMessage(content="u"),
                    UserMessage(content="u"),
                    SystemMessage(content="s"),
                    AssistantMessage(content="a"),
                    UserMessage(content="u"),
                ]
            )
        )
        assert parsed == InstructRequest[ChatMessage, Tool](
            messages=[UserMessage(content="u\n\nu"), AssistantMessage(content="a"), UserMessage(content="u")],
            system_prompt="s\n\ns\n\ns\n\ns",
        )

    def test_message_aggregation_consecutive_assistants(self, normalizer: InstructRequestNormalizer) -> None:
        parsed = normalizer.from_chat_completion_request(
            mock_chat_completion(
                [
                    SystemMessage(content="s"),
                    SystemMessage(content="s"),
                    SystemMessage(content="s"),
                    UserMessage(content="u"),
                    UserMessage(content="u"),
                    AssistantMessage(content="a"),
                    AssistantMessage(content="a"),
                    UserMessage(content="u"),
                ]
            )
        )
        assert parsed == InstructRequest[ChatMessage, Tool](
            messages=[UserMessage(content="u\n\nu"), AssistantMessage(content="a\n\na"), UserMessage(content="u")],
            system_prompt="s\n\ns\n\ns",
        )

    def test_message_aggregation_system_assistant_user(self, normalizer: InstructRequestNormalizer) -> None:
        parsed = normalizer.from_chat_completion_request(
            mock_chat_completion([SystemMessage(content="s"), AssistantMessage(content="a"), UserMessage(content="u")])
        )
        assert parsed == InstructRequest[ChatMessage, Tool](
            messages=[UserMessage(content=""), AssistantMessage(content="a"), UserMessage(content="u")],
            system_prompt="s",
        )

    def test_tool_chunk_aggregation(self, normalizer: InstructRequestNormalizer) -> None:
        messages = [
            ToolMessage(content="C", tool_call_id="1"),
            ToolMessage(content=[TextChunk(text='{"a": 2}')], tool_call_id="2"),
            ToolMessage(content=[TextChunk(text="B"), TextChunk(text="A")], tool_call_id="3"),
        ]

        expected = [
            ToolMessage(content="C", tool_call_id="1"),
            ToolMessage(content=json.dumps({"a": 2}), tool_call_id="2"),
            ToolMessage(content="B\n\nA", tool_call_id="3"),
        ]

        assert (
            normalizer._aggregate_tool_messages(
                messages, [tool_message.tool_call_id for tool_message in messages if tool_message.tool_call_id]
            )
            == expected
        )

    def test_normalize_chunks(self, normalizer: InstructRequestNormalizer) -> None:
        chat_completion_request = mock_chat_completion(
            messages=[
                UserMessage(content="foo"),
                UserMessage(
                    content=[TextChunk(type=ChunkTypes.text, text="chunk")],
                ),
                UserMessage(content="foo"),
                UserMessage(
                    content=[TextChunk(type=ChunkTypes.text, text="chunk")],
                ),
            ],
        )

        parsed_request = normalizer.from_chat_completion_request(chat_completion_request)
        assert parsed_request == InstructRequest[ChatMessage, Tool](
            messages=[UserMessage(content="foo\n\nchunk\n\nfoo\n\nchunk")]
        )

    def test_many_chunks_in_user_message(self, normalizer: InstructRequestNormalizer) -> None:
        chat_completion_request = mock_chat_completion(
            messages=[
                UserMessage(content="foo"),
                UserMessage(
                    content=[
                        TextChunk(type=ChunkTypes.text, text="chunk1"),
                        TextChunk(type=ChunkTypes.text, text="chunk2"),
                        TextChunk(type=ChunkTypes.text, text="chunk3"),
                    ],
                ),
            ],
        )
        parsed_request = normalizer.from_chat_completion_request(chat_completion_request)
        assert parsed_request == InstructRequest[ChatMessage, Tool](
            messages=[UserMessage(content="foo\n\nchunk1\n\nchunk2\n\nchunk3")]
        )

    def test_ignore_middle_empty_text_chunks(self, normalizer: InstructRequestNormalizer) -> None:
        chat_completion_request = mock_chat_completion(
            messages=[
                UserMessage(
                    content=[
                        TextChunk(text="U"),
                        TextChunk(text=""),
                        TextChunk(text="V"),
                    ],
                ),
                AssistantMessage(
                    content=[
                        TextChunk(text="A"),
                        TextChunk(text=""),
                        TextChunk(text="B"),
                    ],
                ),
            ]
        )
        parsed_request = normalizer.from_chat_completion_request(chat_completion_request)
        assert parsed_request == InstructRequest[ChatMessage, Tool](
            messages=[UserMessage(content="U\n\nV"), AssistantMessage(content="A\n\nB")],
        )

    def test_safety_prompt_aggregated(self, normalizer: InstructRequestNormalizer) -> None:
        chat_completion_request = ChatCompletionRequest[ChatMessage](
            model="test",
            messages=[
                UserMessage(content="user"),
                SystemMessage(content="system"),
            ],
            top_p=1.0,
            temperature=0.7,
        )

        parsed_request = normalizer.from_chat_completion_request(chat_completion_request)
        assert parsed_request == InstructRequest[ChatMessage, Tool](
            messages=[UserMessage(content="user")],
            system_prompt="system",
        )

    def test_normalize_tools(self, normalizer: InstructRequestNormalizer) -> None:
        tools = [
            Tool(function=Function(name="tool1", description="1", parameters={})),
            Tool(function=Function(name="tool2", description="2", parameters={})),
        ]

        request = ChatCompletionRequest[ChatMessage](
            model="triton",
            messages=[
                SystemMessage(content="helpful assistant"),
                UserMessage(content="a"),
                AssistantMessage(content="b"),
                UserMessage(content="c"),
            ],
            tools=tools,
        )

        gt = InstructRequest[ChatMessage, Tool](
            messages=[UserMessage(content="a"), AssistantMessage(content="b"), UserMessage(content="c")],
            available_tools=tools,
            system_prompt="helpful assistant",
        )

        normalized = normalizer.from_chat_completion_request(request)
        assert normalized == gt

    def test_normalize_funcalls(self, normalizer: InstructRequestNormalizer) -> None:
        request = ChatCompletionRequest[ChatMessage](
            model="triton",
            messages=[
                SystemMessage(content="helpful assistant"),
                UserMessage(content="a"),
                AssistantMessage(
                    tool_calls=[
                        ToolCall(
                            function=FunctionCall(name="tool1", arguments='\n\n{\n"input":\n      "11"\n}\n\n')
                        ),  # dodgy input json string
                    ],
                ),
                ToolMessage(
                    name="tool1",
                    content='{\n"output":\n      "11"\n}',  # dodgy input json string
                ),
            ],
            tools=[Tool(function=Function(name="tool1", description="1", parameters={}))],
        )

        gt = InstructRequest[ChatMessage, Tool](
            messages=[
                UserMessage(content="a"),
                AssistantMessage(
                    content="",
                    tool_calls=[
                        ToolCall(function=FunctionCall(name="tool1", arguments='{"input": "11"}')),  # clean json string
                    ],
                ),
                ToolMessage(
                    name="tool1",
                    content='{"output": "11"}',  # clean json string
                ),
            ],
            system_prompt="helpful assistant",
            available_tools=[Tool(function=Function(name="tool1", description="1", parameters={}))],
        )

        normalized = normalizer.from_chat_completion_request(request)
        assert normalized == gt

    def test_assert_parsed_settings(
        self,
        normalizer: InstructRequestNormalizer,
    ) -> None:
        chat_completion_request = ChatCompletionRequest(messages=[UserMessage(content="B")])
        parsed_request = normalizer.from_chat_completion_request(chat_completion_request)
        assert parsed_request == InstructRequest[ChatMessage, Tool](messages=[UserMessage(content="B")])

    def test_preserves_assistant_prefix_without_continuation_state(self, normalizer: InstructRequestNormalizer) -> None:
        request = ChatCompletionRequest[ChatMessage](
            messages=[UserMessage(content="a"), AssistantMessage(content="b", prefix=True)],
        )
        result = normalizer.from_chat_completion_request(request)
        assert result == InstructRequest[ChatMessage, Tool](
            messages=[UserMessage(content="a"), AssistantMessage(content="b", prefix=True)],
        )

    def test_json_normalizes_tool_content(self, normalizer: InstructRequestNormalizer) -> None:
        r"""Base normalizer (v1-v3) JSON-normalizes tool message content."""
        messy_json = '{"key" :  "value" ,  "num": 1}'
        request = mock_chat_completion(
            messages=[
                UserMessage(content="query"),
                AssistantMessage(tool_calls=[ToolCall(function=FunctionCall(name="fn", arguments="{}"), id="c1")]),
                ToolMessage(content=messy_json, tool_call_id="c1"),
            ],
        )
        parsed = normalizer.from_chat_completion_request(request)
        assert parsed == InstructRequest[ChatMessage, Tool](
            messages=[
                UserMessage(content="query"),
                AssistantMessage(
                    content="",
                    tool_calls=[ToolCall(function=FunctionCall(name="fn", arguments="{}"), id="c1")],
                ),
                ToolMessage(content='{"key": "value", "num": 1}', tool_call_id="c1"),
            ],
        )


class TestV7Normalization:
    @pytest.fixture(autouse=True)
    def normalizer_v7(self) -> InstructRequestNormalizerV7:
        return InstructRequestNormalizerV7.normalizer()

    def test_system_assistant_user_v7(self, normalizer_v7: InstructRequestNormalizerV7) -> None:
        chat_completion_request = mock_chat_completion(
            messages=[
                SystemMessage(content="S"),
                AssistantMessage(content="A"),
                UserMessage(content="U"),
            ]
        )

        parsed_request: InstructRequest[ChatMessage, Tool] = normalizer_v7.from_chat_completion_request(
            chat_completion_request
        )
        assert parsed_request == InstructRequest[ChatMessage, Tool](
            messages=[SystemMessage(content="S"), AssistantMessage(content="A"), UserMessage(content="U")],
        )

    def test_assistant_assistant_system_v7(self, normalizer_v7: InstructRequestNormalizerV7) -> None:
        chat_completion_request = mock_chat_completion(
            messages=[
                AssistantMessage(content="A"),
                SystemMessage(content="S"),
            ]
        )

        parsed_request: InstructRequest[ChatMessage, Tool] = normalizer_v7.from_chat_completion_request(
            chat_completion_request
        )
        assert parsed_request == InstructRequest[ChatMessage, Tool](
            messages=[AssistantMessage(content="A"), SystemMessage(content="S")],
        )

    def test_assistant_content_with_tool_calls(self, normalizer_v7: InstructRequestNormalizerV7) -> None:
        chat_completion_request = mock_chat_completion(
            messages=[
                AssistantMessage(
                    content="A",
                    tool_calls=[ToolCall(function=FunctionCall(name="tool1", arguments='{"input": "11"}'))],
                )
            ]
        )
        normalized_chat_req: InstructRequest[ChatMessage, Tool] = normalizer_v7.from_chat_completion_request(
            chat_completion_request
        )
        assert normalized_chat_req == InstructRequest[ChatMessage, Tool](
            messages=[
                AssistantMessage(
                    content="A",
                    tool_calls=[ToolCall(function=FunctionCall(name="tool1", arguments='{"input": "11"}'))],
                ),
            ],
        )

    def test_assistant_content_with_more_tool_calls(self, normalizer_v7: InstructRequestNormalizerV7) -> None:
        chat_completion_request = mock_chat_completion(
            messages=[
                UserMessage(content="A1"),
                AssistantMessage(content="B1"),
                AssistantMessage(
                    content="B2",
                    tool_calls=[ToolCall(function=FunctionCall(name="tool1", arguments='{"input": "1"}'))],
                ),
                AssistantMessage(content="B3"),
                AssistantMessage(
                    content="B4",
                    tool_calls=[
                        ToolCall(function=FunctionCall(name="tool21", arguments='{"input": "21"}')),
                        ToolCall(function=FunctionCall(name="tool22", arguments='{"input": "22"}')),
                    ],
                ),
                AssistantMessage(content="B5"),
                UserMessage(content="C1"),
            ]
        )
        normalized_chat_req: InstructRequest[ChatMessage, Tool] = normalizer_v7.from_chat_completion_request(
            chat_completion_request
        )
        assert normalized_chat_req == InstructRequest[ChatMessage, Tool](
            messages=[
                UserMessage(content="A1"),
                AssistantMessage(
                    content="B1\n\nB2\n\nB3\n\nB4\n\nB5",
                    tool_calls=[
                        ToolCall(function=FunctionCall(name="tool1", arguments='{"input": "1"}')),
                        ToolCall(function=FunctionCall(name="tool21", arguments='{"input": "21"}')),
                        ToolCall(function=FunctionCall(name="tool22", arguments='{"input": "22"}')),
                    ],
                ),
                UserMessage(content="C1"),
            ],
        )

    def test_assert_parsed_settings(
        self,
        normalizer_v7: InstructRequestNormalizerV7,
    ) -> None:
        chat_completion_request = ChatCompletionRequest(messages=[UserMessage(content="B")])
        parsed_request: InstructRequest[ChatMessage, Tool] = normalizer_v7.from_chat_completion_request(
            chat_completion_request
        )
        assert parsed_request == InstructRequest[ChatMessage, Tool](messages=[UserMessage(content="B")])

    def test_preserves_assistant_prefix_without_continuation_state(
        self, normalizer_v7: InstructRequestNormalizerV7
    ) -> None:
        request = ChatCompletionRequest[ChatMessage](
            messages=[UserMessage(content="a"), AssistantMessage(content="b", prefix=True)],
        )
        result: InstructRequest[ChatMessage, Tool] = normalizer_v7.from_chat_completion_request(request)
        assert result == InstructRequest[ChatMessage, Tool](
            messages=[UserMessage(content="a"), AssistantMessage(content="b", prefix=True)],
        )

    @pytest.mark.parametrize("num_empty", [0, 1, 2])
    def test_only_empty_text_chunks(self, normalizer_v7: InstructRequestNormalizerV7, num_empty: int) -> None:
        """Messages with only empty TextChunks produce empty content."""
        chat_completion_request = mock_chat_completion(
            messages=[
                UserMessage(content=[TextChunk(text="") for _ in range(num_empty)]),
                AssistantMessage(content=[TextChunk(text="") for _ in range(num_empty)]),
            ]
        )
        parsed_request: InstructRequest[ChatMessage, Tool] = normalizer_v7.from_chat_completion_request(
            chat_completion_request
        )
        assert parsed_request == InstructRequest[ChatMessage, Tool](
            messages=[UserMessage(content=""), AssistantMessage(content="")],
        )

    def test_complex_user_aggregation(self, normalizer_v7: InstructRequestNormalizerV7) -> None:
        """Complex multi-user-message aggregation with mixed str, chunks, empty, and non-text chunks."""

        chat_completion_request = mock_chat_completion(
            messages=[
                UserMessage(content=""),
                UserMessage(content="A"),
                UserMessage(content=""),
                UserMessage(content=[TextChunk(text="B")]),
                UserMessage(content=[TextChunk(text="")]),
                UserMessage(
                    content=[
                        TextChunk(text="C"),
                        TextChunk(text="D"),
                        ImageURLChunk(image_url="E"),
                        TextChunk(text="G"),
                        TextChunk(text=""),
                        TextChunk(text="H"),
                    ]
                ),
            ]
        )
        parsed_request: InstructRequest[ChatMessage, Tool] = normalizer_v7.from_chat_completion_request(
            chat_completion_request
        )
        assert parsed_request == InstructRequest[ChatMessage, Tool](
            messages=[
                UserMessage(
                    content=[
                        TextChunk(text="A\n\nB\n\nC\n\nD"),
                        ImageURLChunk(image_url="E"),
                        TextChunk(text="G\n\nH"),
                    ]
                ),
            ],
        )

    def test_complex_assistant_aggregation(self, normalizer_v7: InstructRequestNormalizerV7) -> None:
        """Complex multi-assistant-message aggregation with mixed str, chunks, and empty content."""
        chat_completion_request = mock_chat_completion(
            messages=[
                AssistantMessage(content=""),
                AssistantMessage(content="A"),
                AssistantMessage(content=""),
                AssistantMessage(content=[TextChunk(text="B")]),
                AssistantMessage(content=[TextChunk(text="")]),
                AssistantMessage(
                    content=[
                        TextChunk(text="C"),
                        TextChunk(text="D"),
                    ]
                ),
            ]
        )
        parsed_request: InstructRequest[ChatMessage, Tool] = normalizer_v7.from_chat_completion_request(
            chat_completion_request
        )
        assert parsed_request == InstructRequest[ChatMessage, Tool](
            messages=[AssistantMessage(content="A\n\nB\n\nC\n\nD")],
        )

    def test_accepts_string_content(self, normalizer_v7: InstructRequestNormalizerV7) -> None:
        r"""V7 normalizer accepts string content in assistant messages."""
        request = mock_chat_completion(
            messages=[
                UserMessage(content="query"),
                AssistantMessage(content="plain text"),
            ],
        )
        parsed: InstructRequest[ChatMessage, Tool] = normalizer_v7.from_chat_completion_request(request)
        assert parsed == InstructRequest[ChatMessage, Tool](
            messages=[UserMessage(content="query"), AssistantMessage(content="plain text")],
        )

    def test_skips_json_normalization_on_tool_content(self, normalizer_v7: InstructRequestNormalizerV7) -> None:
        r"""V7+ normalizers do not JSON-normalize tool message content."""
        messy_json = '{"key" :  "value" ,  "num": 1}'
        request = mock_chat_completion(
            messages=[
                UserMessage(content="query"),
                AssistantMessage(tool_calls=[ToolCall(function=FunctionCall(name="fn", arguments="{}"), id="c1")]),
                ToolMessage(content=messy_json, tool_call_id="c1"),
            ],
        )
        parsed: InstructRequest[ChatMessage, Tool] = normalizer_v7.from_chat_completion_request(request)
        assert parsed == InstructRequest[ChatMessage, Tool](
            messages=[
                UserMessage(content="query"),
                AssistantMessage(
                    content="",
                    tool_calls=[ToolCall(function=FunctionCall(name="fn", arguments="{}"), id="c1")],
                ),
                ToolMessage(content=messy_json, tool_call_id="c1"),
            ],
        )

    def test_preserves_audio_in_system_message(self, normalizer_v7: InstructRequestNormalizerV7) -> None:
        r"""V7 normalizer preserves AudioChunk in system messages."""
        request = mock_chat_completion(
            messages=[
                SystemMessage(content=[TextChunk(text="hello"), AudioChunk(input_audio=b"fake_audio_data")]),
                UserMessage(content="test"),
            ]
        )
        parsed: InstructRequest[ChatMessage, Tool] = normalizer_v7.from_chat_completion_request(request)
        assert parsed == InstructRequest[ChatMessage, Tool](
            messages=[
                SystemMessage(
                    content=[TextChunk(text="hello"), AudioChunk(input_audio=b"fake_audio_data")],
                ),
                UserMessage(content="test"),
            ],
        )


class TestFineTuningNormalizer:
    @pytest.fixture(autouse=True)
    def normalizer(self) -> InstructRequestNormalizer:
        return InstructRequestNormalizer(
            UserMessage, FinetuningAssistantMessage, ToolMessage, SystemMessage, InstructRequest, None
        )

    def test_normalize_weighted_assistant(self, normalizer: InstructRequestNormalizer) -> None:
        request = ChatCompletionRequest[FinetuningMessage](
            messages=[
                SystemMessage(content="helpful assistant"),
                UserMessage(content="a"),
                FinetuningAssistantMessage(content="he", weight=0),
                UserMessage(content="b"),
                FinetuningAssistantMessage(content="ho", weight=1),
                FinetuningAssistantMessage(content="lla", weight=1),
            ],
        )
        expected = InstructRequest[FinetuningMessage, Tool](
            messages=[
                UserMessage(content="a"),
                FinetuningAssistantMessage(content="he", weight=0),
                UserMessage(content="b"),
                FinetuningAssistantMessage(content="ho\n\nlla", weight=1),
            ],
            system_prompt="helpful assistant",
        )
        normalized = normalizer.from_chat_completion_request(request)
        assert expected == normalized

    def test_should_not_aggregate_if_weight_is_different(self, normalizer: InstructRequestNormalizer) -> None:
        request = ChatCompletionRequest[FinetuningMessage](
            messages=[
                SystemMessage(content="helpful assistant"),
                UserMessage(content="a"),
                FinetuningAssistantMessage(content="he", weight=0),
                UserMessage(content="b"),
                FinetuningAssistantMessage(content="ho", weight=0),
                FinetuningAssistantMessage(content="lla", weight=1),
            ],
        )
        expected = InstructRequest[FinetuningMessage, Tool](
            messages=[
                UserMessage(content="a"),
                FinetuningAssistantMessage(content="he", weight=0),
                UserMessage(content="b"),
                FinetuningAssistantMessage(content="ho", weight=0),
                FinetuningAssistantMessage(content="lla", weight=1),
            ],
            system_prompt="helpful assistant",
        )
        normalized = normalizer.from_chat_completion_request(request)
        assert normalized == expected

    def test_return_type_matches_configured_request_class(self) -> None:
        normalizer: InstructRequestNormalizer[
            UserMessage,
            FinetuningAssistantMessage,
            ToolMessage,
            SystemMessage,
            InstructRequest[FinetuningMessage, Tool],
        ] = InstructRequestNormalizer(
            UserMessage, FinetuningAssistantMessage, ToolMessage, SystemMessage, InstructRequest, None
        )
        request = ChatCompletionRequest[FinetuningMessage](messages=[UserMessage(content="a")])

        result = normalizer.from_chat_completion_request(request)

        assert_type(result, InstructRequest[FinetuningMessage, Tool])
        assert result == InstructRequest[FinetuningMessage, Tool](messages=[UserMessage(content="a")])


class TestV13Normalization:
    @pytest.fixture(autouse=True)
    def normalizer_v13(self) -> InstructRequestNormalizerV13:
        return InstructRequestNormalizerV13.normalizer()

    def _mock_chat_completion(self, messages: list[ChatMessage]) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            model="test",
            messages=messages,
            top_p=1.0,
            temperature=0.7,
        )

    def test_no_reorder_tool_messages(self, normalizer_v13: InstructRequestNormalizerV13) -> None:
        chat_completion_request: ChatCompletionRequest = self._mock_chat_completion(
            messages=[
                UserMessage(content="A"),
                AssistantMessage(
                    content="B",
                    tool_calls=[
                        ToolCall(id="1", function=FunctionCall(name="foo", arguments="{}")),
                        ToolCall(id="2", function=FunctionCall(name="foo", arguments="{}")),
                    ],
                ),
                ToolMessage(content="C", tool_call_id="1"),
                ToolMessage(content="D", tool_call_id="2"),
            ]
        )
        parsed_request: InstructRequest[ChatMessage, Tool] = normalizer_v13.from_chat_completion_request(
            chat_completion_request
        )
        assert parsed_request == InstructRequest[ChatMessage, Tool](
            messages=[
                UserMessage(content="A"),
                AssistantMessage(
                    content="B",
                    tool_calls=[
                        ToolCall(id="1", function=FunctionCall(name="foo", arguments="{}")),
                        ToolCall(id="2", function=FunctionCall(name="foo", arguments="{}")),
                    ],
                ),
                ToolMessage(content="C", tool_call_id="1"),
                ToolMessage(content="D", tool_call_id="2"),
            ],
        )

    def test_reorder_last_tool_messages(self, normalizer_v13: InstructRequestNormalizerV13) -> None:
        chat_completion_request: ChatCompletionRequest = self._mock_chat_completion(
            messages=[
                UserMessage(content="A"),
                AssistantMessage(
                    content="B",
                    tool_calls=[
                        ToolCall(id="1", function=FunctionCall(name="foo", arguments="{}")),
                        ToolCall(id="2", function=FunctionCall(name="foo", arguments="{}")),
                    ],
                ),
                ToolMessage(content="D", tool_call_id="2"),
                ToolMessage(content="C", tool_call_id="1"),
            ]
        )
        parsed_request: InstructRequest[ChatMessage, Tool] = normalizer_v13.from_chat_completion_request(
            chat_completion_request
        )
        assert parsed_request == InstructRequest[ChatMessage, Tool](
            messages=[
                UserMessage(content="A"),
                AssistantMessage(
                    content="B",
                    tool_calls=[
                        ToolCall(id="1", function=FunctionCall(name="foo", arguments="{}")),
                        ToolCall(id="2", function=FunctionCall(name="foo", arguments="{}")),
                    ],
                ),
                ToolMessage(content="C", tool_call_id="1"),
                ToolMessage(content="D", tool_call_id="2"),
            ],
        )

    def test_reorder_internal_tool_messages(self, normalizer_v13: InstructRequestNormalizerV13) -> None:
        chat_completion_request: ChatCompletionRequest = self._mock_chat_completion(
            messages=[
                UserMessage(content="A"),
                AssistantMessage(
                    content="B",
                    tool_calls=[
                        ToolCall(id="1", function=FunctionCall(name="foo", arguments="{}")),
                        ToolCall(id="2", function=FunctionCall(name="foo", arguments="{}")),
                    ],
                ),
                ToolMessage(content="D", tool_call_id="2"),
                ToolMessage(content="C", tool_call_id="1"),
                AssistantMessage(content="E"),
            ]
        )
        parsed_request: InstructRequest[ChatMessage, Tool] = normalizer_v13.from_chat_completion_request(
            chat_completion_request
        )
        assert parsed_request == InstructRequest[ChatMessage, Tool](
            messages=[
                UserMessage(content="A"),
                AssistantMessage(
                    content="B",
                    tool_calls=[
                        ToolCall(id="1", function=FunctionCall(name="foo", arguments="{}")),
                        ToolCall(id="2", function=FunctionCall(name="foo", arguments="{}")),
                    ],
                ),
                ToolMessage(content="C", tool_call_id="1"),
                ToolMessage(content="D", tool_call_id="2"),
                AssistantMessage(content="E"),
            ],
        )

    def test_reorder_extra_tool_messages(self, normalizer_v13: InstructRequestNormalizerV13) -> None:
        chat_completion_request: ChatCompletionRequest = self._mock_chat_completion(
            messages=[
                UserMessage(content="A"),
                AssistantMessage(
                    content="B",
                    tool_calls=[
                        ToolCall(id="1", function=FunctionCall(name="foo", arguments="{}")),
                    ],
                ),
                ToolMessage(content="D", tool_call_id="2"),
                ToolMessage(content="C", tool_call_id="1"),
            ]
        )
        parsed_request: InstructRequest[ChatMessage, Tool] = normalizer_v13.from_chat_completion_request(
            chat_completion_request
        )
        assert parsed_request == InstructRequest[ChatMessage, Tool](
            messages=[
                UserMessage(content="A"),
                AssistantMessage(
                    content="B",
                    tool_calls=[
                        ToolCall(id="1", function=FunctionCall(name="foo", arguments="{}")),
                    ],
                ),
                ToolMessage(content="C", tool_call_id="1"),
                ToolMessage(content="D", tool_call_id="2"),
            ],
        )

    def test_reorder_only_from_latest_assistant_message(self, normalizer_v13: InstructRequestNormalizerV13) -> None:
        chat_completion_request: ChatCompletionRequest = self._mock_chat_completion(
            messages=[
                UserMessage(content="A"),
                AssistantMessage(
                    content="B",
                    tool_calls=[
                        ToolCall(id="1", function=FunctionCall(name="foo", arguments="{}")),
                        ToolCall(id="2", function=FunctionCall(name="foo", arguments="{}")),
                    ],
                ),
                ToolMessage(content="C", tool_call_id="1"),
                ToolMessage(content="D", tool_call_id="2"),
                AssistantMessage(
                    content="E",
                    tool_calls=[
                        ToolCall(id="2", function=FunctionCall(name="foo", arguments="{}")),
                        ToolCall(id="1", function=FunctionCall(name="foo", arguments="{}")),
                    ],
                ),
                ToolMessage(content="C", tool_call_id="1"),
                ToolMessage(content="D", tool_call_id="2"),
            ]
        )
        parsed_request: InstructRequest[ChatMessage, Tool] = normalizer_v13.from_chat_completion_request(
            chat_completion_request
        )
        assert parsed_request == InstructRequest[ChatMessage, Tool](
            messages=[
                UserMessage(content="A"),
                AssistantMessage(
                    content="B",
                    tool_calls=[
                        ToolCall(id="1", function=FunctionCall(name="foo", arguments="{}")),
                        ToolCall(id="2", function=FunctionCall(name="foo", arguments="{}")),
                    ],
                ),
                ToolMessage(content="C", tool_call_id="1"),
                ToolMessage(content="D", tool_call_id="2"),
                AssistantMessage(
                    content="E",
                    tool_calls=[
                        ToolCall(id="2", function=FunctionCall(name="foo", arguments="{}")),
                        ToolCall(id="1", function=FunctionCall(name="foo", arguments="{}")),
                    ],
                ),
                ToolMessage(content="D", tool_call_id="2"),
                ToolMessage(content="C", tool_call_id="1"),
            ],
        )

    @pytest.mark.parametrize(
        ["system_message", "expected_system_message"],
        [
            pytest.param(
                SystemMessage(content="A"),
                SystemMessage(content="A"),
                id="string-content",
            ),
            pytest.param(
                SystemMessage(content=[TextChunk(text="A")]),
                SystemMessage(content="A"),
                id="single-text-chunk",
            ),
            pytest.param(
                SystemMessage(content=[TextChunk(text="A"), TextChunk(text="B")]),
                SystemMessage(content="A\n\nB"),
                id="two-text-chunks",
            ),
            pytest.param(
                SystemMessage(content=[TextChunk(text="A"), TextChunk(text="B"), ThinkChunk(thinking="C")]),
                SystemMessage(content=[TextChunk(text="A\n\nB"), ThinkChunk(thinking="C")]),
                id="text-and-think",
            ),
            pytest.param(
                SystemMessage(
                    content=[
                        TextChunk(text="A"),
                        TextChunk(text="B"),
                        ThinkChunk(thinking="C"),
                        ThinkChunk(thinking="D"),
                    ]
                ),
                SystemMessage(content=[TextChunk(text="A\n\nB"), ThinkChunk(thinking="C"), ThinkChunk(thinking="D")]),
                id="text-and-two-thinks",
            ),
        ],
    )
    def test_aggregate_system_prompt_content(
        self,
        normalizer_v13: InstructRequestNormalizerV13,
        system_message: SystemMessage,
        expected_system_message: SystemMessage,
    ) -> None:
        chat_completion_request: ChatCompletionRequest = self._mock_chat_completion(
            messages=[system_message, UserMessage(content="B")]
        )
        parsed_request: InstructRequest[ChatMessage, Tool] = normalizer_v13.from_chat_completion_request(
            chat_completion_request
        )
        assert parsed_request == InstructRequest[ChatMessage, Tool](
            messages=[expected_system_message, UserMessage(content="B")]
        )

    def test_system_messages_no_aggregation(self, normalizer_v13: InstructRequestNormalizerV13) -> None:
        """Consecutive system messages are NOT aggregated into one in V7+."""
        chat_completion_request = self._mock_chat_completion(
            messages=[
                SystemMessage(content="A"),
                SystemMessage(content="B"),
                UserMessage(content="C"),
                SystemMessage(content="D"),
            ]
        )
        parsed_request: InstructRequest[ChatMessage, Tool] = normalizer_v13.from_chat_completion_request(
            chat_completion_request
        )
        assert parsed_request == InstructRequest[ChatMessage, Tool](
            messages=[
                SystemMessage(content="A"),
                SystemMessage(content="B"),
                UserMessage(content="C"),
                SystemMessage(content="D"),
            ],
        )

    def test_system_messages_normalization(self, normalizer_v13: InstructRequestNormalizerV13) -> None:
        """System message chunks within the same message are aggregated with no separator."""
        chat_completion_request = self._mock_chat_completion(
            messages=[
                SystemMessage(content=[TextChunk(text="A"), TextChunk(text="B")]),
                SystemMessage(content=[TextChunk(text="C")]),
            ]
        )
        parsed_request: InstructRequest[ChatMessage, Tool] = normalizer_v13.from_chat_completion_request(
            chat_completion_request
        )
        assert parsed_request == InstructRequest[ChatMessage, Tool](
            messages=[
                SystemMessage(content="A\n\nB"),
                SystemMessage(content="C"),
            ],
        )

    def test_assert_parsed_settings(
        self,
        normalizer_v13: InstructRequestNormalizerV13,
    ) -> None:
        chat_completion_request: ChatCompletionRequest = self._mock_chat_completion(messages=[UserMessage(content="B")])
        parsed_request: InstructRequest[ChatMessage, Tool] = normalizer_v13.from_chat_completion_request(
            chat_completion_request
        )
        assert parsed_request == InstructRequest[ChatMessage, Tool](messages=[UserMessage(content="B")])

    def test_preserves_assistant_prefix_without_continuation_state(
        self, normalizer_v13: InstructRequestNormalizerV13
    ) -> None:
        request = ChatCompletionRequest[ChatMessage](
            messages=[UserMessage(content="a"), AssistantMessage(content="b", prefix=True)],
        )
        result: InstructRequest[ChatMessage, Tool] = normalizer_v13.from_chat_completion_request(request)
        assert result == InstructRequest[ChatMessage, Tool](
            messages=[UserMessage(content="a"), AssistantMessage(content="b", prefix=True)],
        )

    def test_accepts_text_and_think_chunks(self, normalizer_v13: InstructRequestNormalizerV13) -> None:
        r"""V13 normalizer accepts TextChunk and ThinkChunk in assistant messages."""
        request = mock_chat_completion(
            messages=[
                UserMessage(content="query"),
                AssistantMessage(content=[ThinkChunk(thinking="reasoning"), TextChunk(text="answer")]),
            ],
        )
        parsed: InstructRequest[ChatMessage, Tool] = normalizer_v13.from_chat_completion_request(request)
        assert parsed == InstructRequest[ChatMessage, Tool](
            messages=[
                UserMessage(content="query"),
                AssistantMessage(content=[ThinkChunk(thinking="reasoning"), TextChunk(text="answer")]),
            ],
        )

    def test_accepts_string_content(self, normalizer_v13: InstructRequestNormalizerV13) -> None:
        r"""V13 normalizer accepts string content in assistant messages."""
        request = mock_chat_completion(
            messages=[
                UserMessage(content="query"),
                AssistantMessage(content="plain text"),
            ],
        )
        parsed: InstructRequest[ChatMessage, Tool] = normalizer_v13.from_chat_completion_request(request)
        assert parsed == InstructRequest[ChatMessage, Tool](
            messages=[UserMessage(content="query"), AssistantMessage(content="plain text")],
        )

    def test_assistant_think_chunk_inter_message_aggregation(
        self, normalizer_v13: InstructRequestNormalizerV13
    ) -> None:
        r"""V13 normalizer preserves ThinkChunks across multiple assistant messages."""
        chat_completion_request = mock_chat_completion(
            messages=[
                AssistantMessage(content="A"),
                AssistantMessage(content=[TextChunk(text="B")]),
                AssistantMessage(
                    content=[
                        ThinkChunk(thinking="T"),
                        TextChunk(text="C"),
                        TextChunk(text="D"),
                    ]
                ),
            ]
        )
        parsed: InstructRequest[ChatMessage, Tool] = normalizer_v13.from_chat_completion_request(
            chat_completion_request
        )
        assert parsed == InstructRequest[ChatMessage, Tool](
            messages=[
                AssistantMessage(
                    content=[
                        TextChunk(text="A\n\nB"),
                        ThinkChunk(thinking="T"),
                        TextChunk(text="C\n\nD"),
                    ]
                ),
            ],
        )

    def test_assistant_think_chunk_intra_message_aggregation(
        self, normalizer_v13: InstructRequestNormalizerV13
    ) -> None:
        r"""V13 normalizer coalesces TextChunks and preserves multiple ThinkChunks within a single message."""
        chat_completion_request = mock_chat_completion(
            messages=[
                UserMessage(content="u"),
                AssistantMessage(
                    content=[
                        ThinkChunk(thinking="t1"),
                        ThinkChunk(thinking="t2"),
                        TextChunk(text="a1"),
                        TextChunk(text="a2"),
                        TextChunk(text="a3"),
                    ]
                ),
                UserMessage(content="u"),
            ]
        )
        parsed: InstructRequest[ChatMessage, Tool] = normalizer_v13.from_chat_completion_request(
            chat_completion_request
        )
        assert parsed == InstructRequest[ChatMessage, Tool](
            messages=[
                UserMessage(content="u"),
                AssistantMessage(
                    content=[
                        ThinkChunk(thinking="t1"),
                        ThinkChunk(thinking="t2"),
                        TextChunk(text="a1\n\na2\n\na3"),
                    ]
                ),
                UserMessage(content="u"),
            ],
        )

    def test_aggregates_text_tool_content(self, normalizer_v13: InstructRequestNormalizerV13) -> None:
        r"""V13 normalizer aggregates TextChunks in tool messages to a string."""
        request = mock_chat_completion(
            messages=[
                UserMessage(content="query"),
                AssistantMessage(tool_calls=[ToolCall(function=FunctionCall(name="fn", arguments="{}"), id="c1")]),
                ToolMessage(content=[TextChunk(text="hello"), TextChunk(text="world")], tool_call_id="c1"),
            ],
        )
        parsed: InstructRequest[ChatMessage, Tool] = normalizer_v13.from_chat_completion_request(request)
        assert parsed == InstructRequest[ChatMessage, Tool](
            messages=[
                UserMessage(content="query"),
                AssistantMessage(
                    content="",
                    tool_calls=[ToolCall(function=FunctionCall(name="fn", arguments="{}"), id="c1")],
                ),
                ToolMessage(content="hello\n\nworld", tool_call_id="c1"),
            ],
        )


_V15_BUILDER = ModelSettingsBuilder(
    reasoning_effort=ReasoningEffortEnumBuilder(values=list(ReasoningEffort), default=ReasoningEffort.none)
)


class TestV15Normalization:
    @pytest.fixture(autouse=True)
    def normalizer_v15(self) -> InstructRequestNormalizerV15:
        return InstructRequestNormalizerV15.normalizer(_V15_BUILDER)

    @pytest.mark.parametrize("reasoning_effort", [ReasoningEffort.none, ReasoningEffort.high])
    def test_assert_parsed_settings(
        self, normalizer_v15: InstructRequestNormalizerV15, reasoning_effort: ReasoningEffort
    ) -> None:
        chat_completion_request = ChatCompletionRequest(
            messages=[UserMessage(content="B")], reasoning_effort=reasoning_effort
        )
        parsed_request: InstructRequest[ChatMessage, Tool] = normalizer_v15.from_chat_completion_request(
            chat_completion_request
        )
        assert parsed_request == InstructRequest[ChatMessage, Tool](
            messages=[UserMessage(content="B")],
            settings=ModelSettings(reasoning_effort=reasoning_effort),
        )

    def test_unset_effort_resolves_to_configured_default(self, normalizer_v15: InstructRequestNormalizerV15) -> None:
        chat_completion_request = ChatCompletionRequest(messages=[UserMessage(content="B")], reasoning_effort=None)
        parsed_request: InstructRequest[ChatMessage, Tool] = normalizer_v15.from_chat_completion_request(
            chat_completion_request
        )
        assert parsed_request == InstructRequest[ChatMessage, Tool](
            messages=[UserMessage(content="B")],
            settings=ModelSettings(reasoning_effort=ReasoningEffort.none),
        )

    def test_disallowed_effort_rejected(self) -> None:
        normalizer = InstructRequestNormalizerV15.normalizer(
            ModelSettingsBuilder(
                reasoning_effort=ReasoningEffortEnumBuilder(values=[ReasoningEffort.high], default=ReasoningEffort.high)
            )
        )
        chat_completion_request = ChatCompletionRequest(
            messages=[UserMessage(content="B")], reasoning_effort=ReasoningEffort.none
        )
        with pytest.raises(InvalidRequestException, match=r"reasoning_effort should be one of \['high'\], got none\."):
            normalizer.from_chat_completion_request(chat_completion_request)

    def test_preserves_assistant_prefix_without_continuation_state(
        self, normalizer_v15: InstructRequestNormalizerV15
    ) -> None:
        request = ChatCompletionRequest[ChatMessage](
            messages=[UserMessage(content="a"), AssistantMessage(content="b", prefix=True)],
            reasoning_effort=ReasoningEffort.high,
        )
        result: InstructRequest[ChatMessage, Tool] = normalizer_v15.from_chat_completion_request(request)
        assert result == InstructRequest[ChatMessage, Tool](
            messages=[UserMessage(content="a"), AssistantMessage(content="b", prefix=True)],
            settings=ModelSettings(reasoning_effort=ReasoningEffort.high),
        )

    def test_v15_intra_message_chunks_joined_without_separator(
        self, normalizer_v15: InstructRequestNormalizerV15
    ) -> None:
        """V15 joins TextChunks within the same message with no separator (chunk_join_str='')."""
        request = ChatCompletionRequest[ChatMessage](
            messages=[
                UserMessage(content=[TextChunk(text="A"), TextChunk(text="B")]),
                AssistantMessage(content=[TextChunk(text="C"), TextChunk(text="D")]),
            ],
            reasoning_effort=ReasoningEffort.high,
        )
        parsed: InstructRequest[ChatMessage, Tool] = normalizer_v15.from_chat_completion_request(request)
        assert parsed == InstructRequest[ChatMessage, Tool](
            messages=[UserMessage(content="AB"), AssistantMessage(content="CD")],
            settings=ModelSettings(reasoning_effort=ReasoningEffort.high),
        )

    def test_v15_inter_message_join_still_uses_separator(self, normalizer_v15: InstructRequestNormalizerV15) -> None:
        r"""V15 still joins text across different messages with '\n\n'."""
        request = ChatCompletionRequest[ChatMessage](
            messages=[
                UserMessage(content="First"),
                UserMessage(content="Second"),
                AssistantMessage(content="Reply"),
            ],
            reasoning_effort=ReasoningEffort.high,
        )
        parsed: InstructRequest[ChatMessage, Tool] = normalizer_v15.from_chat_completion_request(request)
        assert parsed == InstructRequest[ChatMessage, Tool](
            messages=[UserMessage(content="First\n\nSecond"), AssistantMessage(content="Reply")],
            settings=ModelSettings(reasoning_effort=ReasoningEffort.high),
        )

    def test_v15_mixed_intra_and_inter_message(self, normalizer_v15: InstructRequestNormalizerV15) -> None:
        r"""V15 combines intra-message ('') and inter-message ('\n\n') joining."""
        request = ChatCompletionRequest[ChatMessage](
            messages=[
                UserMessage(content=[TextChunk(text="A"), TextChunk(text="B")]),
                UserMessage(content=[TextChunk(text="C"), TextChunk(text="D")]),
                AssistantMessage(content="Reply"),
            ],
            reasoning_effort=ReasoningEffort.high,
        )
        parsed: InstructRequest[ChatMessage, Tool] = normalizer_v15.from_chat_completion_request(request)
        assert parsed == InstructRequest[ChatMessage, Tool](
            messages=[UserMessage(content="AB\n\nCD"), AssistantMessage(content="Reply")],
            settings=ModelSettings(reasoning_effort=ReasoningEffort.high),
        )

    def test_v15_mixed_intra_and_inter_assistant_messages(self, normalizer_v15: InstructRequestNormalizerV15) -> None:
        r"""V15 combines intra-message ('') and inter-message ('\n\n') joining for assistant messages."""
        request = ChatCompletionRequest[ChatMessage](
            messages=[
                UserMessage(content="Hello"),
                AssistantMessage(content=[TextChunk(text="A"), TextChunk(text="B")]),
                AssistantMessage(content=[TextChunk(text="C"), TextChunk(text="D")]),
            ],
            reasoning_effort=ReasoningEffort.high,
        )
        parsed: InstructRequest[ChatMessage, Tool] = normalizer_v15.from_chat_completion_request(request)
        assert parsed == InstructRequest[ChatMessage, Tool](
            messages=[UserMessage(content="Hello"), AssistantMessage(content="AB\n\nCD")],
            settings=ModelSettings(reasoning_effort=ReasoningEffort.high),
        )

    def test_v15_tool_message_text_chunks_joined_without_separator(
        self, normalizer_v15: InstructRequestNormalizerV15
    ) -> None:
        """V15 tool message TextChunks are joined with no separator."""
        request = ChatCompletionRequest[ChatMessage](
            messages=[
                UserMessage(content="query"),
                AssistantMessage(tool_calls=[ToolCall(function=FunctionCall(name="fn", arguments="{}"), id="c1")]),
                ToolMessage(content=[TextChunk(text="X"), TextChunk(text="Y")], tool_call_id="c1"),
            ],
            reasoning_effort=ReasoningEffort.high,
        )
        parsed: InstructRequest[ChatMessage, Tool] = normalizer_v15.from_chat_completion_request(request)
        assert parsed == InstructRequest[ChatMessage, Tool](
            messages=[
                UserMessage(content="query"),
                AssistantMessage(
                    content="",
                    tool_calls=[ToolCall(function=FunctionCall(name="fn", arguments="{}"), id="c1")],
                ),
                ToolMessage(content="XY", tool_call_id="c1"),
            ],
            settings=ModelSettings(reasoning_effort=ReasoningEffort.high),
        )

    def test_accepts_text_and_think_chunks(self, normalizer_v15: InstructRequestNormalizerV15) -> None:
        r"""V15 normalizer accepts TextChunk and ThinkChunk in assistant messages."""
        request = ChatCompletionRequest[ChatMessage](
            messages=[
                UserMessage(content="query"),
                AssistantMessage(content=[ThinkChunk(thinking="reasoning"), TextChunk(text="answer")]),
            ],
            reasoning_effort=ReasoningEffort.high,
        )
        parsed: InstructRequest[ChatMessage, Tool] = normalizer_v15.from_chat_completion_request(request)
        assert parsed == InstructRequest[ChatMessage, Tool](
            messages=[
                UserMessage(content="query"),
                AssistantMessage(content=[ThinkChunk(thinking="reasoning"), TextChunk(text="answer")]),
            ],
            settings=ModelSettings(reasoning_effort=ReasoningEffort.high),
        )

    def test_accepts_string_content(self, normalizer_v15: InstructRequestNormalizerV15) -> None:
        r"""V15 normalizer accepts string content in assistant messages."""
        request = ChatCompletionRequest[ChatMessage](
            messages=[
                UserMessage(content="query"),
                AssistantMessage(content="plain text"),
            ],
            reasoning_effort=ReasoningEffort.high,
        )
        parsed: InstructRequest[ChatMessage, Tool] = normalizer_v15.from_chat_completion_request(request)
        assert parsed == InstructRequest[ChatMessage, Tool](
            messages=[UserMessage(content="query"), AssistantMessage(content="plain text")],
            settings=ModelSettings(reasoning_effort=ReasoningEffort.high),
        )

    def test_preserves_non_text_tool_content(self, normalizer_v15: InstructRequestNormalizerV15) -> None:
        r"""V15 normalizer preserves non-text chunks in tool messages."""
        image_chunk = ImageURLChunk(image_url="https://example.com/image.png")
        request = ChatCompletionRequest[ChatMessage](
            messages=[
                UserMessage(content="query"),
                AssistantMessage(tool_calls=[ToolCall(function=FunctionCall(name="fn", arguments="{}"), id="c1")]),
                ToolMessage(content=[image_chunk], tool_call_id="c1"),
            ],
            reasoning_effort=ReasoningEffort.high,
        )
        parsed: InstructRequest[ChatMessage, Tool] = normalizer_v15.from_chat_completion_request(request)
        assert parsed == InstructRequest[ChatMessage, Tool](
            messages=[
                UserMessage(content="query"),
                AssistantMessage(
                    content="",
                    tool_calls=[ToolCall(function=FunctionCall(name="fn", arguments="{}"), id="c1")],
                ),
                ToolMessage(content=[image_chunk], tool_call_id="c1"),
            ],
            settings=ModelSettings(reasoning_effort=ReasoningEffort.high),
        )

    def test_sorts_multimodal_tool_messages(self, normalizer_v15: InstructRequestNormalizerV15) -> None:
        r"""V15 normalizer sorts multimodal tool messages by tool call order."""
        image_chunk_1 = ImageURLChunk(image_url="https://example.com/img1.png")
        image_chunk_2 = ImageURLChunk(image_url="https://example.com/img2.png")
        request = ChatCompletionRequest[ChatMessage](
            messages=[
                UserMessage(content="query"),
                AssistantMessage(
                    tool_calls=[
                        ToolCall(function=FunctionCall(name="fn1", arguments="{}"), id="c1"),
                        ToolCall(function=FunctionCall(name="fn2", arguments="{}"), id="c2"),
                    ]
                ),
                ToolMessage(content=[image_chunk_2], tool_call_id="c2"),
                ToolMessage(content=[image_chunk_1], tool_call_id="c1"),
            ],
            reasoning_effort=ReasoningEffort.high,
        )
        parsed: InstructRequest[ChatMessage, Tool] = normalizer_v15.from_chat_completion_request(request)
        assert parsed == InstructRequest[ChatMessage, Tool](
            messages=[
                UserMessage(content="query"),
                AssistantMessage(
                    content="",
                    tool_calls=[
                        ToolCall(function=FunctionCall(name="fn1", arguments="{}"), id="c1"),
                        ToolCall(function=FunctionCall(name="fn2", arguments="{}"), id="c2"),
                    ],
                ),
                ToolMessage(content=[image_chunk_1], tool_call_id="c1"),
                ToolMessage(content=[image_chunk_2], tool_call_id="c2"),
            ],
            settings=ModelSettings(reasoning_effort=ReasoningEffort.high),
        )

    def test_preserves_audio_in_system_message(self, normalizer_v15: InstructRequestNormalizerV15) -> None:
        r"""V15 normalizer preserves AudioChunk in system messages."""
        request = ChatCompletionRequest[ChatMessage](
            messages=[
                SystemMessage(content=[TextChunk(text="hello"), AudioChunk(input_audio=b"fake_audio_data")]),
                UserMessage(content="test"),
            ],
            reasoning_effort=ReasoningEffort.high,
        )
        parsed: InstructRequest[ChatMessage, Tool] = normalizer_v15.from_chat_completion_request(request)
        assert parsed == InstructRequest[ChatMessage, Tool](
            messages=[
                SystemMessage(
                    content=[TextChunk(text="hello"), AudioChunk(input_audio=b"fake_audio_data")],
                ),
                UserMessage(content="test"),
            ],
            settings=ModelSettings(reasoning_effort=ReasoningEffort.high),
        )


class TestGetNormalizer:
    @pytest.mark.parametrize(
        ("version", "expected_class", "model_settings_builder"),
        [
            pytest.param(TokenizerVersion.v1, InstructRequestNormalizer, None, id="v1"),
            pytest.param(TokenizerVersion.v2, InstructRequestNormalizer, None, id="v2"),
            pytest.param(TokenizerVersion.v3, InstructRequestNormalizer, None, id="v3"),
            pytest.param(TokenizerVersion.v7, InstructRequestNormalizerV7, None, id="v7"),
            pytest.param(TokenizerVersion.v11, InstructRequestNormalizerV13, None, id="v11-selects-v13"),
            pytest.param(TokenizerVersion.v13, InstructRequestNormalizerV13, None, id="v13"),
            pytest.param(TokenizerVersion.v15, InstructRequestNormalizerV15, _V15_BUILDER, id="v15"),
        ],
    )
    def test_get_normalizer_selects_exact_class(
        self,
        version: TokenizerVersion,
        expected_class: type[InstructRequestNormalizer],
        model_settings_builder: ModelSettingsBuilder | None,
    ) -> None:
        normalizer = get_normalizer(version, model_settings_builder)
        assert type(normalizer) is expected_class
        assert normalizer._model_settings_builder == model_settings_builder

    @pytest.mark.parametrize(
        "version",
        [
            pytest.param(TokenizerVersion.v1, id="v1"),
            pytest.param(TokenizerVersion.v2, id="v2"),
            pytest.param(TokenizerVersion.v3, id="v3"),
            pytest.param(TokenizerVersion.v7, id="v7"),
            pytest.param(TokenizerVersion.v11, id="v11"),
            pytest.param(TokenizerVersion.v13, id="v13"),
        ],
    )
    def test_get_normalizer_pre_v15_rejects_settings_builder(self, version: TokenizerVersion) -> None:
        with pytest.raises(ValueError, match="model_settings_builder must be None"):
            get_normalizer(version, _V15_BUILDER)

    def test_v15_normalizer_without_builder_rejects_when_building_settings(self) -> None:
        normalizer = get_normalizer(TokenizerVersion.v15)
        request = ChatCompletionRequest[ChatMessage](messages=[UserMessage(content="B")])
        with pytest.raises(InvalidRequestException, match="model_settings_builder must not be None"):
            normalizer.from_chat_completion_request(request)

    @pytest.mark.parametrize(
        "normalizer_cls",
        [
            pytest.param(InstructRequestNormalizer, id="base"),
            pytest.param(InstructRequestNormalizerV7, id="v7"),
            pytest.param(InstructRequestNormalizerV13, id="v13"),
        ],
    )
    def test_directly_constructed_pre_v15_normalizer_rejects_builder(
        self, normalizer_cls: type[InstructRequestNormalizer]
    ) -> None:
        normalizer = normalizer_cls(
            UserMessage, AssistantMessage, ToolMessage, SystemMessage, InstructRequest, _V15_BUILDER
        )
        request = ChatCompletionRequest(messages=[UserMessage(content="B")])
        with pytest.raises(InvalidRequestException, match="model_settings_builder should be None"):
            normalizer.from_chat_completion_request(request)


def _image_chunk() -> ImageChunk:
    return ImageChunk(image=Image.new(mode="RGB", size=(4, 4), color="red"))


def _v3_mm_case_image_1() -> tuple[ChatCompletionRequest, InstructRequest[ChatMessage, Tool]]:
    request = ChatCompletionRequest[ChatMessage](messages=[UserMessage(content=[TextChunk(text="a"), _image_chunk()])])
    expected = InstructRequest[ChatMessage, Tool](messages=[UserMessage(content=[TextChunk(text="a"), _image_chunk()])])
    return request, expected


def _v3_mm_case_image_2() -> tuple[ChatCompletionRequest, InstructRequest[ChatMessage, Tool]]:
    request = ChatCompletionRequest[ChatMessage](
        messages=[
            SystemMessage(content="A B"),
            UserMessage(content=[TextChunk(text="C"), _image_chunk()]),
        ]
    )
    expected = InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content=[TextChunk(text="C"), _image_chunk()])],
        system_prompt="A B",
    )
    return request, expected


def _v3_mm_case_image_3() -> tuple[ChatCompletionRequest, InstructRequest[ChatMessage, Tool]]:
    request = ChatCompletionRequest[ChatMessage](
        messages=[
            SystemMessage(content="A B"),
            UserMessage(content=[_image_chunk(), TextChunk(text="C")]),
        ]
    )
    expected = InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content=[_image_chunk(), TextChunk(text="C")])],
        system_prompt="A B",
    )
    return request, expected


def _v3_mm_case_image_4() -> tuple[ChatCompletionRequest, InstructRequest[ChatMessage, Tool]]:
    request = ChatCompletionRequest[ChatMessage](
        messages=[
            SystemMessage(content="A B"),
            UserMessage(content=[_image_chunk(), _image_chunk(), TextChunk(text="C")]),
            AssistantMessage(content="D"),
            UserMessage(content=[_image_chunk(), TextChunk(text="E"), _image_chunk()]),
        ]
    )
    expected = InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content=[_image_chunk(), _image_chunk(), TextChunk(text="C")]),
            AssistantMessage(content="D"),
            UserMessage(content=[_image_chunk(), TextChunk(text="E"), _image_chunk()]),
        ],
        system_prompt="A B",
    )
    return request, expected


def _v3_mm_case_image_5() -> tuple[ChatCompletionRequest, InstructRequest[ChatMessage, Tool]]:
    request = ChatCompletionRequest[ChatMessage](
        messages=[
            UserMessage(
                content=[
                    TextChunk(text="A"),
                    _image_chunk(),
                    TextChunk(text="B"),
                    TextChunk(text="C"),
                    _image_chunk(),
                    TextChunk(text="D"),
                    TextChunk(text="E"),
                ]
            ),
        ]
    )
    expected = InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(
                content=[
                    TextChunk(text="A"),
                    _image_chunk(),
                    TextChunk(text="B\n\nC"),
                    _image_chunk(),
                    TextChunk(text="D\n\nE"),
                ]
            ),
        ]
    )
    return request, expected


def _v3_mm_case_text_1() -> tuple[ChatCompletionRequest, InstructRequest[ChatMessage, Tool]]:
    request = ChatCompletionRequest[ChatMessage](
        messages=[
            UserMessage(content="hello"),
            AssistantMessage(content="aaa"),
            UserMessage(content="goodbye"),
        ]
    )
    expected = InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="hello"),
            AssistantMessage(content="aaa"),
            UserMessage(content="goodbye"),
        ]
    )
    return request, expected


def _v3_mm_case_text_2() -> tuple[ChatCompletionRequest, InstructRequest[ChatMessage, Tool]]:
    request = ChatCompletionRequest[ChatMessage](messages=[UserMessage(content="hello")])
    expected = InstructRequest[ChatMessage, Tool](messages=[UserMessage(content="hello")])
    return request, expected


def _v3_mm_case_text_3() -> tuple[ChatCompletionRequest, InstructRequest[ChatMessage, Tool]]:
    request = ChatCompletionRequest[ChatMessage](messages=[UserMessage(content=[TextChunk(text="")])])
    expected = InstructRequest[ChatMessage, Tool](messages=[UserMessage(content="")])
    return request, expected


def _v3_mm_case_text_4() -> tuple[ChatCompletionRequest, InstructRequest[ChatMessage, Tool]]:
    request = ChatCompletionRequest[ChatMessage](
        messages=[
            SystemMessage(content="You are an AI assistant"),
            UserMessage(content=[TextChunk(text="aaa"), TextChunk(text="bbb")]),
            AssistantMessage(content="aaa"),
            UserMessage(content="goodbye"),
        ]
    )
    expected = InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="aaa\n\nbbb"),
            AssistantMessage(content="aaa"),
            UserMessage(content="goodbye"),
        ],
        system_prompt="You are an AI assistant",
    )
    return request, expected


class TestV3MultimodalNormalization:
    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(_v3_mm_case_image_1, id="image-1"),
            pytest.param(_v3_mm_case_image_2, id="image-2"),
            pytest.param(_v3_mm_case_image_3, id="image-3"),
            pytest.param(_v3_mm_case_image_4, id="image-4"),
            pytest.param(_v3_mm_case_image_5, id="image-5"),
            pytest.param(_v3_mm_case_text_1, id="text-1"),
            pytest.param(_v3_mm_case_text_2, id="text-2"),
            pytest.param(_v3_mm_case_text_3, id="text-3"),
            pytest.param(_v3_mm_case_text_4, id="text-4"),
        ],
    )
    def test_v3_multimodal_normalizer_preserves_complete_content(
        self,
        case: Callable[[], tuple[ChatCompletionRequest, InstructRequest[ChatMessage, Tool]]],
    ) -> None:
        r"""The v3 (base) normalizer keeps image position/payload, coalesces adjacent text and folds system prompts."""
        request, expected = case()
        normalizer = get_normalizer(TokenizerVersion.v3)
        assert normalizer.from_chat_completion_request(request) == expected
