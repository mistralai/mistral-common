import json

import pytest
from pydantic import ValidationError

from mistral_common.exceptions import InvalidRequestException
from mistral_common.protocol.instruct.chunk import (
    AudioChunk,
    ChunkTypes,
    ContentChunk,
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
from mistral_common.tokens.tokenizers.model_settings_builder import EnumBuilder, ModelSettingsBuilder


def mock_chat_completion(messages: list[ChatMessage]) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test",
        messages=messages,
        top_p=1.0,
        temperature=0.7,
    )


class TestChatCompletionRequestNormalization:
    @pytest.fixture(autouse=True)
    def normalizer(self) -> InstructRequestNormalizer:
        return InstructRequestNormalizer(
            UserMessage, AssistantMessage, ToolMessage, SystemMessage, InstructRequest, None
        )

    def test_user_system_user(self, normalizer: InstructRequestNormalizer) -> None:
        chat_completion_request = mock_chat_completion(
            messages=[
                UserMessage(content="U"),
                SystemMessage(content="S"),
                UserMessage(content="U"),
            ]
        )

        parsed_request = normalizer.from_chat_completion_request(chat_completion_request)
        assert parsed_request.system_prompt == "S"

    def test_multiple_system(self, normalizer: InstructRequestNormalizer) -> None:
        chat_completion_request = mock_chat_completion(
            messages=[
                SystemMessage(content="S"),
                SystemMessage(content="S"),
                SystemMessage(content="S"),
            ]
        )

        parsed_request = normalizer.from_chat_completion_request(chat_completion_request)
        assert parsed_request.system_prompt == "S\n\nS\n\nS"

    def test_single_system(self, normalizer: InstructRequestNormalizer) -> None:
        chat_completion_request = mock_chat_completion(
            messages=[
                SystemMessage(content="S"),
            ]
        )

        parsed_request = normalizer.from_chat_completion_request(chat_completion_request)

        assert parsed_request.system_prompt == "S"

    def test_system_assistant_user(self, normalizer: InstructRequestNormalizer) -> None:
        chat_completion_request = mock_chat_completion(
            messages=[
                SystemMessage(content="S"),
                AssistantMessage(content="A"),
                UserMessage(content="U"),
            ]
        )

        parsed_request = normalizer.from_chat_completion_request(chat_completion_request)

        first_message = parsed_request.messages[0]
        assert isinstance(first_message, UserMessage)
        assert first_message.content == ""
        assert parsed_request.system_prompt == "S"

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

        assert parsed_request.system_prompt == "S"

        assert len(parsed_request.messages) == 3  # 1 user message added, system message removed

        first_message = parsed_request.messages[0]
        assert isinstance(first_message, UserMessage)
        assert first_message.content == ""
        assert parsed_request.system_prompt == "S"

    def check_merge(
        self,
        roles: list[str],
        expected_roles: list[str],
        expected_content: list[list[ContentChunk] | str],
        normalizer: InstructRequestNormalizer,
    ) -> None:
        letter_to_cls: dict[str, ChatMessage] = {
            "s": SystemMessage(content="s"),
            "u": UserMessage(content="u"),
            "a": AssistantMessage(content="a"),
            "u2": UserMessage(content="u2"),
        }

        chat_completion_request = mock_chat_completion(
            messages=[letter_to_cls[r] for r in roles],
        )
        parsed_request = normalizer.from_chat_completion_request(chat_completion_request)
        assert len(parsed_request.messages) == len(expected_roles)
        assert [message.role for message in parsed_request.messages] == [
            letter_to_cls[role].role for role in expected_roles
        ]
        assert len(expected_content) == len(parsed_request.messages)
        for x, expected in zip(parsed_request.messages, expected_content):
            assert isinstance(x, (UserMessage, AssistantMessage))
            assert x.content == expected

    def check_merge_chunks(
        self,
        roles: list[str],
        expected_roles: list[str],
        expected_content: list[list[ContentChunk] | str],
        normalizer: InstructRequestNormalizer,
    ) -> None:
        letter_to_cls: dict[str, ChatMessage] = {
            "s": SystemMessage(content="s"),
            "u": UserMessage(content="u"),
            "a": AssistantMessage(content="a"),
            "a2": AssistantMessage(
                content=[
                    ThinkChunk(thinking="t1"),
                    ThinkChunk(thinking="t2"),
                    TextChunk(text="a1"),
                    TextChunk(text="a2"),
                    TextChunk(text="a3"),
                ]
            ),
        }

        chat_completion_request = mock_chat_completion(
            messages=[letter_to_cls[r] for r in roles],
        )
        parsed_request = normalizer.from_chat_completion_request(chat_completion_request)
        assert len(parsed_request.messages) == len(expected_roles)
        assert [message.role for message in parsed_request.messages] == [
            letter_to_cls[role].role for role in expected_roles
        ]
        assert len(expected_content) == len(parsed_request.messages)
        for x, expected in zip(parsed_request.messages, expected_content):
            assert isinstance(x, (UserMessage, AssistantMessage))
            assert x.content == expected

    def test_message_aggregation(self, normalizer: InstructRequestNormalizer) -> None:
        self.check_merge(["s", "s", "s", "u"], ["u"], ["u"], normalizer)
        self.check_merge(["s", "s", "s", "u", "u"], ["u"], ["u\n\nu"], normalizer)
        self.check_merge(["s", "s", "s", "u", "u", "s", "a", "u"], ["u", "a", "u"], ["u\n\nu", "a", "u"], normalizer)

        self.check_merge(
            ["s", "s", "s", "u", "u", "a", "a", "u"],
            ["u", "a", "u"],
            ["u\n\nu", "a\n\na", "u"],
            normalizer,
        )

        self.check_merge(
            ["s", "a", "u"],
            ["u", "a", "u"],
            ["", "a", "u"],
            normalizer,
        )

        self.check_merge_chunks(
            ["u", "a2", "u"],
            ["u", "a", "u"],
            [
                "u",
                [
                    ThinkChunk(thinking="t1"),
                    ThinkChunk(thinking="t2"),
                    TextChunk(text="a1\n\na2\n\na3"),
                ],
                "u",
            ],
            normalizer,
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

        first_message = parsed_request.messages[0]
        assert isinstance(first_message, UserMessage)
        assert first_message.content == "foo\n\nchunk\n\nfoo\n\nchunk"

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
        first_message = parsed_request.messages[0]
        assert isinstance(first_message, UserMessage)
        assert first_message.content == "foo\n\nchunk1\n\nchunk2\n\nchunk3"

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
        first_message = parsed_request.messages[0]
        assert isinstance(first_message, UserMessage)
        assert first_message.content == "U\n\nV"
        second_message = parsed_request.messages[1]
        assert isinstance(second_message, AssistantMessage)
        assert second_message.content == "A\n\nB"

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
        first_message = parsed_request.messages[0]
        assert isinstance(first_message, UserMessage)
        assert first_message.content == "user"
        assert parsed_request.system_prompt == "system"

    def test_normalize_tools(self, normalizer: InstructRequestNormalizer) -> None:
        """
        Test doesn't really "normalize" anything but it checks that the tools are added to the
        InstructRequest during from_chat_completion_request
        """
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
        parsed_request: InstructRequest[ChatMessage, Tool] = normalizer.from_chat_completion_request(
            chat_completion_request
        )
        assert parsed_request.settings == ModelSettings.none()

    def test_continue_final_message_forwarded(self, normalizer: InstructRequestNormalizer) -> None:
        request = ChatCompletionRequest[ChatMessage](
            messages=[UserMessage(content="a"), AssistantMessage(content="b")],
            continue_final_message=True,
        )
        result: InstructRequest[ChatMessage, Tool] = normalizer.from_chat_completion_request(request)
        assert result.continue_final_message is True


class TestChatCompletionRequestNormalizationV7:
    @pytest.fixture(autouse=True)
    def normalizer_v7(self) -> InstructRequestNormalizerV7:
        return InstructRequestNormalizerV7(
            UserMessage, AssistantMessage, ToolMessage, SystemMessage, InstructRequest, None
        )

    def test_system_assistant_user_v7(self, normalizer_v7: InstructRequestNormalizerV7) -> None:
        chat_completion_request = mock_chat_completion(
            messages=[
                SystemMessage(content="S"),
                AssistantMessage(content="A"),
                UserMessage(content="U"),
            ]
        )

        parsed_request: InstructRequest = normalizer_v7.from_chat_completion_request(chat_completion_request)

        first_message = parsed_request.messages[0]
        assert isinstance(first_message, SystemMessage)
        assert first_message.content == "S"

        second_message = parsed_request.messages[1]
        assert isinstance(second_message, AssistantMessage)
        assert second_message.content == "A"

        assert parsed_request.system_prompt is None

    def test_assistant_assistant_system_v7(self, normalizer_v7: InstructRequestNormalizer) -> None:
        chat_completion_request = mock_chat_completion(
            messages=[
                AssistantMessage(content="A"),
                SystemMessage(content="S"),
            ]
        )

        parsed_request = normalizer_v7.from_chat_completion_request(chat_completion_request)

        assert parsed_request.system_prompt is None

        assert len(parsed_request.messages) == 2

        first_message = parsed_request.messages[0]
        assert isinstance(first_message, AssistantMessage)
        assert first_message.content == "A"

        second_message = parsed_request.messages[1]
        assert isinstance(second_message, SystemMessage)
        assert second_message.content == "S"

    def test_assistant_content_with_tool_calls(self, normalizer_v7: InstructRequestNormalizer) -> None:
        chat_completion_request = mock_chat_completion(
            messages=[
                AssistantMessage(
                    content="A",
                    tool_calls=[ToolCall(function=FunctionCall(name="tool1", arguments='{"input": "11"}'))],
                )
            ]
        )
        normalized_chat_req = normalizer_v7.from_chat_completion_request(chat_completion_request)

        assert normalized_chat_req.messages[0].content == "A", normalized_chat_req.messages[0].content
        assert len(normalized_chat_req.messages[0].tool_calls) == 1, normalized_chat_req.messages[0].tool_calls
        assert normalized_chat_req.messages[0].tool_calls[0].function.name == "tool1", (
            normalized_chat_req.messages[0].tool_calls[0].function.name
        )

    def test_assistant_content_with_more_tool_calls(self, normalizer_v7: InstructRequestNormalizer) -> None:
        chat_completion_request = mock_chat_completion(
            messages=[
                UserMessage(content="A1"),
                AssistantMessage(
                    content="B1",
                ),
                AssistantMessage(
                    content="B2",
                    tool_calls=[ToolCall(function=FunctionCall(name="tool1", arguments='{"input": "1"}'))],
                ),
                AssistantMessage(
                    content="B3",
                ),
                AssistantMessage(
                    content="B4",
                    tool_calls=[
                        ToolCall(function=FunctionCall(name="tool21", arguments='{"input": "21"}')),
                        ToolCall(function=FunctionCall(name="tool22", arguments='{"input": "22"}')),
                    ],
                ),
                AssistantMessage(
                    content="B5",
                ),
                UserMessage(content="C1"),
            ]
        )
        normalized_chat_req = normalizer_v7.from_chat_completion_request(chat_completion_request)

        assert normalized_chat_req.messages[0].content == "A1"
        assert normalized_chat_req.messages[1].content.split("\n\n") == [f"B{i}" for i in range(1, 6)]

        tool_calls = normalized_chat_req.messages[1].tool_calls

        assert len(tool_calls) == 3

        tool_key = ["1", "21", "22"]
        assert all([t.function.name == f"tool{tool_key[i]}" for i, t in enumerate(tool_calls)])
        assert all([json.loads(t.function.arguments)["input"] == tool_key[i] for i, t in enumerate(tool_calls)])

    def test_assert_parsed_settings(
        self,
        normalizer_v7: InstructRequestNormalizerV7,
    ) -> None:
        chat_completion_request = ChatCompletionRequest(messages=[UserMessage(content="B")])
        parsed_request: InstructRequest[ChatMessage, Tool] = normalizer_v7.from_chat_completion_request(
            chat_completion_request
        )
        assert parsed_request.settings == ModelSettings.none()

    def test_continue_final_message_forwarded(self, normalizer_v7: InstructRequestNormalizerV7) -> None:
        request = ChatCompletionRequest[ChatMessage](
            messages=[UserMessage(content="a"), AssistantMessage(content="b")],
            continue_final_message=True,
        )
        result: InstructRequest[ChatMessage, Tool] = normalizer_v7.from_chat_completion_request(request)
        assert result.continue_final_message is True

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
        first_message = parsed_request.messages[0]
        assert isinstance(first_message, UserMessage)
        assert first_message.content == ""
        second_message = parsed_request.messages[1]
        assert isinstance(second_message, AssistantMessage)
        # Empty string content is passed through directly
        assert second_message.content == ""

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
        first_message = parsed_request.messages[0]
        assert isinstance(first_message, UserMessage)
        assert first_message.content == [
            TextChunk(text="A\n\nB\n\nC\n\nD"),
            ImageURLChunk(image_url="E"),
            TextChunk(text="G\n\nH"),
        ]

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
                        ThinkChunk(thinking="T"),
                        TextChunk(text="C"),
                        TextChunk(text="D"),
                    ]
                ),
            ]
        )
        parsed_request: InstructRequest[ChatMessage, Tool] = normalizer_v7.from_chat_completion_request(
            chat_completion_request
        )
        first_message = parsed_request.messages[0]
        assert isinstance(first_message, AssistantMessage)
        assert first_message.content == [
            TextChunk(text="A\n\nB"),
            ThinkChunk(thinking="T"),
            TextChunk(text="C\n\nD"),
        ]


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


class TestChatCompletionRequestNormalizationV13:
    @pytest.fixture(autouse=True)
    def normalizer_v13(self) -> InstructRequestNormalizerV13:
        return InstructRequestNormalizerV13(
            UserMessage, AssistantMessage, ToolMessage, SystemMessage, InstructRequest, None
        )

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
        assert parsed_request.messages == [
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
        assert parsed_request.messages == [
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
        assert parsed_request.messages == [
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
        ]

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
        assert parsed_request.messages == [
            UserMessage(content="A"),
            AssistantMessage(
                content="B",
                tool_calls=[
                    ToolCall(id="1", function=FunctionCall(name="foo", arguments="{}")),
                ],
            ),
            ToolMessage(content="C", tool_call_id="1"),
            ToolMessage(content="D", tool_call_id="2"),
        ]

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
        assert parsed_request.messages == [
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
        ]

    @pytest.mark.parametrize(
        ["system_message", "expected_system_message"],
        [
            (
                SystemMessage(content="A"),
                SystemMessage(content="A"),
            ),
            (
                SystemMessage(content=[TextChunk(text="A")]),
                SystemMessage(content="A"),
            ),
            (
                SystemMessage(content=[TextChunk(text="A"), TextChunk(text="B")]),
                SystemMessage(content="A\n\nB"),
            ),
            (
                SystemMessage(content=[TextChunk(text="A"), TextChunk(text="B"), ThinkChunk(thinking="C")]),
                SystemMessage(content=[TextChunk(text="A\n\nB"), ThinkChunk(thinking="C")]),
            ),
            (
                SystemMessage(
                    content=[
                        TextChunk(text="A"),
                        TextChunk(text="B"),
                        ThinkChunk(thinking="C"),
                        ThinkChunk(thinking="D"),
                    ]
                ),
                SystemMessage(content=[TextChunk(text="A\n\nB"), ThinkChunk(thinking="C"), ThinkChunk(thinking="D")]),
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
        assert parsed_request.messages == [expected_system_message, UserMessage(content="B")]

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
        assert parsed_request.messages == [
            SystemMessage(content="A"),
            SystemMessage(content="B"),
            UserMessage(content="C"),
            SystemMessage(content="D"),
        ]

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
        assert parsed_request.messages == [
            SystemMessage(content="A\n\nB"),
            SystemMessage(content="C"),
        ]

    def test_assert_parsed_settings(
        self,
        normalizer_v13: InstructRequestNormalizerV13,
    ) -> None:
        chat_completion_request: ChatCompletionRequest = self._mock_chat_completion(messages=[UserMessage(content="B")])
        parsed_request: InstructRequest[ChatMessage, Tool] = normalizer_v13.from_chat_completion_request(
            chat_completion_request
        )
        assert parsed_request.settings == ModelSettings.none()

    def test_continue_final_message_forwarded(self, normalizer_v13: InstructRequestNormalizerV13) -> None:
        request = ChatCompletionRequest[ChatMessage](
            messages=[UserMessage(content="a"), AssistantMessage(content="b")],
            continue_final_message=True,
        )
        result: InstructRequest[ChatMessage, Tool] = normalizer_v13.from_chat_completion_request(request)
        assert result.continue_final_message is True


class TestChatCompletionRequestNormalizationV15:
    @pytest.fixture(autouse=True)
    def normalizer_v15(self) -> InstructRequestNormalizerV15:
        return InstructRequestNormalizerV15(
            UserMessage,
            AssistantMessage,
            ToolMessage,
            SystemMessage,
            InstructRequest,
            ModelSettingsBuilder(
                reasoning_effort=EnumBuilder[ReasoningEffort](
                    values=list(ReasoningEffort), accepts_none=False, default=None
                )
            ),
        )

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
        assert parsed_request.settings == ModelSettings(reasoning_effort=reasoning_effort)

    def test_continue_final_message_forwarded(self, normalizer_v15: InstructRequestNormalizerV15) -> None:
        request = ChatCompletionRequest[ChatMessage](
            messages=[UserMessage(content="a"), AssistantMessage(content="b")],
            continue_final_message=True,
            reasoning_effort=ReasoningEffort.high,
        )
        result: InstructRequest[ChatMessage, Tool] = normalizer_v15.from_chat_completion_request(request)
        assert result.continue_final_message is True

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
        user_msg = parsed.messages[0]
        assert isinstance(user_msg, UserMessage)
        assert user_msg.content == "AB"
        assistant_msg = parsed.messages[1]
        assert isinstance(assistant_msg, AssistantMessage)
        assert assistant_msg.content == "CD"

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
        user_msg = parsed.messages[0]
        assert isinstance(user_msg, UserMessage)
        assert user_msg.content == "First\n\nSecond"

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
        user_msg = parsed.messages[0]
        assert isinstance(user_msg, UserMessage)
        assert user_msg.content == "AB\n\nCD"

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
        assistant_msg = parsed.messages[1]
        assert isinstance(assistant_msg, AssistantMessage)
        assert assistant_msg.content == "AB\n\nCD"

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
        tool_msg = parsed.messages[2]
        assert isinstance(tool_msg, ToolMessage)
        assert tool_msg.content == "XY"


@pytest.mark.parametrize(
    "version,expected_class,model_settings_builder",
    [
        (TokenizerVersion.v1, InstructRequestNormalizer, None),
        (TokenizerVersion.v2, InstructRequestNormalizer, None),
        (TokenizerVersion.v3, InstructRequestNormalizer, None),
        (TokenizerVersion.v7, InstructRequestNormalizerV7, None),
        (TokenizerVersion.v11, InstructRequestNormalizerV7, None),
        (TokenizerVersion.v13, InstructRequestNormalizerV13, None),
        (
            TokenizerVersion.v15,
            InstructRequestNormalizerV15,
            ModelSettingsBuilder(
                reasoning_effort=EnumBuilder[ReasoningEffort](
                    values=list(ReasoningEffort), accepts_none=False, default=None
                )
            ),
        ),
    ],
)
def test_get_normalizer_version_mapping(
    version: TokenizerVersion, expected_class: type, model_settings_builder: ModelSettingsBuilder
) -> None:
    normalizer = get_normalizer(version, model_settings_builder)
    assert isinstance(normalizer, expected_class)
    assert normalizer._model_settings_builder == model_settings_builder


class TestAssistantContentNarrowing:
    def test_accepts_text_and_think_chunks(self) -> None:
        r"""Normalizer accepts TextChunk and ThinkChunk in assistant messages."""
        normalizer = get_normalizer(TokenizerVersion.v13)
        request = mock_chat_completion(
            messages=[
                UserMessage(content="query"),
                AssistantMessage(content=[ThinkChunk(thinking="reasoning"), TextChunk(text="answer")]),
            ],
        )
        parsed: InstructRequest[ChatMessage, Tool] = normalizer.from_chat_completion_request(request)
        assistant_msg = parsed.messages[1]
        assert isinstance(assistant_msg, AssistantMessage)
        assert isinstance(assistant_msg.content, list)
        assert len(assistant_msg.content) == 2

    def test_accepts_string_content(self) -> None:
        r"""Normalizer accepts string content in assistant messages."""
        normalizer = get_normalizer(
            TokenizerVersion.v15,
            model_settings_builder=ModelSettingsBuilder(
                reasoning_effort=EnumBuilder[ReasoningEffort](
                    values=list(ReasoningEffort), accepts_none=False, default=None
                )
            ),
        )
        request = ChatCompletionRequest[ChatMessage](
            messages=[
                UserMessage(content="query"),
                AssistantMessage(content="plain text"),
            ],
            reasoning_effort=ReasoningEffort.high,
        )
        parsed: InstructRequest[ChatMessage, Tool] = normalizer.from_chat_completion_request(request)
        assistant_msg = parsed.messages[1]
        assert isinstance(assistant_msg, AssistantMessage)
        assert assistant_msg.content == "plain text"


class TestToolMessageContentChunk:
    @pytest.fixture()
    def normalizer_v15(self) -> InstructRequestNormalizerV15:
        return InstructRequestNormalizerV15(
            UserMessage,
            AssistantMessage,
            ToolMessage,
            SystemMessage,
            InstructRequest,
            ModelSettingsBuilder(
                reasoning_effort=EnumBuilder[ReasoningEffort](
                    values=list(ReasoningEffort), accepts_none=False, default=None
                )
            ),
        )

    @pytest.fixture()
    def normalizer_v13(self) -> InstructRequestNormalizerV13:
        return InstructRequestNormalizerV13(
            UserMessage, AssistantMessage, ToolMessage, SystemMessage, InstructRequest, None
        )

    def test_v15_preserves_non_text_tool_content(self, normalizer_v15: InstructRequestNormalizerV15) -> None:
        image_chunk = ImageURLChunk(image_url="https://example.com/image.png")
        request = ChatCompletionRequest(  # type: ignore[type-var]
            messages=[
                UserMessage(content="query"),
                AssistantMessage(tool_calls=[ToolCall(function=FunctionCall(name="fn", arguments="{}"), id="c1")]),
                ToolMessage(content=[image_chunk], tool_call_id="c1"),
            ],
            reasoning_effort=ReasoningEffort.high,
        )
        parsed: InstructRequest[ChatMessage, Tool] = normalizer_v15.from_chat_completion_request(request)
        tool_msg = parsed.messages[2]
        assert isinstance(tool_msg, ToolMessage)
        assert isinstance(tool_msg.content, list)
        assert tool_msg.content == [image_chunk]

    def test_v15_sorts_multimodal_tool_messages(self, normalizer_v15: InstructRequestNormalizerV15) -> None:
        image_chunk_1 = ImageURLChunk(image_url="https://example.com/img1.png")
        image_chunk_2 = ImageURLChunk(image_url="https://example.com/img2.png")
        request = ChatCompletionRequest(  # type: ignore[type-var]
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

        tool_msg_1 = parsed.messages[2]
        assert isinstance(tool_msg_1, ToolMessage)
        assert tool_msg_1.tool_call_id == "c1"
        assert tool_msg_1.content == [image_chunk_1]

        tool_msg_2 = parsed.messages[3]
        assert isinstance(tool_msg_2, ToolMessage)
        assert tool_msg_2.tool_call_id == "c2"
        assert tool_msg_2.content == [image_chunk_2]

    def test_pre_v15_rejects_non_text_tool_content(self) -> None:
        r"""Pre-V15 normalizer raises InvalidRequestException for non-text tool content."""
        normalizer = get_normalizer(TokenizerVersion.v13)
        request = mock_chat_completion(
            messages=[
                UserMessage(content="hi"),
                AssistantMessage(
                    tool_calls=[ToolCall(function=FunctionCall(name="fn", arguments="{}"), id="test12345")],
                ),
                ToolMessage(
                    content=[ImageURLChunk(image_url="data:image/png;base64,iVBORw0")],
                    tool_call_id="test12345",
                ),
            ]
        )
        with pytest.raises(InvalidRequestException, match="Unexpected content chunk types in tool message"):
            normalizer.from_chat_completion_request(request)

    def test_pre_v15_aggregates_text_tool_content(self, normalizer_v13: InstructRequestNormalizerV13) -> None:
        request = mock_chat_completion(
            messages=[
                UserMessage(content="query"),
                AssistantMessage(tool_calls=[ToolCall(function=FunctionCall(name="fn", arguments="{}"), id="c1")]),
                ToolMessage(content=[TextChunk(text="hello"), TextChunk(text="world")], tool_call_id="c1"),
            ],
        )
        parsed: InstructRequest[ChatMessage, Tool] = normalizer_v13.from_chat_completion_request(request)
        tool_msg = parsed.messages[2]
        assert isinstance(tool_msg, ToolMessage)
        assert isinstance(tool_msg.content, str)
        assert tool_msg.content == "hello\n\nworld"

    def test_pre_v15_rejects_audio_in_tool_content(self) -> None:
        r"""Pre-V15 normalizer raises InvalidRequestException for audio tool content."""
        normalizer = get_normalizer(TokenizerVersion.v13)
        request = mock_chat_completion(
            messages=[
                UserMessage(content="hi"),
                AssistantMessage(
                    tool_calls=[ToolCall(function=FunctionCall(name="fn", arguments="{}"), id="test12345")],
                ),
                ToolMessage(
                    content=[AudioChunk(input_audio=b"fake_audio_data")],
                    tool_call_id="test12345",
                ),
            ]
        )
        with pytest.raises(InvalidRequestException, match="Unexpected content chunk types in tool message"):
            normalizer.from_chat_completion_request(request)

    def test_base_normalizer_json_normalizes_tool_content(self) -> None:
        r"""Base normalizer (v1-v3) JSON-normalizes tool message content."""
        normalizer = InstructRequestNormalizer(
            UserMessage, AssistantMessage, ToolMessage, SystemMessage, InstructRequest, None
        )
        messy_json = '{"key" :  "value" ,  "num": 1}'
        request = mock_chat_completion(
            messages=[
                UserMessage(content="query"),
                AssistantMessage(tool_calls=[ToolCall(function=FunctionCall(name="fn", arguments="{}"), id="c1")]),
                ToolMessage(content=messy_json, tool_call_id="c1"),
            ],
        )
        parsed: InstructRequest[ChatMessage, Tool] = normalizer.from_chat_completion_request(request)
        tool_msg = parsed.messages[2]
        assert isinstance(tool_msg, ToolMessage)
        assert tool_msg.content == '{"key": "value", "num": 1}'

    def test_v7_skips_json_normalization_on_tool_content(self) -> None:
        r"""V7+ normalizers do not JSON-normalize tool message content."""
        normalizer = InstructRequestNormalizerV7(
            UserMessage, AssistantMessage, ToolMessage, SystemMessage, InstructRequest, None
        )
        messy_json = '{"key" :  "value" ,  "num": 1}'
        request = mock_chat_completion(
            messages=[
                UserMessage(content="query"),
                AssistantMessage(tool_calls=[ToolCall(function=FunctionCall(name="fn", arguments="{}"), id="c1")]),
                ToolMessage(content=messy_json, tool_call_id="c1"),
            ],
        )
        parsed: InstructRequest[ChatMessage, Tool] = normalizer.from_chat_completion_request(request)
        tool_msg = parsed.messages[2]
        assert isinstance(tool_msg, ToolMessage)
        assert tool_msg.content == messy_json


class TestSystemMessageContentChunk:
    def test_system_message_accepts_audio_chunk(self) -> None:
        msg = SystemMessage(content=[AudioChunk(input_audio="dGVzdA==")])
        assert isinstance(msg.content, list)
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], AudioChunk)

    def test_system_message_rejects_image_chunk(self) -> None:
        with pytest.raises(ValidationError):
            SystemMessage(content=[ImageURLChunk(image_url="https://example.com/image.png")])  # type: ignore[list-item]

    def test_v15_rejects_think_in_system_message(self) -> None:
        r"""V15 normalizer rejects ThinkChunk in system messages."""
        normalizer = get_normalizer(
            TokenizerVersion.v15,
            model_settings_builder=ModelSettingsBuilder(
                reasoning_effort=EnumBuilder[ReasoningEffort](
                    values=list(ReasoningEffort), accepts_none=False, default=None
                )
            ),
        )
        request = mock_chat_completion(
            messages=[
                SystemMessage(content=[TextChunk(text="hello"), ThinkChunk(thinking="thinking", closed=True)]),
                UserMessage(content="test"),
            ]
        )
        with pytest.raises(InvalidRequestException, match="ThinkChunk"):
            normalizer.from_chat_completion_request(request)

    def test_v7_preserves_audio_in_system_message(self) -> None:
        r"""V7 normalizer preserves AudioChunk in system messages."""
        normalizer = InstructRequestNormalizerV7.normalizer()
        request = mock_chat_completion(
            messages=[
                SystemMessage(content=[TextChunk(text="hello"), AudioChunk(input_audio=b"fake_audio_data")]),
                UserMessage(content="test"),
            ]
        )
        parsed: InstructRequest[ChatMessage, Tool] = normalizer.from_chat_completion_request(request)
        system_msg = parsed.messages[0]
        assert isinstance(system_msg, SystemMessage)
        assert isinstance(system_msg.content, list)
        assert len(system_msg.content) == 2
        assert isinstance(system_msg.content[0], TextChunk)
        assert isinstance(system_msg.content[1], AudioChunk)

    @pytest.mark.parametrize(
        "chunk",
        [
            pytest.param(AudioChunk(input_audio=b"fake_audio_data"), id="audio"),
            pytest.param(ThinkChunk(thinking="thinking", closed=True), id="think"),
        ],
    )
    def test_pre_v7_rejects_non_text_in_system_message(self, chunk: AudioChunk | ThinkChunk) -> None:
        r"""Pre-V7 normalizer rejects non-text chunks in system messages."""
        normalizer = InstructRequestNormalizer(
            UserMessage, AssistantMessage, ToolMessage, SystemMessage, InstructRequest, None
        )
        request = mock_chat_completion(
            messages=[
                SystemMessage(content=[TextChunk(text="hello"), chunk]),
                UserMessage(content="query"),
                AssistantMessage(content="answer"),
            ]
        )
        with pytest.raises(AssertionError):
            normalizer.from_chat_completion_request(request)

    def test_v15_preserves_audio_in_system_message(self) -> None:
        r"""V15 normalizer preserves AudioChunk in system messages."""
        normalizer = get_normalizer(
            TokenizerVersion.v15,
            model_settings_builder=ModelSettingsBuilder(
                reasoning_effort=EnumBuilder[ReasoningEffort](
                    values=list(ReasoningEffort), accepts_none=False, default=None
                )
            ),
        )
        request = ChatCompletionRequest[ChatMessage](
            messages=[
                SystemMessage(content=[TextChunk(text="hello"), AudioChunk(input_audio=b"fake_audio_data")]),
                UserMessage(content="test"),
            ],
            reasoning_effort=ReasoningEffort.high,
        )
        parsed: InstructRequest[ChatMessage, Tool] = normalizer.from_chat_completion_request(request)
        system_msg = parsed.messages[0]
        assert isinstance(system_msg, SystemMessage)
        assert isinstance(system_msg.content, list)
        assert len(system_msg.content) == 2
        assert isinstance(system_msg.content[0], TextChunk)
        assert isinstance(system_msg.content[1], AudioChunk)
