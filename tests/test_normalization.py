import json
from typing import Dict, List, Union

import pytest

from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    ChatMessage,
    ChunkTypes,
    ContentChunk,
    FinetuningAssistantMessage,
    FinetuningMessage,
    SystemMessage,
    TextChunk,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.normalize import (
    InstructRequestNormalizer,
    InstructRequestNormalizerV7,
    InstructRequestNormalizerV13,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import Function, FunctionCall, Tool, ToolCall
from mistral_common.tokens.instruct.request import InstructRequest


def mock_chat_completion(messages: List[ChatMessage]) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test",
        messages=messages,
        top_p=1.0,
        temperature=0.7,
    )


class TestChatCompletionRequestNormalization:
    @pytest.fixture(autouse=True)
    def normalizer(self) -> InstructRequestNormalizer:
        return InstructRequestNormalizer(UserMessage, AssistantMessage, ToolMessage, SystemMessage, InstructRequest)

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
        roles: List[str],
        expected_roles: List[str],
        expected_content: List[Union[List[ContentChunk], str]],
        normalizer: InstructRequestNormalizer,
    ) -> None:
        letter_to_cls: Dict[str, ChatMessage] = {
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
                    tool_calls=[
                        ToolCall(function=FunctionCall(name="tool1", arguments='{"input": "11"}')),  # clean json string
                    ]
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


class TestChatCompletionRequestNormalizationV7:
    @pytest.fixture(autouse=True)
    def normalizer_v7(self) -> InstructRequestNormalizerV7:
        return InstructRequestNormalizerV7(UserMessage, AssistantMessage, ToolMessage, SystemMessage, InstructRequest)

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


class TestFineTuningNormalizer:
    @pytest.fixture(autouse=True)
    def normalizer(self) -> InstructRequestNormalizer:
        return InstructRequestNormalizer(
            UserMessage, FinetuningAssistantMessage, ToolMessage, SystemMessage, InstructRequest
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
        return InstructRequestNormalizerV13(UserMessage, AssistantMessage, ToolMessage, SystemMessage, InstructRequest)

    def _mock_chat_completion(self, messages: List[ChatMessage]) -> ChatCompletionRequest:
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
