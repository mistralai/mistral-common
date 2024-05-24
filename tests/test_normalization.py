from typing import Dict, List

import pytest
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    ChatMessage,
    ChunkTypes,
    ContentChunk,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.normalize import InstructRequestNormalizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import Function, FunctionCall, Tool, ToolCall
from mistral_common.tokens.instruct.request import InstructRequest


class TestChatCompletionRequestNormalization:
    @pytest.fixture(autouse=True)
    def normalizer(self) -> InstructRequestNormalizer:
        return InstructRequestNormalizer(UserMessage, AssistantMessage, ToolMessage, SystemMessage, InstructRequest)

    def mock_chat_completion(self, messages: List[ChatMessage]) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            model="test",
            messages=messages,
            top_p=1.0,
            temperature=0.7,
        )

    def test_user_system_user(self, normalizer: InstructRequestNormalizer) -> None:
        chat_completion_request = self.mock_chat_completion(
            messages=[
                UserMessage(content="U"),
                SystemMessage(content="S"),
                UserMessage(content="U"),
            ]
        )

        parsed_request = normalizer.from_chat_completion_request(chat_completion_request)
        assert parsed_request.system_prompt == "S"

    def test_multiple_system(self, normalizer: InstructRequestNormalizer) -> None:
        chat_completion_request = self.mock_chat_completion(
            messages=[
                SystemMessage(content="S"),
                SystemMessage(content="S"),
                SystemMessage(content="S"),
            ]
        )

        parsed_request = normalizer.from_chat_completion_request(chat_completion_request)
        assert parsed_request.system_prompt == "S\n\nS\n\nS"

    def test_single_system(self, normalizer: InstructRequestNormalizer) -> None:
        chat_completion_request = self.mock_chat_completion(
            messages=[
                SystemMessage(content="S"),
            ]
        )

        parsed_request = normalizer.from_chat_completion_request(chat_completion_request)

        assert parsed_request.system_prompt == "S"

    def test_system_assistant_user(self, normalizer: InstructRequestNormalizer) -> None:
        chat_completion_request = self.mock_chat_completion(
            messages=[
                SystemMessage(content="S"),
                AssistantMessage(content="A"),
                UserMessage(content="U"),
            ]
        )

        parsed_request = normalizer.from_chat_completion_request(chat_completion_request)

        assert parsed_request.system_prompt == "S"

        first_message = parsed_request.messages[0]
        assert isinstance(first_message, UserMessage)
        assert first_message.content == ""
        assert parsed_request.system_prompt == "S"

    def test_assistant_system_user_adds_user(self, normalizer: InstructRequestNormalizer) -> None:
        chat_completion_request = self.mock_chat_completion(
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
        expected_content: List[str],
        normalizer: InstructRequestNormalizer,
    ) -> None:
        letter_to_cls: Dict[str, ChatMessage] = {
            "s": SystemMessage(content="s"),
            "u": UserMessage(content="u"),
            "a": AssistantMessage(content="a"),
            "u2": UserMessage(content="u2"),
        }

        chat_completion_request = self.mock_chat_completion(
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
        chat_completion_request = self.mock_chat_completion(
            messages=[
                UserMessage(content="foo"),
                UserMessage(
                    content=[ContentChunk(type=ChunkTypes.text, text="chunk")],
                ),
                UserMessage(content="foo"),
                UserMessage(
                    content=[ContentChunk(type=ChunkTypes.text, text="chunk")],
                ),
            ],
        )

        parsed_request = normalizer.from_chat_completion_request(chat_completion_request)

        first_message = parsed_request.messages[0]
        assert isinstance(first_message, UserMessage)
        assert first_message.content == "foo\n\nchunk\n\nfoo\n\nchunk"

    def test_many_chunks_in_user_message(self, normalizer: InstructRequestNormalizer) -> None:
        chat_completion_request = self.mock_chat_completion(
            messages=[
                UserMessage(content="foo"),
                UserMessage(
                    content=[
                        ContentChunk(type=ChunkTypes.text, text="chunk1"),
                        ContentChunk(type=ChunkTypes.text, text="chunk2"),
                        ContentChunk(type=ChunkTypes.text, text="chunk3"),
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

    def test_system_prompt_chunks_aggregated(self, normalizer: InstructRequestNormalizer) -> None:
        chat_completion_request = self.mock_chat_completion(
            messages=[
                UserMessage(content="foo"),
                SystemMessage(
                    content=[
                        ContentChunk(type=ChunkTypes.text, text="chunk1"),
                        ContentChunk(type=ChunkTypes.text, text="chunk2"),
                        ContentChunk(type=ChunkTypes.text, text="chunk3"),
                    ],
                ),
            ],
        )

        parsed_request = normalizer.from_chat_completion_request(chat_completion_request)
        assert parsed_request.system_prompt == "chunk1\n\nchunk2\n\nchunk3"

    def test_normalize_tools(self, normalizer: InstructRequestNormalizer) -> None:
        """
        Test doesnt really "normalize" anything but it checks that the tools are added to the
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
