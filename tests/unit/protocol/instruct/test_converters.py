import pytest

from mistral_common.protocol.instruct.chunk import TextChunk
from mistral_common.protocol.instruct.converters import convert_openai_messages
from mistral_common.protocol.instruct.messages import SystemMessage, UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest


def test_convert_openai_messages_developer_role_maps_to_system() -> None:
    messages = convert_openai_messages([{"role": "developer", "content": "You are a helpful assistant."}])

    assert messages == [SystemMessage(content="You are a helpful assistant.")]


def test_convert_openai_messages_developer_role_with_chunk_content() -> None:
    messages = convert_openai_messages(
        [{"role": "developer", "content": [{"type": "text", "text": "You are a helpful assistant."}]}]
    )

    assert messages == [SystemMessage(content=[TextChunk(text="You are a helpful assistant.")])]


def test_convert_openai_messages_developer_role_does_not_mutate_input() -> None:
    openai_message = {"role": "developer", "content": "You are a helpful assistant."}

    convert_openai_messages([openai_message])

    assert openai_message == {"role": "developer", "content": "You are a helpful assistant."}


def test_from_openai_developer_role_maps_to_system() -> None:
    request = ChatCompletionRequest.from_openai(
        messages=[
            {"role": "developer", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
    )

    assert request.messages == [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="Hello"),
    ]


def test_convert_openai_messages_unknown_role_raises() -> None:
    with pytest.raises(ValueError, match="Unknown message role: bogus"):
        convert_openai_messages([{"role": "bogus", "content": "Hello"}])
