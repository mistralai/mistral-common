from mistral_common.protocol.instruct.converters import convert_openai_messages
from mistral_common.protocol.instruct.messages import AssistantMessage, SystemMessage, UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest


def test_convert_openai_messages_developer_role_maps_to_system_without_mutating_input() -> None:
    openai_message = {"role": "developer", "content": "You are a helpful assistant."}

    messages = convert_openai_messages([openai_message])

    assert messages == [SystemMessage(content="You are a helpful assistant.")]
    assert openai_message == {"role": "developer", "content": "You are a helpful assistant."}


def test_from_openai_chat_completion_request_with_developer_role() -> None:
    openai_request = {
        "model": "gpt-5",
        "messages": [
            {"role": "developer", "content": "You are a concise coding assistant."},
            {"role": "user", "content": "How do I reverse a list in Python?"},
            {"role": "assistant", "content": "Use slicing: `lst[::-1]`."},
            {"role": "user", "content": "And in place?"},
        ],
        "temperature": 0.2,
        "max_tokens": 256,
    }

    request = ChatCompletionRequest.from_openai(**openai_request)

    assert request.messages == [
        SystemMessage(content="You are a concise coding assistant."),
        UserMessage(content="How do I reverse a list in Python?"),
        AssistantMessage(content="Use slicing: `lst[::-1]`."),
        UserMessage(content="And in place?"),
    ]
    assert request.temperature == 0.2
    assert request.max_tokens == 256
