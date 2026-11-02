from mistral_common.protocol.instruct.converters import convert_openai_messages
from mistral_common.protocol.instruct.messages import SystemMessage


def test_convert_openai_messages_developer_role_maps_to_system_without_mutating_input() -> None:
    openai_message = {"role": "developer", "content": "You are a helpful assistant."}

    messages = convert_openai_messages([openai_message])

    assert messages == [SystemMessage(content="You are a helpful assistant.")]
    assert openai_message == {"role": "developer", "content": "You are a helpful assistant."}
