import pytest

from mistral_common.protocol.instruct.chunk import TextChunk, ThinkChunk
from mistral_common.protocol.instruct.messages import AssistantMessage, ReasoningFieldFormat, UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest


@pytest.mark.parametrize("reasoning_field_format", ["reasoning", "reasoning_content"])
def test_assistant_message_to_openai_accepts_plain_string_reasoning_field_format(
    reasoning_field_format: str,
) -> None:
    message = AssistantMessage(content=[ThinkChunk(thinking="Let me think", closed=True), TextChunk(text="Done")])

    result = message.to_openai(reasoning_field_format=reasoning_field_format)  # type: ignore[arg-type]

    assert result == {"role": "assistant", reasoning_field_format: "Let me think", "content": "Done"}


def test_assistant_message_to_openai_plain_string_matches_enum() -> None:
    message = AssistantMessage(content=[ThinkChunk(thinking="Let me think", closed=True), TextChunk(text="Done")])

    assert message.to_openai(reasoning_field_format="reasoning") == message.to_openai(  # type: ignore[arg-type]
        reasoning_field_format=ReasoningFieldFormat.reasoning
    )


def test_chat_completion_request_to_openai_accepts_plain_string_reasoning_field_format() -> None:
    request = ChatCompletionRequest(
        messages=[
            UserMessage(content="Think then answer."),
            AssistantMessage(content=[ThinkChunk(thinking="Let me think", closed=True), TextChunk(text="Done")]),
        ]
    )

    result = request.to_openai(reasoning_field_format="reasoning")  # type: ignore[arg-type]

    assert result["messages"][1] == {"role": "assistant", "reasoning": "Let me think", "content": "Done"}
