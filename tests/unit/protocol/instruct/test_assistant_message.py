import pytest

from mistral_common.protocol.instruct.chunk import TextChunk, ThinkChunk
from mistral_common.protocol.instruct.messages import AssistantMessage


@pytest.mark.parametrize("reasoning_field_format", ["reasoning", "reasoning_content"])
def test_assistant_message_to_openai_accepts_plain_string_reasoning_field_format(
    reasoning_field_format: str,
) -> None:
    message = AssistantMessage(content=[ThinkChunk(thinking="Let me think", closed=True), TextChunk(text="Done")])

    result = message.to_openai(reasoning_field_format=reasoning_field_format)  # type: ignore[arg-type]

    assert result == {"role": "assistant", reasoning_field_format: "Let me think", "content": "Done"}
