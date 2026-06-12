import pytest
from pydantic import ValidationError

from mistral_common.protocol.instruct.chunk import (
    AudioChunk,
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


class TestMessageContentChunkUnions:
    def test_assistant_rejects_image(self) -> None:
        with pytest.raises(ValidationError):
            AssistantMessage(content=[ImageURLChunk(image_url="data:image/png;base64,iVBORw0")])

    def test_assistant_rejects_audio(self) -> None:
        with pytest.raises(ValidationError):
            AssistantMessage(content=[AudioChunk(input_audio=b"fake")])

    def test_assistant_accepts_text_and_think(self) -> None:
        AssistantMessage(content=[ThinkChunk(thinking="r"), TextChunk(text="a")])

    def test_system_rejects_image(self) -> None:
        with pytest.raises(ValidationError):
            SystemMessage(content=[ImageURLChunk(image_url="data:image/png;base64,iVBORw0")])

    def test_system_accepts_audio(self) -> None:
        SystemMessage(content=[TextChunk(text="x"), AudioChunk(input_audio=b"fake")])

    def test_user_rejects_think(self) -> None:
        with pytest.raises(ValidationError):
            UserMessage(content=[ThinkChunk(thinking="r")])

    def test_user_accepts_image(self) -> None:
        UserMessage(content=[ImageURLChunk(image_url="data:image/png;base64,iVBORw0")])

    def test_tool_accepts_arbitrary_chunks(self) -> None:
        ToolMessage(content=[ImageURLChunk(image_url="data:image/png;base64,iVBORw0")], tool_call_id="c1")
