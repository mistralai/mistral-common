from collections.abc import Callable
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from mistral_common.protocol.instruct.chunk import TextChunk
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)


@pytest.mark.parametrize(
    ("openai_message", "expected"),
    [
        pytest.param(
            {"role": "user", "content": "Hello", "name": "user1"},
            UserMessage(content="Hello"),
            id="user-text-message-name",
        ),
        pytest.param(
            {"role": "user", "content": [{"type": "text", "text": "Hello"}], "name": "user1"},
            UserMessage(content=[TextChunk(text="Hello")]),
            id="user-chunked-message-name",
        ),
        pytest.param(
            {"role": "system", "content": "Be helpful", "name": "sys"},
            SystemMessage(content="Be helpful"),
            id="system-message-name",
        ),
        pytest.param(
            {"role": "tool", "content": "42", "tool_call_id": "c1", "extra": "ignored"},
            ToolMessage(content="42", tool_call_id="c1"),
            id="tool-message-extra-field",
        ),
        pytest.param(
            {"role": "assistant", "content": "Hi", "refusal": None, "audio": None},
            AssistantMessage(content="Hi"),
            id="assistant-message-openai-fields",
        ),
    ],
)
def test_message_from_openai_drops_unknown_fields(openai_message: dict[str, Any], expected: BaseMessage) -> None:
    message_type = expected.__class__

    assert message_type.from_openai(openai_message) == expected


@pytest.mark.parametrize(
    ("constructor", "field"),
    [
        pytest.param(
            lambda: UserMessage(content="Hello", name="user1"),  # type: ignore[call-arg]
            "name",
            id="user-message-name-rejected",
        ),
        pytest.param(
            lambda: SystemMessage(content="Be helpful", name="sys"),  # type: ignore[call-arg]
            "name",
            id="system-message-name-rejected",
        ),
        pytest.param(
            lambda: TextChunk(text="Hello", extra="bad"),  # type: ignore[call-arg]
            "extra",
            id="text-chunk-extra",
        ),
    ],
)
def test_direct_construction_rejects_openai_extra_fields(constructor: Callable[[], BaseModel], field: str) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted") as error:
        constructor()

    assert error.value.errors()[0]["type"] == "extra_forbidden"
    assert error.value.errors()[0]["loc"] == (field,)
