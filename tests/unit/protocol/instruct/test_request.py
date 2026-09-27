import pytest
from pydantic import ValidationError

from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest


def test_request_from_openai_drops_unsupported_fields() -> None:
    request = ChatCompletionRequest.from_openai(
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.5,
        stream=False,
        n=2,
        logprobs=True,
        frequency_penalty=0.1,
        unknown_field="value",
    )

    assert request == ChatCompletionRequest(messages=[UserMessage(content="Hello")], temperature=0.5)


def test_request_from_openai_rejects_conflicting_seed_names() -> None:
    with pytest.raises(ValueError, match="Cannot specify both `seed` and `random_seed`"):
        ChatCompletionRequest.from_openai(
            messages=[{"role": "user", "content": "Hello"}],
            seed=7,
            random_seed=7,
        )


def test_request_from_openai_rejects_invalid_recognized_value() -> None:
    with pytest.raises(ValidationError, match="temperature"):
        ChatCompletionRequest.from_openai(
            messages=[{"role": "user", "content": "Hello"}],
            temperature="not-a-number",
        )
