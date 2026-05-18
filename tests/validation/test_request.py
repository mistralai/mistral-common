import pytest
from pydantic import ValidationError

from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest


class TestValidateRequest:
    @pytest.fixture
    def chat_request_raw(self) -> dict:
        return {"model": "test-model", "message": [UserMessage(content="foo")]}

    def test_request_random_seed_negative(self, chat_request_raw: dict) -> None:
        with pytest.raises(ValidationError):
            ChatCompletionRequest(**chat_request_raw, random_seed=-1)

    def test_from_openai_preserves_zero_seed(self) -> None:
        request = ChatCompletionRequest.from_openai(
            messages=[{"role": "user", "content": "hello"}],
            seed=0,
        )

        assert request.random_seed == 0
