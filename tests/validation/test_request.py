import pytest
from pydantic import ValidationError

from mistral_common.protocol.instruct.messages import AssistantMessage, SystemMessage, UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest, InstructRequest


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

    def test_legacy_continuation_sets_final_assistant_prefix_and_warns_once(self) -> None:
        raw_assistant = {"role": "assistant", "content": "bar", "prefix": False}
        raw_request = {
            "messages": [{"role": "user", "content": "foo"}, raw_assistant],
            "continue_final_message": True,
        }

        with pytest.warns(DeprecationWarning, match="continue_final_message") as caught:
            request = ChatCompletionRequest(**raw_request)
            ChatCompletionRequest(**raw_request)

        assert len(caught) == 1
        assert isinstance(request.messages[-1], AssistantMessage)
        assert request.messages[-1].prefix is True
        assert "continue_final_message" not in ChatCompletionRequest.model_fields
        assert "continue_final_message" not in request.model_dump()
        assert raw_assistant["prefix"] is False
        assert raw_request["continue_final_message"] is True
        assert "continue_final_message" not in InstructRequest.model_fields

    def test_legacy_false_preserves_existing_assistant_prefix(self) -> None:
        assistant = AssistantMessage(content="bar", prefix=True)

        request = ChatCompletionRequest(
            messages=[UserMessage(content="foo"), assistant],
            continue_final_message=False,
        )

        assert isinstance(request.messages[-1], AssistantMessage)
        assert request.messages[-1].prefix is True
        assert assistant.prefix is True
        assert "continue_final_message" not in ChatCompletionRequest.model_fields
        assert "continue_final_message" not in request.model_dump()

    def test_legacy_true_does_not_mutate_assistant_model(self) -> None:
        assistant = AssistantMessage(content="bar", prefix=False)

        request = ChatCompletionRequest(
            messages=[UserMessage(content="foo"), assistant],
            continue_final_message=True,
        )

        assert isinstance(request.messages[-1], AssistantMessage)
        assert request.messages[-1].prefix is True
        assert assistant.prefix is False

    @pytest.mark.parametrize(
        "messages",
        [
            [UserMessage(content="foo"), SystemMessage(content="bar")],
            [],
        ],
    )
    def test_legacy_true_requires_final_assistant(self, messages: list[object]) -> None:
        with pytest.raises(ValidationError, match="requires final message to be an assistant"):
            ChatCompletionRequest(messages=messages, continue_final_message=True)

    def test_legacy_prefix_validation_is_preserved(self) -> None:
        with pytest.raises(ValidationError, match="prefix"):
            ChatCompletionRequest(
                messages=[{"role": "assistant", "content": "bar", "prefix": "invalid"}],
                continue_final_message=False,
            )
