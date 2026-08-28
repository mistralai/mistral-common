import warnings
from collections.abc import Iterator

import pytest
from pydantic import ValidationError

import mistral_common.deprecation
from mistral_common.protocol.instruct.messages import AssistantMessage, SystemMessage, UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest, InstructRequest


class TestValidateRequest:
    @pytest.fixture
    def clear_continue_warning(self) -> Iterator[None]:
        key = "ChatCompletionRequest.continue_final_message"
        mistral_common.deprecation._warned_keys.discard(key)
        yield
        mistral_common.deprecation._warned_keys.discard(key)

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

    def test_legacy_continuation_sets_final_assistant_prefix_and_warns_once(self, clear_continue_warning: None) -> None:
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

    def test_legacy_false_preserves_existing_assistant_prefix(self, clear_continue_warning: None) -> None:
        assistant = AssistantMessage(content="bar", prefix=True)

        with pytest.warns(DeprecationWarning, match="continue_final_message"):
            request = ChatCompletionRequest(
                messages=[UserMessage(content="foo"), assistant],
                continue_final_message=False,
            )

        assert isinstance(request.messages[-1], AssistantMessage)
        assert request.messages[-1].prefix is True
        assert assistant.prefix is True
        assert "continue_final_message" not in ChatCompletionRequest.model_fields
        assert "continue_final_message" not in request.model_dump()

    def test_legacy_true_does_not_mutate_assistant_model(self, clear_continue_warning: None) -> None:
        assistant = AssistantMessage(content="bar", prefix=False)

        with pytest.warns(DeprecationWarning, match="continue_final_message"):
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
    def test_legacy_true_requires_final_assistant(self, messages: list[object], clear_continue_warning: None) -> None:
        with pytest.warns(DeprecationWarning, match="continue_final_message"):
            with pytest.raises(ValidationError, match="requires final message to be an assistant"):
                ChatCompletionRequest(messages=messages, continue_final_message=True)

    def test_legacy_prefix_validation_is_preserved(self, clear_continue_warning: None) -> None:
        with pytest.warns(DeprecationWarning, match="continue_final_message"):
            with pytest.raises(ValidationError, match="prefix"):
                ChatCompletionRequest(
                    messages=[{"role": "assistant", "content": "bar", "prefix": "invalid"}],
                    continue_final_message=False,
                )

    def test_instruct_request_retains_continuation_until_migration(self) -> None:
        request = InstructRequest(messages=[UserMessage(content="foo")], continue_final_message=True)

        assert request.continue_final_message is True

    def test_legacy_invalid_value_validates_before_warning(self, clear_continue_warning: None) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValidationError, match="valid boolean"):
                ChatCompletionRequest(
                    messages=[UserMessage(content="foo")],
                    continue_final_message="not-a-bool",
                )
