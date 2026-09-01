import warnings
from collections.abc import Iterator
from typing import Any

import pytest
from pydantic import ValidationError

import mistral_common.deprecation
from mistral_common.protocol.instruct.messages import AssistantMessage, ChatMessage, SystemMessage, UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest


class TestValidateRequest:
    @pytest.fixture
    def clear_continue_warning(self) -> Iterator[None]:
        key = "ChatCompletionRequest.continue_final_message"
        mistral_common.deprecation._warned_keys.discard(key)
        yield
        mistral_common.deprecation._warned_keys.discard(key)

    @pytest.fixture
    def chat_request_raw(self) -> dict[str, Any]:
        return {"model": "test-model", "message": [UserMessage(content="foo")]}

    def test_request_random_seed_negative(self, chat_request_raw: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            ChatCompletionRequest(**chat_request_raw, random_seed=-1)

    def test_from_openai_preserves_zero_seed(self) -> None:
        request = ChatCompletionRequest.from_openai(
            messages=[{"role": "user", "content": "hello"}],
            seed=0,
        )

        assert request.random_seed == 0

    def test_legacy_continuation_sets_final_assistant_prefix_and_warns_once(self, clear_continue_warning: None) -> None:
        raw_assistant: dict[str, Any] = {"role": "assistant", "content": "bar", "prefix": False}
        raw_request: dict[str, Any] = {
            "messages": [{"role": "user", "content": "foo"}, raw_assistant],
            "continue_final_message": True,
        }

        with pytest.warns(DeprecationWarning, match="continue_final_message") as caught:
            request: ChatCompletionRequest[ChatMessage] = ChatCompletionRequest(**raw_request)
            ChatCompletionRequest(**raw_request)

        assert len(caught) == 1
        assert request.messages == [UserMessage(content="foo"), AssistantMessage(content="bar", prefix=True)]
        assert "continue_final_message" not in ChatCompletionRequest.model_fields
        assert "continue_final_message" not in request.model_dump()
        assert raw_assistant["prefix"] is False
        assert raw_request["continue_final_message"] is True

    @pytest.mark.parametrize(
        "legacy_value, initial_prefix, expected_prefix",
        [(False, True, True), (True, False, True), (True, True, True)],
    )
    def test_legacy_continuation_maps_and_copies_assistant_model(
        self,
        legacy_value: bool,
        initial_prefix: bool,
        expected_prefix: bool,
        clear_continue_warning: None,
    ) -> None:
        assistant = AssistantMessage(content="bar", prefix=initial_prefix)

        with pytest.warns(DeprecationWarning, match="continue_final_message"):
            request = ChatCompletionRequest[ChatMessage](  # type: ignore[call-arg]
                messages=[UserMessage(content="foo"), assistant],
                continue_final_message=legacy_value,
            )

        assert request.messages == [
            UserMessage(content="foo"),
            AssistantMessage(content="bar", prefix=expected_prefix),
        ]
        assert assistant.prefix == initial_prefix

    @pytest.mark.parametrize(
        ["legacy_value", "expected_prefix"],
        [(1, True), ("true", True), (0, False), ("false", False)],
    )
    def test_legacy_boolean_coercion_is_preserved(
        self,
        legacy_value: bool | int | str,
        expected_prefix: bool,
        clear_continue_warning: None,
    ) -> None:
        with pytest.warns(DeprecationWarning, match="continue_final_message"):
            request = ChatCompletionRequest[ChatMessage](  # type: ignore[call-arg]
                messages=[UserMessage(content="foo"), AssistantMessage(content="bar")],
                continue_final_message=legacy_value,
            )

        assert isinstance(request.messages[-1], AssistantMessage)
        assert request.messages[-1].prefix == expected_prefix

    def test_legacy_tuple_messages_maps_final_assistant(self, clear_continue_warning: None) -> None:
        messages = (UserMessage(content="foo"), AssistantMessage(content="bar"))

        with pytest.warns(DeprecationWarning, match="continue_final_message"):
            request = ChatCompletionRequest[ChatMessage](  # type: ignore[call-arg]
                messages=messages,  # type: ignore[arg-type]
                continue_final_message=True,
            )

        assert request.messages == [UserMessage(content="foo"), AssistantMessage(content="bar", prefix=True)]

    @pytest.mark.parametrize(
        "messages",
        [
            [UserMessage(content="foo"), SystemMessage(content="bar")],
            [],
        ],
    )
    def test_legacy_true_requires_final_assistant(
        self, messages: list[ChatMessage], clear_continue_warning: None
    ) -> None:
        with pytest.warns(DeprecationWarning, match="continue_final_message"):
            with pytest.raises(ValidationError, match="requires final message to be an assistant"):
                ChatCompletionRequest[ChatMessage](  # type: ignore[call-arg]
                    messages=messages, continue_final_message=True
                )

    @pytest.mark.parametrize("legacy_value, error_match", [(False, "prefix"), (True, "valid boolean")])
    def test_legacy_invalid_raw_prefix_is_validated(
        self, legacy_value: bool, error_match: str, clear_continue_warning: None
    ) -> None:
        legacy_messages: list[dict[str, Any]] = [{"role": "assistant", "content": "bar", "prefix": "invalid"}]

        with pytest.warns(DeprecationWarning, match="continue_final_message"):
            with pytest.raises(ValidationError, match=error_match):
                ChatCompletionRequest[ChatMessage](  # type: ignore[call-arg]
                    messages=legacy_messages,  # type: ignore[arg-type]
                    continue_final_message=legacy_value,
                )

        assert legacy_messages[-1]["prefix"] == "invalid"

    def test_legacy_invalid_value_validates_before_warning(self, clear_continue_warning: None) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("error")
            with pytest.raises(ValidationError, match="valid boolean"):
                ChatCompletionRequest[UserMessage](  # type: ignore[call-arg]
                    messages=[UserMessage(content="foo")],
                    continue_final_message="not-a-bool",
                )

        assert caught == []
