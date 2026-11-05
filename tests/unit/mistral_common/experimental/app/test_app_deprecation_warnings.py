import warnings
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import mistral_common.deprecation
from mistral_common.experimental.app.main import create_app
from mistral_common.experimental.app.models import EngineChatCompletionRequest
from mistral_common.protocol.instruct.messages import AssistantMessage, UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.experimental.test_app import MockResponse

_TEKKEN = MistralTokenizer._data_path() / "tekken_240911.json"


@pytest.mark.parametrize(
    "sampling",
    [
        pytest.param({"temperature": 0.1, "top_p": 0.9, "max_tokens": 5, "random_seed": 3}, id="mistral"),
        pytest.param({"temperature": 0.1, "top_p": 0.9, "max_tokens": 5, "seed": 3}, id="openai"),
    ],
)
def test_app_forwards_sampling_fields_without_deprecation_warnings(sampling: dict[str, Any]) -> None:
    deprecations = set(ChatCompletionRequest._deprecated_fields.values())
    mistral_common.deprecation._warned_keys.difference_update(deprecations)
    tokenizer = MistralTokenizer.from_file(_TEKKEN)
    client = TestClient(create_app(tokenizer))
    body = {"messages": [{"role": "user", "content": "Hello, world!"}], **sampling}
    output_tokens = tokenizer.instruct_tokenizer.encode_assistant_message(  # type: ignore[attr-defined]
        AssistantMessage(content="Hi"), False
    )

    assert client.post("/v1/tokenize", json=body).status_code == 200
    with patch("mistral_common.experimental.app.routers.requests.post") as mock_generate:
        mock_generate.return_value = MockResponse(200, {"tokens": output_tokens})
        response = client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    sent = mock_generate.call_args.kwargs["json"]
    assert {key: sent[key] for key in sampling} == sampling
    assert not deprecations & mistral_common.deprecation._warned_keys


def test_engine_request_keeps_sampling_fields_without_warnings() -> None:
    sampling: dict[str, Any] = {"temperature": 0.1, "top_p": 0.9, "max_tokens": 5, "random_seed": 3}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        request = EngineChatCompletionRequest(messages=[UserMessage(content="Hi")], **sampling)

    assert request.model_dump(include=set(sampling)) == sampling
    assert set(sampling) <= set(EngineChatCompletionRequest.model_fields)
