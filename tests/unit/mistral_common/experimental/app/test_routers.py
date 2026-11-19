from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from mistral_common.experimental.app.main import create_app
from mistral_common.protocol.instruct.validator import ValidationMode
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

_TEKKEN = MistralTokenizer._data_path() / "tekken_240911.json"
_USER: dict[str, Any] = {"role": "user", "content": "Hello"}
_ASSISTANT: dict[str, Any] = {"role": "assistant", "content": "Hi"}
_TOOL: dict[str, Any] = {"type": "function", "function": {"name": "f", "parameters": {}}}
_TOOL_CALL: dict[str, Any] = {"id": "call", "type": "function", "function": {"name": "f", "arguments": "{}"}}
_UNDECODABLE_IMAGE: dict[str, Any] = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}

# `max_completion_tokens` is not a `ChatCompletionRequest` field, so these bodies are parsed as OpenAI requests.
_REJECTED_REQUESTS = [
    pytest.param(
        {"messages": [{"role": "function", "content": "Hello"}]},
        "Unknown message role: function",
        id="openai-conversion-value-error",
    ),
    pytest.param(
        {"messages": [_USER], "continue_final_message": True, "max_completion_tokens": 16},
        "continue_final_message=True requires final message to be an assistant.",
        id="openai-conversion-mistral-common-exception",
    ),
    pytest.param(
        {"messages": [_USER, _ASSISTANT], "max_completion_tokens": 16},
        "Expected last role User or Tool (or Assistant with prefix) for serving but got assistant",
        id="openai-invalid-message-structure",
    ),
    pytest.param(
        {"messages": [_USER, _ASSISTANT]},
        "Expected last role User or Tool (or Assistant with prefix) for serving but got assistant",
        id="mistral-invalid-message-structure",
    ),
    pytest.param(
        {"messages": [_USER, {"role": "tool", "content": "42", "tool_call_id": "abcdefghi"}]},
        "Unexpected role 'tool' after role 'user'",
        id="tool-result-without-tool-call",
    ),
    pytest.param(
        {
            "messages": [
                _USER,
                {"role": "assistant", "tool_calls": [_TOOL_CALL]},
                {"role": "tool", "content": "42", "tool_call_id": "call"},
            ],
            "tools": [_TOOL],
        },
        "Tool call id was call but must be a-z, A-Z, 0-9, with a length of 9.",
        id="invalid-tool-call-id",
    ),
    pytest.param(
        {"messages": [{"role": "user", "content": [_UNDECODABLE_IMAGE]}]},
        "Invalid image data URL data:image/png;base64,AAAA: payload is not a decodable image.",
        id="undecodable-image",
    ),
]


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(_TEKKEN, ValidationMode.test))


@pytest.mark.parametrize(("body", "detail"), _REJECTED_REQUESTS)
def test_tokenize_returns_400_for_rejected_request(client: TestClient, body: dict[str, Any], detail: str) -> None:
    response = client.post("/v1/tokenize", json=body)

    assert response.status_code == 400
    assert response.json() == {"detail": detail}


@pytest.mark.parametrize(("body", "detail"), _REJECTED_REQUESTS)
def test_generate_returns_400_for_rejected_request_without_calling_the_engine(
    client: TestClient, body: dict[str, Any], detail: str
) -> None:
    with patch("mistral_common.experimental.app.routers.requests.post") as mock_engine:
        response = client.post("/v1/chat/completions", json=body)

    assert response.status_code == 400
    assert response.json() == {"detail": detail}
    mock_engine.assert_not_called()


def test_tokenize_and_generate_still_accept_valid_requests(client: TestClient) -> None:
    body = {"messages": [_USER], "max_completion_tokens": 16}

    tokenized = client.post("/v1/tokenize", json=body)
    with patch("mistral_common.experimental.app.routers.requests.post") as mock_engine:
        mock_engine.return_value.status_code = 200
        mock_engine.return_value.json.return_value = {"tokens": [1267, 2]}
        generated = client.post("/v1/chat/completions", json=body)

    assert tokenized.status_code == 200
    assert generated.status_code == 200
    assert mock_engine.call_args.kwargs["json"]["prompt"] == tokenized.json()
    assert mock_engine.call_args.kwargs["json"]["max_completion_tokens"] == 16
