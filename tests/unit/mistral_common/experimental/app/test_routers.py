from typing import cast
from unittest.mock import patch

import pytest
import requests
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.testclient import TestClient

from mistral_common.experimental.app.main import create_app
from mistral_common.experimental.app.models import OpenAIChatCompletionRequest, get_settings
from mistral_common.protocol.instruct.chunk import ContentChunk, TextChunk, ThinkChunk
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    ChatMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.normalize import InstructRequestNormalizerV13
from mistral_common.protocol.instruct.request import ChatCompletionRequest, InstructRequest
from mistral_common.protocol.instruct.tool_calls import Function, FunctionCall, Tool, ToolCall
from mistral_common.protocol.instruct.validator import MistralRequestValidatorV13, ValidationMode
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy, TokenizerVersion
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV13
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.utils import get_special_tokens, quick_vocab

_TEKKEN_FILE = MistralTokenizer._data_path() / "tekken_240911.json"
_SPM_FILE = MistralTokenizer._data_path() / "mistral_instruct_tokenizer_241114.model.v7m1"


@pytest.fixture(scope="module")
def tekken_app() -> FastAPI:
    return create_app(_TEKKEN_FILE, ValidationMode.test)


@pytest.fixture(scope="module")
def tekken_tokenizer() -> MistralTokenizer:
    return MistralTokenizer.from_file(_TEKKEN_FILE, ValidationMode.test)


@pytest.fixture(scope="module")
def spm_tokenizer() -> MistralTokenizer:
    return MistralTokenizer.from_file(_SPM_FILE, ValidationMode.finetuning)


@pytest.fixture(scope="module")
def mistral_tokenizer_v13() -> MistralTokenizer:
    return MistralTokenizer(
        instruct_tokenizer=InstructTokenizerV13(
            Tekkenizer(
                quick_vocab(
                    [
                        b"Hello",
                        b",",
                        b" ",
                        b"world",
                        b"!",
                        b"How",
                        b"can",
                        b"I",
                        b"assist",
                        b"you",
                        b"today",
                        b"?",
                        b'"',
                        b"a",
                        b"b",
                        b"c",
                        b"d",
                        b"{",
                        b"}",
                        b"1",
                        b"2",
                        b"call",
                        b"_",
                        b":",
                    ]
                ),
                special_tokens=get_special_tokens(TokenizerVersion.v13, add_think=True),
                pattern=r".+",  # single token, whole string
                vocab_size=256 + 100,
                num_special_tokens=100,
                version=TokenizerVersion.v13,
            )
        ),
        validator=MistralRequestValidatorV13(),
        request_normalizer=InstructRequestNormalizerV13(
            UserMessage, AssistantMessage, ToolMessage, SystemMessage, InstructRequest, None
        ),
    )


@pytest.fixture(scope="module")
def spm_app() -> FastAPI:
    return create_app(_SPM_FILE, ValidationMode.finetuning)


@pytest.fixture(scope="module")
def tekken_client(tekken_app: FastAPI) -> TestClient:
    return TestClient(tekken_app)


@pytest.fixture(scope="module")
def tekken_v13_app(mistral_tokenizer_v13: MistralTokenizer) -> FastAPI:
    return create_app(mistral_tokenizer_v13)


@pytest.fixture(scope="module")
def tekken_v13_client(tekken_v13_app: FastAPI) -> TestClient:
    return TestClient(tekken_v13_app)


@pytest.fixture(scope="module")
def spm_client(spm_app: FastAPI) -> TestClient:
    return TestClient(spm_app)


def _client(request: pytest.FixtureRequest, app_kind: str) -> TestClient:
    if app_kind == "tekken":
        return cast("TestClient", request.getfixturevalue("tekken_client"))
    assert app_kind == "spm"
    return cast("TestClient", request.getfixturevalue("spm_client"))


def _tokenizer_of(client: TestClient) -> MistralTokenizer:
    app = cast("FastAPI", client.app)
    return cast("MistralTokenizer", app.dependency_overrides[get_settings]().tokenizer)


def _chat_request(app_kind: str) -> ChatCompletionRequest:
    r"""Build a fresh chat request for the given app kind (tekken requests carry a tool)."""
    if app_kind == "tekken":
        messages: list[ChatMessage] = [
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="Hello, world!"),
        ]
        tools = [
            Tool(
                function=Function(
                    name="get_current_weather",
                    description="Get the current weather in a given location",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                    },
                )
            )
        ]
        return ChatCompletionRequest(messages=messages, tools=tools)
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="Hello, how are you ?!"),
        AssistantMessage(content="I'm fine, thank you ! How can I help you ?"),
    ]
    return ChatCompletionRequest(messages=messages)


def _weather_tool_calls() -> list[ToolCall]:
    return [
        ToolCall(
            id="call_1",
            function=FunctionCall(name="get_current_weather", arguments='{"location": "Paris"}'),
        ),
        ToolCall(
            id="call_2",
            function=FunctionCall(
                name="git_clone", arguments='{"repository": "https://github.com/mistralai/mistral-common"}'
            ),
        ),
    ]


class MockResponse:
    def __init__(self, status_code: int, json_data: list | dict | None = None, text: str | None = None) -> None:
        self.json_data = json_data
        self.status_code = status_code
        self.text = text

    def json(self) -> list | dict:
        if self.json_data is None:
            raise ValueError("No JSON data available")
        return self.json_data


@pytest.mark.parametrize("app_kind", ["tekken", "spm"])
def test_redirect_to_docs(app_kind: str, request: pytest.FixtureRequest) -> None:
    response = _client(request, app_kind).get("/")
    assert response.status_code == 200


@pytest.mark.parametrize("app_kind", ["tekken", "spm"])
@pytest.mark.parametrize("request_format", ["mistral", "openai"])
def test_tokenize_request(app_kind: str, request_format: str, request: pytest.FixtureRequest) -> None:
    client = _client(request, app_kind)
    chat_request = _chat_request(app_kind)
    expected_tokens = _tokenizer_of(client).encode_chat_completion(chat_request).tokens
    if request_format == "mistral":
        body = jsonable_encoder(chat_request)
    else:
        body = jsonable_encoder(OpenAIChatCompletionRequest.model_validate(chat_request.to_openai()))

    response = client.post("/v1/tokenize", json=body)

    assert response.status_code == 200
    assert response.json() == expected_tokens


@pytest.mark.parametrize("app_kind", ["tekken", "spm"])
def test_tokenize_request_empty_messages(app_kind: str, request: pytest.FixtureRequest) -> None:
    response = _client(request, app_kind).post("/v1/tokenize", json={"messages": []})
    assert response.status_code == 400
    assert response.json()["detail"] == "Messages list cannot be empty."


@pytest.mark.parametrize("app_kind", ["tekken", "spm"])
@pytest.mark.parametrize("policy", [SpecialTokenPolicy.KEEP, SpecialTokenPolicy.IGNORE])
def test_detokenize_string(app_kind: str, policy: SpecialTokenPolicy, request: pytest.FixtureRequest) -> None:
    client = _client(request, app_kind)
    tokenizer = _tokenizer_of(client)
    prompt = "Hello, world!"
    encoded_prompt = tokenizer.instruct_tokenizer.tokenizer.encode(prompt, bos=True, eos=True)

    response = client.post("/v1/detokenize/string", json={"tokens": encoded_prompt, "special_token_policy": policy})

    assert response.status_code == 200
    assert response.json() == tokenizer.instruct_tokenizer.tokenizer.decode(encoded_prompt, special_token_policy=policy)


@pytest.mark.parametrize("app_kind", ["tekken", "spm"])
def test_detokenize_string_empty_tokens(app_kind: str, request: pytest.FixtureRequest) -> None:
    response = _client(request, app_kind).post(
        "/v1/detokenize/string", json={"tokens": [], "special_token_policy": SpecialTokenPolicy.IGNORE}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Tokens list cannot be empty."


@pytest.mark.parametrize("app_kind", ["tekken", "spm"])
def test_detokenize_string_raise_policy(app_kind: str, request: pytest.FixtureRequest) -> None:
    client = _client(request, app_kind)
    tokenizer = _tokenizer_of(client)
    prompt = "Hello, world!"
    encoded_prompt = tokenizer.instruct_tokenizer.tokenizer.encode(prompt, bos=True, eos=True)
    with pytest.raises(ValueError) as exc_info:
        tokenizer.instruct_tokenizer.tokenizer.decode(encoded_prompt, special_token_policy=SpecialTokenPolicy.RAISE)

    response = client.post(
        "/v1/detokenize/string", json={"tokens": encoded_prompt, "special_token_policy": SpecialTokenPolicy.RAISE}
    )

    assert response.status_code == 400
    assert response.json()["detail"] == str(exc_info.value)


def _encode_tool_calls_only(tokenizer: MistralTokenizer, prefix: bool) -> list[int]:
    encoded_tool_calls: list[int] = tokenizer.instruct_tokenizer._encode_tool_calls_in_assistant_message(  # type: ignore[attr-defined]
        AssistantMessage(tool_calls=_weather_tool_calls())
    )
    if not prefix:
        encoded_tool_calls.append(tokenizer.instruct_tokenizer.tokenizer.eos_id)
    return encoded_tool_calls


@pytest.mark.parametrize("app_kind", ["tekken", "spm"])
@pytest.mark.parametrize("prefix", [False, True])
def test_detokenize_assistant_content_only(app_kind: str, prefix: bool, request: pytest.FixtureRequest) -> None:
    client = _client(request, app_kind)
    raw_tokenizer = _tokenizer_of(client).instruct_tokenizer.tokenizer
    content = "Hello, world!"
    tokens = raw_tokenizer.encode(content, bos=True, eos=not prefix)

    response = client.post("/v1/detokenize/", json=tokens)

    assert response.status_code == 200
    assert AssistantMessage.model_validate(response.json()) == AssistantMessage(content=content, prefix=prefix)


@pytest.mark.parametrize("app_kind", ["tekken", "spm"])
@pytest.mark.parametrize("prefix", [False, True])
def test_detokenize_assistant_content_and_tool_calls(
    app_kind: str, prefix: bool, request: pytest.FixtureRequest
) -> None:
    client = _client(request, app_kind)
    tokenizer = _tokenizer_of(client)
    content = "Hello, world!"
    tool_calls = _weather_tool_calls()
    tokens = tokenizer.instruct_tokenizer.tokenizer.encode(content, bos=True, eos=False)
    encoded_tool_calls: list[int] = tokenizer.instruct_tokenizer._encode_tool_calls_in_assistant_message(  # type: ignore[attr-defined]
        AssistantMessage(tool_calls=tool_calls)
    )
    tokens += encoded_tool_calls
    if not prefix:
        tokens.append(tokenizer.instruct_tokenizer.tokenizer.eos_id)

    response = client.post("/v1/detokenize/", json=tokens)

    assert response.status_code == 200
    assert AssistantMessage.model_validate(response.json()) == AssistantMessage(
        content=content, tool_calls=tool_calls, prefix=prefix
    )


@pytest.mark.parametrize("app_kind", ["tekken", "spm"])
@pytest.mark.parametrize("prefix", [False, True])
def test_detokenize_assistant_tool_calls_only(app_kind: str, prefix: bool, request: pytest.FixtureRequest) -> None:
    client = _client(request, app_kind)
    tool_calls = _weather_tool_calls()
    tokens = _encode_tool_calls_only(_tokenizer_of(client), prefix)

    response = client.post("/v1/detokenize/", json=tokens)

    assert response.status_code == 200
    assert AssistantMessage.model_validate(response.json()) == AssistantMessage(tool_calls=tool_calls, prefix=prefix)


@pytest.mark.parametrize("app_kind", ["tekken", "spm"])
@pytest.mark.parametrize("prefix", [False, True])
def test_detokenize_assistant_empty_tokens(app_kind: str, prefix: bool, request: pytest.FixtureRequest) -> None:
    response = _client(request, app_kind).post("/v1/detokenize/", json=[])
    assert response.status_code == 400
    assert response.json()["detail"] == "Tokens list cannot be empty."


@pytest.mark.parametrize("app_kind", ["tekken", "spm"])
@pytest.mark.parametrize("prefix", [False, True])
def test_detokenize_assistant_malformed_tool_call(app_kind: str, prefix: bool, request: pytest.FixtureRequest) -> None:
    client = _client(request, app_kind)
    encoded_tool_calls = _encode_tool_calls_only(_tokenizer_of(client), prefix)
    truncated = encoded_tool_calls[: (-2 if not prefix else -1)]

    response = client.post("/v1/detokenize/", json=truncated)

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid tool call tokenization. Expected a JSON list of tool calls."


def _think_chunk_case(case_id: str) -> tuple[AssistantMessage, AssistantMessage]:
    r"""Build a fresh (message to encode, expected decoded message) pair for a think-chunk case."""
    think_1 = ThinkChunk(thinking="Let me think about this...")
    think_2 = ThinkChunk(thinking="I need to consider all options.")
    if case_id == "single-text-chunk":
        return AssistantMessage(content=[TextChunk(text="Hello, world!")]), AssistantMessage(content="Hello, world!")
    if case_id == "single-think-chunk":
        return AssistantMessage(content=[think_1]), AssistantMessage(
            content=[ThinkChunk(thinking="Let me think about this...")]
        )
    if case_id == "think-think-text":
        content: list[ContentChunk] = [think_1, think_2, TextChunk(text="Here is my final answer.")]
        expected_content: list[ContentChunk] = [
            ThinkChunk(thinking="Let me think about this..."),
            ThinkChunk(thinking="I need to consider all options."),
            TextChunk(text="Here is my final answer."),
        ]
        return AssistantMessage(content=content), AssistantMessage(content=expected_content)
    unclosed: list[ContentChunk] = [think_1, ThinkChunk(thinking="I need to consider all options.", closed=False)]
    unclosed_expected: list[ContentChunk] = [
        ThinkChunk(thinking="Let me think about this..."),
        ThinkChunk(thinking="I need to consider all options.", closed=False),
    ]
    if case_id == "unclosed-think-prefix":
        return AssistantMessage(content=unclosed, prefix=True), AssistantMessage(
            content=unclosed_expected,
            prefix=True,
        )
    assert case_id == "unclosed-think-prefix-with-tool-call"
    tool_call = ToolCall(
        id="call_1", function=FunctionCall(name="get_current_weather", arguments='{"location": "Paris"}')
    )
    return AssistantMessage(content=unclosed, prefix=True, tool_calls=[tool_call]), AssistantMessage(
        content=unclosed_expected,
        prefix=True,
        # v13 does not encode the call ID; the decoded call gets the default id.
        tool_calls=[ToolCall(function=FunctionCall(name="get_current_weather", arguments='{"location": "Paris"}'))],
    )


@pytest.mark.parametrize(
    "case_id",
    [
        "single-text-chunk",
        "single-think-chunk",
        "think-think-text",
        "unclosed-think-prefix",
        "unclosed-think-prefix-with-tool-call",
    ],
)
def test_detokenize_assistant_message_think_chunks(
    case_id: str, mistral_tokenizer_v13: MistralTokenizer, tekken_v13_client: TestClient
) -> None:
    message, expected = _think_chunk_case(case_id)
    encoded_tokens = mistral_tokenizer_v13.instruct_tokenizer.encode_assistant_message(message, False)  # type: ignore[attr-defined]

    response = tekken_v13_client.post("/v1/detokenize/", json=encoded_tokens)

    assert response.status_code == 200
    assert AssistantMessage.model_validate(response.json()) == expected


def _generate_request_body(request_shape: str) -> tuple[object, ChatCompletionRequest]:
    r"""Build a fresh (HTTP body, expected converted request) pair for a generate request shape."""
    if request_shape == "raw-dict":
        body = {"messages": [{"role": "user", "content": "Hello, world!"}], "temperature": 0.1}
        return body, ChatCompletionRequest(messages=[UserMessage(content="Hello, world!")], temperature=0.1)
    if request_shape == "mistral-request":
        chat_request = ChatCompletionRequest(messages=[UserMessage(content="Hello, world!")], temperature=0.1)
        return jsonable_encoder(chat_request), chat_request
    assert request_shape == "openai-request-with-tools"
    body = {
        "messages": [{"role": "user", "content": "Hello, world!"}],
        "tools": [{"type": "function", "function": {"name": "get_current_weather", "parameters": {}}}],
    }
    openai_request = OpenAIChatCompletionRequest.model_validate(body)
    return body, ChatCompletionRequest.from_openai(**openai_request.model_dump())


def _generate_output_case(output_id: str) -> tuple[AssistantMessage, AssistantMessage]:
    r"""Build a fresh (engine output message to encode, expected decoded message) pair."""
    if output_id == "plain-content":
        message = AssistantMessage(content="Hello, world!")
        return message, message
    if output_id == "content-and-tool-call":
        message = AssistantMessage(
            content="Hello, world!",
            tool_calls=[ToolCall(id="1a2bc3d4e", function=FunctionCall(name="get_current_weather", arguments="{}"))],
        )
        # v13 does not encode the call ID; the decoded call gets the default id.
        expected = AssistantMessage(
            content="Hello, world!",
            tool_calls=[ToolCall(function=FunctionCall(name="get_current_weather", arguments="{}"))],
        )
        return message, expected
    assert output_id == "think-chunks"
    message = AssistantMessage(
        content=[
            ThinkChunk(thinking="Let me think about this..."),
            ThinkChunk(thinking="I need to consider all options."),
            TextChunk(text="Here is my final answer."),
        ]
    )
    expected = AssistantMessage(
        content=[
            ThinkChunk(thinking="Let me think about this..."),
            ThinkChunk(thinking="I need to consider all options."),
            TextChunk(text="Here is my final answer."),
        ]
    )
    return message, expected


@pytest.mark.parametrize("request_shape", ["raw-dict", "mistral-request", "openai-request-with-tools"])
@pytest.mark.parametrize("output_id", ["plain-content", "content-and-tool-call", "think-chunks"])
def test_generate(
    request_shape: str,
    output_id: str,
    mistral_tokenizer_v13: MistralTokenizer,
    tekken_v13_client: TestClient,
) -> None:
    body, expected_request = _generate_request_body(request_shape)
    output_message, expected_decoded = _generate_output_case(output_id)
    output_tokens = mistral_tokenizer_v13.instruct_tokenizer.encode_assistant_message(output_message, False)  # type: ignore[attr-defined]
    expected_prompt = mistral_tokenizer_v13.encode_chat_completion(expected_request).tokens

    with patch("mistral_common.experimental.app.routers.requests.post") as mock_generate:
        mock_generate.return_value = MockResponse(200, {"tokens": output_tokens})
        response = tekken_v13_client.post("/v1/chat/completions", json=jsonable_encoder(body))

    assert response.status_code == 200
    assert AssistantMessage.model_validate(response.json()) == expected_decoded
    # The engine receives the encoded prompt plus the request settings, never messages/tools.
    assert mock_generate.call_count == 1
    call = mock_generate.call_args
    assert call.args == ("http://127.0.0.1:8080/completions",)
    assert call.kwargs["timeout"] == 60.0
    engine_json = call.kwargs["json"]
    assert engine_json["prompt"] == expected_prompt
    assert engine_json["return_tokens"] is True
    assert "messages" not in engine_json
    assert "tools" not in engine_json
    if request_shape in ("raw-dict", "mistral-request"):
        assert engine_json["temperature"] == 0.1


@pytest.mark.parametrize(
    "engine_json",
    [
        pytest.param(None, id="non-json-body"),
        pytest.param({"content": "Hello"}, id="missing-tokens"),
        pytest.param([1, 2], id="list-body"),
        pytest.param({"tokens": None}, id="null-tokens"),
        pytest.param({"tokens": "abc"}, id="string-tokens"),
        pytest.param({"tokens": [[1]]}, id="nested-tokens"),
        pytest.param({"tokens": [1.5]}, id="float-tokens"),
        pytest.param({"tokens": ["1"]}, id="str-tokens"),
        pytest.param({"tokens": [True]}, id="bool-tokens"),
    ],
)
@pytest.mark.parametrize("app_kind", ["tekken", "spm"])
def test_generate_invalid_engine_response(
    engine_json: list | dict | None, app_kind: str, request: pytest.FixtureRequest
) -> None:
    client = _client(request, app_kind)
    chat_request = _chat_request(app_kind)
    with patch("mistral_common.experimental.app.routers.requests.post") as mock_generate:
        mock_generate.return_value = MockResponse(200, engine_json, text="<html>Bad Gateway</html>")
        response = client.post("/v1/chat/completions", json=jsonable_encoder(chat_request))
    assert response.status_code == 502
    assert response.json()["detail"].startswith("Engine returned an invalid response")


def test_generate_engine_error_status(tekken_v13_client: TestClient) -> None:
    with patch("mistral_common.experimental.app.routers.requests.post") as mock_generate:
        mock_generate.return_value = MockResponse(400, text="Error")
        response = tekken_v13_client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "Hello, world!"}]}
        )
    assert response.status_code == 400
    assert response.json() == {"detail": "Error"}


def test_generate_engine_request_exception(tekken_v13_client: TestClient) -> None:
    with patch("mistral_common.experimental.app.routers.requests.post") as mock_generate:
        mock_generate.side_effect = requests.exceptions.RequestException("Error")
        response = tekken_v13_client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "Hello, world!"}]}
        )
    assert response.status_code == 500
    assert response.json() == {"detail": "Error"}


def test_generate_engine_timeout(tekken_v13_client: TestClient) -> None:
    with patch("mistral_common.experimental.app.routers.requests.post") as mock_generate:
        mock_generate.side_effect = requests.exceptions.Timeout()
        response = tekken_v13_client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "Hello, world!"}]}
        )
    assert response.status_code == 504
    assert response.json() == {"detail": "Timeout"}


@pytest.mark.parametrize("app_kind", ["tekken", "spm"])
def test_generate_streaming_rejected_before_engine(app_kind: str, request: pytest.FixtureRequest) -> None:
    client = _client(request, app_kind)
    messages = [{"role": "user", "content": "Hello, world!"}]
    if app_kind == "spm":
        # The spm app runs in finetuning mode, which requires an assistant-last message.
        messages.append({"role": "assistant", "content": "Hi! How can I help you?"})
    body = {"messages": messages, "stream": True}  # extra field: parsed as an OpenAI request
    with patch("mistral_common.experimental.app.routers.requests.post") as mock_generate:
        response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 400
    assert response.json() == {"detail": "Streaming is not supported."}
    assert mock_generate.call_count == 0
