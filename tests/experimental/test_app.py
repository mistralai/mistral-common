from unittest.mock import patch

import pytest
import requests
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.testclient import TestClient

from mistral_common.experimental.app.main import create_app
from mistral_common.experimental.app.models import OpenAIChatCompletionRequest
from mistral_common.protocol.instruct.chunk import (
    TextChunk,
    ThinkChunk,
)
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
from mistral_common.protocol.instruct.validator import (
    MistralRequestValidatorV13,
    ValidationMode,
)
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy, TokenizerVersion
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV13
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.test_tekken import quick_vocab, get_special_tokens


@pytest.fixture(scope="module")
def tekken_app() -> FastAPI:
    return create_app(MistralTokenizer._data_path() / "tekken_240911.json", ValidationMode.test)


@pytest.fixture(scope="module")
def tekken_tokenizer() -> MistralTokenizer:
    return MistralTokenizer.from_file(MistralTokenizer._data_path() / "tekken_240911.json", ValidationMode.test)


@pytest.fixture(scope="module")
def spm_tokenizer() -> MistralTokenizer:
    return MistralTokenizer.from_file(
        MistralTokenizer._data_path() / "mistral_instruct_tokenizer_241114.model.v7m1", ValidationMode.finetuning
    )


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
            UserMessage, AssistantMessage, ToolMessage, SystemMessage, InstructRequest
        ),
    )


@pytest.fixture(scope="module")
def spm_app() -> FastAPI:
    return create_app(
        MistralTokenizer._data_path() / "mistral_instruct_tokenizer_241114.model.v7m1", ValidationMode.finetuning
    )


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


@pytest.fixture(scope="module")
def tekken_messages() -> list[ChatMessage]:
    return [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="Hello, world!"),
    ]


@pytest.fixture(scope="module")
def tekken_message_tokens(tekken_tokenizer: MistralTokenizer, tekken_messages: list[ChatMessage]) -> list[int]:
    tokens: list[int] = tekken_tokenizer.encode_chat_completion(ChatCompletionRequest(messages=tekken_messages)).tokens
    return tokens


@pytest.fixture(scope="module")
def spm_messages() -> list[ChatMessage]:
    return [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="Hello, how are you ?!"),
        AssistantMessage(content="I'm fine, thank you ! How can I help you ?"),
    ]


@pytest.fixture(scope="module")
def spm_message_tokens(spm_tokenizer: MistralTokenizer, spm_messages: list[ChatMessage]) -> list[int]:
    tokens: list[int] = spm_tokenizer.encode_chat_completion(ChatCompletionRequest(messages=spm_messages)).tokens
    return tokens


@pytest.fixture(scope="module")
def tekken_request(tekken_messages: list[ChatMessage]) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        messages=tekken_messages,
        tools=[
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
        ],
    )


@pytest.fixture(scope="module")
def tekken_oai_request(tekken_request: ChatCompletionRequest) -> OpenAIChatCompletionRequest:
    return OpenAIChatCompletionRequest.model_validate(tekken_request.to_openai())


@pytest.fixture(scope="module")
def tekken_request_tokens(tekken_tokenizer: MistralTokenizer, tekken_request: ChatCompletionRequest) -> list[int]:
    tokens: list[int] = tekken_tokenizer.encode_chat_completion(tekken_request).tokens
    return tokens


@pytest.fixture(scope="module")
def spm_request(spm_messages: list[ChatMessage]) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        messages=spm_messages,
    )


@pytest.fixture(scope="module")
def spm_oai_request(spm_request: ChatCompletionRequest) -> OpenAIChatCompletionRequest:
    return OpenAIChatCompletionRequest.model_validate(spm_request.to_openai())


@pytest.fixture(scope="module")
def spm_request_tokens(spm_tokenizer: MistralTokenizer, spm_request: ChatCompletionRequest) -> list[int]:
    tokens: list[int] = spm_tokenizer.encode_chat_completion(spm_request).tokens
    return tokens


@pytest.mark.parametrize(["client_fixture"], [("tekken_client",), ("spm_client",)])
def test_redirect_to_docs(client_fixture: str, request: pytest.FixtureRequest) -> None:
    client: TestClient = request.getfixturevalue(client_fixture)

    response = client.get("/")
    assert response.status_code == 200


@pytest.mark.parametrize(
    ["client_fixture", "request_fixture", "tokens_fixture"],
    [
        (
            "tekken_client",
            "tekken_request",
            "tekken_request_tokens",
        ),
        ("spm_client", "spm_request", "spm_request_tokens"),
        (
            "tekken_client",
            "tekken_oai_request",
            "tekken_request_tokens",
        ),
        ("spm_client", "spm_oai_request", "spm_request_tokens"),
    ],
)
def test_tokenize_request(
    client_fixture: str, request_fixture: str, tokens_fixture: str, request: pytest.FixtureRequest
) -> None:
    chat_request: ChatCompletionRequest | OpenAIChatCompletionRequest = request.getfixturevalue(request_fixture)
    tokens: list[int] = request.getfixturevalue(tokens_fixture)
    client: TestClient = request.getfixturevalue(client_fixture)

    response = client.post("/v1/tokenize", json=jsonable_encoder(chat_request))
    assert response.status_code == 200
    assert response.json() == tokens


@pytest.mark.parametrize(
    ["tokenizer_fixture", "client_fixture"], [("tekken_tokenizer", "tekken_client"), ("spm_tokenizer", "spm_client")]
)
def test_detokenize_string(tokenizer_fixture: str, client_fixture: str, request: pytest.FixtureRequest) -> None:
    prompt = "Hello, world!"
    tokenizer: MistralTokenizer = request.getfixturevalue(tokenizer_fixture)
    client: TestClient = request.getfixturevalue(client_fixture)
    encoded_prompt = tokenizer.instruct_tokenizer.tokenizer.encode(prompt, bos=True, eos=True)

    response_with_special = client.post(
        "/v1/detokenize/string", json={"tokens": encoded_prompt, "special_token_policy": SpecialTokenPolicy.KEEP}
    )
    assert response_with_special.status_code == 200
    assert response_with_special.json() == tokenizer.instruct_tokenizer.tokenizer.decode(
        encoded_prompt, special_token_policy=SpecialTokenPolicy.KEEP
    )
    response_without_special = client.post(
        "/v1/detokenize/string", json={"tokens": encoded_prompt, "special_token_policy": SpecialTokenPolicy.IGNORE}
    )
    assert response_without_special.status_code == 200
    assert response_without_special.json() == tokenizer.instruct_tokenizer.tokenizer.decode(
        encoded_prompt, special_token_policy=SpecialTokenPolicy.IGNORE
    )

    response_empty_tokens = client.post(
        "/v1/detokenize/string", json={"tokens": [], "special_token_policy": SpecialTokenPolicy.IGNORE}
    )
    assert response_empty_tokens.status_code == 400
    assert response_empty_tokens.json()["detail"] == "Tokens list cannot be empty."

    response_special_error = client.post(
        "/v1/detokenize/string", json={"tokens": encoded_prompt, "special_token_policy": SpecialTokenPolicy.RAISE}
    )
    assert response_special_error.status_code == 400


@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize(
    ["tokenizer_fixture", "client_fixture"], [("tekken_tokenizer", "tekken_client"), ("spm_tokenizer", "spm_client")]
)
def test_detokenize_assistant_message(
    prefix: bool, tokenizer_fixture: str, client_fixture: str, request: pytest.FixtureRequest
) -> None:
    # Test 1:
    # Detokenize only content
    tokenizer: MistralTokenizer = request.getfixturevalue(tokenizer_fixture)
    client: TestClient = request.getfixturevalue(client_fixture)
    content = "Hello, world!"
    encoded_content = tokenizer.instruct_tokenizer.tokenizer.encode(content, bos=True, eos=not prefix)
    response = client.post("/v1/detokenize/", json=encoded_content)
    assert response.status_code == 200

    assert AssistantMessage.model_validate(response.json()) == AssistantMessage(content=content, prefix=prefix)

    # Test 2:
    # Detokenize content and tool calls
    content = "Hello, world!"
    tool_calls = [
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
    encoded_content = tokenizer.instruct_tokenizer.tokenizer.encode(content, bos=True, eos=False)
    encoded_tool_calls: list[int] = tokenizer.instruct_tokenizer._encode_tool_calls_in_assistant_message(  # type: ignore[attr-defined]
        AssistantMessage(tool_calls=tool_calls)
    )
    if not prefix:
        encoded_tool_calls.append(tokenizer.instruct_tokenizer.tokenizer.eos_id)
    encoded_tokens = encoded_content + encoded_tool_calls

    response = client.post("/v1/detokenize/", json=encoded_tokens)
    assert response.status_code == 200
    assert AssistantMessage.model_validate(response.json()) == AssistantMessage(
        content=content, tool_calls=tool_calls, prefix=prefix
    )

    # Test 3:
    # Detokenize only tool calls
    encoded_tokens = tokenizer.instruct_tokenizer._encode_tool_calls_in_assistant_message(  # type: ignore[attr-defined]
        AssistantMessage(tool_calls=tool_calls)
    )
    if not prefix:
        encoded_tokens.append(tokenizer.instruct_tokenizer.tokenizer.eos_id)

    response = client.post("/v1/detokenize/", json=encoded_tokens)
    assert response.status_code == 200
    assert AssistantMessage.model_validate(response.json()) == AssistantMessage(tool_calls=tool_calls, prefix=prefix)

    # Test 4:
    # Detokenize empty tokens
    response = client.post("/v1/detokenize/", json=[])
    assert response.status_code == 400
    assert response.json()["detail"] == "Tokens list cannot be empty."

    # Test 6:
    # Wrong tool call format
    response = client.post("/v1/detokenize/", json=encoded_tool_calls[: (-2 if not prefix else -1)])
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid tool call tokenization. Expected a JSON list of tool calls."


@pytest.mark.parametrize(
    "assistant_message",
    [
        AssistantMessage(
            content=[
                TextChunk(text="Hello, world!"),
            ]
        ),
        AssistantMessage(
            content=[
                ThinkChunk(thinking="Let me think about this..."),
            ]
        ),
        AssistantMessage(
            content=[
                TextChunk(text="Hello, world!"),
                ThinkChunk(thinking="Let me think about this..."),
                TextChunk(text="This is a complex question."),
                ThinkChunk(thinking="I need to consider all options."),
                TextChunk(text="Here is my final answer."),
            ],
        ),
        AssistantMessage(
            content=[
                TextChunk(text="Hello, world!"),
                ThinkChunk(thinking="Let me think about this..."),
                TextChunk(text="This is a complex question."),
                ThinkChunk(thinking="I need to consider all options.", closed=False),
            ],
            prefix=True,
        ),
        AssistantMessage(
            content=[
                TextChunk(text="Hello, world!"),
                ThinkChunk(thinking="Let me think about this..."),
                TextChunk(text="This is a complex question."),
                ThinkChunk(thinking="I need to consider all options.", closed=False),
            ],
            prefix=True,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    function=FunctionCall(name="get_current_weather", arguments='{"location": "Paris"}'),
                ),
            ],
        ),
    ],
)
def test_detokenize_assistant_message_think_chunks(
    assistant_message: AssistantMessage, mistral_tokenizer_v13: MistralTokenizer, tekken_v13_client: TestClient
) -> None:
    encoded_tokens = mistral_tokenizer_v13.instruct_tokenizer.encode_assistant_message(assistant_message, False, False)  # type: ignore[attr-defined]

    response = tekken_v13_client.post("/v1/detokenize/", json=encoded_tokens)
    assert response.status_code == 200

    assistant_message.tool_calls = (
        [ToolCall.model_validate(tool_call.model_dump(exclude={"id"})) for tool_call in assistant_message.tool_calls]
        if assistant_message.tool_calls
        else None
    )
    if (
        isinstance(assistant_message.content, list)
        and len(assistant_message.content) == 1
        and isinstance(assistant_message.content[0], TextChunk)
    ):
        assistant_message.content = assistant_message.content[0].text
    assert AssistantMessage.model_validate(response.json()) == assistant_message


class MockResponse:
    def __init__(self, status_code: int, json_data: list | dict | None = None, text: str | None = None) -> None:
        self.json_data = json_data
        self.status_code = status_code
        self.text = text

    def json(self) -> list | dict:
        if self.json_data is None:
            raise ValueError("No JSON data available")
        return self.json_data


@pytest.mark.parametrize(
    "engine_request",
    [
        {
            "messages": [{"role": "user", "content": "Hello, world!"}],
            "temperature": 0.1,
        },
        ChatCompletionRequest(messages=[UserMessage(content="Hello, world!")], temperature=0.1),
        OpenAIChatCompletionRequest(
            messages=[{"role": "user", "content": "Hello, world!"}],
            tools=[{"type": "function", "function": {"name": "get_current_weather", "parameters": {}}}],
        ),
    ],
)
@pytest.mark.parametrize(
    "output_assistant_message",
    [
        AssistantMessage(content="Hello, world!"),
        AssistantMessage(
            content="Hello, world!",
            tool_calls=[ToolCall(id="1a2bc3d4e", function=FunctionCall(name="get_current_weather", arguments="{}"))],
        ),
        AssistantMessage(
            content=[
                TextChunk(text="Hello, world!"),
                ThinkChunk(thinking="Let me think about this..."),
                TextChunk(text="This is a complex question."),
                ThinkChunk(thinking="I need to consider all options."),
                TextChunk(text="Here is my final answer."),
            ],
        ),
    ],
)
def test_generate(
    mistral_tokenizer_v13: MistralTokenizer,
    tekken_v13_client: TestClient,
    engine_request: dict | ChatCompletionRequest | OpenAIChatCompletionRequest,
    output_assistant_message: AssistantMessage,
) -> None:
    output_tokens = mistral_tokenizer_v13.instruct_tokenizer.encode_assistant_message(  # type: ignore[attr-defined]
        output_assistant_message, False, False
    )
    if output_assistant_message.tool_calls:
        output_assistant_message = AssistantMessage(
            content=output_assistant_message.content,
            tool_calls=[
                ToolCall(**tool_call.model_dump(exclude={"id"})) for tool_call in output_assistant_message.tool_calls
            ],
        )

    with patch("mistral_common.experimental.app.routers.requests.post") as mock_generate:
        mock_generate.return_value = MockResponse(200, {"tokens": output_tokens})
        response = tekken_v13_client.post("/v1/chat/completions", json=jsonable_encoder(engine_request))
    assert response.status_code == 200
    assert AssistantMessage(**response.json()) == output_assistant_message


def test_generate_error(tekken_v13_client: TestClient) -> None:
    with patch("mistral_common.experimental.app.routers.requests.post") as mock_generate:
        mock_generate.return_value = MockResponse(400, text="Error")
        response = tekken_v13_client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "Hello, world!"}]}
        )
    assert response.status_code == 400
    assert response.json() == {"detail": "Error"}

    with patch("mistral_common.experimental.app.routers.requests.post") as mock_generate:
        mock_generate.side_effect = requests.exceptions.RequestException("Error")
        response = tekken_v13_client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "Hello, world!"}]}
        )
    assert response.status_code == 500
    assert response.json() == {"detail": "Error"}

    with patch("mistral_common.experimental.app.routers.requests.post") as mock_generate:
        mock_generate.side_effect = requests.exceptions.Timeout()
        response = tekken_v13_client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "Hello, world!"}]}
        )
    assert response.status_code == 504
    assert response.json() == {"detail": "Timeout"}
