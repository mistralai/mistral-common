from typing import Union

import pytest
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.testclient import TestClient

from mistral_common.app.main import OpenAIChatCompletionRequest, Settings, create_app, get_settings
from mistral_common.protocol.instruct.messages import AssistantMessage, ChatMessage, SystemMessage, UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import Function, Tool
from mistral_common.protocol.instruct.validator import ValidationMode
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer


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
def spm_app() -> FastAPI:
    return create_app(
        MistralTokenizer._data_path() / "mistral_instruct_tokenizer_241114.model.v7m1", ValidationMode.finetuning
    )


@pytest.fixture(scope="module")
def tekken_client(tekken_app: FastAPI) -> TestClient:
    return TestClient(tekken_app)


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
def test_read_main(client_fixture: str, request: pytest.FixtureRequest) -> None:
    client: TestClient = request.getfixturevalue(client_fixture)

    response = client.get("/")
    assert response.status_code == 200


@pytest.mark.parametrize(
    ["app_fixture", "client_fixture"], [("tekken_app", "tekken_client"), ("spm_app", "spm_client")]
)
def test_get_info(app_fixture: str, client_fixture: str, request: pytest.FixtureRequest) -> None:
    app: FastAPI = request.getfixturevalue(app_fixture)
    client: TestClient = request.getfixturevalue(client_fixture)
    settings: Settings = app.dependency_overrides[get_settings]()

    response = client.get("/info")
    assert response.status_code == 200
    assert response.json() == {"app_name": settings.app_name, "app_version": settings.app_version}


@pytest.mark.parametrize(
    ["client_fixture", "messages_fixture", "tokens_fixture"],
    [
        (
            "tekken_client",
            "tekken_messages",
            "tekken_message_tokens",
        ),
        ("spm_client", "spm_messages", "spm_message_tokens"),
    ],
)
def test_tokenize_messages(
    client_fixture: str, messages_fixture: str, tokens_fixture: str, request: pytest.FixtureRequest
) -> None:
    messages: list[ChatMessage] = request.getfixturevalue(messages_fixture)
    tokens: list[int] = request.getfixturevalue(tokens_fixture)
    client: TestClient = request.getfixturevalue(client_fixture)

    response = client.post("/tokenize/messages", json=jsonable_encoder(messages))
    assert response.status_code == 200
    assert response.json() == tokens


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
    chat_request: Union[ChatCompletionRequest, OpenAIChatCompletionRequest] = request.getfixturevalue(request_fixture)
    tokens: list[int] = request.getfixturevalue(tokens_fixture)
    client: TestClient = request.getfixturevalue(client_fixture)

    response = client.post("/tokenize/request", json=jsonable_encoder(chat_request))
    assert response.status_code == 200
    assert response.json() == tokens


def test_tokenize_request_with_empty_messages(tekken_client: TestClient) -> None:
    response = tekken_client.post("/tokenize/request", json=jsonable_encoder(ChatCompletionRequest(messages=[])))
    assert response.status_code == 400
    assert response.json()["detail"] == "Messages list cannot be empty."


def test_tokenize_messages_with_empty_messages(tekken_client: TestClient) -> None:
    response = tekken_client.post("/tokenize/messages", json=jsonable_encoder([]))
    assert response.status_code == 400
    assert response.json()["detail"] == "Messages list cannot be empty."


@pytest.mark.parametrize(
    ["tokenizer_fixture", "client_fixture"], [("tekken_tokenizer", "tekken_client"), ("spm_tokenizer", "spm_client")]
)
def test_tokenize_prompt(tokenizer_fixture: str, client_fixture: str, request: pytest.FixtureRequest) -> None:
    tokenizer: MistralTokenizer = request.getfixturevalue(tokenizer_fixture)
    client: TestClient = request.getfixturevalue(client_fixture)

    prompt = "Hello, world!"
    tokens_with_special = tokenizer.instruct_tokenizer.tokenizer.encode(prompt, bos=True, eos=True)
    tokens_without_special = tokenizer.instruct_tokenizer.tokenizer.encode(prompt, bos=False, eos=False)
    response_with_special = client.post("/tokenize/prompt", json={"prompt": prompt, "add_special": True})
    response_without_special = client.post("/tokenize/prompt", json={"prompt": prompt, "add_special": False})
    assert response_with_special.status_code == 200
    assert response_without_special.status_code == 200
    assert response_with_special.json() == tokens_with_special
    assert response_without_special.json() == tokens_without_special

    response_prompt_empty = client.post("/tokenize/prompt", json={"prompt": "", "add_special": True})
    assert response_prompt_empty.status_code == 400
    assert response_prompt_empty.json()["detail"] == "Prompt cannot be empty."


@pytest.mark.parametrize(
    ["tokenizer_fixture", "client_fixture"], [("tekken_tokenizer", "tekken_client"), ("spm_tokenizer", "spm_client")]
)
def test_detokenize_tokens(tokenizer_fixture: str, client_fixture: str, request: pytest.FixtureRequest) -> None:
    prompt = "Hello, world!"
    tokenizer: MistralTokenizer = request.getfixturevalue(tokenizer_fixture)
    client: TestClient = request.getfixturevalue(client_fixture)
    encoded_prompt = tokenizer.instruct_tokenizer.tokenizer.encode(prompt, bos=True, eos=True)

    response_with_special = client.post(
        "/detokenize", json={"tokens": encoded_prompt, "special_token_policy": SpecialTokenPolicy.KEEP}
    )
    assert response_with_special.status_code == 200
    assert response_with_special.json() == tokenizer.instruct_tokenizer.tokenizer.decode(
        encoded_prompt, special_token_policy=SpecialTokenPolicy.KEEP
    )
    response_without_special = client.post(
        "/detokenize", json={"tokens": encoded_prompt, "special_token_policy": SpecialTokenPolicy.IGNORE}
    )
    assert response_without_special.status_code == 200
    assert response_without_special.json() == tokenizer.instruct_tokenizer.tokenizer.decode(
        encoded_prompt, special_token_policy=SpecialTokenPolicy.IGNORE
    )

    response_empty_tokens = client.post(
        "/detokenize", json={"tokens": [], "special_token_policy": SpecialTokenPolicy.IGNORE}
    )
    assert response_empty_tokens.status_code == 400
    assert response_empty_tokens.json()["detail"] == "Tokens list cannot be empty."

    response_special_error = client.post(
        "/detokenize", json={"tokens": encoded_prompt, "special_token_policy": SpecialTokenPolicy.RAISE}
    )
    assert response_special_error.status_code == 400
