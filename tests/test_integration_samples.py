import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from mistral_common.exceptions import (
    InvalidMessageStructureException,
    MistralCommonException,
    UnsupportedTokenizerFeatureException,
)
from mistral_common.protocol.instruct.messages import AssistantMessage, ChatMessage, ToolMessage, UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import FunctionCall, ToolCall
from mistral_common.protocol.instruct.validator import ValidationMode
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.utils import decode_keep

_BUNDLED_V2_PATH = (
    Path(__file__).resolve().parents[1] / "src/mistral_common/data/mistral_instruct_tokenizer_240216.model.v2"
)
_BUNDLED_V2_SHA256 = "37f00374dea48658ee8f5d0f21895b9bc55cb0103939607c8185bfd1c6ca1f89"
_PARALLEL_RESULTS_MESSAGE = r"v2.*multiple tool results.*assistant turn"


@dataclass(frozen=True)
class BundledV2Configuration:
    configuration_id: str
    mode: ValidationMode


@dataclass(frozen=True)
class PublicRejectionCase:
    case_id: str
    configuration: BundledV2Configuration
    call_count: int
    result_count: int
    expected_exception: type[MistralCommonException]
    message_pattern: str


_SERVING_CONFIGURATION = BundledV2Configuration(configuration_id="bundled-spm-v2-serving", mode=ValidationMode.serving)
_TEST_CONFIGURATION = BundledV2Configuration(configuration_id="bundled-spm-v2-test", mode=ValidationMode.test)
_FINETUNING_CONFIGURATION = BundledV2Configuration(
    configuration_id="bundled-spm-v2-finetuning", mode=ValidationMode.finetuning
)
_AGNOSTIC_CONFIGURATION = BundledV2Configuration(
    configuration_id="bundled-spm-v2-agnostic", mode=ValidationMode.agnostic
)

_PUBLIC_REJECTION_CASES = (
    PublicRejectionCase(
        case_id="chat-v2-parallel-results-rejected-serving",
        configuration=_SERVING_CONFIGURATION,
        call_count=2,
        result_count=2,
        expected_exception=UnsupportedTokenizerFeatureException,
        message_pattern=_PARALLEL_RESULTS_MESSAGE,
    ),
    PublicRejectionCase(
        case_id="chat-v2-parallel-results-rejected-test",
        configuration=_TEST_CONFIGURATION,
        call_count=2,
        result_count=2,
        expected_exception=UnsupportedTokenizerFeatureException,
        message_pattern=_PARALLEL_RESULTS_MESSAGE,
    ),
    PublicRejectionCase(
        case_id="chat-v2-parallel-results-rejected-finetuning",
        configuration=_FINETUNING_CONFIGURATION,
        call_count=2,
        result_count=2,
        expected_exception=UnsupportedTokenizerFeatureException,
        message_pattern=_PARALLEL_RESULTS_MESSAGE,
    ),
    PublicRejectionCase(
        case_id="chat-v2-parallel-results-rejected-agnostic",
        configuration=_AGNOSTIC_CONFIGURATION,
        call_count=2,
        result_count=2,
        expected_exception=UnsupportedTokenizerFeatureException,
        message_pattern=_PARALLEL_RESULTS_MESSAGE,
    ),
    PublicRejectionCase(
        case_id="chat-v2-extra-tool-result-structure-serving",
        configuration=_SERVING_CONFIGURATION,
        call_count=1,
        result_count=2,
        expected_exception=InvalidMessageStructureException,
        message_pattern=r"Not the same number of function calls and responses",
    ),
    PublicRejectionCase(
        case_id="chat-v2-extra-tool-result-structure-finetuning",
        configuration=_FINETUNING_CONFIGURATION,
        call_count=1,
        result_count=2,
        expected_exception=InvalidMessageStructureException,
        message_pattern=r"Not the same number of function calls and responses",
    ),
)


@pytest.fixture()
def samples_dir() -> Path:
    return Path(__file__).parent.joinpath("data").joinpath("samples")


def load_sample(samples_dir: Path, sample_name: str, version: int) -> tuple[Any, str, Any]:
    with open(samples_dir / f"{sample_name}/sample.json", "r") as f:
        sample = json.load(f)

    with open(samples_dir / f"{sample_name}/text_v{version}.txt", "r") as f:
        text = f.read()

    with open(samples_dir / f"{sample_name}/tokens_v{version}.json", "r") as f:
        tokens = json.load(f)

    return sample, text, tokens["tokens"]


def get_tokenizer(version: int) -> MistralTokenizer:
    if version == 1:
        return MistralTokenizer.v1()
    elif version == 2:
        return MistralTokenizer.v2()
    elif version == 3:
        return MistralTokenizer.v3()
    else:
        raise ValueError(f"Invalid version: {version}")


@pytest.mark.parametrize(
    "sample",
    [
        {"sample_name": "get_weather_full", "versions": [2, 3]},
        {"sample_name": "get_weather_no_history", "versions": [2, 3]},
        {"sample_name": "several_calls", "versions": [2, 3]},
        {"sample_name": "no_tools", "versions": [1, 2, 3]},
        {"sample_name": "get_weather_no_system_prompt", "versions": [2, 3]},
        {"sample_name": "parallel_calls", "versions": [3]},
    ],
    ids=[
        "get_weather_full",
        "get_weather_no_history",
        "several_calls",
        "no_tools",
        "get_weather_no_system_prompt",
        "parallel_calls",
    ],
)
@pytest.mark.parametrize("version", [1, 2, 3], ids=["v1", "v2", "v3"])
def test_samples(sample: dict[str, Any], version: int, samples_dir: Path) -> None:
    if version not in sample["versions"]:
        pytest.skip(f"Sample {sample['sample_name']} not available for version {version}")

    mistral_tokenizer = get_tokenizer(version)
    instruct_request, text, tokens = load_sample(samples_dir, sample["sample_name"], version)
    chat_completion_string = json.dumps(
        {"model": "debug", "messages": instruct_request["messages"], "tools": instruct_request["tools"]}
    )
    chat_completion_request = ChatCompletionRequest[ChatMessage].model_validate_json(chat_completion_string)
    tokenized = mistral_tokenizer.encode_chat_completion(chat_completion_request)
    decoded_text = decode_keep(mistral_tokenizer, tokenized)
    assert decoded_text == text
    assert tokenized.tokens == tokens


def build_chat_request(
    *, call_count: int, result_count: int, mode: ValidationMode
) -> ChatCompletionRequest[ChatMessage]:
    tool_calls = [
        ToolCall(id=f"call0000{index}", function=FunctionCall(name=f"tool_{index}", arguments="{}"))
        for index in range(1, call_count + 1)
    ]
    messages: list[ChatMessage] = [
        UserMessage(content="Run these tools."),
        AssistantMessage(content=None, tool_calls=tool_calls),
    ]
    messages.extend(
        ToolMessage(
            name=f"tool_{index}",
            content=f"result {index}",
            tool_call_id=f"call0000{index}",
        )
        for index in range(1, result_count + 1)
    )
    if mode == ValidationMode.finetuning:
        messages.append(AssistantMessage(content="The tool work is complete."))

    return ChatCompletionRequest[ChatMessage](model="test", messages=messages)


def load_bundled_v2(configuration: BundledV2Configuration) -> MistralTokenizer:
    digest = hashlib.sha256(_BUNDLED_V2_PATH.read_bytes()).hexdigest()
    assert digest == _BUNDLED_V2_SHA256
    return MistralTokenizer.from_file(tokenizer_filename=_BUNDLED_V2_PATH, mode=configuration.mode)


@pytest.mark.parametrize("case", _PUBLIC_REJECTION_CASES, ids=lambda case: case.case_id)
def test_public_v2_rejections(case: PublicRejectionCase) -> None:
    request = build_chat_request(
        call_count=case.call_count,
        result_count=case.result_count,
        mode=case.configuration.mode,
    )
    tokenizer = load_bundled_v2(configuration=case.configuration)

    with pytest.raises(case.expected_exception, match=case.message_pattern):
        tokenizer.encode_chat_completion(request)
