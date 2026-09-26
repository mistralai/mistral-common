import hashlib
from dataclasses import dataclass
from pathlib import Path

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

_BUNDLED_V2_PATH = (
    Path(__file__).resolve().parents[2] / "src/mistral_common/data/mistral_instruct_tokenizer_240216.model.v2"
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
