import re

import pytest

from mistral_common.experimental.tools import (
    InvalidArgsToolCallError,
    InvalidToolCallError,
    _decode_tool_calls,
    _split_content_and_tool_calls,
    _split_integer_list_by_value,
)
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.normalize import InstructRequestNormalizerV7, InstructRequestNormalizerV13
from mistral_common.protocol.instruct.request import InstructRequest
from mistral_common.protocol.instruct.tool_calls import FunctionCall, ToolCall
from mistral_common.protocol.instruct.validator import (
    MistralRequestValidatorV5,
    MistralRequestValidatorV13,
)
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV11, InstructTokenizerV13
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.utils.json import is_unloadable_json
from tests.utils.special_tokens import get_special_tokens, quick_vocab

_TOOL_CALL_TOKEN_ID = 6


def _mistral_tokenizer_v11() -> MistralTokenizer:
    return MistralTokenizer(
        instruct_tokenizer=InstructTokenizerV11(
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
                        b":",
                    ]
                ),
                special_tokens=get_special_tokens(TokenizerVersion.v11),
                pattern=r".+",  # single token, whole string
                vocab_size=256 + 100,
                num_special_tokens=100,
                version=TokenizerVersion.v11,
            ),
        ),
        validator=MistralRequestValidatorV5(),
        request_normalizer=InstructRequestNormalizerV7(
            UserMessage, AssistantMessage, ToolMessage, SystemMessage, InstructRequest, None
        ),
    )


def _mistral_tokenizer_v13() -> MistralTokenizer:
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
                special_tokens=get_special_tokens(TokenizerVersion.v13),
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


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        pytest.param([1, 2, 3, 4, 5], ([1, 2, 3, 4, 5], ()), id="no-tool-calls"),
        pytest.param([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ([1, 2, 3, 4, 5], ([6, 7, 8, 9, 10],)), id="one-tool-call"),
        pytest.param(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 6, 11, 12, 13, 14],
            ([1, 2, 3, 4, 5], ([6, 7, 8, 9, 10], [6, 11, 12, 13, 14])),
            id="multiple-tool-calls",
        ),
        pytest.param([6, 7, 8, 9, 10], ([], ([6, 7, 8, 9, 10],)), id="no-content"),
    ],
)
def test_split_content_and_tool_calls(tokens: list[int], expected: tuple[list[int], tuple[list[int], ...]]) -> None:
    assert _split_content_and_tool_calls(tokens, _TOOL_CALL_TOKEN_ID) == expected


_ONE_TOOL_CALL = [ToolCall(id="call_1", function=FunctionCall(name="ab", arguments="{}"))]
_TWO_TOOL_CALLS = [
    ToolCall(id="call_1", function=FunctionCall(name="ab", arguments="{}")),
    ToolCall(id="call_2", function=FunctionCall(name="cd", arguments='{"a": "b", "c": "d"}')),
]

_VERSIONED_TOKENIZERS = [
    pytest.param(MistralTokenizer.v2(), id="v2"),
    pytest.param(MistralTokenizer.v3(), id="v3"),
    pytest.param(MistralTokenizer.v7(), id="v7"),
    pytest.param(_mistral_tokenizer_v11(), id="v11"),
    pytest.param(_mistral_tokenizer_v13(), id="v13"),
]

_PRE_V11_VERSIONS = [TokenizerVersion.v2, TokenizerVersion.v3, TokenizerVersion.v7]


def _encode_tool_calls(tokenizer: MistralTokenizer, tool_calls: list[ToolCall]) -> tuple[list[int], ...]:
    encoded_tool_calls = tokenizer.instruct_tokenizer._encode_tool_calls_in_assistant_message(  # type: ignore[attr-defined]
        AssistantMessage(tool_calls=tool_calls)
    )
    return _split_integer_list_by_value(
        encoded_tool_calls, tokenizer.instruct_tokenizer.tokenizer.get_special_token("[TOOL_CALLS]")
    )


@pytest.mark.parametrize("tokenizer", _VERSIONED_TOKENIZERS)
@pytest.mark.parametrize(
    "tool_calls", [pytest.param(_ONE_TOOL_CALL, id="one-call"), pytest.param(_TWO_TOOL_CALLS, id="two-calls")]
)
def test_decode_tool_calls_round_trip(tokenizer: MistralTokenizer, tool_calls: list[ToolCall]) -> None:
    splitted_tool_calls = _encode_tool_calls(tokenizer, tool_calls)

    decoded_tool_calls = _decode_tool_calls(splitted_tool_calls, tokenizer.instruct_tokenizer.tokenizer)
    if tokenizer.instruct_tokenizer.tokenizer.version != TokenizerVersion.v11:
        assert len(decoded_tool_calls) == len(tool_calls)
        for decoded_tool_call, tool_call in zip(decoded_tool_calls, tool_calls):
            assert decoded_tool_call.model_dump(exclude={"id"}) == tool_call.model_dump(exclude={"id"})
    else:
        # v11 is the only version that encodes the id
        assert decoded_tool_calls == tool_calls


@pytest.mark.parametrize("tokenizer", _VERSIONED_TOKENIZERS)
@pytest.mark.parametrize(
    "tool_calls", [pytest.param(_ONE_TOOL_CALL, id="one-call"), pytest.param(_TWO_TOOL_CALLS, id="two-calls")]
)
def test_decode_tool_calls_corrupted_arguments(tokenizer: MistralTokenizer, tool_calls: list[ToolCall]) -> None:
    splitted_tool_calls = _encode_tool_calls(tokenizer, tool_calls)

    # Remove a JSON token right after the args marker to make the arguments undecodable.
    if tokenizer.instruct_tokenizer.tokenizer.version in _PRE_V11_VERSIONS:
        arg_index = tokenizer.instruct_tokenizer.tokenizer.get_special_token("[TOOL_CALLS]")
    else:
        arg_index = tokenizer.instruct_tokenizer.tokenizer.get_special_token("[ARGS]")

    first_args_index = splitted_tool_calls[0].index(arg_index)
    corrupted_tool_calls = (
        splitted_tool_calls[0][: first_args_index + 1] + splitted_tool_calls[0][first_args_index + 2 :],
        *splitted_tool_calls[1:],
    )

    with pytest.raises(
        InvalidToolCallError
        if tokenizer.instruct_tokenizer.tokenizer.version in _PRE_V11_VERSIONS
        else InvalidArgsToolCallError
    ):
        _decode_tool_calls(corrupted_tool_calls, tokenizer.instruct_tokenizer.tokenizer)


_DEEP_NESTING_JSON = "[" * 100_000 + "]" * 100_000


@pytest.mark.parametrize(
    "tokenizer",
    (
        pytest.param(MistralTokenizer.v2(), id="v2"),
        pytest.param(MistralTokenizer.v3(is_tekken=True), id="v3-tekken"),
        pytest.param(MistralTokenizer.v7(), id="v7"),
    ),
)
@pytest.mark.parametrize(
    ("tool_calls_json", "expected_error", "expected_message"),
    (
        pytest.param("5", InvalidToolCallError, "Expected a list of tool calls.", id="int"),
        pytest.param("null", InvalidToolCallError, "Expected a list of tool calls.", id="null"),
        pytest.param(
            '{"name": "f", "arguments": {}}', InvalidToolCallError, "Expected a list of tool calls.", id="dict"
        ),
        pytest.param("[1]", InvalidToolCallError, "Expected a dict with a name.", id="list-of-int"),
        pytest.param('[{"arguments": {}}]', InvalidToolCallError, "Expected a dict with a name.", id="missing-name"),
        pytest.param('[{"name": "f"}]', InvalidArgsToolCallError, "Expected a dict.", id="missing-args"),
        pytest.param(
            '[{"name": "f", "arguments": 1}]', InvalidArgsToolCallError, "Expected a dict.", id="non-dict-args"
        ),
        # Valid JSON that `json.loads` rejects without a JSONDecodeError.
        pytest.param(
            '[{"name": "f", "arguments": {"a": ' + "1" * 5000 + "}}]",
            InvalidToolCallError,
            "Expected a JSON list",
            id="oversized-int",
        ),
        # RecursionError where the recursion limit applies; on Python 3.14+ it parses only with a large stack.
        pytest.param(
            _DEEP_NESTING_JSON,
            InvalidToolCallError,
            "Expected a JSON list",
            marks=pytest.mark.skipif(
                not is_unloadable_json(_DEEP_NESTING_JSON),
                reason="json.loads accepts deeply nested JSON on this interpreter (Python 3.14+ with a large stack)",
            ),
            id="deep-nesting",
        ),
    ),
)
def test_decode_tool_calls_up_to_v7_malformed(
    tokenizer: MistralTokenizer,
    tool_calls_json: str,
    expected_error: type[InvalidToolCallError],
    expected_message: str,
) -> None:
    raw_tokenizer = tokenizer.instruct_tokenizer.tokenizer
    tool_call_tokens = [
        raw_tokenizer.get_special_token("[TOOL_CALLS]"),
        *raw_tokenizer.encode(tool_calls_json, bos=False, eos=False),
    ]

    with pytest.raises(expected_error, match=re.escape(expected_message)):
        _decode_tool_calls(tool_call_tokens=[tool_call_tokens], tokenizer=raw_tokenizer)


def test_decode_tool_calls_v11_without_id() -> None:
    tokenizer = _mistral_tokenizer_v11()
    tool_call = ToolCall(id="call_1", function=FunctionCall(name="ab", arguments="{}"))

    encoded_tool_call: list[int] = tokenizer.instruct_tokenizer._encode_tool_calls_in_assistant_message(  # type: ignore[attr-defined]
        AssistantMessage(tool_calls=[tool_call])
    )

    call_id_token = tokenizer.instruct_tokenizer.tokenizer.get_special_token("[CALL_ID]")
    args_token = tokenizer.instruct_tokenizer.tokenizer.get_special_token("[ARGS]")

    call_id_index = encoded_tool_call.index(call_id_token)
    args_token_index = encoded_tool_call.index(args_token)

    encoded_tool_call = encoded_tool_call[:call_id_index] + encoded_tool_call[args_token_index:]  # remove the id

    decoded_tool_call = _decode_tool_calls([encoded_tool_call], tokenizer.instruct_tokenizer.tokenizer)
    assert len(decoded_tool_call) == 1
    assert decoded_tool_call[0].model_dump(exclude={"id"}) == tool_call.model_dump(exclude={"id"})
