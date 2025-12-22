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
from mistral_common.protocol.instruct.tool_calls import FunctionCall, ToolCall
from mistral_common.protocol.instruct.validator import (
    MistralRequestValidatorV5,
    MistralRequestValidatorV13,
)
from mistral_common.tokens.instruct.request import InstructRequest
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV11, InstructTokenizerV13
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.test_tekken import get_special_tokens, quick_vocab


def fixture_mistral_tokenizer_v11() -> MistralTokenizer:
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
            UserMessage, AssistantMessage, ToolMessage, SystemMessage, InstructRequest
        ),
    )


def fixture_mistral_tokenizer_v13() -> MistralTokenizer:
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
            UserMessage, AssistantMessage, ToolMessage, SystemMessage, InstructRequest
        ),
    )


def test_find_content_tool_calls() -> None:
    # Test 1: No tool calls
    tokens = [1, 2, 3, 4, 5]
    assert _split_content_and_tool_calls(tokens, 6) == ([1, 2, 3, 4, 5], ())

    # Test 2: One tool call
    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert _split_content_and_tool_calls(tokens, 6) == ([1, 2, 3, 4, 5], ([6, 7, 8, 9, 10],))

    # Test 3: Multiple tool calls
    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 6, 11, 12, 13, 14]
    assert _split_content_and_tool_calls(tokens, 6) == ([1, 2, 3, 4, 5], ([6, 7, 8, 9, 10], [6, 11, 12, 13, 14]))

    # Test 4: No content
    tokens = [6, 7, 8, 9, 10]
    assert _split_content_and_tool_calls(tokens, 6) == ([], ([6, 7, 8, 9, 10],))


@pytest.mark.parametrize(
    "tokenizer",
    (
        MistralTokenizer.v2(),
        MistralTokenizer.v3(),
        MistralTokenizer.v7(),
        MistralTokenizer(
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
                UserMessage, AssistantMessage, ToolMessage, SystemMessage, InstructRequest
            ),
        ),
        fixture_mistral_tokenizer_v11(),
        fixture_mistral_tokenizer_v13(),
    ),
)
@pytest.mark.parametrize(
    "tool_calls",
    (
        [ToolCall(id="call_1", function=FunctionCall(name="ab", arguments="{}"))],
        [
            ToolCall(id="call_1", function=FunctionCall(name="ab", arguments="{}")),
            ToolCall(id="call_2", function=FunctionCall(name="cd", arguments='{"a": "b", "c": "d"}')),
        ],
    ),
)
def test_decode_tool_calls(tokenizer: MistralTokenizer, tool_calls: list[ToolCall]) -> None:
    # Test 1:
    # valid tool calls
    encoded_tool_calls = tokenizer.instruct_tokenizer._encode_tool_calls_in_assistant_message(  # type: ignore[attr-defined]
        AssistantMessage(tool_calls=tool_calls)
    )

    splitted_tool_calls = _split_integer_list_by_value(
        encoded_tool_calls, tokenizer.instruct_tokenizer.tokenizer.get_special_token("[TOOL_CALLS]")
    )

    decoded_tool_calls = _decode_tool_calls(splitted_tool_calls, tokenizer.instruct_tokenizer.tokenizer)
    if tokenizer.instruct_tokenizer.tokenizer.version != TokenizerVersion.v11:
        assert len(decoded_tool_calls) == len(tool_calls)
        for decoded_tool_call, tool_call in zip(decoded_tool_calls, tool_calls):
            assert decoded_tool_call.model_dump(exclude={"id"}) == tool_call.model_dump(exclude={"id"})
    else:
        # v11 is the only version that encode the id
        assert decoded_tool_calls == tool_calls

    versions_inf_v11 = [
        TokenizerVersion.v2,
        TokenizerVersion.v3,
        TokenizerVersion.v7,
    ]

    # Test 2:
    # invalid tool calls (remove a JSON needed token from the args)
    if tokenizer.instruct_tokenizer.tokenizer.version in versions_inf_v11:
        arg_index = tokenizer.instruct_tokenizer.tokenizer.get_special_token("[TOOL_CALLS]")
    else:
        arg_index = tokenizer.instruct_tokenizer.tokenizer.get_special_token("[ARGS]")

    first_args_index = splitted_tool_calls[0].index(arg_index)
    splitted_tool_calls = (
        splitted_tool_calls[0][: first_args_index + 1]
        + splitted_tool_calls[0][first_args_index + 2 :],  # make the args invalid
        *splitted_tool_calls[1:],
    )

    with pytest.raises(
        InvalidToolCallError
        if tokenizer.instruct_tokenizer.tokenizer.version in versions_inf_v11
        else InvalidArgsToolCallError
    ):
        _decode_tool_calls(splitted_tool_calls, tokenizer.instruct_tokenizer.tokenizer)


def test_decode_tool_calls_v11_without_id() -> None:
    tokenizer = fixture_mistral_tokenizer_v11()
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
