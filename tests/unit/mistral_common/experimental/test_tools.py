import re

import pytest

from mistral_common.experimental.tools import InvalidArgsToolCallError, InvalidToolCallError, _decode_tool_calls
from mistral_common.protocol.instruct.tool_calls import FunctionCall, ToolCall
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.test_tekken import get_special_tokens, quick_vocab


def _tekkenizer(version: TokenizerVersion) -> Tekkenizer:
    return Tekkenizer(
        vocab=quick_vocab(),
        special_tokens=get_special_tokens(tokenizer_version=version),
        pattern=r".+",
        vocab_size=256 + 100,
        num_special_tokens=100,
        version=version,
    )


def _tool_call_tokens(tokenizer: Tekkenizer, parts: list[str]) -> list[int]:
    r"""Build `[TOOL_CALLS]` followed by `parts`, where bracketed control tokens are mapped to their IDs."""
    tokens = [tokenizer.get_special_token("[TOOL_CALLS]")]
    for part in parts:
        if part in ("[ARGS]", "[CALL_ID]"):
            tokens.append(tokenizer.get_special_token(part))
        else:
            tokens.extend(tokenizer.encode(part, bos=False, eos=False))
    return tokens


V11_AND_V13_FORMATS = (
    pytest.param(TokenizerVersion.v11, ["f"], id="v11"),
    pytest.param(TokenizerVersion.v11, ["f", "[CALL_ID]", "abcdefghi"], id="v11_with_call_id"),
    pytest.param(TokenizerVersion.v13, ["f"], id="v13"),
)


@pytest.mark.parametrize(("version", "prefix"), V11_AND_V13_FORMATS)
@pytest.mark.parametrize("arguments", ("[1, 2]", "5", "true", '"x"', "null"))
def test_decode_tool_calls_v11_non_dict_arguments(version: TokenizerVersion, prefix: list[str], arguments: str) -> None:
    tokenizer = _tekkenizer(version=version)
    tool_call_tokens = _tool_call_tokens(tokenizer=tokenizer, parts=[*prefix, "[ARGS]", arguments])

    with pytest.raises(InvalidArgsToolCallError, match=re.escape("Expected a dict.")):
        _decode_tool_calls(tool_call_tokens=[tool_call_tokens], tokenizer=tokenizer)


@pytest.mark.parametrize(("version", "prefix"), V11_AND_V13_FORMATS)
def test_decode_tool_calls_v11_invalid_json_arguments(version: TokenizerVersion, prefix: list[str]) -> None:
    tokenizer = _tekkenizer(version=version)
    tool_call_tokens = _tool_call_tokens(tokenizer=tokenizer, parts=[*prefix, "[ARGS]", '{"a": '])

    with pytest.raises(InvalidArgsToolCallError) as exc_info:
        _decode_tool_calls(tool_call_tokens=[tool_call_tokens], tokenizer=tokenizer)
    assert isinstance(exc_info.value.__cause__, ValueError)


@pytest.mark.parametrize(
    ("version", "parts", "expected_message"),
    (
        pytest.param(TokenizerVersion.v11, ["f", "{}"], "[ARGS] not found", id="v11_missing_args"),
        pytest.param(
            TokenizerVersion.v11,
            ["f", "[ARGS]", "{}", "[ARGS]"],
            "[ARGS] found more than once",
            id="v11_duplicate_args",
        ),
        pytest.param(
            TokenizerVersion.v11,
            ["f", "[CALL_ID]", "abcdefghi", "{}"],
            "[ARGS] not found",
            id="v11_with_call_id_missing_args",
        ),
        pytest.param(
            TokenizerVersion.v11,
            ["f", "[CALL_ID]", "abcdefghi", "[CALL_ID]", "abcdefghi", "[ARGS]", "{}"],
            "[CALL_ID] found more than once",
            id="v11_duplicate_call_id",
        ),
        pytest.param(TokenizerVersion.v13, ["f", "{}"], "[ARGS] not found", id="v13_missing_args"),
    ),
)
def test_decode_tool_calls_v11_malformed_control_tokens(
    version: TokenizerVersion, parts: list[str], expected_message: str
) -> None:
    tokenizer = _tekkenizer(version=version)
    tool_call_tokens = _tool_call_tokens(tokenizer=tokenizer, parts=parts)

    with pytest.raises(InvalidToolCallError, match=re.escape(expected_message)) as exc_info:
        _decode_tool_calls(tool_call_tokens=[tool_call_tokens], tokenizer=tokenizer)
    assert not isinstance(exc_info.value, InvalidArgsToolCallError)
    assert isinstance(exc_info.value.__cause__, ValueError)


@pytest.mark.parametrize(("version", "prefix"), V11_AND_V13_FORMATS)
def test_decode_tool_calls_v11_dict_arguments(version: TokenizerVersion, prefix: list[str]) -> None:
    tokenizer = _tekkenizer(version=version)
    tool_call_tokens = _tool_call_tokens(tokenizer=tokenizer, parts=[*prefix, "[ARGS]", '{"a": [1, "b"]}'])

    decoded = _decode_tool_calls(tool_call_tokens=[tool_call_tokens], tokenizer=tokenizer)

    expected = ToolCall(function=FunctionCall(name="f", arguments='{"a": [1, "b"]}'))
    assert [tool_call.model_dump(exclude={"id"}) for tool_call in decoded] == [expected.model_dump(exclude={"id"})]
