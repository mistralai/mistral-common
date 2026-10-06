import base64
import json
import re

import pytest

from mistral_common.experimental.tools import InvalidArgsToolCallError, InvalidToolCallError, _decode_tool_calls
from mistral_common.tokens.tokenizers.base import SpecialTokens, TokenizerVersion
from mistral_common.tokens.tokenizers.tekken import SpecialTokenInfo, Tekkenizer, TokenInfo

_NUM_SPECIAL_TOKENS = 100


def _byte_level_tekkenizer(version: TokenizerVersion) -> Tekkenizer:
    r"""Build a byte-level Tekkenizer exposing the v11+ tool call control tokens."""
    vocab = [TokenInfo(rank=i, token_bytes=base64.b64encode(bytes([i])).decode(), token_str=chr(i)) for i in range(256)]
    special_tokens = [
        *Tekkenizer.DEPRECATED_SPECIAL_TOKENS,
        SpecialTokenInfo(rank=len(Tekkenizer.DEPRECATED_SPECIAL_TOKENS), token_str=SpecialTokens.args, is_control=True),
        SpecialTokenInfo(
            rank=len(Tekkenizer.DEPRECATED_SPECIAL_TOKENS) + 1, token_str=SpecialTokens.call_id, is_control=True
        ),
    ]
    return Tekkenizer(
        vocab=vocab,
        special_tokens=special_tokens,
        pattern=r".",
        vocab_size=256 + _NUM_SPECIAL_TOKENS,
        num_special_tokens=_NUM_SPECIAL_TOKENS,
        version=version,
    )


def _tool_call_tokens(tokenizer: Tekkenizer, *parts: str) -> list[int]:
    r"""Encode parts, mapping control token strings to their IDs and encoding the rest as text."""
    tokens = [tokenizer.get_special_token(SpecialTokens.tool_calls)]
    for part in parts:
        if tokenizer.is_special(part):
            tokens.append(tokenizer.get_special_token(part))
        else:
            tokens.extend(tokenizer.encode(part, bos=False, eos=False))
    return tokens


@pytest.mark.parametrize("version", [TokenizerVersion.v11, TokenizerVersion.v13])
@pytest.mark.parametrize("arguments", ["[1, 2]", "5", '"x"', "null", "true"])
def test_decode_tool_calls_v11_plus_rejects_non_object_arguments(version: TokenizerVersion, arguments: str) -> None:
    tokenizer = _byte_level_tekkenizer(version=version)
    tool_call_tokens = _tool_call_tokens(tokenizer, "f", SpecialTokens.args, arguments)

    with pytest.raises(InvalidArgsToolCallError, match=re.escape("Expected a dict.")):
        _decode_tool_calls(tool_call_tokens=[tool_call_tokens], tokenizer=tokenizer)


def test_decode_tool_calls_v11_with_call_id_rejects_non_object_arguments() -> None:
    tokenizer = _byte_level_tekkenizer(version=TokenizerVersion.v11)
    tool_call_tokens = _tool_call_tokens(tokenizer, "f", SpecialTokens.call_id, "abc123def", SpecialTokens.args, "[1]")

    with pytest.raises(InvalidArgsToolCallError, match=re.escape("Expected a dict.")):
        _decode_tool_calls(tool_call_tokens=[tool_call_tokens], tokenizer=tokenizer)


@pytest.mark.parametrize(
    ("parts", "expected_message"),
    [
        (("f", "{}"), "Control token [ARGS] not found"),
        (("f", SpecialTokens.args, "{}", SpecialTokens.args), "Control token [ARGS] found more than once"),
    ],
)
@pytest.mark.parametrize("version", [TokenizerVersion.v11, TokenizerVersion.v13])
def test_decode_tool_calls_v11_plus_malformed_structure(
    version: TokenizerVersion, parts: tuple[str, ...], expected_message: str
) -> None:
    tokenizer = _byte_level_tekkenizer(version=version)
    tool_call_tokens = _tool_call_tokens(tokenizer, *parts)

    with pytest.raises(InvalidToolCallError, match=re.escape(expected_message)):
        _decode_tool_calls(tool_call_tokens=[tool_call_tokens], tokenizer=tokenizer)


def test_decode_tool_calls_v11_with_call_id_missing_args() -> None:
    tokenizer = _byte_level_tekkenizer(version=TokenizerVersion.v11)
    tool_call_tokens = _tool_call_tokens(tokenizer, "f", SpecialTokens.call_id, "abc123def", "{}")

    with pytest.raises(InvalidToolCallError, match=re.escape("Control token [ARGS] not found")):
        _decode_tool_calls(tool_call_tokens=[tool_call_tokens], tokenizer=tokenizer)


@pytest.mark.parametrize("version", [TokenizerVersion.v11, TokenizerVersion.v13])
def test_decode_tool_calls_v11_plus_object_arguments(version: TokenizerVersion) -> None:
    tokenizer = _byte_level_tekkenizer(version=version)
    tool_call_tokens = _tool_call_tokens(tokenizer, "f", SpecialTokens.args, '{"a": [1, "é"]}')

    (tool_call,) = _decode_tool_calls(tool_call_tokens=[tool_call_tokens], tokenizer=tokenizer)

    assert tool_call.function.name == "f"
    assert json.loads(tool_call.function.arguments) == {"a": [1, "é"]}
