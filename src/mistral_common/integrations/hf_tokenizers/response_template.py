import re
from enum import StrEnum, auto
from typing import Any

from mistral_common.tokens.tokenizers.base import (
    PLAIN_BEGIN_THINK,
    PLAIN_END_THINK,
    SpecialTokens,
    TokenizerVersion,
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer


class ThinkingMode(StrEnum):
    r"""How the model encodes thinking in assistant messages.

    Attributes:
        none: No thinking support.
        special: [THINK]/[/THINK] special tokens (v13+).
        plain: Plain `<think>`/`</think>` text (v11 only).
    """

    none = auto()
    special = auto()
    plain = auto()


def _thinking_mode(tokenizer_version: TokenizerVersion) -> ThinkingMode:
    r"""Thinking mode implied by the tokenizer version alone.

    v11 is the only version that can use plain thinking, but a v11 tokenizer
    file cannot distinguish reasoning from non-reasoning models, so callers
    must confirm plain thinking explicitly when they want it.

    Args:
        tokenizer_version: Tokenizer version.

    Returns:
        `special` for v13+, `plain` for v11, `none` otherwise.
    """
    if tokenizer_version >= TokenizerVersion.v13:
        return ThinkingMode.special
    if tokenizer_version == TokenizerVersion.v11:
        return ThinkingMode.plain
    return ThinkingMode.none


def _build_response_template(version: TokenizerVersion, thinking_mode: ThinkingMode) -> dict[str, Any]:
    r"""Build response-template fields from tokenizer capabilities.

    Args:
        version: Tokenizer version used to select field formats.
        thinking_mode: Thinking encoding to emit a `reasoning_content` field for.

    Returns:
        A JSON-serializable Hugging Face response-template mapping.

    Examples:
        >>> _build_response_template(
        ...     version=TokenizerVersion.v13,
        ...     thinking_mode=ThinkingMode.special,
        ... )["defaults"]
        {'role': 'assistant'}
    """
    fields: dict[str, Any] = {}

    if thinking_mode is ThinkingMode.special:
        fields["reasoning_content"] = {
            "open": SpecialTokens.begin_think.value,
            "close": SpecialTokens.end_think.value,
            "content": "text",
        }
    elif thinking_mode is ThinkingMode.plain:
        fields["reasoning_content"] = {
            "open": PLAIN_BEGIN_THINK,
            "close": PLAIN_END_THINK,
            "content": "text",
        }

    if version.has_tools:
        close_pattern = "(?=" + re.escape(SpecialTokens.tool_calls.value) + ")|" + re.escape(SpecialTokens.eos.value)
        if version >= TokenizerVersion.v13:
            open_pattern = (
                re.escape(SpecialTokens.tool_calls.value) + "(?P<name>.+?)" + re.escape(SpecialTokens.args.value)
            )
            fields["tool_calls"] = {
                "open_pattern": open_pattern,
                "close_pattern": close_pattern,
                "repeats": True,
                "content": "json",
                "transform": {
                    "type": "function",
                    "function": {"name": "{name}", "arguments": "{content}"},
                },
            }
        elif version == TokenizerVersion.v11:
            # v11 emits [CALL_ID] only when the call has a non-null id; the lookbehind
            # makes the id group match empty for id-less calls (id="").
            call_id = re.escape(SpecialTokens.call_id.value)
            fields["tool_calls"] = {
                "open_pattern": (
                    re.escape(SpecialTokens.tool_calls.value)
                    + "(?P<name>.+?)"
                    + "(?:"
                    + call_id
                    + ")?"
                    + "(?P<id>(?<="
                    + call_id
                    + ").+?|)"
                    + re.escape(SpecialTokens.args.value)
                ),
                "close_pattern": close_pattern,
                "repeats": True,
                "content": "json",
                "transform": {
                    "type": "function",
                    "id": "{id}",
                    "function": {"name": "{name}", "arguments": "{content}"},
                },
            }
        else:
            assert version.uses_pre_v11_tool_format, f"Unsupported tool-call format for tokenizer version {version}"
            transform: dict[str, Any] = {"id": "{id}"} if version != TokenizerVersion.v2 else {}
            transform["type"] = "function"
            transform["function"] = {"name": "{name}", "arguments": "{arguments}"}
            fields["tool_calls"] = {
                "open": SpecialTokens.tool_calls.value,
                "close": SpecialTokens.eos.value,
                "content": "json",
                "transform_each": True,
                "transform": transform,
            }

        fields["content"] = {
            "close": [SpecialTokens.tool_calls.value, SpecialTokens.eos.value],
            "content": "text",
            "repeats": True,
            "join": "",
        }
        start_anchor: list[str] = [
            SpecialTokens.end_system.value,
            SpecialTokens.end_inst.value,
            SpecialTokens.end_tool_results.value,
        ]
    else:
        fields["content"] = {"close": SpecialTokens.eos.value, "content": "text"}
        start_anchor = [SpecialTokens.end_system.value, SpecialTokens.end_inst.value]

    return {"defaults": {"role": "assistant"}, "start_anchor": start_anchor, "fields": fields}


def generate_response_template(tokenizer_version: TokenizerVersion, thinking_mode: ThinkingMode) -> dict[str, Any]:
    r"""Generate the Hugging Face `response_template` mapping.

    See https://huggingface.co/docs/transformers/en/chat_response_parsing for the
    Transformers response parsing schema.

    Args:
        tokenizer_version: Tokenizer version used to select field formats.
        thinking_mode: Thinking encoding to emit; must be compatible with the version.
            Note the audio-compatibility rule for `plain` is only checked by
            `convert_tokenizer_to_response_template`, which sees the tokenizer.

    Returns:
        A JSON-serializable mapping suitable for a tokenizer configuration.

    Raises:
        ValueError: If `thinking_mode` is unsupported for the tokenizer version.

    Examples:
        >>> generate_response_template(
        ...     tokenizer_version=TokenizerVersion.v13,
        ...     thinking_mode=ThinkingMode.special,
        ... )["defaults"]
        {'role': 'assistant'}
    """
    supported = _thinking_mode(tokenizer_version=tokenizer_version)
    if thinking_mode is not ThinkingMode.none and thinking_mode is not supported:
        raise ValueError(f"Thinking mode {thinking_mode} is not supported for tokenizer version {tokenizer_version}")

    return _build_response_template(version=tokenizer_version, thinking_mode=thinking_mode)


def convert_tokenizer_to_response_template(
    tokenizer: MistralTokenizer, plain_thinking_support: bool | None = None
) -> dict[str, Any]:
    r"""Generate the Hugging Face response template for a loaded `MistralTokenizer`.

    Special-token thinking is detected from the tokenizer's special tokens.
    Plain thinking cannot be reliably detected: reasoning and non-reasoning
    v11 models can have identical tokenizer files, so when
    `plain_thinking_support` is `None` the same heuristic as
    `convert_tokenizer_to_chat_template` applies (v11 without audio).

    See https://huggingface.co/docs/transformers/en/chat_response_parsing for the
    Transformers response parsing schema.

    Args:
        tokenizer: An already-loaded `MistralTokenizer`.
        plain_thinking_support: Whether to emit plain-thinking parsing for v11,
            or `None` for the v11-without-audio heuristic. Pass `False` to suppress.

    Returns:
        A JSON-serializable mapping suitable for a tokenizer configuration.

    Raises:
        ValueError: If plain thinking is requested for a non-v11 tokenizer or one
            with audio support.
    """
    instruct_tokenizer = tokenizer.instruct_tokenizer
    inner_tokenizer = instruct_tokenizer.tokenizer
    version = tokenizer.version
    audio_support = instruct_tokenizer.audio_encoder is not None
    has_special_thinking = inner_tokenizer.is_special(SpecialTokens.begin_think.value) and inner_tokenizer.is_special(
        SpecialTokens.end_think.value
    )

    if plain_thinking_support is None:
        plain_thinking_support = version == TokenizerVersion.v11 and not audio_support

    if plain_thinking_support:
        if version != TokenizerVersion.v11:
            raise ValueError("Plain thinking support is only available for tokenizer version v11")
        if audio_support:
            raise ValueError("Plain thinking support is not available for tokenizers with audio support")
        thinking_mode = ThinkingMode.plain
    elif has_special_thinking:
        thinking_mode = ThinkingMode.special
    else:
        thinking_mode = ThinkingMode.none

    return generate_response_template(tokenizer_version=version, thinking_mode=thinking_mode)
