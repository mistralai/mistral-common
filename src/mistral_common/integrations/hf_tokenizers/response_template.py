import re
from pathlib import Path
from typing import Any

from mistral_common.tokens.tokenizers.base import SpecialTokens, TokenizerVersion
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

_PLAIN_THINKING_OPEN = "<think>"
_PLAIN_THINKING_CLOSE = "</think>"


def _build_response_template(version: TokenizerVersion, plain_thinking_support: bool) -> dict[str, Any]:
    r"""Build response-template fields from tokenizer capabilities.

    Args:
        version: Tokenizer version used to select field formats.
        plain_thinking_support: Whether plain-text thinking is supported.

    Returns:
        A JSON-serializable Hugging Face response-template mapping.

    Examples:
        >>> _build_response_template(
        ...     version=TokenizerVersion.v13,
        ...     plain_thinking_support=False,
        ... )["defaults"]
        {'role': 'assistant'}
    """
    fields: dict[str, Any] = {}

    if version.uses_v13_tool_format:
        fields["reasoning_content"] = {
            "open": SpecialTokens.begin_think.value,
            "close": SpecialTokens.end_think.value,
            "content": "text",
        }
    elif version.uses_plain_thinking and plain_thinking_support:
        fields["reasoning_content"] = {
            "open": _PLAIN_THINKING_OPEN,
            "close": _PLAIN_THINKING_CLOSE,
            "content": "text",
        }

    if version.has_tools:
        close_pattern = "(?=" + re.escape(SpecialTokens.tool_calls.value) + ")|" + re.escape(SpecialTokens.eos.value)
        if version.uses_v13_tool_format:
            open_pattern = re.escape(SpecialTokens.tool_calls.value) + "(?P<name>.+?)"
            open_pattern += re.escape(SpecialTokens.args.value)
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
        elif version.uses_call_id_in_tool_calls:
            fields["tool_calls"] = {
                "open_pattern": (
                    re.escape(SpecialTokens.tool_calls.value)
                    + "(?P<name>.+?)"
                    + re.escape(SpecialTokens.call_id.value)
                    + "(?P<id>.+?)"
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
        elif version.uses_pre_v11_tool_format:
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

    if version.has_tools:
        fields["content"] = {
            "close": [SpecialTokens.tool_calls.value, SpecialTokens.eos.value],
            "content": "text",
            "repeats": True,
            "join": "",
        }
        start_anchor: str | list[str] = [
            SpecialTokens.end_inst.value,
            SpecialTokens.end_tool_results.value,
        ]
    else:
        fields["content"] = {"close": SpecialTokens.eos.value, "content": "text"}
        start_anchor = SpecialTokens.end_inst.value

    return {"defaults": {"role": "assistant"}, "start_anchor": start_anchor, "fields": fields}


def generate_response_template(
    tokenizer_version: TokenizerVersion, thinking_support: bool, plain_thinking_support: bool
) -> dict[str, Any]:
    r"""Generate the Hugging Face `response_template` mapping.

    See https://huggingface.co/docs/transformers/en/chat_response_parsing for the
    Transformers response parsing schema.

    Args:
        tokenizer_version: Tokenizer version used to select field formats.
        thinking_support: Whether special-token thinking is supported.
        plain_thinking_support: Whether plain-text thinking is supported.

    Returns:
        A JSON-serializable mapping suitable for a tokenizer configuration.

    Raises:
        ValueError: If the requested thinking modes are unsupported or combined.

    Examples:
        >>> generate_response_template(
        ...     tokenizer_version=TokenizerVersion.v13,
        ...     thinking_support=True,
        ...     plain_thinking_support=False,
        ... )["defaults"]
        {'role': 'assistant'}
    """
    if thinking_support and plain_thinking_support:
        raise ValueError("thinking_support and plain_thinking_support are mutually exclusive")
    if plain_thinking_support and not tokenizer_version.uses_plain_thinking:
        raise ValueError("Plain thinking support is only available for tokenizer version v11")
    if thinking_support and not tokenizer_version.uses_v13_tool_format:
        raise ValueError("Special-token thinking support requires tokenizer version v13 or above")

    return _build_response_template(
        version=tokenizer_version,
        plain_thinking_support=plain_thinking_support,
    )


def convert_tokenizer_to_response_template(
    tokenizer_file: str | Path, plain_thinking_support: bool | None
) -> dict[str, Any]:
    r"""Load a tokenizer and generate its Hugging Face response template.

    Special-token thinking is detected by checking for both thinking delimiters.
    Plain thinking cannot be reliably detected from a tokenizer file: reasoning
    and non-reasoning v11 models can have identical tokenizer data. When
    `plain_thinking_support` is `None`, the heuristic selects v11 tokenizers
    without audio and can therefore over-trigger on non-reasoning v11 models.
    Pass `False` to suppress that guess.

    See https://huggingface.co/docs/transformers/en/chat_response_parsing for the
    Transformers response parsing schema.

    Args:
        tokenizer_file: Path to a tokenizer file.
        plain_thinking_support: Whether to enable plain thinking, or `None` to
            use the v11-without-audio heuristic.

    Returns:
        A JSON-serializable mapping suitable for a tokenizer configuration.

    Raises:
        ValueError: If thinking support is unsupported by the tokenizer version or both thinking
            modes are enabled.
    """
    mistral_tokenizer = MistralTokenizer.from_file(tokenizer_filename=tokenizer_file)
    instruct_tokenizer = mistral_tokenizer.instruct_tokenizer
    tokenizer = instruct_tokenizer.tokenizer
    version = mistral_tokenizer.version
    audio_support = instruct_tokenizer.audio_encoder is not None
    thinking_support = tokenizer.is_special(SpecialTokens.begin_think.value) and tokenizer.is_special(
        SpecialTokens.end_think.value
    )

    if plain_thinking_support is None:
        plain_thinking_support = version.uses_plain_thinking and not audio_support

    return generate_response_template(
        tokenizer_version=version,
        thinking_support=thinking_support,
        plain_thinking_support=plain_thinking_support,
    )
