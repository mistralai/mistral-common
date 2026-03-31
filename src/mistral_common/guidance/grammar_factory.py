import json
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from mistral_common.guidance.tokenizer import from_mistral_tokenizer
from mistral_common.imports import (
    assert_jinja2_installed,
    assert_llguidance_installed,
    is_jinja2_installed,
    is_llguidance_installed,
)
from mistral_common.protocol.instruct.tool_calls import NamedToolChoice, Tool, ToolChoice, ToolChoiceEnum
from mistral_common.tokens.tokenizers.base import SpecialTokens, TokenizerVersion
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.tekken import is_tekkenizer

if is_llguidance_installed():
    import llguidance as llg

if is_jinja2_installed():
    from jinja2 import Template

JINJA_DIR = Path(__file__).parent / "data"


def _validate_mode_and_tools(mode: ToolChoice, tools: list[Tool] | None) -> None:
    if isinstance(mode, NamedToolChoice) and all(mode.function.name != tool.function.name for tool in (tools or [])):
        raise ValueError(
            f"Tool choice requires the {mode.function.name} tool but no tools with this name has been passed."
        )
    elif mode in [ToolChoiceEnum.any, ToolChoiceEnum.required] and not tools:
        raise ValueError(f"When {mode=} please ensure to pass tools, got {tools=}.")


@lru_cache()
def _cached_get_jinja_template(tokenizer_version: TokenizerVersion, reasoning: bool) -> str:
    if not reasoning:
        jinja_key = _GrammarVariant.base
    elif tokenizer_version < TokenizerVersion.v13:
        jinja_key = _GrammarVariant.plain_think
    else:
        jinja_key = _GrammarVariant.think

    return JINJA_PATHS[jinja_key].read_text(encoding="utf-8")


@lru_cache()
def _cached_get_lark_from_jinja(
    template: str,
    mode: str,
    fcall: str,
    json_schema_str: str | None,
    parallel_tool_calls: bool,
    json_only: bool,
    think_with_json: bool,
    begin_think_token: str | None,
    end_think_token: str | None,
) -> str:
    jinja_template = Template(template)
    lark_grammar = jinja_template.render(
        mode=mode,
        fcall=fcall,
        json_schema_str=json_schema_str,
        parallel_tool_calls=parallel_tool_calls,
        json_only=json_only,
        think_with_json=think_with_json,
        begin_think_token=begin_think_token,
        end_think_token=end_think_token,
    )
    return lark_grammar


class _GrammarVariant(str, Enum):
    base = "base"
    plain_think = "plain_think"
    think = "think"


JINJA_PATHS = {
    _GrammarVariant.base: JINJA_DIR / "base_grammar.lark.jinja",
    _GrammarVariant.plain_think: JINJA_DIR / "plain_text_think_grammar.lark.jinja",
    _GrammarVariant.think: JINJA_DIR / "think_grammar.lark.jinja",
}


_TOOL_CALL_GRAMMAR = "{tool_calls_token} SAFE_WS? {tool_name} {args_token} SAFE_WS? %json {args_json} SAFE_WS?"


def _get_tool_args_json(tool: Tool) -> dict[str, Any]:
    r"""Returns the JSON schema for a tool's arguments."""
    args = tool.function.parameters if tool.function.strict else {"type": "object"}
    return args or {"type": "object", "properties": {}, "additionalProperties": False}


def _convert_tool_calls(
    tools: list[Tool] | None,
    mode: ToolChoice,
    parallel_tool_calls: bool,
    get_special_token_id: Callable[[str], str],
) -> str:
    r"""Converts tool calls to a lark grammar string.

    Args:
        tools: The list of tools available.
        mode: The tool choice mode.
        parallel_tool_calls: Whether parallel tool calls are allowed.
        get_special_token_id: Callable that maps a special token name to its lark grammar syntax.

    Returns:
        The lark grammar string for tool calls.
    """
    if mode == "none":
        return ""

    tool_calls_token = get_special_token_id(SpecialTokens.tool_calls.value)
    args_token = get_special_token_id(SpecialTokens.args.value)

    any_strict_true = any(tool.function.strict for tool in tools) if tools else False

    if not tools or not any_strict_true:
        tool_name = f'"{mode.function.name}"' if isinstance(mode, NamedToolChoice) else "/.+/"
        tool_entries = [(tool_name, '{"type": "object"}')]
    else:
        filtered_tools = (
            [next(tool for tool in tools if tool.function.name == mode.function.name)]
            if isinstance(mode, NamedToolChoice)
            else tools
        )
        tool_entries = [
            (f'"{tool.function.name}"', json.dumps(_get_tool_args_json(tool), ensure_ascii=False))
            for tool in filtered_tools
        ]

    grammar_parts = [
        _TOOL_CALL_GRAMMAR.format(
            tool_calls_token=tool_calls_token,
            args_token=args_token,
            tool_name=name,
            args_json=args_json,
        )
        for name, args_json in tool_entries
    ]

    grammar_tool_call = (
        " | ".join(f"({part})" for part in grammar_parts) if len(grammar_parts) > 1 else grammar_parts[0]
    )

    return f"({grammar_tool_call})+" if parallel_tool_calls else grammar_tool_call


class GrammarFactory:
    r"""Generates grammars for a given tokenizer."""

    @staticmethod
    def is_supported(tokenizer: MistralTokenizer) -> bool:
        r"""Checks whether the given tokenizer is supported by guidance.

        Guidance requires a Tekken tokenizer with version >= v11.

        Args:
            tokenizer: The Mistral tokenizer to check.

        Returns:
            Whether the tokenizer is supported.
        """
        inner = tokenizer.instruct_tokenizer.tokenizer
        return is_tekkenizer(inner) and not inner.version < TokenizerVersion.v11

    def __init__(self, tokenizer: MistralTokenizer) -> None:
        r"""Initialize the grammar factory.

        Args:
            tokenizer: The Mistral tokenizer to generate grammars for.

        Raises:
            ValueError: If the tokenizer is not supported (see
                [`is_supported`][mistral_common.guidance.grammar_factory.GrammarFactory.is_supported]).
        """
        assert_llguidance_installed()
        assert_jinja2_installed()
        self._tokenizer = tokenizer.instruct_tokenizer.tokenizer
        if not self.is_supported(tokenizer):
            raise ValueError(
                f"Guidance requires a Tekken tokenizer with version >= v11, "
                f"got {type(self._tokenizer).__name__} {self._tokenizer.version.value}"
            )
        self._llg_tokenizer = from_mistral_tokenizer(tokenizer)
        self._special_token_map = self._build_special_token_map()

    def _build_special_token_map(self) -> dict[str, str]:
        """Build a mapping from special token names to their grammar syntax."""
        return {self._tokenizer.id_to_piece(i): f"<[{i}]>" for i in range(self._tokenizer.num_special_tokens)}

    def _special_token_lark(self, token_name: str) -> str:
        """Convert special token name to lark grammar syntax."""
        assert token_name in self._special_token_map, f"Unknown special token: {token_name}"
        return self._special_token_map[token_name]

    def _get_optional_special_token_lark(self, token_name: str) -> str | None:
        r"""Returns lark grammar syntax for a special token, or `None` if absent."""
        return self._special_token_map.get(token_name)

    @property
    def llg_tokenizer(self) -> "llg.LLTokenizer":
        return self._llg_tokenizer

    def select_jinja_template(self, reasoning: bool) -> str:
        r"""Selects and returns the appropriate jinja template content based on tokenizer version and reasoning mode.

        Args:
            reasoning: Whether reasoning/thinking mode is enabled.

        Returns:
            The jinja template content as a string.
        """
        return _cached_get_jinja_template(tokenizer_version=self._tokenizer.version, reasoning=reasoning)

    def get_lark_from_jinja(
        self,
        template: str,
        mode: ToolChoice,
        tools: list[Tool] | None,
        json_schema: dict[str, Any] | None,
        parallel_tool_calls: bool,
        json_only: bool = False,
    ) -> str:
        r"""Renders a lark grammar from a jinja template.

        Args:
            template: Jinja template to render as a string.
            mode: The function calling mode (auto, any, none).
            tools: The list of tools available.
            json_schema: JSON schema to additionally allow, unioned with the grammar.
            parallel_tool_calls: Whether parallel tool calls are allowed.
            json_only: If True, generates only JSON schema grammar without text/tool call alternatives.

        Returns:
            The rendered lark grammar string.
        """
        # Verifies that the NamedToolChoice has a valid tool and "any", "required" has tools.
        _validate_mode_and_tools(mode=mode, tools=tools)

        fcall = _convert_tool_calls(tools, mode, parallel_tool_calls, self._special_token_lark)
        json_schema_str = json.dumps(json_schema, ensure_ascii=False) if json_schema else None
        # NamedToolChoice forces a specific tool, which maps to "required" grammar.
        template_mode = ToolChoiceEnum.required if isinstance(mode, NamedToolChoice) else ToolChoiceEnum(mode)
        think_with_json = self._tokenizer.version.supports_model_settings

        begin_think_token = self._get_optional_special_token_lark(SpecialTokens.begin_think.value)
        end_think_token = self._get_optional_special_token_lark(SpecialTokens.end_think.value)

        return _cached_get_lark_from_jinja(
            template=template,
            mode=template_mode.value,
            fcall=fcall,
            json_schema_str=json_schema_str,
            parallel_tool_calls=parallel_tool_calls,
            json_only=json_only,
            think_with_json=think_with_json,
            begin_think_token=begin_think_token,
            end_think_token=end_think_token,
        )

    def get_lark_for_json_schema(self, template: str, json_schema: dict[str, Any]) -> str:
        r"""Returns a lark grammar that only accepts JSON objects matching the given schema.

        Args:
            template: Jinja template to render as a string.
            json_schema: The JSON schema to validate against.

        Returns:
            The rendered lark grammar string that only matches the given JSON schema.
        """
        return self.get_lark_from_jinja(
            template=template,
            mode=ToolChoiceEnum.none,
            tools=None,
            json_schema=json_schema,
            parallel_tool_calls=True,
            json_only=True,
        )

    def get_matcher(self, lark: str) -> "llg.LLMatcher":
        r"""Creates an LLMatcher from a lark grammar string.

        Args:
            lark: The lark grammar string.

        Returns:
            The LLMatcher instance.

        Raises:
            ValueError: If the grammar is invalid.
        """
        error = llg.LLMatcher.validate_grammar(lark)
        if error:
            raise ValueError(f"Invalid grammar: {error}")
        return llg.LLMatcher(self._llg_tokenizer, lark)
