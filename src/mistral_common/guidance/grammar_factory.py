import json
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

from mistral_common.guidance.tokenizer import from_mistral_tokenizer
from mistral_common.imports import (
    assert_jinja2_installed,
    assert_llguidance_installed,
    is_jinja2_installed,
    is_llguidance_installed,
)
from mistral_common.protocol.instruct.tool_calls import NamedToolChoice, Tool, ToolChoice, ToolChoiceEnum
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.tekken import is_tekkenizer

if is_llguidance_installed():
    import llguidance as llg

if is_jinja2_installed():
    from jinja2 import Template

JINJA_DIR = Path(__file__).parent / "data"


@lru_cache()
def _cached_get_jinja_template(tokenizer_version: TokenizerVersion, reasoning: bool) -> str:
    if tokenizer_version < TokenizerVersion.v13:
        jinja_key = _GrammarVariant.plain_think if reasoning else _GrammarVariant.base
    else:
        jinja_key = _GrammarVariant.think if reasoning else _GrammarVariant.base
    jinja_path = JINJA_PATHS[jinja_key]
    return jinja_path.read_text(encoding="utf-8")


@lru_cache()
def _cached_get_lark_from_jinja(
    template: str,
    mode: str,
    fcall: str,
    json_schema_str: str | None,
    parallel_tool_calls: bool,
) -> str:
    jinja_template = Template(template)
    lark_grammar = jinja_template.render(
        mode=mode,
        fcall=fcall,
        json_schema_str=json_schema_str,
        parallel_tool_calls=parallel_tool_calls,
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


def _get_tool_args_json(tool: Tool) -> dict[str, Any]:
    r"""Returns the JSON schema for a tool's arguments."""
    args = tool.function.parameters if tool.function.strict else {"type": "object"}
    if args == {}:
        args = {"type": "object", "properties": {}, "additionalProperties": False}
    return args


def convert_tool_calls(
    tools: list[Tool] | None,
    mode: ToolChoice,
    parallel_tool_calls: bool,
) -> str:
    r"""Converts tool calls to a lark grammar string.

    Args:
        tools: The list of tools available.
        mode: The tool choice mode.
        parallel_tool_calls: Whether parallel tool calls are allowed.

    Returns:
        The lark grammar string for tool calls.
    """
    if mode == "none":
        return ""

    any_strict_true = any(tool.function.strict for tool in tools) if tools else False

    if not tools or not any_strict_true:
        if not isinstance(mode, NamedToolChoice):
            grammar_tool_call = '<TOOL_CALLS> SAFE_WS? /.+/ <ARGS> SAFE_WS? %json {"type": "object"} SAFE_WS?'
        else:
            grammar_tool_call = (
                f'<TOOL_CALLS> SAFE_WS? "{mode.function.name}" <ARGS> SAFE_WS? %json {{"type": "object"}} SAFE_WS?'
            )
    else:
        grammar_per_tool = []
        tools = (
            [next(tool for tool in tools if tool.function.name == mode.function.name)]
            if isinstance(mode, NamedToolChoice)
            else tools
        )
        for tool in tools:
            args = _get_tool_args_json(tool)
            grammar_per_tool.append(
                f'(<TOOL_CALLS> SAFE_WS? "{tool.function.name}" <ARGS> SAFE_WS? %json '
                f"{json.dumps(args, ensure_ascii=False)} SAFE_WS?)"
            )
        grammar_tool_call = f"{' | '.join(grammar_per_tool)}"

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
    ) -> str:
        r"""Renders a lark grammar from a jinja template.

        Args:
            template: Jinja template to render as a string.
            mode: The function calling mode (auto, any, none).
            tools: The list of tools available.
            json_schema: JSON schema to additionally allow, unioned with the grammar.
            parallel_tool_calls: Whether parallel tool calls are allowed.

        Returns:
            The rendered lark grammar string.
        """
        fcall = convert_tool_calls(tools, mode, parallel_tool_calls)
        json_schema_str = json.dumps(json_schema, ensure_ascii=False) if json_schema else None
        # NamedToolChoice forces a specific tool, which maps to "required" grammar
        template_mode = ToolChoiceEnum.required if isinstance(mode, NamedToolChoice) else ToolChoiceEnum(mode)
        return _cached_get_lark_from_jinja(
            template=template,
            mode=template_mode.value,
            fcall=fcall,
            json_schema_str=json_schema_str,
            parallel_tool_calls=parallel_tool_calls,
        )

    def get_lark_for_json_schema(self, json_schema: dict[str, Any]) -> str:
        r"""Returns a lark grammar that only accepts JSON objects matching the given schema."""
        return f"start: SAFE_WS? %json {json.dumps(json_schema, ensure_ascii=False)} \nSAFE_WS: /[ \t\r\n]+/"

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
