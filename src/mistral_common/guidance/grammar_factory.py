import json
from dataclasses import dataclass
from enum import StrEnum, auto
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from typing_extensions import assert_never

from mistral_common.deprecation import warn_once
from mistral_common.exceptions import InvalidRequestException
from mistral_common.guidance.tokenizer import from_mistral_tokenizer
from mistral_common.imports import (
    assert_jinja2_installed,
    assert_llguidance_installed,
    is_jinja2_installed,
    is_llguidance_installed,
)
from mistral_common.protocol.instruct.request import (
    ChatCompletionRequest,
    ResponseFormats,
    SchemaRenderingMode,
)
from mistral_common.protocol.instruct.tool_calls import NamedToolChoice, Tool, ToolChoiceEnum
from mistral_common.tokens.tokenizers.base import SpecialTokens, TokenizerVersion
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.tekken import is_tekkenizer

if is_llguidance_installed():
    import llguidance as llg

if is_jinja2_installed():
    from jinja2 import Template

JINJA_DIR = Path(__file__).parent / "data"


class _GrammarMode(StrEnum):
    r"""Template branch selected by a tool choice; `required` and `any` both force a tool call."""

    auto = "auto"
    any = "any"
    none = "none"


_TOOL_CHOICE_TO_GRAMMAR_MODE: dict[ToolChoiceEnum, _GrammarMode] = {
    ToolChoiceEnum.auto: _GrammarMode.auto,
    ToolChoiceEnum.any: _GrammarMode.any,
    ToolChoiceEnum.required: _GrammarMode.any,
    ToolChoiceEnum.none: _GrammarMode.none,
}


@lru_cache()
def _cached_get_jinja_template(tokenizer_version: TokenizerVersion, has_think_tokens: bool) -> str:
    if tokenizer_version >= TokenizerVersion.v13 and has_think_tokens:
        jinja_key = _GrammarVariant.think
    else:
        jinja_key = _GrammarVariant.base
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
    return str(lark_grammar)


class _GrammarVariant(StrEnum):
    base = auto()
    plain_think = auto()
    think = auto()


JINJA_PATHS = {
    _GrammarVariant.base: JINJA_DIR / "base_grammar.lark.jinja",
    _GrammarVariant.plain_think: JINJA_DIR / "plain_text_think_grammar.lark.jinja",
    _GrammarVariant.think: JINJA_DIR / "think_grammar.lark.jinja",
}


_TOOL_CALL_GRAMMAR = "{tool_calls_token} SAFE_WS? {tool_name} {args_token} SAFE_WS? %json {args_json} SAFE_WS?"


@dataclass(frozen=True)
class _GrammarSpec:
    r"""Carry request-derived settings through one grammar render."""

    mode: _GrammarMode
    tools: list[Tool] | None
    json_schema: dict[str, Any] | None
    parallel_tool_calls: bool
    json_only: bool
    force_tool_names: bool


def _get_grammar_spec(request: ChatCompletionRequest) -> _GrammarSpec | None:
    r"""Select grammar settings based on the request's response format."""
    match request.response_format.type:
        case ResponseFormats.text:
            return _text_grammar_spec(request=request)
        case ResponseFormats.json_object | ResponseFormats.json_schema:
            return _json_grammar_spec(request=request)
        case _ as unreachable:
            assert_never(unreachable)


def _text_grammar_spec(request: ChatCompletionRequest) -> _GrammarSpec | None:
    r"""Resolve text requests while preserving their tool-choice semantics."""
    tools = request.tools or None
    choice = request.tool_choice

    if tools is None:
        match choice:
            case ToolChoiceEnum.auto:
                return None
            case ToolChoiceEnum.none:
                mode = _GrammarMode.none
            case ToolChoiceEnum.any | ToolChoiceEnum.required:
                raise InvalidRequestException(f"No tools provided for tool choice `{choice}`")
            case NamedToolChoice():
                raise InvalidRequestException(f"No tools provided for tool choice `{choice.function.name}`")
            case _ as unreachable:
                assert_never(unreachable)
        force_tool_names = False
    else:
        match choice:
            case NamedToolChoice():
                name = choice.function.name
                available = [tool.function.name for tool in tools]
                if name not in available:
                    raise InvalidRequestException(f"Tool choice `{name}` not found in available tools: {available}.")
                tools = [tool for tool in tools if tool.function.name == name]
                mode = _GrammarMode.any
                force_tool_names = True
            case ToolChoiceEnum.auto | ToolChoiceEnum.none | ToolChoiceEnum.any | ToolChoiceEnum.required:
                mode = _TOOL_CHOICE_TO_GRAMMAR_MODE[choice]
                force_tool_names = False
            case _ as unreachable:
                assert_never(unreachable)

    return _GrammarSpec(
        mode=mode,
        tools=tools,
        json_schema=None,
        parallel_tool_calls=request.parallel_tool_calls,
        json_only=False,
        force_tool_names=force_tool_names,
    )


def _json_grammar_spec(request: ChatCompletionRequest) -> _GrammarSpec:
    r"""Resolve JSON response settings and any compatible tool-call union."""
    tools = request.tools or None
    choice = request.tool_choice
    if tools is not None and choice not in (ToolChoiceEnum.none, ToolChoiceEnum.auto):
        raise InvalidRequestException(
            f"`{request.response_format.type}` response type with tools is only compatible with `tool_choice: auto`"
        )

    json_schema = request.response_format.get_schema(purpose=SchemaRenderingMode.grammar)
    assert json_schema is not None

    if tools is not None and choice != ToolChoiceEnum.none:
        return _GrammarSpec(
            mode=_GrammarMode.auto,
            tools=tools,
            json_schema=json_schema,
            parallel_tool_calls=request.parallel_tool_calls,
            json_only=False,
            force_tool_names=False,
        )

    return _GrammarSpec(
        mode=_GrammarMode.none,
        tools=None,
        json_schema=json_schema,
        parallel_tool_calls=request.parallel_tool_calls,
        json_only=True,
        force_tool_names=False,
    )


def _get_tool_args_json(tool: Tool) -> dict[str, Any]:
    r"""Return the JSON schema of a tool's arguments.

    Strict tools use their declared parameters schema; non-strict tools accept
    any JSON object. Always returns a non-empty object schema.

    Args:
        tool: The tool whose argument schema is extracted.

    Returns:
        The JSON schema for the tool's arguments. Never `None`; falls back to a
        permissive empty-object schema when the tool declares none.
    """
    args = tool.function.parameters if tool.function.strict else {"type": "object"}
    return args or {"type": "object", "properties": {}, "additionalProperties": False}


def _convert_tool_calls(
    tools: list[Tool] | None,
    mode: _GrammarMode,
    parallel_tool_calls: bool,
    get_special_token_id: Callable[[str], str],
    force_tool_names: bool,
) -> str:
    r"""Convert tool definitions into a lark grammar fragment.

    Builds a grammar that matches one or more tool calls. Each tool entry maps
    the tool name to its JSON argument schema. Non-strict tools accept any JSON
    object arguments.

    Args:
        tools: The list of tools available. Ignored when mode is `none`.
        mode: The grammar mode; `none` disables tool calls.
        parallel_tool_calls: If `True`, the grammar allows repeated tool calls
            (one or more); if `False`, exactly one.
        get_special_token_id: Callable that maps a special token name to its
            lark grammar syntax.
        force_tool_names: If `True`, pin non-strict tool calls to the names in
            `tools` rather than accepting any tool name.

    Returns:
        The lark grammar string for tool calls, or an empty string when mode
        is `none`.
    """
    if mode == _GrammarMode.none:
        return ""

    tool_calls_token = get_special_token_id(SpecialTokens.tool_calls.value)
    args_token = get_special_token_id(SpecialTokens.args.value)

    any_strict_true = any(tool.function.strict for tool in tools) if tools else False

    if not tools or not any_strict_true:
        if force_tool_names:
            assert tools
            tool_name = " | ".join(f'"{tool.function.name}"' for tool in tools)
            if len(tools) > 1:
                tool_name = f"({tool_name})"
        else:
            tool_name = "/.+/"
        tool_entries = [(tool_name, '{"type": "object"}')]
    else:
        tool_entries = [
            (f'"{tool.function.name}"', json.dumps(_get_tool_args_json(tool), ensure_ascii=False)) for tool in tools
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
    r"""Generates Lark grammars that constrain model output for a given tokenizer.

    Grammars cover tool calls, JSON schema output, and thinking sections, and are
    rendered from tokenizer-version-specific jinja templates using the tokenizer's
    special tokens.
    """

    @staticmethod
    def is_supported(tokenizer: MistralTokenizer) -> bool:
        r"""Check whether the given tokenizer is supported by guidance.

        Guidance requires a Tekken tokenizer with version >= v11.

        Args:
            tokenizer: The Mistral tokenizer to check.

        Returns:
            `True` if the tokenizer is a Tekkenizer of version >= v11, `False` otherwise.
        """
        inner = tokenizer.instruct_tokenizer.tokenizer
        return is_tekkenizer(inner) and not inner.version < TokenizerVersion.v11

    def __init__(self, tokenizer: MistralTokenizer) -> None:
        r"""Initialize the grammar factory.

        Requires llguidance and jinja2 to be installed.

        Args:
            tokenizer: The Mistral tokenizer to generate grammars for. Must be a
                Tekken tokenizer with version >= v11 (see
                [`is_supported`][mistral_common.guidance.grammar_factory.GrammarFactory.is_supported]).

        Raises:
            ValueError: If the tokenizer is not supported.
            ImportError: If llguidance or jinja2 is not installed.
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
        r"""Map every special token string to its llguidance lark syntax.

        Returns:
            Dictionary mapping each special token string to its lark token
            reference, e.g. "<s>" -> "<[1]>".
        """
        return {self._tokenizer.id_to_piece(i): f"<[{i}]>" for i in range(self._tokenizer.num_special_tokens)}

    def _special_token_lark(self, token_name: str) -> str:
        r"""Return the lark grammar syntax for a special token.

        Args:
            token_name: The special token string (e.g., "[TOOL_CALLS]").

        Returns:
            The lark token reference for this token.

        Raises:
            AssertionError: If the token name is not a registered special token.
        """
        assert token_name in self._special_token_map, f"Unknown special token: {token_name}"
        return self._special_token_map[token_name]

    def _get_optional_special_token_lark(self, token_name: str) -> str | None:
        r"""Return lark grammar syntax for a special token, or `None` if absent.

        Unlike `_special_token_lark`, missing tokens do not raise.

        Args:
            token_name: The special token string (e.g., "[THINK]").

        Returns:
            The lark token reference, or `None` if the token is not registered
            in this tokenizer.
        """
        return self._special_token_map.get(token_name)

    @property
    def llg_tokenizer(self) -> "llg.LLTokenizer":
        r"""The llguidance tokenizer used to validate grammars.

        Returns:
            The LLTokenizer instance adapted from this Mistral tokenizer.
        """
        return self._llg_tokenizer

    def select_jinja_template(self, reasoning: bool | None = None) -> str:
        r"""Selects and returns the appropriate jinja template content.

        Selection derives from the tokenizer version and presence of think tokens:
        - Returns the `think` template when the tokenizer version is >= v13 and both
          `[THINK]` and `[/THINK]` special tokens are registered.
        - Returns the `base` template in all other cases.

        Args:
            reasoning: Deprecated and ignored. Template selection is determined solely
                by the tokenizer version and presence of think tokens. Will be removed
                in 1.13.0.

        Returns:
            The jinja template content as a string.
        """
        if reasoning is not None:
            warn_once(
                "select_jinja_template.reasoning",
                "The reasoning parameter of select_jinja_template is deprecated, "
                "no longer has any effect, and will be removed in 1.13.0.",
                DeprecationWarning,
                stacklevel=2,
            )
        has_begin_think = SpecialTokens.begin_think.value in self._special_token_map
        has_end_think = SpecialTokens.end_think.value in self._special_token_map
        assert has_begin_think == has_end_think, (
            f"both {SpecialTokens.begin_think.value} and {SpecialTokens.end_think.value} "
            "should be defined or none of them."
        )
        return _cached_get_jinja_template(tokenizer_version=self._tokenizer.version, has_think_tokens=has_begin_think)

    def get_lark_from_jinja(
        self,
        template: str,
        mode: ToolChoiceEnum,
        tools: list[Tool] | None,
        json_schema: dict[str, Any] | None,
        parallel_tool_calls: bool,
        json_only: bool,
        force_tool_names: bool,
    ) -> str:
        r"""Render a lark grammar from a jinja template.

        This is the general entry point for grammar generation. Combines tool call,
        JSON schema, and thinking sections according to the requested mode.

        Args:
            template: Jinja template to render, as obtained from `select_jinja_template`.
            mode: The tool choice. `none` disables tool calls; `any` and `required`
                both force a tool call. Named tool choices go through
                `get_grammar_for_chat_request`.
            tools: The list of tools available. Required when mode is any/required;
                ignored when mode is none.
            json_schema: Optional JSON schema additionally allowed by the grammar,
                unioned with tool call and text alternatives. If `None`, no JSON
                section is added. With tools and a mode other than `none`, the
                `any` branch is rendered so plain text cannot replace the JSON
                answer; think templates still allow text that does not start
                like JSON before a tool call.
            parallel_tool_calls: If `True`, the grammar allows one or more tool
                calls in sequence; if `False`, exactly one.
            json_only: If `True`, generates only JSON schema grammar without
                text/tool call alternatives.
            force_tool_names: If `True`, pins non-strict tool names to those in
                `tools`.

        Returns:
            The rendered lark grammar string.

        Raises:
            ValueError: If `mode` is a `NamedToolChoice`, or mode is any/required
                with no tools provided.
        """
        if isinstance(mode, NamedToolChoice):
            raise ValueError("Named tool choices are only supported by `get_grammar_for_chat_request`.")
        return self._render(
            template=template,
            mode=_TOOL_CHOICE_TO_GRAMMAR_MODE[ToolChoiceEnum(mode)],
            tools=tools,
            json_schema=json_schema,
            parallel_tool_calls=parallel_tool_calls,
            json_only=json_only,
            force_tool_names=force_tool_names,
        )

    def _render(
        self,
        template: str,
        mode: _GrammarMode,
        tools: list[Tool] | None,
        json_schema: dict[str, Any] | None,
        parallel_tool_calls: bool,
        json_only: bool,
        force_tool_names: bool,
    ) -> str:
        r"""Render the lark grammar for a grammar mode; see `get_lark_from_jinja` for the arguments."""
        if mode == _GrammarMode.any and not tools:
            raise ValueError(f"A forced tool call requires tools, please ensure to pass tools, got {tools=}.")

        json_schema_str = json.dumps(json_schema, ensure_ascii=False) if json_schema else None
        if json_schema_str is not None and tools and not json_only and mode == _GrammarMode.auto:
            # Free text before a tool call would also match the JSON answer and defeat the schema,
            # so the union with tools only allows JSON or tool calls.
            mode = _GrammarMode.any

        fcall = _convert_tool_calls(
            tools=tools,
            mode=mode,
            parallel_tool_calls=parallel_tool_calls,
            get_special_token_id=self._special_token_lark,
            force_tool_names=force_tool_names,
        )
        think_with_json = self._tokenizer.version.supports_model_settings

        begin_think_token = self._get_optional_special_token_lark(SpecialTokens.begin_think.value)
        end_think_token = self._get_optional_special_token_lark(SpecialTokens.end_think.value)

        return _cached_get_lark_from_jinja(
            template=template,
            mode=mode.value,
            fcall=fcall,
            json_schema_str=json_schema_str,
            parallel_tool_calls=parallel_tool_calls,
            json_only=json_only,
            think_with_json=think_with_json,
            begin_think_token=begin_think_token,
            end_think_token=end_think_token,
        )

    def get_grammar_for_chat_request(self, request: ChatCompletionRequest, template: str) -> str | None:
        r"""Resolve a chat request to its lark grammar in one pass.

        Returns `None` when the request needs no grammar and raises
        `InvalidRequestException` for inconsistent request combinations.
        `template` is used as given; callers select their custom or default
        template before calling this method.

        Args:
            request: Chat completion request to resolve.
            template: Jinja template used to render the grammar.

        Returns:
            The rendered lark grammar, or `None` when no grammar is needed.

        Raises:
            InvalidRequestException: If the request contains an incompatible
                tool choice or response format.
        """
        spec = _get_grammar_spec(request=request)
        if spec is None:
            return None
        return self._render(
            template=template,
            mode=spec.mode,
            tools=spec.tools,
            json_schema=spec.json_schema,
            parallel_tool_calls=spec.parallel_tool_calls,
            json_only=spec.json_only,
            force_tool_names=spec.force_tool_names,
        )

    def get_lark_for_json_schema(self, template: str, json_schema: dict[str, Any]) -> str:
        r"""Return a lark grammar that only accepts JSON matching the given schema.

        Convenience wrapper around `get_lark_from_jinja` that disables tool calls
        and text alternatives, constraining output to the JSON schema alone.

        Args:
            template: Jinja template to render, as obtained from `select_jinja_template`.
            json_schema: The JSON schema the output must conform to.

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
            force_tool_names=False,
        )
