import inspect
import re
import warnings
from pathlib import Path
from typing import Any, Literal

import llguidance as llg
import pytest
from pydantic import BaseModel, ConfigDict, Field, model_validator

import mistral_common.deprecation
from mistral_common.exceptions import InvalidRequestException
from mistral_common.guidance.grammar_factory import (
    JINJA_PATHS,
    GrammarFactory,
    _convert_tool_calls,
    _GrammarMode,
    _GrammarVariant,
)
from mistral_common.protocol.instruct.chunk import TextChunk, ThinkChunk
from mistral_common.protocol.instruct.messages import AssistantMessage, UserMessage
from mistral_common.protocol.instruct.normalize import get_normalizer
from mistral_common.protocol.instruct.request import (
    ChatCompletionRequest,
    JsonSchema,
    ReasoningEffort,
    ResponseFormat,
    ResponseFormats,
)
from mistral_common.protocol.instruct.tool_calls import (
    Function,
    FunctionCall,
    FunctionName,
    NamedToolChoice,
    Tool,
    ToolCall,
    ToolChoice,
    ToolChoiceEnum,
    ToolTypes,
)
from mistral_common.protocol.instruct.validator import ValidationMode, get_validator
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy, SpecialTokens, Tokenizer, TokenizerVersion
from mistral_common.tokens.tokenizers.instruct import (
    InstructTokenizerBase,
    InstructTokenizerV11,
    InstructTokenizerV13,
    InstructTokenizerV15,
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.model_settings_builder import EnumBuilder, ModelSettingsBuilder
from mistral_common.tokens.tokenizers.tekken import Tekkenizer, is_tekkenizer
from tests.test_tekken import get_special_tokens, quick_vocab

EMOJI_LARK_PATH = Path(__file__).parent.parent / "data" / "emoji.lark"

Mode = Literal["auto", "any", "none", "required"]


_NUM_SPECIAL_TOKENS = 100
_EXTRA_TOKENS = [
    b"de",
    b"he",
    b"llo",
    "😃".encode("utf-8"),
    "😂".encode("utf-8"),
    "😊".encode("utf-8"),
    "😍".encode("utf-8"),
    "😘".encode("utf-8"),
    "😗".encode("utf-8"),
    "😙".encode("utf-8"),
    "😚".encode("utf-8"),
    "😋".encode("utf-8"),
    "😛".encode("utf-8"),
    "😜".encode("utf-8"),
    "😝".encode("utf-8"),
    "🤑".encode("utf-8"),
    "🤗".encode("utf-8"),
    "🤔".encode("utf-8"),
    "🤐".encode("utf-8"),
    "😐".encode("utf-8"),
    "😑".encode("utf-8"),
    "😶".encode("utf-8"),
    "😬".encode("utf-8"),
    "こ".encode("utf-8"),
    "ん".encode("utf-8"),
    "に".encode("utf-8"),
    "ち".encode("utf-8"),
    "は".encode("utf-8"),
    "مرحبا".encode("utf-8"),
    "بكم".encode("utf-8"),
    "في".encode("utf-8"),
    "عالم".encode("utf-8"),
    "الذكاء".encode("utf-8"),
    "الاصطناعي".encode("utf-8"),
]


def _build_tekken_mistral_tokenizer(
    version: TokenizerVersion,
    add_think: bool = False,
    model_settings_builder: ModelSettingsBuilder | None = None,
) -> MistralTokenizer:
    r"""Builds a MistralTokenizer wrapping a programmatic Tekkenizer."""
    special_tokens = get_special_tokens(version, add_think=add_think)
    vocab = quick_vocab(_EXTRA_TOKENS)

    tekkenizer = Tekkenizer(
        vocab,
        special_tokens=special_tokens,
        pattern=r"(?s:.+)",
        vocab_size=len(vocab) + _NUM_SPECIAL_TOKENS,
        num_special_tokens=_NUM_SPECIAL_TOKENS,
        version=version,
        model_settings_builder=model_settings_builder,
    )

    match version:
        case TokenizerVersion.v11:
            instruct_tokenizer = InstructTokenizerV11(tekkenizer)
        case TokenizerVersion.v13:
            instruct_tokenizer = InstructTokenizerV13(tekkenizer)
        case TokenizerVersion.v15:
            instruct_tokenizer = InstructTokenizerV15(tekkenizer)
        case _:
            raise ValueError(f"Unsupported version for programmatic Tekken build: {version}")

    normalizer = get_normalizer(version, tekkenizer.model_settings_builder)
    validator = get_validator(version, mode=ValidationMode.test)
    return MistralTokenizer(instruct_tokenizer, validator=validator, request_normalizer=normalizer)


_V15_MODEL_SETTINGS_BUILDER = ModelSettingsBuilder(
    reasoning_effort=EnumBuilder[ReasoningEffort](
        values=list(ReasoningEffort),
        accepts_none=True,
        default=None,
    ),
)

# One tokenizer per supported version, shared by the generated cases, fixtures and request tests.
_ALL_TOKENIZERS: dict[TokenizerVersion, MistralTokenizer] = {
    TokenizerVersion.v11: _build_tekken_mistral_tokenizer(version=TokenizerVersion.v11),
    TokenizerVersion.v13: _build_tekken_mistral_tokenizer(version=TokenizerVersion.v13, add_think=True),
    TokenizerVersion.v15: _build_tekken_mistral_tokenizer(
        version=TokenizerVersion.v15, add_think=True, model_settings_builder=_V15_MODEL_SETTINGS_BUILDER
    ),
}
_GRAMMAR_FACTORIES: dict[TokenizerVersion, GrammarFactory] = {}


def _factory_for(version: TokenizerVersion) -> GrammarFactory:
    r"""Return the cached grammar factory of the shared tokenizer for `version`."""
    if version not in _GRAMMAR_FACTORIES:
        _GRAMMAR_FACTORIES[version] = GrammarFactory(tokenizer=_ALL_TOKENIZERS[version])
    return _GRAMMAR_FACTORIES[version]


_V13_NO_THINK_TOKENIZER = _build_tekken_mistral_tokenizer(version=TokenizerVersion.v13, add_think=False)


_PAYMENT_PARAMS: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"transaction_id": {"type": "string", "description": "The transaction id."}},
    "required": ["transaction_id"],
}


class ToolProvider:
    @staticmethod
    def make(name: str, strict: bool = False, parameters: dict[str, Any] | None = None, description: str = "") -> Tool:
        if parameters is None:
            parameters = {"type": "object", "properties": {}, "additionalProperties": False}
        return Tool(function=Function(name=name, description=description, parameters=parameters, strict=strict))

    @staticmethod
    def payment(name: str, strict: bool) -> Tool:
        r"""`retrieve_payment_<field>` tool taking a `transaction_id`."""
        field = name.removeprefix("retrieve_payment_")
        return ToolProvider.make(
            name=name,
            strict=strict,
            parameters=_PAYMENT_PARAMS,
            description=f"Get payment {field} of a transaction",
        )

    @staticmethod
    def payments(strict: bool) -> list[Tool]:
        r"""`retrieve_payment_date` and `retrieve_payment_status`, sharing one argument schema."""
        return [
            ToolProvider.payment(name="retrieve_payment_date", strict=strict),
            ToolProvider.payment(name="retrieve_payment_status", strict=strict),
        ]


class SchemaProvider:
    @staticmethod
    def basic_person() -> dict[str, Any]:
        class Person(BaseModel):
            model_config = ConfigDict(extra="forbid")
            name: str
            age: int

        return Person.model_json_schema()

    @staticmethod
    def basic_dict_of_list() -> dict[str, Any]:
        class DoMerge(BaseModel):
            model_config = ConfigDict(extra="forbid")
            new_clusters: dict[str, list[str]] = Field(default_factory=dict)

        return DoMerge.model_json_schema()


class TestCase(BaseModel):
    __test__ = False
    model_config = ConfigDict(arbitrary_types_allowed=True)
    tokenizer: Tokenizer
    mode: Mode | NamedToolChoice
    tokens: list[int]
    should_fail_on: int | None
    case_name: str
    parallel_tool_calls: bool = True
    tools: list[Tool] | None = None
    response_format: ResponseFormat = Field(default_factory=ResponseFormat)
    # When set, this grammar is used instead of the one rendered for the chat request.
    raw_lark: str | None = None

    @model_validator(mode="after")
    def validate_should_fail_on(self) -> "TestCase":
        if self.should_fail_on is not None and self.should_fail_on < 0:
            self.should_fail_on += len(self.tokens)
            if self.should_fail_on < 0:
                raise ValueError(
                    f"should_fail_on={self.should_fail_on + len(self.tokens)} "
                    f"is out of bounds for tokens of length {len(self.tokens)}"
                )
        return self

    @property
    def name(self) -> str:
        return f"tekken_{self.tokenizer.version.value}_{self.mode}_{self.case_name}"


def _encode_content(
    instruct_tokenizer: InstructTokenizerBase,
    content: str | list[Any],
) -> list[int]:
    tokenizer = instruct_tokenizer.tokenizer

    if isinstance(content, str):
        return instruct_tokenizer.encode_assistant_message(
            AssistantMessage(content=content), is_before_last_user_message=False
        )

    tool_calls = [x for x in content if isinstance(x, ToolCall)]
    content_chunks = [x for x in content if not isinstance(x, ToolCall)]

    tokens: list[int] = []
    if content_chunks:
        tokens = instruct_tokenizer.encode_assistant_message(
            AssistantMessage(content=content_chunks),
            is_before_last_user_message=False,
        )
    # The instruct tokenizer appends EOS after content, but when tool calls follow,
    # the EOS should come after the last tool call, not after the content. Strip it
    # here so the tool call tokens are appended directly after content tokens.
    while tokens and tokens[-1] == tokenizer.eos_id:
        tokens.pop()

    for tc in tool_calls:
        tokens += [
            tokenizer.get_special_token("[TOOL_CALLS]"),
            *tokenizer.encode(tc.function.name, bos=False, eos=False),
            tokenizer.get_special_token("[ARGS]"),
            *tokenizer.encode(tc.function.arguments, bos=False, eos=False),
        ]
    tokens.append(tokenizer.eos_id)
    return tokens


_HELLO_ARGS = '{"arg1": "val1", "arg2": "val2"}'
_TRANSACTION_ARGS = '{"transaction_id": "12345"}'


def _tool_call(name: str, arguments: str) -> ToolCall:
    return ToolCall(function=FunctionCall(name=name, arguments=arguments))


def _instruct(mistral_tokenizer: MistralTokenizer) -> InstructTokenizerBase:
    instruct_tokenizer = mistral_tokenizer.instruct_tokenizer
    assert isinstance(instruct_tokenizer, InstructTokenizerBase)
    return instruct_tokenizer


def _think_tokens(instruct_tokenizer: InstructTokenizerBase, text: str) -> list[int]:
    assert isinstance(instruct_tokenizer, InstructTokenizerV13)
    return instruct_tokenizer.encode_think(ThinkChunk(thinking=text))


def _chat_request(**kwargs: Any) -> ChatCompletionRequest:
    return ChatCompletionRequest(messages=[UserMessage(content="hello")], **kwargs)


def _strict_schema(schema: dict[str, Any]) -> ResponseFormat:
    return ResponseFormat(
        type=ResponseFormats.json_schema, json_schema=JsonSchema(name="test_schema", schema=schema, strict=True)
    )


_JSON_OBJECT = ResponseFormat(type=ResponseFormats.json_object)
_NON_STRICT_SCHEMA = ResponseFormat(
    type=ResponseFormats.json_schema,
    json_schema=JsonSchema(name="test_schema", schema=SchemaProvider.basic_person(), strict=False),
)


def _chat_grammar(
    factory: GrammarFactory,
    tool_choice: ToolChoice | Mode,
    tools: list[Tool] | None,
    response_format: ResponseFormat,
    parallel_tool_calls: bool,
) -> str:
    r"""Render the grammar of a chat request through the public request entry point.

    Token-acceptance tests all go through `get_grammar_for_chat_request`, so they also
    cover template selection, request-to-spec mapping and `get_lark_from_jinja`.

    `auto` without tools needs no grammar on the request path; a placeholder tool is
    added because with `auto` tool names are not forced, so the rendered grammar is
    identical to the tool-less one while still accepting calls to any tool name.
    """
    choice = tool_choice if isinstance(tool_choice, NamedToolChoice) else ToolChoiceEnum(tool_choice)
    if not tools and choice == ToolChoiceEnum.auto and response_format.type == ResponseFormats.text:
        tools = [ToolProvider.make(name="placeholder")]
    request = _chat_request(
        tools=tools, tool_choice=choice, response_format=response_format, parallel_tool_calls=parallel_tool_calls
    )
    grammar = factory.get_grammar_for_chat_request(request=request, template=factory.select_jinja_template())
    assert grammar is not None
    return grammar


def _grammar_accepts(
    tokenizer: MistralTokenizer, factory: GrammarFactory, grammar: str, content: str | list[Any]
) -> bool:
    r"""Check that the encoded assistant `content` is fully accepted by `grammar`."""
    tokens = _encode_content(instruct_tokenizer=_instruct(mistral_tokenizer=tokenizer), content=content)
    matcher = llg.LLMatcher(tokenizer=factory.llg_tokenizer, grammar=grammar)
    return all(matcher.consume_token(token) for token in tokens) and matcher.is_stopped()


def _find_first_rejection(
    factory: GrammarFactory,
    tokens: list[int],
    tool_choice: ToolChoice | Mode,
    tools: list[Tool] | None,
    response_format: ResponseFormat,
) -> int:
    r"""Finds the index of the first token rejected by the request grammar.

    Used where the rejection index depends on llguidance's byte-level parsing.

    Raises:
        ValueError: If all tokens are accepted.
    """
    grammar = _chat_grammar(
        factory=factory,
        tool_choice=tool_choice,
        tools=tools,
        response_format=response_format,
        parallel_tool_calls=True,
    )
    matcher = llg.LLMatcher(tokenizer=factory.llg_tokenizer, grammar=grammar)
    for i, token in enumerate(tokens):
        if not matcher.consume_token(token):
            return i
    raise ValueError("All tokens were accepted — expected a rejection")


def _token_debug_repr(tokenizer: Tekkenizer, token_id: int) -> str:
    return repr(tokenizer.id_to_byte_piece(token_id, SpecialTokenPolicy.KEEP))


def _count_prefix_tokens(tokenizer: Tokenizer, full_text: str, prefix: str) -> int:
    r"""Counts the number of tokens that cover the prefix bytes in the full-text tokenization.

    BPE tokenization is context-dependent, so encoding a prefix in isolation may produce
    different tokens than encoding the full string. This helper encodes the full string
    and counts how many tokens are needed to cover the byte-length of the prefix.
    """
    assert is_tekkenizer(tokenizer)
    prefix_byte_len = len(prefix.encode("utf-8"))
    tokens = tokenizer.encode(full_text, bos=False, eos=False)
    byte_count = 0
    for i, t in enumerate(tokens):
        byte_count += len(tokenizer.id_to_byte_piece(t, SpecialTokenPolicy.IGNORE))
        if byte_count >= prefix_byte_len:
            return i + 1
    return len(tokens)


def _mode_cases(
    tokenizer: Tokenizer,
    case_name: str,
    tokens: list[int],
    outcomes: dict[Mode, int | None],
    tools: list[Tool] | None = None,
    response_format: ResponseFormat | None = None,
    parallel_tool_calls: bool = True,
) -> list[TestCase]:
    r"""Expand one token sequence into a case per mode, mapped to its expected rejection index."""
    return [
        TestCase(
            tokenizer=tokenizer,
            tokens=tokens,
            should_fail_on=should_fail_on,
            case_name=case_name,
            mode=mode,
            tools=tools,
            response_format=response_format or ResponseFormat(),
            parallel_tool_calls=parallel_tool_calls,
        )
        for mode, should_fail_on in outcomes.items()
    ]


def _generate_text_cases(mistral_tokenizer: MistralTokenizer) -> list[TestCase]:
    instruct_tokenizer = _instruct(mistral_tokenizer=mistral_tokenizer)
    texts = {
        "newline": "\n",
        "blank": "_",
        "text": "Hello!",
        "text_with_newlines": "Hello!\n\nHow are you?\nI'm fine, thanks!",
        "emojis": "😃😂😊😍😘😗😙😚😋😛😜😝🤑🤗🤔🤐😐😑😶😬",
        "japanese": "こんにちは",
        "arabic": "مرحبا بكم في عالم الذكاء الاصطناعي",
    }
    cases: list[TestCase] = []
    for case_name, text in texts.items():
        tokens = _encode_content(instruct_tokenizer=instruct_tokenizer, content=text)
        cases += _mode_cases(
            tokenizer=instruct_tokenizer.tokenizer, case_name=case_name, tokens=tokens, outcomes={"auto": None}
        )
    return cases


def _generate_emoji_cases(mistral_tokenizer: MistralTokenizer) -> list[TestCase]:
    r"""Sanity-check llguidance with a raw emoji grammar, the only cases not rendered by `GrammarFactory`."""
    tokenizer = _instruct(mistral_tokenizer=mistral_tokenizer).tokenizer
    emoji_lark = EMOJI_LARK_PATH.read_text(encoding="utf-8")
    items: dict[str, tuple[str, int | None]] = {
        "emojis_valid_a": ("😃😂😊😍😘😗😙😚😋😛😜😝🤑🤗🤔🤐😐😑😶😬", None),
        "emojis_valid_b": ("😃😃😃", None),
        "emojis_invalid_text": ("😃smile", _count_prefix_tokens(tokenizer=tokenizer, full_text="😃smile", prefix="😃")),
        "emojis_invalid_space": ("😃 ", _count_prefix_tokens(tokenizer=tokenizer, full_text="😃 ", prefix="😃")),
    }
    return [
        TestCase(
            tokenizer=tokenizer,
            tokens=tokenizer.encode(text, bos=False, eos=False),
            case_name=case_name,
            should_fail_on=should_fail_on,
            mode="auto",
            raw_lark=emoji_lark,
        )
        for case_name, (text, should_fail_on) in items.items()
    ]


def _generate_tool_call_cases(mistral_tokenizer: MistralTokenizer) -> list[TestCase]:
    r"""Tool calls without tool definitions: any name is accepted by `auto`, rejected by `none`."""
    instruct_tokenizer = _instruct(mistral_tokenizer=mistral_tokenizer)
    tokenizer = instruct_tokenizer.tokenizer

    calls: dict[str, list[Any]] = {
        "single_fcall": [_tool_call(name="hello", arguments=_HELLO_ARGS)],
        "multi_fcall": [
            _tool_call(name="hello", arguments=_HELLO_ARGS),
            _tool_call(name="hello_1", arguments=_HELLO_ARGS),
            _tool_call(name="hello_2_3", arguments=_HELLO_ARGS),
        ],
        "emoji_fcall": [_tool_call(name="he🧦🧦o", arguments='{"arg1": "🐱", "arg2": "🐶", "arg🧦": "🧦"}')],
        "pretty_printed_args": [
            _tool_call(name="hello", arguments='{\n      "arg1": "val1",\n      "arg2": "val2"\n    }\n')
        ],
        "japanese_fcall": [_tool_call(name="こんにちは", arguments='{"こん": "にちは"}')],
    }
    cases: list[TestCase] = []
    for case_name, content in calls.items():
        cases += _mode_cases(
            tokenizer=tokenizer,
            case_name=case_name,
            tokens=_encode_content(instruct_tokenizer=instruct_tokenizer, content=content),
            outcomes={"auto": None, "none": 0},
        )

    text_len = len(tokenizer.encode("Hello!", bos=False, eos=False))
    text_fcall = _encode_content(
        instruct_tokenizer=instruct_tokenizer,
        content=[TextChunk(text="Hello!"), _tool_call(name="hello", arguments=_HELLO_ARGS)],
    )
    cases += _mode_cases(
        tokenizer=tokenizer, case_name="text_fcall", tokens=text_fcall, outcomes={"auto": None, "none": text_len}
    )

    call_prefix = [
        tokenizer.get_special_token("[TOOL_CALLS]"),
        *tokenizer.encode("hello", bos=False, eos=False),
        tokenizer.get_special_token("[ARGS]"),
    ]
    broken: dict[str, list[int]] = {
        "fcall_broken_args": [*call_prefix, *tokenizer.encode('{"a', bos=False, eos=False), tokenizer.eos_id],
        "fcall_missing_args": [*call_prefix, tokenizer.get_special_token("[TOOL_CALLS]")],
    }
    for case_name, tokens in broken.items():
        cases += _mode_cases(tokenizer=tokenizer, case_name=case_name, tokens=tokens, outcomes={"auto": -1, "none": 0})

    single = _encode_content(
        instruct_tokenizer=instruct_tokenizer, content=[_tool_call(name="hello", arguments=_HELLO_ARGS)]
    )
    cases += _mode_cases(
        tokenizer=tokenizer,
        case_name="single_tool_call",
        tokens=single,
        outcomes={"auto": None, "none": 0},
        parallel_tool_calls=False,
    )
    first_call = _encode_content(
        instruct_tokenizer=instruct_tokenizer, content=[_tool_call(name="fn1", arguments=_HELLO_ARGS)]
    )
    two_calls = _encode_content(
        instruct_tokenizer=instruct_tokenizer,
        content=[_tool_call(name="fn1", arguments=_HELLO_ARGS), _tool_call(name="fn2", arguments=_HELLO_ARGS)],
    )
    cases += _mode_cases(
        tokenizer=tokenizer,
        case_name="multi_tool_call_disallowed",
        tokens=two_calls,
        outcomes={"auto": len(first_call) - 1},
        parallel_tool_calls=False,
    )
    return cases


def _generate_strict_tool_call_cases(mistral_tokenizer: MistralTokenizer, factory: GrammarFactory) -> list[TestCase]:
    instruct_tokenizer = _instruct(mistral_tokenizer=mistral_tokenizer)
    strict_date = [ToolProvider.payment(name="retrieve_payment_date", strict=True)]
    strict_both = ToolProvider.payments(strict=True)

    def encode(*calls: ToolCall) -> list[int]:
        return _encode_content(instruct_tokenizer=instruct_tokenizer, content=list(calls))

    def first_rejection(tokens: list[int]) -> int:
        return _find_first_rejection(
            factory=factory,
            tokens=tokens,
            tool_choice=ToolChoiceEnum.auto,
            tools=strict_date,
            response_format=ResponseFormat(),
        )

    wrong_args = encode(_tool_call(name="retrieve_payment_date", arguments='{"bogus": "12345"}'))
    wrong_name = encode(_tool_call(name="fn1", arguments=_TRANSACTION_ARGS))
    rows: list[tuple[str, list[int], int | None, list[Tool]]] = [
        (
            "single_non_strict_tool_call",
            encode(_tool_call(name="fn1", arguments=_HELLO_ARGS)),
            None,
            [ToolProvider.payment(name="retrieve_payment_date", strict=False)],
        ),
        (
            "single_strict_tool_call",
            encode(_tool_call(name="retrieve_payment_date", arguments=_TRANSACTION_ARGS)),
            None,
            strict_date,
        ),
        ("strict_tool_call_wrong_args", wrong_args, first_rejection(tokens=wrong_args), strict_date),
        ("strict_tool_call_wrong_name", wrong_name, first_rejection(tokens=wrong_name), strict_date),
        (
            "multiple_strict_tool_calls",
            encode(
                _tool_call(name="retrieve_payment_date", arguments=_TRANSACTION_ARGS),
                _tool_call(name="retrieve_payment_status", arguments=_TRANSACTION_ARGS),
            ),
            None,
            strict_both,
        ),
    ]
    cases: list[TestCase] = []
    for case_name, tokens, should_fail_on, tools in rows:
        outcomes: dict[Mode, int | None] = dict.fromkeys(("auto", "required"), should_fail_on)
        cases += _mode_cases(
            tokenizer=instruct_tokenizer.tokenizer, case_name=case_name, tokens=tokens, outcomes=outcomes, tools=tools
        )

    # A forced call renders the `required` branch: the base template rejects text before it,
    # think templates (`fcalls: content? <fcall>`) accept it.
    text_then_call = _encode_content(
        instruct_tokenizer=instruct_tokenizer,
        content=[TextChunk(text="Hello!"), _tool_call(name="retrieve_payment_date", arguments=_TRANSACTION_ARGS)],
    )
    forced_outcome = None if instruct_tokenizer.tokenizer.version >= TokenizerVersion.v13 else 0
    cases += _mode_cases(
        tokenizer=instruct_tokenizer.tokenizer,
        case_name="text_then_strict_tool_call",
        tokens=text_then_call,
        outcomes={"auto": None, "required": forced_outcome},
        tools=strict_date,
    )
    return cases


def _generate_named_tool_choice_cases(mistral_tokenizer: MistralTokenizer, factory: GrammarFactory) -> list[TestCase]:
    instruct_tokenizer = _instruct(mistral_tokenizer=mistral_tokenizer)
    named = NamedToolChoice(type=ToolTypes.function, function=FunctionName(name="retrieve_payment_date"))
    non_strict = ToolProvider.payments(strict=False)
    strict = [ToolProvider.payment(name="retrieve_payment_date", strict=True)]

    def encode(call: ToolCall) -> list[int]:
        return _encode_content(instruct_tokenizer=instruct_tokenizer, content=[call])

    def first_rejection(tokens: list[int], tools: list[Tool]) -> int:
        return _find_first_rejection(
            factory=factory, tokens=tokens, tool_choice=named, tools=tools, response_format=ResponseFormat()
        )

    correct = encode(_tool_call(name="retrieve_payment_date", arguments=_TRANSACTION_ARGS))
    arbitrary_args = encode(
        _tool_call(name="retrieve_payment_date", arguments='{"completely": "arbitrary", "keys": [1, 2, 3]}')
    )
    wrong_tool = encode(_tool_call(name="retrieve_payment_status", arguments=_TRANSACTION_ARGS))
    wrong_args = encode(_tool_call(name="retrieve_payment_date", arguments='{"wrong_arg": "12345"}'))
    wrong_tool_index = first_rejection(tokens=wrong_tool, tools=non_strict)
    wrong_args_index = first_rejection(tokens=wrong_args, tools=strict)
    rows: list[tuple[str, list[int], int | None, list[Tool], bool]] = [
        ("named_tool_choice_correct", correct, None, non_strict, True),
        ("named_tool_choice_correct_no_parallel", correct, None, non_strict, False),
        ("named_tool_choice_non_strict_arbitrary_args", arbitrary_args, None, non_strict, True),
        ("named_tool_choice_wrong_tool", wrong_tool, wrong_tool_index, non_strict, True),
        ("named_tool_choice_strict_correct", correct, None, strict, True),
        ("named_tool_choice_strict_wrong_args", wrong_args, wrong_args_index, strict, True),
    ]
    return [
        TestCase(
            tokenizer=instruct_tokenizer.tokenizer,
            tokens=tokens,
            should_fail_on=should_fail_on,
            case_name=case_name,
            mode=named,
            tools=tools,
            parallel_tool_calls=parallel_tool_calls,
        )
        for case_name, tokens, should_fail_on, tools, parallel_tool_calls in rows
    ]


def _generate_thinking_cases(mistral_tokenizer: MistralTokenizer) -> list[TestCase]:
    r"""Thinking cases; `any`/`required` need tools, so only `auto`/`none` are covered."""
    instruct_tokenizer = _instruct(mistral_tokenizer=mistral_tokenizer)
    tokenizer = instruct_tokenizer.tokenizer
    call = _tool_call(name="hello", arguments=_HELLO_ARGS)

    def length(text: str) -> int:
        return len(tokenizer.encode(text, bos=False, eos=False))

    rows: list[tuple[str, list[Any], dict[Mode, int | None]]]
    if tokenizer.version >= TokenizerVersion.v13:
        # Think template: [THINK]...[/THINK] must be followed by an answer.
        think_len = len(_think_tokens(instruct_tokenizer=instruct_tokenizer, text="Hello!"))
        rows = [
            ("plain_text", [TextChunk(text="Hello world!")], {"auto": None, "none": None}),
            ("plain_think", [ThinkChunk(thinking="Hello!")], {"auto": -1, "none": -1}),
            (
                "plain_think_with_response",
                [ThinkChunk(thinking="Hello!"), TextChunk(text="World!")],
                {"auto": None, "none": None},
            ),
            ("think_with_tool_call", [ThinkChunk(thinking="Hello!"), call], {"auto": None, "none": think_len}),
            (
                "think_text_tool_call",
                [ThinkChunk(thinking="Hello!"), TextChunk(text="World!"), call],
                {"auto": None, "none": think_len + length("World!")},
            ),
        ]
    else:
        # Base template: a literal <think>...</think> is plain text content.
        rows = [
            ("force_think", [TextChunk(text="Hello world!")], {"auto": None, "none": None}),
            ("think_without_response", [TextChunk(text="<think>Hello!</think>")], {"auto": None, "none": None}),
            ("unclosed_think", [TextChunk(text="<think>Hello!")], {"auto": None, "none": None}),
            (
                "plain_think_with_response",
                [TextChunk(text="<think>Hello!</think>World!")],
                {"auto": None, "none": None},
            ),
            (
                "think_with_tool_call",
                [TextChunk(text="<think>Hello!</think>"), call],
                {"auto": None, "none": length("<think>Hello!</think>")},
            ),
            (
                "think_with_text_and_tool_call",
                [TextChunk(text="<think>Hello!</think>Ho!"), call],
                {"auto": None, "none": length("<think>Hello!</think>Ho!")},
            ),
        ]
    cases: list[TestCase] = []
    for case_name, content, outcomes in rows:
        cases += _mode_cases(
            tokenizer=tokenizer,
            case_name=case_name,
            tokens=_encode_content(instruct_tokenizer=instruct_tokenizer, content=content),
            outcomes=outcomes,
        )
    return cases


def _generate_json_schema_cases(mistral_tokenizer: MistralTokenizer, factory: GrammarFactory) -> list[TestCase]:
    instruct_tokenizer = _instruct(mistral_tokenizer=mistral_tokenizer)
    tokenizer = instruct_tokenizer.tokenizer
    person = SchemaProvider.basic_person()
    any_object: dict[str, Any] = {"type": "object"}
    valid_person = '{"name": "John", "age": 30}'

    def encode(content: str | list[Any]) -> list[int]:
        return _encode_content(instruct_tokenizer=instruct_tokenizer, content=content)

    swapped_person = encode('{"age": "John", "name": 30}')
    rows: list[tuple[str, list[int], int | None, dict[str, Any]]] = [
        ("basic_person_valid", encode(valid_person), None, person),
        (
            "basic_person_invalid",
            swapped_person,
            _find_first_rejection(
                factory=factory,
                tokens=swapped_person,
                tool_choice="auto",
                tools=None,
                response_format=_strict_schema(schema=person),
            ),
            person,
        ),
        ("basic_person_non_strict_valid", swapped_person, None, any_object),
        (
            "domerge_valid",
            encode('{"new_clusters": {"b": ["a", "b", "c"], "d": ["e"]} }'),
            None,
            SchemaProvider.basic_dict_of_list(),
        ),
        ("valid_json_white_spaces", encode("\n " + valid_person), None, any_object),
        ("invalid_json_missing_curly_bracket", encode('"name": "John", "age": 30}'), 0, any_object),
        ("invalid_json_backslash_f", encode("\f" + valid_person), 0, any_object),
        ("json_only_rejects_text", encode("Hello world!"), 0, any_object),
        (
            "json_only_rejects_tool_call",
            encode([_tool_call(name="hello", arguments='{"arg1": "val1"}')]),
            0,
            any_object,
        ),
    ]
    if tokenizer.version.supports_model_settings:
        # v15+ allows thinking before the JSON answer in json_only mode.
        think_prefix = _think_tokens(instruct_tokenizer=instruct_tokenizer, text="Let me think...")
        rows += [
            (
                "think_with_json_valid",
                [
                    *_think_tokens(instruct_tokenizer=instruct_tokenizer, text="Let me think about this..."),
                    *tokenizer.encode(valid_person, bos=False, eos=False),
                    tokenizer.eos_id,
                ],
                None,
                person,
            ),
            (
                "think_with_json_rejects_text_after_think",
                [*think_prefix, *tokenizer.encode("Hello world!", bos=False, eos=False), tokenizer.eos_id],
                len(think_prefix),
                person,
            ),
        ]
    cases: list[TestCase] = []
    for case_name, tokens, should_fail_on, json_schema in rows:
        cases += _mode_cases(
            tokenizer=tokenizer,
            case_name=case_name,
            tokens=tokens,
            outcomes={"auto": should_fail_on},
            response_format=_strict_schema(schema=json_schema),
        )

    # `json_object` and non-strict `json_schema` constrain to any JSON object or array.
    for format_name, response_format in (("json_object", _JSON_OBJECT), ("non_strict_schema", _NON_STRICT_SCHEMA)):
        loose_rows: dict[str, tuple[str | list[Any], int | None]] = {
            "object": ('{"b": 1}', None),
            "array": ("[1]", None),
            "rejects_text": ("Hello world!", 0),
        }
        for row_name, (content, should_fail_on) in loose_rows.items():
            cases += _mode_cases(
                tokenizer=tokenizer,
                case_name=f"{format_name}_{row_name}",
                tokens=encode(content),
                outcomes={"auto": should_fail_on},
                response_format=response_format,
            )

    # Tools combined with a JSON schema: `none` drops the tools and only allows JSON;
    # `auto` allows JSON or tool calls but not plain text, which would defeat the schema,
    # and keeps the `parallel_tool_calls` setting.
    tools = [ToolProvider.make(name="get_weather")]
    call = _tool_call(name="get_weather", arguments="{}")
    single_call = encode([call])

    def auto_rejection(tokens: list[int]) -> int:
        return _find_first_rejection(
            factory=factory,
            tokens=tokens,
            tool_choice="auto",
            tools=tools,
            response_format=_strict_schema(schema=person),
        )

    text = encode("Hello world!")
    invalid_person = swapped_person
    with_tools: dict[str, tuple[list[int], dict[Mode, int | None], bool]] = {
        "json_schema_with_tools": (single_call, {"auto": None, "none": 0}, True),
        "json_schema_with_tools_json": (encode(valid_person), {"auto": None, "none": None}, True),
        "json_schema_with_tools_text": (text, {"auto": auto_rejection(tokens=text), "none": 0}, True),
        "json_schema_with_tools_invalid_json": (
            invalid_person,
            {"auto": auto_rejection(tokens=invalid_person)},
            True,
        ),
        "json_schema_with_tools_parallel_disallowed": (
            encode([call, call]),
            {"auto": len(single_call) - 1},
            False,
        ),
    }
    for case_name, (tokens, outcomes, parallel_tool_calls) in with_tools.items():
        cases += _mode_cases(
            tokenizer=tokenizer,
            case_name=case_name,
            tokens=tokens,
            outcomes=outcomes,
            tools=tools,
            response_format=_strict_schema(schema=person),
            parallel_tool_calls=parallel_tool_calls,
        )
    return cases


def _generate_cases(mistral_tokenizer: MistralTokenizer, factory: GrammarFactory) -> list[TestCase]:
    return [
        *_generate_text_cases(mistral_tokenizer=mistral_tokenizer),
        *_generate_emoji_cases(mistral_tokenizer=mistral_tokenizer),
        *_generate_tool_call_cases(mistral_tokenizer=mistral_tokenizer),
        *_generate_strict_tool_call_cases(mistral_tokenizer=mistral_tokenizer, factory=factory),
        *_generate_named_tool_choice_cases(mistral_tokenizer=mistral_tokenizer, factory=factory),
        *_generate_thinking_cases(mistral_tokenizer=mistral_tokenizer),
        *_generate_json_schema_cases(mistral_tokenizer=mistral_tokenizer, factory=factory),
    ]


_ALL_CASES: list[TestCase] = [
    case
    for version, mistral_tokenizer in _ALL_TOKENIZERS.items()
    for case in _generate_cases(mistral_tokenizer=mistral_tokenizer, factory=_factory_for(version=version))
]


class TestGrammarFactory:
    @pytest.mark.parametrize("test_case", _ALL_CASES, ids=lambda tc: tc.name)
    def test_grammar(self, test_case: TestCase) -> None:
        factory = _factory_for(version=test_case.tokenizer.version)

        if test_case.raw_lark is not None:
            grammar = test_case.raw_lark
        else:
            grammar = _chat_grammar(
                factory=factory,
                tool_choice=test_case.mode,
                tools=test_case.tools,
                response_format=test_case.response_format,
                parallel_tool_calls=test_case.parallel_tool_calls,
            )

        matcher = llg.LLMatcher(tokenizer=factory.llg_tokenizer, grammar=grammar)

        assert is_tekkenizer(test_case.tokenizer)
        debug_tokens = [_token_debug_repr(tokenizer=test_case.tokenizer, token_id=t) for t in test_case.tokens]
        for i, token in enumerate(test_case.tokens):
            debug_bytes = _token_debug_repr(tokenizer=test_case.tokenizer, token_id=token)
            if token != test_case.tokenizer.eos_id:
                assert not matcher.is_stopped(), (
                    f"Matcher stopped before token {i} id={token}\n\n"
                    f"Grammar:\n{grammar}\n\nTokens: {debug_tokens}\n\nIds: {test_case.tokens}"
                )
            accepted = matcher.consume_token(token)
            if i == test_case.should_fail_on:
                if accepted:
                    raise AssertionError(
                        f"Token {token}={debug_bytes} at pos {i} was accepted but should have been rejected."
                        f"\n\nGrammar:\n{grammar}\n\nTokens: {debug_tokens}\n\nIds: {test_case.tokens}"
                    )
                break
            elif not accepted:
                raise AssertionError(
                    f"Token {token}={debug_bytes} at pos {i} was rejected but should have been accepted."
                    f"\n\nGrammar:\n{grammar}\n\nTokens: {debug_tokens}\n\nIds: {test_case.tokens}"
                )

        # For fully accepted sequences, verify the matcher reached a valid terminal state.
        # Raw lark grammars (e.g., emoji matcher) don't consume EOS, so they may not reach
        # a stopped state — skip the check for those.
        if test_case.should_fail_on is None and test_case.raw_lark is None:
            assert matcher.is_stopped(), (
                f"Matcher did not reach terminal state after consuming all tokens.\n\n"
                f"Grammar:\n{grammar}\n\nTokens: {debug_tokens}\n\nIds: {test_case.tokens}"
            )

    @pytest.mark.parametrize(
        ("tokenizer", "expected"),
        [
            (MistralTokenizer.v1(), False),
            (MistralTokenizer.v3(is_tekken=True), False),
            *((tokenizer, True) for tokenizer in _ALL_TOKENIZERS.values()),
        ],
    )
    def test_grammar_factory_is_supported(self, tokenizer: MistralTokenizer, expected: bool) -> None:
        assert GrammarFactory.is_supported(tokenizer) is expected

    @pytest.mark.parametrize(
        "tokenizer",
        [MistralTokenizer.v1(), MistralTokenizer.v3(is_tekken=True)],
    )
    def test_grammar_factory_init_rejects_unsupported(self, tokenizer: MistralTokenizer) -> None:
        with pytest.raises(ValueError, match="Guidance requires a Tekken tokenizer with version >= v11"):
            GrammarFactory(tokenizer=tokenizer)

    @pytest.mark.parametrize(
        ("mode", "tools", "error_message"),
        [
            (ToolChoiceEnum.any, None, "please ensure to pass tools"),
            (ToolChoiceEnum.any, [], "please ensure to pass tools"),
            (ToolChoiceEnum.required, None, "please ensure to pass tools"),
            (ToolChoiceEnum.required, [], "please ensure to pass tools"),
            (
                NamedToolChoice(function=FunctionName(name="retrieve_payment_date")),
                [ToolProvider.payment(name="retrieve_payment_date", strict=True)],
                "only supported by `get_grammar_for_chat_request`",
            ),
        ],
    )
    def test_get_lark_from_jinja_rejects_invalid_mode_and_tools(
        self, mode: ToolChoice, tools: list[Tool] | None, error_message: str
    ) -> None:
        factory = _factory_for(version=TokenizerVersion.v11)
        with pytest.raises(ValueError, match=error_message):
            factory.get_lark_from_jinja(
                template=factory.select_jinja_template(),
                mode=mode,  # type: ignore[arg-type]  # untyped callers may still pass a NamedToolChoice
                tools=tools,
                json_schema=None,
                parallel_tool_calls=True,
                json_only=False,
                force_tool_names=False,
            )


class TestSelectJinjaTemplate:
    @pytest.mark.parametrize(
        ("tokenizer", "expected_variant"),
        [
            pytest.param(_ALL_TOKENIZERS[TokenizerVersion.v11], _GrammarVariant.base, id="v11"),
            pytest.param(_V13_NO_THINK_TOKENIZER, _GrammarVariant.base, id="v13_no_think"),
            pytest.param(_ALL_TOKENIZERS[TokenizerVersion.v13], _GrammarVariant.think, id="v13_think"),
            pytest.param(_ALL_TOKENIZERS[TokenizerVersion.v15], _GrammarVariant.think, id="v15_think"),
        ],
    )
    def test_select_jinja_template(self, tokenizer: MistralTokenizer, expected_variant: _GrammarVariant) -> None:
        factory = GrammarFactory(tokenizer=tokenizer)

        assert factory.select_jinja_template() == JINJA_PATHS[expected_variant].read_text(encoding="utf-8")

    @pytest.mark.parametrize("think_token", [SpecialTokens.begin_think, SpecialTokens.end_think])
    def test_select_jinja_template_raises_on_single_think_token(self, think_token: SpecialTokens) -> None:
        factory = GrammarFactory(tokenizer=_V13_NO_THINK_TOKENIZER)
        factory._special_token_map[think_token.value] = "<[99]>"
        with pytest.raises(AssertionError, match=r"both \[THINK\] and \[/THINK\] should be defined or none of them\."):
            factory.select_jinja_template()

    @pytest.mark.parametrize("reasoning", [True, False])
    def test_select_jinja_template_reasoning_warns_and_is_ignored(self, reasoning: bool) -> None:
        factory = _factory_for(version=TokenizerVersion.v11)
        expected = factory.select_jinja_template()
        mistral_common.deprecation._warned_keys.discard("select_jinja_template.reasoning")

        with pytest.warns(DeprecationWarning):
            assert factory.select_jinja_template(reasoning=reasoning) == expected

    def test_select_jinja_template_reasoning_none_no_warn(self) -> None:
        factory = _factory_for(version=TokenizerVersion.v11)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            factory.select_jinja_template()

    @pytest.mark.parametrize(
        ("tokenizer", "content"),
        [
            pytest.param(_V13_NO_THINK_TOKENIZER, "hello world", id="v13_no_think_plain_content"),
            pytest.param(
                _ALL_TOKENIZERS[TokenizerVersion.v11], "<think>reasoning</think>answer", id="v11_literal_think"
            ),
        ],
    )
    def test_base_grammar_accepts_content(self, tokenizer: MistralTokenizer, content: str) -> None:
        r"""Base-selected grammars accept plain content, including a literal `<think>` tag."""
        factory = GrammarFactory(tokenizer=tokenizer)
        grammar = _chat_grammar(
            factory=factory,
            tool_choice=ToolChoiceEnum.auto,
            tools=None,
            response_format=ResponseFormat(),
            parallel_tool_calls=True,
        )

        assert _grammar_accepts(tokenizer=tokenizer, factory=factory, grammar=grammar, content=content)


def _stub_get_special_token_id(token_name: str) -> str:
    r"""Returns a stub lark grammar token for testing."""
    return f"<[{token_name}]>"


_STRICT_PAYMENT_SCHEMA = (
    '{"type": "object", "additionalProperties": false, "properties": {"transaction_id": '
    '{"type": "string", "description": "The transaction id."}}, "required": ["transaction_id"]}'
)


def _expected_fcall(name: str, schema: str = '{"type": "object"}') -> str:
    return f"<[[TOOL_CALLS]]> SAFE_WS? {name} <[[ARGS]]> SAFE_WS? %json {schema} SAFE_WS?"


@pytest.mark.parametrize(
    ("tools", "mode", "parallel_tool_calls", "force_tool_names", "expected"),
    [
        pytest.param(None, _GrammarMode.none, False, False, "", id="none"),
        pytest.param(
            [ToolProvider.payment(name="retrieve_payment_date", strict=True)],
            _GrammarMode.none,
            True,
            False,
            "",
            id="none_with_tools",
        ),
        pytest.param(None, _GrammarMode.auto, False, False, _expected_fcall(name="/.+/"), id="auto_no_tools"),
        pytest.param(
            None, _GrammarMode.auto, True, False, "(" + _expected_fcall(name="/.+/") + ")+", id="auto_parallel"
        ),
        pytest.param(
            [ToolProvider.payment(name="retrieve_payment_date", strict=False)],
            _GrammarMode.auto,
            False,
            False,
            _expected_fcall(name="/.+/"),
            id="auto_non_strict",
        ),
        pytest.param(
            [ToolProvider.payment(name="retrieve_payment_date", strict=True)],
            _GrammarMode.auto,
            False,
            False,
            _expected_fcall(name='"retrieve_payment_date"', schema=_STRICT_PAYMENT_SCHEMA),
            id="auto_single_strict",
        ),
        pytest.param(
            ToolProvider.payments(strict=True),
            _GrammarMode.auto,
            False,
            False,
            " | ".join(
                "(" + _expected_fcall(name=f'"{name}"', schema=_STRICT_PAYMENT_SCHEMA) + ")"
                for name in ("retrieve_payment_date", "retrieve_payment_status")
            ),
            id="auto_strict",
        ),
        pytest.param(
            [Tool(function=Function(name="empty_fn", parameters={}, strict=True))],
            _GrammarMode.auto,
            False,
            False,
            _expected_fcall(
                name='"empty_fn"', schema='{"type": "object", "properties": {}, "additionalProperties": false}'
            ),
            id="auto_strict_empty_params",
        ),
        pytest.param(
            [ToolProvider.make(name="get_weather"), ToolProvider.make(name="get_time")],
            _GrammarMode.required,
            False,
            False,
            _expected_fcall(name="/.+/"),
            id="required_unforced",
        ),
        pytest.param(
            [ToolProvider.make(name="get_weather"), ToolProvider.make(name="get_time")],
            _GrammarMode.required,
            False,
            True,
            _expected_fcall(name='("get_weather" | "get_time")'),
            id="required_forced",
        ),
        pytest.param(
            [ToolProvider.make(name="get_weather")],
            _GrammarMode.required,
            False,
            True,
            _expected_fcall(name='"get_weather"'),
            id="required_forced_single",
        ),
    ],
)
def test_convert_tool_calls(
    tools: list[Tool] | None, mode: _GrammarMode, parallel_tool_calls: bool, force_tool_names: bool, expected: str
) -> None:
    result = _convert_tool_calls(
        tools=tools,
        mode=mode,
        parallel_tool_calls=parallel_tool_calls,
        get_special_token_id=_stub_get_special_token_id,
        force_tool_names=force_tool_names,
    )
    assert result == expected


def test_convert_tool_calls_force_tool_names_requires_tools() -> None:
    with pytest.raises(AssertionError):
        _convert_tool_calls(
            tools=[],
            mode=_GrammarMode.required,
            parallel_tool_calls=False,
            get_special_token_id=_stub_get_special_token_id,
            force_tool_names=True,
        )


@pytest.mark.parametrize(
    ("kwargs", "expects_lark"),
    [
        ({"tool_choice": ToolChoiceEnum.auto}, False),
        ({"tool_choice": ToolChoiceEnum.none}, True),
        ({"tools": [ToolProvider.make(name="get_weather")], "tool_choice": ToolChoiceEnum.auto}, True),
        ({"tools": [ToolProvider.make(name="get_weather")], "tool_choice": ToolChoiceEnum.none}, True),
        ({"tools": [ToolProvider.make(name="get_weather")], "tool_choice": ToolChoiceEnum.required}, True),
        ({"tools": [ToolProvider.make(name="get_weather")], "tool_choice": ToolChoiceEnum.any}, True),
        (
            {
                "tools": [ToolProvider.make(name="get_weather")],
                "tool_choice": NamedToolChoice(function=FunctionName(name="get_weather")),
            },
            True,
        ),
        ({"response_format": ResponseFormat(type=ResponseFormats.json_object)}, True),
        (
            {
                "response_format": ResponseFormat(
                    type=ResponseFormats.json_schema,
                    json_schema=JsonSchema(name="result", schema={"type": "object"}, strict=True),
                )
            },
            True,
        ),
        (
            {
                "response_format": ResponseFormat(
                    type=ResponseFormats.json_schema,
                    json_schema=JsonSchema(name="result", schema={"type": "object"}, strict=False),
                )
            },
            True,
        ),
        (
            {
                "response_format": ResponseFormat(type=ResponseFormats.json_object),
                "tools": [ToolProvider.make(name="get_weather")],
                "tool_choice": ToolChoiceEnum.auto,
            },
            True,
        ),
        (
            {
                "response_format": ResponseFormat(type=ResponseFormats.json_object),
                "tools": [ToolProvider.make(name="get_weather")],
                "tool_choice": ToolChoiceEnum.none,
            },
            True,
        ),
        (
            {
                "response_format": ResponseFormat(
                    type=ResponseFormats.json_schema,
                    json_schema=JsonSchema(name="result", schema={"type": "object"}),
                ),
                "tools": [ToolProvider.make(name="get_weather")],
                "tool_choice": ToolChoiceEnum.auto,
            },
            True,
        ),
        (
            {
                "response_format": ResponseFormat(
                    type=ResponseFormats.json_schema,
                    json_schema=JsonSchema(name="result", schema={"type": "object"}, strict=True),
                ),
                "tools": [ToolProvider.make(name="get_weather")],
                "tool_choice": ToolChoiceEnum.none,
            },
            True,
        ),
    ],
)
def test_get_grammar_for_chat_request_returns_lark(kwargs: dict[str, Any], expects_lark: bool) -> None:
    factory = _factory_for(version=TokenizerVersion.v11)

    lark = factory.get_grammar_for_chat_request(
        request=_chat_request(**kwargs), template=factory.select_jinja_template()
    )

    assert (lark is not None) == expects_lark
    if lark is not None:
        llg.LLMatcher(tokenizer=factory.llg_tokenizer, grammar=lark)


@pytest.mark.parametrize(
    ("kwargs", "error_message"),
    [
        ({"tool_choice": ToolChoiceEnum.any}, "No tools provided for tool choice `any`"),
        ({"tool_choice": ToolChoiceEnum.required}, "No tools provided for tool choice `required`"),
        (
            {"tool_choice": NamedToolChoice(function=FunctionName(name="get_weather"))},
            "No tools provided for tool choice `get_weather`",
        ),
        (
            {
                "tools": [ToolProvider.make(name="get_weather"), ToolProvider.make(name="get_time")],
                "tool_choice": NamedToolChoice(function=FunctionName(name="missing")),
            },
            "Tool choice `missing` not found in available tools: ['get_weather', 'get_time'].",
        ),
        (
            {
                "response_format": ResponseFormat(type=ResponseFormats.json_object),
                "tools": [ToolProvider.make(name="get_weather")],
                "tool_choice": ToolChoiceEnum.any,
            },
            "`json_object` response type with tools is only compatible with `tool_choice: auto`",
        ),
        (
            {
                "response_format": ResponseFormat(type=ResponseFormats.json_object),
                "tools": [ToolProvider.make(name="get_weather")],
                "tool_choice": ToolChoiceEnum.required,
            },
            "`json_object` response type with tools is only compatible with `tool_choice: auto`",
        ),
        (
            {
                "response_format": ResponseFormat(
                    type=ResponseFormats.json_schema,
                    json_schema=JsonSchema(name="result", schema={"type": "object"}),
                ),
                "tools": [ToolProvider.make(name="get_weather")],
                "tool_choice": NamedToolChoice(function=FunctionName(name="get_weather")),
            },
            "`json_schema` response type with tools is only compatible with `tool_choice: auto`",
        ),
    ],
)
def test_get_grammar_for_chat_request_raises(kwargs: dict[str, Any], error_message: str) -> None:
    factory = _factory_for(version=TokenizerVersion.v11)

    with pytest.raises(InvalidRequestException, match=re.escape(error_message)):
        factory.get_grammar_for_chat_request(request=_chat_request(**kwargs), template=factory.select_jinja_template())


def test_get_grammar_for_chat_request_any_equals_required() -> None:
    factory = _factory_for(version=TokenizerVersion.v11)
    template = factory.select_jinja_template()
    tools = [ToolProvider.make(name="get_weather")]

    any_lark, required_lark = (
        factory.get_grammar_for_chat_request(request=_chat_request(tools=tools, tool_choice=choice), template=template)
        for choice in (ToolChoiceEnum.any, ToolChoiceEnum.required)
    )

    assert any_lark == required_lark


@pytest.mark.parametrize("version", [TokenizerVersion.v11, TokenizerVersion.v13])
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("parallel_tool_calls", [True, False])
def test_named_tool_choice_grammar(version: TokenizerVersion, strict: bool, parallel_tool_calls: bool) -> None:
    r"""Named tool choice behaviour not covered by the generated `named_tool_choice_*` cases."""
    factory = _factory_for(version=version)
    grammar = _chat_grammar(
        factory=factory,
        tool_choice=NamedToolChoice(function=FunctionName(name="get_weather")),
        tools=[ToolProvider.make(name="get_weather", strict=strict), ToolProvider.make(name="get_time", strict=strict)],
        response_format=ResponseFormat(),
        parallel_tool_calls=parallel_tool_calls,
    )
    named_call = _tool_call(name="get_weather", arguments="{}")

    def accepts(content: str | list[Any]) -> bool:
        return _grammar_accepts(tokenizer=_ALL_TOKENIZERS[version], factory=factory, grammar=grammar, content=content)

    assert not accepts(content="plain text")
    assert accepts(content=[named_call, named_call]) == parallel_tool_calls
    # A forced call renders the `required` branch: the base template allows no text before it, the
    # think template keeps `fcalls: content? <fcall>`.
    assert accepts(content=[TextChunk(text="before"), named_call]) == (version >= TokenizerVersion.v13)
    if version >= TokenizerVersion.v13:
        assert accepts(content=[ThinkChunk(thinking="reasoning"), TextChunk(text="before"), named_call])


@pytest.mark.parametrize("version", list(_ALL_TOKENIZERS))
def test_get_lark_from_jinja_json_schema_with_tools_forces_required(version: TokenizerVersion) -> None:
    r"""Direct callers get the `required` union for a JSON schema with tools, whatever the non-`none` mode."""
    factory = _factory_for(version=version)

    def render(mode: ToolChoiceEnum) -> str:
        return factory.get_lark_from_jinja(
            template=factory.select_jinja_template(),
            mode=mode,
            tools=[ToolProvider.make(name="get_weather")],
            json_schema=SchemaProvider.basic_person(),
            parallel_tool_calls=True,
            json_only=False,
            force_tool_names=False,
        )

    assert render(mode=ToolChoiceEnum.auto) == render(mode=ToolChoiceEnum.required) == render(mode=ToolChoiceEnum.any)


@pytest.mark.parametrize("version", list(_ALL_TOKENIZERS))
def test_get_lark_from_jinja_empty_json_schema_is_no_schema(version: TokenizerVersion) -> None:
    r"""An empty schema renders no JSON branch, so it must not force the `required` branch either."""
    factory = _factory_for(version=version)

    def render(json_schema: dict[str, Any] | None) -> str:
        return factory.get_lark_from_jinja(
            template=factory.select_jinja_template(),
            mode=ToolChoiceEnum.auto,
            tools=[ToolProvider.make(name="get_weather")],
            json_schema=json_schema,
            parallel_tool_calls=True,
            json_only=False,
            force_tool_names=False,
        )

    assert render(json_schema={}) == render(json_schema=None)


@pytest.mark.parametrize("version", [TokenizerVersion.v13, TokenizerVersion.v15])
def test_json_schema_with_tools_think_grammar(version: TokenizerVersion) -> None:
    r"""Think grammars keep text before a tool call, unless it starts like JSON (INFENG-3774)."""
    factory = _factory_for(version=version)
    grammar = _chat_grammar(
        factory=factory,
        tool_choice="auto",
        tools=[ToolProvider.make(name="get_weather")],
        response_format=_strict_schema(schema=SchemaProvider.basic_person()),
        parallel_tool_calls=True,
    )
    call = _tool_call(name="get_weather", arguments="{}")

    def accepts(content: str | list[Any]) -> bool:
        return _grammar_accepts(tokenizer=_ALL_TOKENIZERS[version], factory=factory, grammar=grammar, content=content)

    assert accepts(content=[TextChunk(text="Looking it up."), call])
    assert accepts(content=[ThinkChunk(thinking="hmm"), TextChunk(text=" Looking it up."), call])
    # Thinking before the JSON answer needs model settings support (v15+), as without tools.
    think_then_json = [ThinkChunk(thinking="hmm"), TextChunk(text='{"name": "John", "age": 30}')]
    assert accepts(content=think_then_json) == version.supports_model_settings
    assert not accepts(content=[TextChunk(text='{"name": "John"}'), call])
    assert not accepts(content=[TextChunk(text="[1]"), call])
    assert not accepts(content=[TextChunk(text="Hello world!")])


@pytest.mark.parametrize(
    ("version", "text_before_call_allowed"), [(TokenizerVersion.v11, False), (TokenizerVersion.v13, True)]
)
def test_get_lark_from_jinja_forced_tool_call(version: TokenizerVersion, text_before_call_allowed: bool) -> None:
    r"""`required` and the deprecated `any` render the same `required` template branch."""
    factory = _factory_for(version=version)

    def render(mode: ToolChoiceEnum) -> str:
        return factory.get_lark_from_jinja(
            template=factory.select_jinja_template(),
            mode=mode,
            tools=[ToolProvider.make(name="get_weather")],
            json_schema=None,
            parallel_tool_calls=True,
            json_only=False,
            force_tool_names=False,
        )

    grammar = render(mode=ToolChoiceEnum.required)
    call = _tool_call(name="get_weather", arguments="{}")

    def accepts(content: str | list[Any]) -> bool:
        return _grammar_accepts(tokenizer=_ALL_TOKENIZERS[version], factory=factory, grammar=grammar, content=content)

    assert grammar == render(mode=ToolChoiceEnum.any)
    assert accepts(content=[call])
    assert not accepts(content="plain text")
    assert accepts(content=[TextChunk(text="before"), call]) == text_before_call_allowed


def test_get_lark_for_json_schema_matches_request_path() -> None:
    factory = _factory_for(version=TokenizerVersion.v11)
    schema = SchemaProvider.basic_person()

    direct = factory.get_lark_for_json_schema(template=factory.select_jinja_template(), json_schema=schema)
    via_request = _chat_grammar(
        factory=factory,
        tool_choice="auto",
        tools=None,
        response_format=_strict_schema(schema=schema),
        parallel_tool_calls=True,
    )

    assert direct == via_request


def test_get_grammar_for_chat_request_no_default_kwargs() -> None:
    signature = inspect.signature(GrammarFactory.get_grammar_for_chat_request)

    assert all(parameter.default is inspect.Parameter.empty for parameter in signature.parameters.values())
