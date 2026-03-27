from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mistral_common.guidance.grammar_factory import GrammarFactory, convert_tool_calls
from mistral_common.protocol.instruct.chunk import TextChunk, ThinkChunk
from mistral_common.protocol.instruct.messages import AssistantMessage
from mistral_common.protocol.instruct.normalize import get_normalizer
from mistral_common.protocol.instruct.request import ReasoningEffort
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
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy, Tokenizer, TokenizerVersion
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
_AUTO_ANY_REQUIRED: tuple[Mode, Mode, Mode] = ("auto", "any", "required")


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


@pytest.fixture(scope="module")
def v11_tekken() -> MistralTokenizer:
    return _build_tekken_mistral_tokenizer(TokenizerVersion.v11)


@pytest.fixture(scope="module")
def v13_tekken() -> MistralTokenizer:
    return _build_tekken_mistral_tokenizer(TokenizerVersion.v13, add_think=True)


_V15_MODEL_SETTINGS_BUILDER = ModelSettingsBuilder(
    reasoning_effort=EnumBuilder[ReasoningEffort](
        values=list(ReasoningEffort),
        accepts_none=True,
        default=None,
    ),
)


@pytest.fixture(scope="module")
def v15_tekken() -> MistralTokenizer:
    return _build_tekken_mistral_tokenizer(
        TokenizerVersion.v15, add_think=True, model_settings_builder=_V15_MODEL_SETTINGS_BUILDER
    )


_PAYMENT_PARAMS: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"transaction_id": {"type": "string", "description": "The transaction id."}},
    "required": ["transaction_id"],
}


class ToolProvider:
    @staticmethod
    def retrieve_payment_status(strict: bool) -> Tool:
        return Tool(
            function=Function(
                name="retrieve_payment_status",
                description="Get payment status of a transaction",
                strict=strict,
                parameters=_PAYMENT_PARAMS,
            )
        )

    @staticmethod
    def retrieve_payment_date(strict: bool) -> Tool:
        return Tool(
            function=Function(
                name="retrieve_payment_date",
                description="Get payment date of a transaction",
                strict=strict,
                parameters=_PAYMENT_PARAMS,
            )
        )


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
    mode: Literal["auto", "any", "none", "required"] | NamedToolChoice
    tokens: list[int]
    should_fail_on: int | None
    case_name: str
    reasoning: bool = False
    parallel_tool_calls: bool = True
    tools: list[Tool] | None = None
    json_schema: dict[str, Any] | None = None
    # When set, uses this raw lark grammar instead of GrammarFactory.get_lark
    raw_lark: str | None = None

    @model_validator(mode="after")
    def validate_should_fail_on(self) -> TestCase:
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
            AssistantMessage(content=content), is_before_last_user_message=False, continue_message=False
        )

    tool_calls = [x for x in content if isinstance(x, ToolCall)]
    content_chunks = [x for x in content if not isinstance(x, ToolCall)]

    tokens: list[int] = []
    if content_chunks:
        tokens = instruct_tokenizer.encode_assistant_message(
            AssistantMessage(content=content_chunks),
            is_before_last_user_message=False,
            continue_message=False,
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


def _find_first_rejection(
    factory: GrammarFactory,
    tokens: list[int],
    mode: ToolChoice,
    tools: list[Tool] | None,
) -> int:
    r"""Finds the index of the first token rejected by the grammar.

    Args:
        factory: The grammar factory.
        tokens: The token sequence to test.
        mode: The tool choice mode (literal or NamedToolChoice).
        tools: The tools to pass to grammar generation.

    Returns:
        The index of the first rejected token.

    Raises:
        ValueError: If all tokens are accepted.
    """
    template = factory.select_jinja_template(reasoning=False)
    grammar = factory.get_lark_from_jinja(
        template=template, mode=mode, tools=tools, json_schema=None, parallel_tool_calls=True
    )
    matcher = factory.get_matcher(grammar)
    for i, token in enumerate(tokens):
        if not matcher.consume_token(token):
            return i
    raise ValueError("All tokens were accepted — expected a rejection")


def _token_debug_repr(tokenizer: Tekkenizer, token_id: int) -> str:
    return repr(tokenizer.id_to_byte_piece(token_id, SpecialTokenPolicy.KEEP))


def _generate_general_cases(mistral_tokenizer: MistralTokenizer) -> list[TestCase]:
    instruct_tokenizer = mistral_tokenizer.instruct_tokenizer
    tokenizer = instruct_tokenizer.tokenizer
    assert isinstance(instruct_tokenizer, InstructTokenizerBase)
    assert isinstance(tokenizer, Tekkenizer)

    cases: list[TestCase] = []
    items = {
        "newline": "\n",
        "blank": "_",
        "text": "Hello!",
        "text_with_newlines": "Hello!\n\nHow are you?\nI'm fine, thanks!",
        "emojis": "😃😂😊😍😘😗😙😚😋😛😜😝🤑🤗🤔🤐😐😑😶😬",
        "japanese": "こんにちは",
        "arabic": "مرحبا بكم في عالم الذكاء الاصطناعي",
    }
    for case_name, content in items.items():
        tokens = _encode_content(instruct_tokenizer, content)
        cases.append(
            TestCase(
                tokenizer=tokenizer,
                tokens=tokens,
                case_name=case_name,
                should_fail_on=None,
                mode="auto",
                reasoning=False,
            )
        )
        if tokenizer.version < TokenizerVersion.v13:
            # Count how many leading whitespace-only tokens the SAFE_WS? rule will consume
            # before the grammar rejects (expecting <think>).
            content_tokens = tokenizer.encode(content, bos=False, eos=False)
            ws_prefix_len = 0
            for t in content_tokens:
                piece = tokenizer.id_to_byte_piece(t, SpecialTokenPolicy.IGNORE)
                if piece.strip(b" \t\r\n") == b"":
                    ws_prefix_len += 1
                else:
                    break
            reasoning_fail = ws_prefix_len
        else:
            reasoning_fail = None
        cases.append(
            TestCase(
                tokenizer=tokenizer,
                tokens=tokens,
                case_name=f"{case_name}_reasoning",
                should_fail_on=reasoning_fail,
                mode="auto",
                reasoning=True,
            )
        )
    return cases


def _count_prefix_tokens(tokenizer: Tokenizer, full_text: str, prefix: str) -> int:
    r"""Counts the number of tokens that cover the prefix bytes in the full-text tokenization.

    BPE tokenization is context-dependent, so encoding a prefix in isolation may produce
    different tokens than encoding the full string. This helper encodes the full string
    and counts how many tokens are needed to cover the byte-length of the prefix.

    Args:
        tokenizer: The tokenizer to use.
        full_text: The complete text to tokenize.
        prefix: The prefix whose byte-length determines the token count.

    Returns:
        The number of tokens covering the prefix bytes.
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


def _generate_emoji_cases(mistral_tokenizer: MistralTokenizer) -> list[TestCase]:
    instruct_tokenizer = mistral_tokenizer.instruct_tokenizer
    tokenizer = instruct_tokenizer.tokenizer
    assert isinstance(instruct_tokenizer, InstructTokenizerBase)
    emoji_lark = EMOJI_LARK_PATH.read_text(encoding="utf-8")
    cases: list[TestCase] = []
    items: dict[str, tuple[str, int | None]] = {
        "emojis_valid_a": ("😃😂😊😍😘😗😙😚😋😛😜😝🤑🤗🤔🤐😐😑😶😬", None),
        "emojis_valid_b": ("😃😃😃", None),
        "emojis_invalid_text": ("😃smile", _count_prefix_tokens(tokenizer, "😃smile", "😃")),
        "emojis_invalid_space": ("😃 ", _count_prefix_tokens(tokenizer, "😃 ", "😃")),
    }
    for case_name, (text, should_fail_on) in items.items():
        tokens = tokenizer.encode(text, bos=False, eos=False)
        cases.append(
            TestCase(
                tokenizer=tokenizer,
                tokens=tokens,
                case_name=case_name,
                should_fail_on=should_fail_on,
                mode="auto",
                raw_lark=emoji_lark,
            )
        )
    return cases


def _generate_cases_tool_calls(mistral_tokenizer: MistralTokenizer) -> list[TestCase]:
    instruct_tokenizer = mistral_tokenizer.instruct_tokenizer
    tokenizer = instruct_tokenizer.tokenizer
    assert isinstance(instruct_tokenizer, InstructTokenizerBase)

    cases: list[TestCase] = []
    items: list[tuple[str, list[ToolCall], dict[Mode, int | None]]] = [
        (
            "single_fcall",
            [ToolCall(function=FunctionCall(name="hello", arguments='{"arg1": "val1", "arg2": "val2"}'))],
            {"auto": None, "any": None, "none": 0, "required": None},
        ),
        (
            "multi_fcall",
            [
                ToolCall(function=FunctionCall(name="hello", arguments='{"arg1": "val1", "arg2": "val2"}')),
                ToolCall(function=FunctionCall(name="hello_1", arguments='{"arg1": "val1", "arg2": "val2"}')),
                ToolCall(function=FunctionCall(name="hello_2_3", arguments='{"arg1": "val1", "arg2": "val2"}')),
            ],
            {"auto": None, "any": None, "none": 0, "required": None},
        ),
        (
            "emoji_fcall",
            [
                ToolCall(
                    function=FunctionCall(name="he🧦🧦o", arguments='{"arg1": "🐱", "arg2": "🐶", "arg🧦": "🧦"}'),
                )
            ],
            {"auto": None, "any": None, "none": 0, "required": None},
        ),
        (
            "pretty_printed_args",
            [
                ToolCall(
                    function=FunctionCall(
                        name="hello",
                        arguments='{\n      "arg1": "val1",\n      "arg2": "val2"\n    }\n',
                    ),
                )
            ],
            {"auto": None, "any": None, "none": 0, "required": None},
        ),
        (
            "japanese_fcall",
            [ToolCall(function=FunctionCall(name="こんにちは", arguments='{"こん": "にちは"}'))],
            {"auto": None, "any": None, "none": 0, "required": None},
        ),
    ]
    for case_name, content, valid_for in items:
        tokens = _encode_content(instruct_tokenizer, content)
        for mode, should_fail_on in valid_for.items():
            cases.append(
                TestCase(
                    tokenizer=tokenizer,
                    tokens=tokens,
                    should_fail_on=should_fail_on,
                    case_name=case_name,
                    mode=mode,
                )
            )

        if tokenizer.version < TokenizerVersion.v13:
            reasoning_valid_for: dict[Mode, int | None] = {"auto": 0, "any": 0, "none": 0, "required": 0}
        else:
            reasoning_valid_for = {"auto": None, "any": None, "none": 0, "required": None}
        for mode, should_fail_on in reasoning_valid_for.items():
            cases.append(
                TestCase(
                    tokenizer=tokenizer,
                    tokens=tokens,
                    should_fail_on=should_fail_on,
                    case_name=f"{case_name}_reasoning",
                    mode=mode,
                    reasoning=True,
                )
            )

    # Broken / missing args edge cases
    token_items: list[tuple[str, list[int], dict[Mode, int | None]]] = [
        (
            "fcall_broken_args",
            [
                tokenizer.get_special_token("[TOOL_CALLS]"),
                *tokenizer.encode("hello", bos=False, eos=False),
                tokenizer.get_special_token("[ARGS]"),
                *tokenizer.encode('{"a', bos=False, eos=False),
                tokenizer.eos_id,
            ],
            {"auto": -1, "any": -1, "none": 0, "required": -1},
        ),
        (
            "fcall_missing_args",
            [
                tokenizer.get_special_token("[TOOL_CALLS]"),
                *tokenizer.encode("hello", bos=False, eos=False),
                tokenizer.get_special_token("[ARGS]"),
                tokenizer.get_special_token("[TOOL_CALLS]"),
            ],
            {"auto": -1, "any": -1, "none": 0, "required": -1},
        ),
    ]

    for case_name, tokens, valid_for in token_items:
        for mode, should_fail_on in valid_for.items():
            cases.append(
                TestCase(
                    tokenizer=tokenizer,
                    tokens=tokens,
                    should_fail_on=should_fail_on,
                    case_name=case_name,
                    mode=mode,
                )
            )
    return cases


def _generate_cases_text_and_tool_calls(mistral_tokenizer: MistralTokenizer) -> list[TestCase]:
    instruct_tokenizer = mistral_tokenizer.instruct_tokenizer
    tokenizer = instruct_tokenizer.tokenizer
    assert isinstance(instruct_tokenizer, InstructTokenizerBase)

    cases: list[TestCase] = []
    content = [
        TextChunk(text="Hello!"),
        ToolCall(function=FunctionCall(name="hello", arguments='{"arg1": "val1", "arg2": "val2"}')),
    ]
    tokens = _encode_content(instruct_tokenizer, content)
    text_len = len(tokenizer.encode("Hello!", bos=False, eos=False))

    valid_for: dict[Mode, int | None] = {"auto": None, "any": 0, "none": text_len, "required": None}

    for mode, should_fail_on in valid_for.items():
        cases.append(
            TestCase(
                tokenizer=tokenizer,
                tokens=tokens,
                should_fail_on=should_fail_on,
                case_name="text_fcall",
                mode=mode,
            )
        )

    if tokenizer.version < TokenizerVersion.v13:
        reasoning_valid_for: dict[Mode, int | None] = {"auto": 0, "any": 0, "none": 0, "required": 0}
    else:
        reasoning_valid_for = {"auto": None, "any": None, "none": text_len, "required": None}
    for mode, should_fail_on in reasoning_valid_for.items():
        cases.append(
            TestCase(
                tokenizer=tokenizer,
                tokens=tokens,
                should_fail_on=should_fail_on,
                case_name="text_fcall_reasoning",
                mode=mode,
                reasoning=True,
            )
        )

    return cases


def _generate_cases_thinking_v11(mistral_tokenizer: MistralTokenizer) -> list[TestCase]:
    r"""Generate thinking test cases for v11 (plain text think grammar)."""
    instruct_tokenizer = mistral_tokenizer.instruct_tokenizer
    tokenizer = instruct_tokenizer.tokenizer
    assert isinstance(instruct_tokenizer, InstructTokenizerBase)

    cases: list[TestCase] = []
    thinks: list[tuple[str, list[Any], dict[Mode, int | None]]] = [
        (
            "force_think",
            [TextChunk(text="Hello world!")],
            {"auto": 0, "any": 0, "none": 0, "required": 0},
        ),
        (
            "think_without_response",
            [TextChunk(text="<think>Hello!</think>")],
            {"auto": -1, "any": -1, "none": -1, "required": -1},
        ),
        (
            "unclosed_think",
            [TextChunk(text="<think>Hello!")],
            {"auto": -1, "any": -1, "none": -1, "required": -1},
        ),
        (
            "plain_think_with_response",
            [TextChunk(text="<think>Hello!</think>World!")],
            {
                "auto": None,
                # any/required: think fcalls — after think, "World!" doesn't match fcalls
                "any": len(tokenizer.encode("<think>Hello!</think>", bos=False, eos=False)),
                "none": None,
                "required": len(tokenizer.encode("<think>Hello!</think>", bos=False, eos=False)),
            },
        ),
        (
            "think_with_tool_call",
            [
                TextChunk(text="<think>Hello!</think>"),
                ToolCall(function=FunctionCall(name="hello", arguments='{"arg1": "val1", "arg2": "val2"}')),
            ],
            {
                "auto": None,
                "any": None,
                "none": len(tokenizer.encode("<think>Hello!</think>", bos=False, eos=False)),
                "required": None,
            },
        ),
        (
            "think_with_text_and_tool_call",
            [
                TextChunk(text="<think>Hello!</think>Ho!"),
                ToolCall(function=FunctionCall(name="hello", arguments='{"arg1": "val1", "arg2": "val2"}')),
            ],
            {
                # auto: think (content | fcalls) — picks content for "Ho!", then tool call rejected
                "auto": len(tokenizer.encode("<think>Hello!</think>Ho!", bos=False, eos=False)),
                # any/required: think fcalls — after think, "Ho!" doesn't match fcalls
                "any": len(tokenizer.encode("<think>Hello!</think>", bos=False, eos=False)),
                "none": len(tokenizer.encode("<think>Hello!</think>Ho!", bos=False, eos=False)),
                "required": len(tokenizer.encode("<think>Hello!</think>", bos=False, eos=False)),
            },
        ),
    ]
    for case_name, content, valid_for in thinks:
        tokens = _encode_content(instruct_tokenizer, content)
        for mode, should_fail_on in valid_for.items():
            cases.append(
                TestCase(
                    tokenizer=tokenizer,
                    tokens=tokens,
                    should_fail_on=should_fail_on,
                    case_name=case_name,
                    mode=mode,
                    reasoning=True,
                )
            )
    return cases


def _generate_cases_thinking(mistral_tokenizer: MistralTokenizer) -> list[TestCase]:
    r"""Generate thinking test cases for v13+ (structured think grammar with [THINK]/[/THINK] tokens)."""
    instruct_tokenizer = mistral_tokenizer.instruct_tokenizer
    tokenizer = instruct_tokenizer.tokenizer
    assert isinstance(instruct_tokenizer, InstructTokenizerBase)

    cases: list[TestCase] = []

    def _think_tokens(text: str) -> list[int]:
        r"""Helper to encode a ThinkChunk."""
        assert isinstance(instruct_tokenizer, InstructTokenizerV13)
        return instruct_tokenizer.encode_think(ThinkChunk(thinking=text))

    thinks: list[tuple[str, list[Any], dict[Mode, int | None]]] = [
        (
            "plain_text",
            [TextChunk(text="Hello world!")],
            {"auto": None, "any": -1, "none": None, "required": -1},
        ),
        (
            "plain_think",
            [ThinkChunk(thinking="Hello!")],
            {"auto": -1, "any": -1, "none": -1, "required": -1},
        ),
        (
            "plain_think_with_response",
            [ThinkChunk(thinking="Hello!"), TextChunk(text="World!")],
            {"auto": None, "any": -1, "none": None, "required": -1},
        ),
        (
            "think_with_tool_call",
            [
                ThinkChunk(thinking="Hello!"),
                ToolCall(function=FunctionCall(name="hello", arguments='{"arg1": "val1", "arg2": "val2"}')),
            ],
            {
                "auto": None,
                "any": None,
                "none": len(_think_tokens("Hello!")),
                "required": None,
            },
        ),
        (
            "think_text_tool_call",
            [
                ThinkChunk(thinking="Hello!"),
                TextChunk(text="World!"),
                ToolCall(function=FunctionCall(name="hello", arguments='{"arg1": "val1", "arg2": "val2"}')),
            ],
            {
                "auto": None,
                "any": None,
                "none": len(_think_tokens("Hello!")) + len(tokenizer.encode("World!", bos=False, eos=False)),
                # required: think? content? fcalls — think+content+fcalls all present, passes
                "required": None,
            },
        ),
    ]
    for case_name, content, valid_for in thinks:
        tokens = _encode_content(instruct_tokenizer, content)
        for mode, should_fail_on in valid_for.items():
            cases.append(
                TestCase(
                    tokenizer=tokenizer,
                    tokens=tokens,
                    should_fail_on=should_fail_on,
                    case_name=case_name,
                    mode=mode,
                    reasoning=True,
                )
            )
    return cases


def _generate_single_tool_call(mistral_tokenizer: MistralTokenizer) -> list[TestCase]:
    instruct_tokenizer = mistral_tokenizer.instruct_tokenizer
    tokenizer = instruct_tokenizer.tokenizer
    assert isinstance(instruct_tokenizer, InstructTokenizerBase)

    cases: list[TestCase] = []
    single_call = [ToolCall(function=FunctionCall(name="hello", arguments='{"arg1": "val1", "arg2": "val2"}'))]
    single_tokens = _encode_content(instruct_tokenizer, single_call)
    for mode in _AUTO_ANY_REQUIRED:
        cases.append(
            TestCase(
                tokenizer=tokenizer,
                tokens=single_tokens,
                should_fail_on=None,
                case_name="single_tool_call",
                mode=mode,
                parallel_tool_calls=False,
            )
        )
    cases.append(
        TestCase(
            tokenizer=tokenizer,
            tokens=single_tokens,
            should_fail_on=0,
            case_name="single_tool_call",
            mode="none",
            parallel_tool_calls=False,
        )
    )

    if tokenizer.version < TokenizerVersion.v13:
        reasoning_valid_for: dict[Mode, int | None] = {"auto": 0, "any": 0, "none": 0, "required": 0}
    else:
        reasoning_valid_for = {"auto": None, "any": None, "none": 0, "required": None}
    for mode, should_fail_on in reasoning_valid_for.items():
        cases.append(
            TestCase(
                tokenizer=tokenizer,
                tokens=single_tokens,
                should_fail_on=should_fail_on,
                case_name="single_tool_call_reasoning",
                mode=mode,
                parallel_tool_calls=False,
                reasoning=True,
            )
        )

    multi_calls = [
        ToolCall(function=FunctionCall(name="fn1", arguments='{"arg1": "val1", "arg2": "val2"}')),
        ToolCall(function=FunctionCall(name="fn2", arguments='{"arg1": "val1", "arg2": "val2"}')),
    ]
    multi_tokens = _encode_content(instruct_tokenizer, multi_calls)
    single_tokens_with_eos = _encode_content(instruct_tokenizer, [multi_calls[0]])
    fail_idx = len(single_tokens_with_eos) - 1

    for mode in _AUTO_ANY_REQUIRED:
        cases.append(
            TestCase(
                tokenizer=tokenizer,
                tokens=multi_tokens,
                should_fail_on=fail_idx,
                case_name="multi_tool_call_disallowed",
                mode=mode,
                parallel_tool_calls=False,
            )
        )

    return cases


def _generate_strict_tool_calls(mistral_tokenizer: MistralTokenizer, factory: GrammarFactory) -> list[TestCase]:
    instruct_tokenizer = mistral_tokenizer.instruct_tokenizer
    tokenizer = instruct_tokenizer.tokenizer
    assert isinstance(instruct_tokenizer, InstructTokenizerBase)

    cases: list[TestCase] = []
    tools_strict = [ToolProvider.retrieve_payment_date(strict=True)]

    # 1. Non-strict tools — any function name/args accepted
    non_strict_call = [ToolCall(function=FunctionCall(name="fn1", arguments='{"arg1": "val1", "arg2": "val2"}'))]
    non_strict_tokens = _encode_content(instruct_tokenizer, non_strict_call)
    for mode in _AUTO_ANY_REQUIRED:
        cases.append(
            TestCase(
                tokenizer=tokenizer,
                tokens=non_strict_tokens,
                should_fail_on=None,
                case_name="single_non_strict_tool_call",
                mode=mode,
                tools=[ToolProvider.retrieve_payment_date(strict=False)],
            )
        )

    # 2. Correct strict tool call
    strict_call = [
        ToolCall(function=FunctionCall(name="retrieve_payment_date", arguments='{"transaction_id": "12345"}'))
    ]
    strict_tokens = _encode_content(instruct_tokenizer, strict_call)
    for mode in _AUTO_ANY_REQUIRED:
        cases.append(
            TestCase(
                tokenizer=tokenizer,
                tokens=strict_tokens,
                should_fail_on=None,
                case_name="single_strict_tool_call",
                mode=mode,
                tools=[ToolProvider.retrieve_payment_date(strict=True)],
            )
        )

    # 3. Wrong args for strict tool — it must fail somewhere before the end.
    wrong_args_call = [ToolCall(function=FunctionCall(name="retrieve_payment_date", arguments='{"bogus": "12345"}'))]
    wrong_args_tokens = _encode_content(instruct_tokenizer, wrong_args_call)
    bogus_start = _find_first_rejection(factory, wrong_args_tokens, mode=ToolChoiceEnum.auto, tools=tools_strict)
    for mode in _AUTO_ANY_REQUIRED:
        cases.append(
            TestCase(
                tokenizer=tokenizer,
                tokens=wrong_args_tokens,
                should_fail_on=bogus_start,
                case_name="strict_tool_call_wrong_args",
                mode=mode,
                tools=tools_strict,
            )
        )

    # 4. Wrong name for strict tool
    wrong_name_call = [ToolCall(function=FunctionCall(name="fn1", arguments='{"transaction_id": "12345"}'))]
    wrong_name_tokens = _encode_content(instruct_tokenizer, wrong_name_call)
    fail_on_name = _find_first_rejection(factory, wrong_name_tokens, mode=ToolChoiceEnum.auto, tools=tools_strict)
    for mode in _AUTO_ANY_REQUIRED:
        cases.append(
            TestCase(
                tokenizer=tokenizer,
                tokens=wrong_name_tokens,
                should_fail_on=fail_on_name,
                case_name="strict_tool_call_wrong_name",
                mode=mode,
                tools=tools_strict,
            )
        )

    # 5. Multiple strict tool calls (both correct)
    multi_strict = [
        ToolCall(function=FunctionCall(name="retrieve_payment_date", arguments='{"transaction_id": "12345"}')),
        ToolCall(function=FunctionCall(name="retrieve_payment_status", arguments='{"transaction_id": "12345"}')),
    ]
    multi_strict_tokens = _encode_content(instruct_tokenizer, multi_strict)
    for mode in _AUTO_ANY_REQUIRED:
        cases.append(
            TestCase(
                tokenizer=tokenizer,
                tokens=multi_strict_tokens,
                should_fail_on=None,
                case_name="multiple_strict_tool_calls",
                mode=mode,
                tools=[
                    ToolProvider.retrieve_payment_date(strict=True),
                    ToolProvider.retrieve_payment_status(strict=True),
                ],
            )
        )

    # 6. reasoning=True variants
    if tokenizer.version < TokenizerVersion.v13:
        for mode in _AUTO_ANY_REQUIRED:
            cases.append(
                TestCase(
                    tokenizer=tokenizer,
                    tokens=strict_tokens,
                    should_fail_on=0,
                    case_name="strict_tool_call_reasoning",
                    mode=mode,
                    tools=tools_strict,
                    reasoning=True,
                )
            )
    else:
        for mode in _AUTO_ANY_REQUIRED:
            cases.append(
                TestCase(
                    tokenizer=tokenizer,
                    tokens=strict_tokens,
                    should_fail_on=None,
                    case_name="strict_tool_call_reasoning",
                    mode=mode,
                    tools=tools_strict,
                    reasoning=True,
                )
            )
        for mode in _AUTO_ANY_REQUIRED:
            cases.append(
                TestCase(
                    tokenizer=tokenizer,
                    tokens=multi_strict_tokens,
                    should_fail_on=None,
                    case_name="multiple_strict_tool_calls_reasoning",
                    mode=mode,
                    tools=[
                        ToolProvider.retrieve_payment_date(strict=True),
                        ToolProvider.retrieve_payment_status(strict=True),
                    ],
                    reasoning=True,
                )
            )

    return cases


def _generate_named_tool_choice(mistral_tokenizer: MistralTokenizer, factory: GrammarFactory) -> list[TestCase]:
    r"""Generate test cases for NamedToolChoice (forcing a specific tool)."""
    instruct_tokenizer = mistral_tokenizer.instruct_tokenizer
    tokenizer = instruct_tokenizer.tokenizer
    assert isinstance(instruct_tokenizer, InstructTokenizerBase)

    cases: list[TestCase] = []

    tools = [
        ToolProvider.retrieve_payment_date(strict=False),
        ToolProvider.retrieve_payment_status(strict=False),
    ]

    # 1. NamedToolChoice for retrieve_payment_date with correct function call
    named_tool_date = NamedToolChoice(
        type=ToolTypes.function,
        function=FunctionName(name="retrieve_payment_date"),
    )
    correct_date_call = [
        ToolCall(
            function=FunctionCall(
                name="retrieve_payment_date",
                arguments='{"transaction_id": "12345"}',
            )
        )
    ]
    correct_date_tokens = _encode_content(instruct_tokenizer, correct_date_call)

    cases.append(
        TestCase(
            tokenizer=tokenizer,
            tokens=correct_date_tokens,
            should_fail_on=None,
            case_name="named_tool_choice_correct",
            mode=named_tool_date,
            tools=tools,
            parallel_tool_calls=True,
        )
    )
    cases.append(
        TestCase(
            tokenizer=tokenizer,
            tokens=correct_date_tokens,
            should_fail_on=None,
            case_name="named_tool_choice_correct",
            mode=named_tool_date,
            tools=tools,
            parallel_tool_calls=False,
        )
    )

    # 2. Non-strict NamedToolChoice should NOT enforce JSON arguments schema —
    arbitrary_args_call = [
        ToolCall(
            function=FunctionCall(
                name="retrieve_payment_date",
                arguments='{"completely": "arbitrary", "keys": [1, 2, 3]}',
            )
        )
    ]
    arbitrary_args_tokens = _encode_content(instruct_tokenizer, arbitrary_args_call)
    cases.append(
        TestCase(
            tokenizer=tokenizer,
            tokens=arbitrary_args_tokens,
            should_fail_on=None,
            case_name="named_tool_choice_non_strict_arbitrary_args",
            mode=named_tool_date,
            tools=tools,
            parallel_tool_calls=True,
        )
    )

    # 3. NamedToolChoice should reject a different tool name
    wrong_tool_call = [
        ToolCall(
            function=FunctionCall(
                name="retrieve_payment_status",
                arguments='{"transaction_id": "12345"}',
            )
        )
    ]
    wrong_tool_tokens = _encode_content(instruct_tokenizer, wrong_tool_call)

    fail_idx = _find_first_rejection(
        factory,
        wrong_tool_tokens,
        mode=named_tool_date,
        tools=tools,
    )
    cases.append(
        TestCase(
            tokenizer=tokenizer,
            tokens=wrong_tool_tokens,
            should_fail_on=fail_idx,
            case_name="named_tool_choice_wrong_tool",
            mode=named_tool_date,
            tools=tools,
            parallel_tool_calls=True,
        )
    )

    # 4. NamedToolChoice with reasoning mode
    if tokenizer.version < TokenizerVersion.v13:
        cases.append(
            TestCase(
                tokenizer=tokenizer,
                tokens=correct_date_tokens,
                should_fail_on=0,
                case_name="named_tool_choice_reasoning",
                mode=named_tool_date,
                tools=tools,
                reasoning=True,
            )
        )
    else:
        cases.append(
            TestCase(
                tokenizer=tokenizer,
                tokens=correct_date_tokens,
                should_fail_on=None,
                case_name="named_tool_choice_reasoning",
                mode=named_tool_date,
                tools=tools,
                reasoning=True,
            )
        )

    # 5. NamedToolChoice with strict tool should validate arguments
    strict_tools = [ToolProvider.retrieve_payment_date(strict=True)]
    named_tool_strict = NamedToolChoice(
        type=ToolTypes.function,
        function=FunctionName(name="retrieve_payment_date"),
    )

    cases.append(
        TestCase(
            tokenizer=tokenizer,
            tokens=correct_date_tokens,
            should_fail_on=None,
            case_name="named_tool_choice_strict_correct",
            mode=named_tool_strict,
            tools=strict_tools,
        )
    )

    wrong_args_call = [
        ToolCall(
            function=FunctionCall(
                name="retrieve_payment_date",
                arguments='{"wrong_arg": "12345"}',
            )
        )
    ]
    wrong_args_tokens = _encode_content(instruct_tokenizer, wrong_args_call)
    fail_on_args = _find_first_rejection(
        factory,
        wrong_args_tokens,
        mode=named_tool_strict,
        tools=strict_tools,
    )
    cases.append(
        TestCase(
            tokenizer=tokenizer,
            tokens=wrong_args_tokens,
            should_fail_on=fail_on_args,
            case_name="named_tool_choice_strict_wrong_args",
            mode=named_tool_strict,
            tools=strict_tools,
        )
    )

    # 6. NamedToolChoice with non-existent tool in tools list
    named_nonexistent = NamedToolChoice(
        type=ToolTypes.function,
        function=FunctionName(name="non_existent_tool"),
    )
    nonexistent_call = [
        ToolCall(
            function=FunctionCall(
                name="non_existent_tool",
                arguments='{"arg": "value"}',
            )
        )
    ]
    nonexistent_tokens = _encode_content(instruct_tokenizer, nonexistent_call)

    cases.append(
        TestCase(
            tokenizer=tokenizer,
            tokens=nonexistent_tokens,
            should_fail_on=None,
            case_name="named_tool_choice_nonexistent_tool",
            mode=named_nonexistent,
            tools=[],
        )
    )

    return cases


def _find_first_json_schema_rejection(
    factory: GrammarFactory,
    tokens: list[int],
    json_schema: dict[str, Any],
) -> int:
    r"""Finds the index of the first token rejected by the JSON schema grammar.

    Args:
        factory: The grammar factory.
        tokens: The token sequence to test.
        json_schema: The JSON schema to validate against.

    Returns:
        The index of the first rejected token.

    Raises:
        ValueError: If all tokens are accepted.
    """
    template = factory.select_jinja_template(reasoning=False)
    grammar = factory.get_lark_for_json_schema(template=template, json_schema=json_schema)
    matcher = factory.get_matcher(grammar)
    for i, token in enumerate(tokens):
        if not matcher.consume_token(token):
            return i
    raise ValueError("All tokens were accepted — expected a rejection")


def _generate_json_schema(mistral_tokenizer: MistralTokenizer, factory: GrammarFactory) -> list[TestCase]:
    instruct_tokenizer = mistral_tokenizer.instruct_tokenizer
    tokenizer = instruct_tokenizer.tokenizer
    assert isinstance(instruct_tokenizer, InstructTokenizerBase)

    cases: list[TestCase] = []
    items: list[tuple[str, str, int | None, dict[str, Any]]] = [
        (
            "basic_person_valid",
            '{"name": "John", "age": 30}',
            None,
            SchemaProvider.basic_person(),
        ),
        (
            "invalid_json_missing_curly_bracket",
            '"name": "John", "age": 30}',
            0,
            {"type": "object"},
        ),
        (
            "valid_json_white_spaces",
            '\n {"name": "John", "age": 30}',
            None,
            {"type": "object"},
        ),
        (
            "invalid_json_backslash_f",
            '\f{"name": "John", "age": 30}',
            0,
            {"type": "object"},
        ),
        (
            "basic_person_non_strict_valid",
            '{"age": "John", "name": 30}',
            None,
            {"type": "object"},
        ),
        (
            "domerge_valid",
            '{"new_clusters": {"b": ["a", "b", "c"], "d": ["e"]} }',
            None,
            SchemaProvider.basic_dict_of_list(),
        ),
    ]

    for case_name, text, should_fail_on, json_schema in items:
        tokens = _encode_content(instruct_tokenizer, text)
        cases.append(
            TestCase(
                tokenizer=tokenizer,
                tokens=tokens,
                should_fail_on=should_fail_on,
                case_name=case_name,
                mode="auto",
                json_schema=json_schema,
            )
        )

    # Cases where should_fail_on must be computed dynamically because the exact
    # rejection index depends on the grammar engine's internal byte-level parsing.
    person_invalid_tokens = _encode_content(instruct_tokenizer, '{"age": "John", "name": 30}')
    person_schema = SchemaProvider.basic_person()
    cases.append(
        TestCase(
            tokenizer=tokenizer,
            tokens=person_invalid_tokens,
            should_fail_on=_find_first_json_schema_rejection(factory, person_invalid_tokens, person_schema),
            case_name="basic_person_invalid",
            mode="auto",
            json_schema=person_schema,
        )
    )

    return cases


def _generate_json_schema_reasoning(mistral_tokenizer: MistralTokenizer, factory: GrammarFactory) -> list[TestCase]:
    instruct_tokenizer = mistral_tokenizer.instruct_tokenizer
    tokenizer = instruct_tokenizer.tokenizer
    assert isinstance(instruct_tokenizer, InstructTokenizerBase)

    cases: list[TestCase] = []

    # Valid JSON should be accepted even under reasoning templates with json_only
    items: list[tuple[str, str, int | None, dict[str, Any]]] = [
        (
            "json_schema_reasoning_valid",
            '{"name": "John", "age": 30}',
            None,
            SchemaProvider.basic_person(),
        ),
        (
            "json_schema_reasoning_whitespace_valid",
            '\n {"name": "John", "age": 30}',
            None,
            {"type": "object"},
        ),
        (
            "json_schema_reasoning_invalid_bracket",
            '"name": "John", "age": 30}',
            0,
            {"type": "object"},
        ),
    ]
    for case_name, text, should_fail_on, json_schema in items:
        tokens = _encode_content(instruct_tokenizer, text)
        cases.append(
            TestCase(
                tokenizer=tokenizer,
                tokens=tokens,
                should_fail_on=should_fail_on,
                case_name=case_name,
                mode="auto",
                json_schema=json_schema,
                reasoning=True,
            )
        )

    return cases


def _generate_json_only_negative(mistral_tokenizer: MistralTokenizer, factory: GrammarFactory) -> list[TestCase]:
    instruct_tokenizer = mistral_tokenizer.instruct_tokenizer
    tokenizer = instruct_tokenizer.tokenizer
    assert isinstance(instruct_tokenizer, InstructTokenizerBase)

    cases: list[TestCase] = []

    # Plain text should be rejected by json_only grammar
    text_tokens = _encode_content(instruct_tokenizer, "Hello world!")
    cases.append(
        TestCase(
            tokenizer=tokenizer,
            tokens=text_tokens,
            should_fail_on=0,
            case_name="json_only_rejects_text",
            mode="auto",
            json_schema={"type": "object"},
            reasoning=False,
        )
    )

    # Tool calls should be rejected by json_only grammar
    tool_call_tokens = _encode_content(
        instruct_tokenizer,
        [ToolCall(function=FunctionCall(name="hello", arguments='{"arg1": "val1"}'))],
    )
    cases.append(
        TestCase(
            tokenizer=tokenizer,
            tokens=tool_call_tokens,
            should_fail_on=0,
            case_name="json_only_rejects_tool_call",
            mode="auto",
            json_schema={"type": "object"},
            reasoning=False,
        )
    )

    # Plain text should also be rejected with reasoning=True
    cases.append(
        TestCase(
            tokenizer=tokenizer,
            tokens=text_tokens,
            should_fail_on=0,
            case_name="json_only_reasoning_rejects_text",
            mode="auto",
            json_schema={"type": "object"},
            reasoning=True,
        )
    )

    return cases


def _generate_json_schema_think_with_json(
    mistral_tokenizer: MistralTokenizer, factory: GrammarFactory
) -> list[TestCase]:
    instruct_tokenizer = mistral_tokenizer.instruct_tokenizer
    tokenizer = instruct_tokenizer.tokenizer
    assert isinstance(instruct_tokenizer, InstructTokenizerBase)
    assert isinstance(instruct_tokenizer, InstructTokenizerV13)

    cases: list[TestCase] = []

    json_schema = SchemaProvider.basic_person()
    valid_json = '{"name": "John", "age": 30}'

    def _think_tokens(text: str) -> list[int]:
        return instruct_tokenizer.encode_think(ThinkChunk(thinking=text))

    # Think + valid JSON should be accepted with json_only + reasoning
    think_json_tokens = [
        *_think_tokens("Let me think about this..."),
        *tokenizer.encode(valid_json, bos=False, eos=False),
        tokenizer.eos_id,
    ]
    cases.append(
        TestCase(
            tokenizer=tokenizer,
            tokens=think_json_tokens,
            should_fail_on=None,
            case_name="think_with_json_valid",
            mode="auto",
            json_schema=json_schema,
            reasoning=True,
        )
    )

    json_only_tokens = _encode_content(instruct_tokenizer, valid_json)
    cases.append(
        TestCase(
            tokenizer=tokenizer,
            tokens=json_only_tokens,
            should_fail_on=None,
            case_name="think_with_json_no_think_valid",
            mode="auto",
            json_schema=json_schema,
            reasoning=True,
        )
    )

    think_text_tokens = [
        *_think_tokens("Let me think..."),
        *tokenizer.encode("Hello world!", bos=False, eos=False),
        tokenizer.eos_id,
    ]

    fail_idx = len(_think_tokens("Let me think..."))
    cases.append(
        TestCase(
            tokenizer=tokenizer,
            tokens=think_text_tokens,
            should_fail_on=fail_idx,
            case_name="think_with_json_rejects_text_after_think",
            mode="auto",
            json_schema=json_schema,
            reasoning=True,
        )
    )

    return cases


def _generate_cases(mistral_tokenizer: MistralTokenizer, factory: GrammarFactory) -> list[TestCase]:
    instruct_tokenizer = mistral_tokenizer.instruct_tokenizer
    assert isinstance(instruct_tokenizer, InstructTokenizerBase)
    tokenizer_version = instruct_tokenizer.tokenizer.version

    cases = _generate_general_cases(mistral_tokenizer)
    cases += _generate_emoji_cases(mistral_tokenizer)
    cases += _generate_json_schema(mistral_tokenizer, factory)
    cases += _generate_json_schema_reasoning(mistral_tokenizer, factory)
    cases += _generate_json_only_negative(mistral_tokenizer, factory)
    cases += _generate_cases_tool_calls(mistral_tokenizer)
    cases += _generate_single_tool_call(mistral_tokenizer)
    cases += _generate_strict_tool_calls(mistral_tokenizer, factory)
    cases += _generate_named_tool_choice(mistral_tokenizer, factory)
    cases += _generate_cases_text_and_tool_calls(mistral_tokenizer)

    if tokenizer_version >= TokenizerVersion.v13:
        cases += _generate_cases_thinking(mistral_tokenizer)
    else:
        cases += _generate_cases_thinking_v11(mistral_tokenizer)

    # v15+ supports think_with_json (thinking before JSON in json_only mode)
    if tokenizer_version.supports_model_settings:
        cases += _generate_json_schema_think_with_json(mistral_tokenizer, factory)

    return cases


_grammar_factories: dict[int, GrammarFactory] = {}


def _get_grammar_factory(mistral_tokenizer: MistralTokenizer) -> GrammarFactory:
    tok_id = id(mistral_tokenizer)
    if tok_id not in _grammar_factories:
        _grammar_factories[tok_id] = GrammarFactory(mistral_tokenizer)
    return _grammar_factories[tok_id]


_ALL_TOKENIZERS: list[MistralTokenizer] = [
    _build_tekken_mistral_tokenizer(TokenizerVersion.v11),
    _build_tekken_mistral_tokenizer(TokenizerVersion.v13, add_think=True),
    _build_tekken_mistral_tokenizer(
        TokenizerVersion.v15, add_think=True, model_settings_builder=_V15_MODEL_SETTINGS_BUILDER
    ),
]

_ALL_CASES: list[TestCase] = []
_ALL_MISTRAL_TOKENIZERS: dict[int, MistralTokenizer] = {}
for mistral_tokenizer in _ALL_TOKENIZERS:
    tokenizer = mistral_tokenizer.instruct_tokenizer.tokenizer
    _ALL_MISTRAL_TOKENIZERS[id(tokenizer)] = mistral_tokenizer
    factory = _get_grammar_factory(mistral_tokenizer)
    _ALL_CASES.extend(_generate_cases(mistral_tokenizer, factory))


class TestGrammarFactory:
    @pytest.mark.parametrize("test_case", _ALL_CASES, ids=lambda tc: tc.name)
    def test_grammar(self, test_case: TestCase) -> None:
        mistral_tokenizer = _ALL_MISTRAL_TOKENIZERS[id(test_case.tokenizer)]
        factory = _get_grammar_factory(mistral_tokenizer)

        if test_case.raw_lark is not None:
            grammar = test_case.raw_lark
        elif test_case.json_schema is not None and test_case.tools is None:
            template = factory.select_jinja_template(reasoning=test_case.reasoning)
            grammar = factory.get_lark_for_json_schema(template=template, json_schema=test_case.json_schema)
        else:
            template = factory.select_jinja_template(reasoning=test_case.reasoning)
            resolved_mode: ToolChoice
            if isinstance(test_case.mode, NamedToolChoice):
                resolved_mode = test_case.mode
            else:
                resolved_mode = ToolChoiceEnum(test_case.mode)
            grammar = factory.get_lark_from_jinja(
                template=template,
                mode=resolved_mode,
                tools=test_case.tools,
                json_schema=test_case.json_schema,
                parallel_tool_calls=test_case.parallel_tool_calls,
            )

        matcher = factory.get_matcher(grammar)

        assert is_tekkenizer(test_case.tokenizer)
        debug_tokens = [_token_debug_repr(test_case.tokenizer, t) for t in test_case.tokens]
        for i, token in enumerate(test_case.tokens):
            debug_bytes = _token_debug_repr(test_case.tokenizer, token)
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
            (_build_tekken_mistral_tokenizer(TokenizerVersion.v11), True),
            (_build_tekken_mistral_tokenizer(TokenizerVersion.v13, add_think=True), True),
            (
                _build_tekken_mistral_tokenizer(
                    TokenizerVersion.v15, add_think=True, model_settings_builder=_V15_MODEL_SETTINGS_BUILDER
                ),
                True,
            ),
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
            GrammarFactory(tokenizer)

    def test_get_matcher_rejects_invalid_grammar(self, v11_tekken: MistralTokenizer) -> None:
        factory = GrammarFactory(v11_tekken)
        with pytest.raises(ValueError, match="Invalid grammar"):
            factory.get_matcher("start: INVALID_RULE_REF_THAT_DOES_NOT_EXIST")


class TestConvertToolCalls:
    def test_none_mode(self) -> None:
        result = convert_tool_calls(tools=None, mode=ToolChoiceEnum.none, parallel_tool_calls=False)
        assert result == ""

    def test_none_mode_with_tools(self) -> None:
        tools = [ToolProvider.retrieve_payment_date(strict=True)]
        result = convert_tool_calls(tools=tools, mode=ToolChoiceEnum.none, parallel_tool_calls=True)
        assert result == ""

    def test_auto_mode_no_tools(self) -> None:
        result = convert_tool_calls(tools=None, mode=ToolChoiceEnum.auto, parallel_tool_calls=False)
        assert "<TOOL_CALLS>" in result
        assert "<ARGS>" in result
        assert "/.+/" in result
        assert not result.endswith(")+")

    def test_auto_mode_non_strict(self) -> None:
        tools = [ToolProvider.retrieve_payment_date(strict=False)]
        result = convert_tool_calls(tools=tools, mode=ToolChoiceEnum.auto, parallel_tool_calls=False)
        assert "<TOOL_CALLS>" in result
        assert "/.+/" in result
        assert not result.endswith(")+")

    def test_auto_mode_strict(self) -> None:
        tools = [
            ToolProvider.retrieve_payment_date(strict=True),
            ToolProvider.retrieve_payment_status(strict=True),
        ]
        result = convert_tool_calls(tools=tools, mode=ToolChoiceEnum.auto, parallel_tool_calls=False)
        assert '"retrieve_payment_date"' in result
        assert '"retrieve_payment_status"' in result
        assert not result.endswith(")+")

    def test_named_tool_choice_non_strict(self) -> None:
        named = NamedToolChoice(function=FunctionName(name="retrieve_payment_date"))
        tools = [ToolProvider.retrieve_payment_date(strict=False)]
        result = convert_tool_calls(tools=tools, mode=named, parallel_tool_calls=False)
        assert '"retrieve_payment_date"' in result
        assert "/.+/" not in result
        assert not result.endswith(")+")

    def test_named_tool_choice_strict(self) -> None:
        named = NamedToolChoice(function=FunctionName(name="retrieve_payment_date"))
        tools = [
            ToolProvider.retrieve_payment_date(strict=True),
            ToolProvider.retrieve_payment_status(strict=True),
        ]
        result = convert_tool_calls(tools=tools, mode=named, parallel_tool_calls=False)
        assert '"retrieve_payment_date"' in result
        assert '"retrieve_payment_status"' not in result
        assert not result.endswith(")+")

    def test_parallel_tool_calls(self) -> None:
        result = convert_tool_calls(tools=None, mode=ToolChoiceEnum.auto, parallel_tool_calls=True)
        assert result.startswith("(") and result.endswith(")+")

    def test_empty_params_strict_tool(self) -> None:
        tool = Tool(function=Function(name="empty_fn", parameters={}, strict=True))
        result = convert_tool_calls(tools=[tool], mode=ToolChoiceEnum.auto, parallel_tool_calls=False)
        assert '"additionalProperties": false' in result
        assert '"properties": {}' in result
        assert not result.endswith(")+")

    def test_named_tool_not_in_strict_tools_raises(self) -> None:
        named = NamedToolChoice(function=FunctionName(name="non_existent_tool"))
        tools = [ToolProvider.retrieve_payment_date(strict=True)]
        with pytest.raises(StopIteration):
            convert_tool_calls(tools=tools, mode=named, parallel_tool_calls=False)
