from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mistral_common.guidance.grammar_factory import GrammarFactory
from mistral_common.protocol.instruct.chunk import TextChunk, ThinkChunk
from mistral_common.protocol.instruct.messages import AssistantMessage
from mistral_common.protocol.instruct.normalize import get_normalizer
from mistral_common.protocol.instruct.tool_calls import Function, FunctionCall, Tool, ToolCall, ToolChoice
from mistral_common.protocol.instruct.validator import ValidationMode, get_validator
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy, Tokenizer, TokenizerVersion
from mistral_common.tokens.tokenizers.instruct import (
    InstructTokenizerBase,
    InstructTokenizerV11,
    InstructTokenizerV13,
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.tekken import Tekkenizer, is_tekkenizer
from tests.test_tekken import get_special_tokens, quick_vocab

EMOJI_LARK_PATH = Path(__file__).parent.parent / "data" / "emoji.lark"

Mode = Literal["auto", "any", "none"]
_AUTO_ANY: tuple[Mode, Mode] = ("auto", "any")


_NUM_SPECIAL_TOKENS = 100
_EXTRA_TOKENS = [b"a", b"b", b"c", b"f", b"de", b"he", b"llo"]


def _build_tekken_mistral_tokenizer(
    version: TokenizerVersion,
    add_think: bool = False,
) -> MistralTokenizer:
    r"""Builds a MistralTokenizer wrapping a programmatic Tekkenizer."""
    special_tokens = get_special_tokens(version, add_think=add_think)

    tekkenizer = Tekkenizer(
        quick_vocab(_EXTRA_TOKENS),
        special_tokens=special_tokens,
        pattern=r".+",
        vocab_size=256 + _NUM_SPECIAL_TOKENS,
        num_special_tokens=_NUM_SPECIAL_TOKENS,
        version=version,
    )

    match version:
        case TokenizerVersion.v11:
            instruct_tokenizer = InstructTokenizerV11(tekkenizer)
        case TokenizerVersion.v13:
            instruct_tokenizer = InstructTokenizerV13(tekkenizer)
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
            name: str
            age: int

            class Config:
                extra = "forbid"

        return Person.model_json_schema()

    @staticmethod
    def basic_dict_of_list() -> dict[str, Any]:
        class DoMerge(BaseModel):
            new_clusters: dict[str, list[str]] = Field(default_factory=dict)

            class Config:
                extra = "forbid"

        return DoMerge.model_json_schema()


class TestCase(BaseModel):
    __test__ = False
    model_config = ConfigDict(arbitrary_types_allowed=True)
    tokenizer: Tokenizer
    mode: Literal["auto", "any", "none"]
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
    r"""Encodes assistant message content into token ids.

    Args:
        instruct_tokenizer: The instruct tokenizer to use.
        content: Either a plain string or a list of TextChunk / ThinkChunk / ToolCall.

    Returns:
        The encoded token ids.
    """
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
    # Strip trailing EOS before adding tool calls
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
    mode: Mode,
    tools: list[Tool] | None,
) -> int:
    r"""Finds the index of the first token rejected by the grammar.

    Args:
        factory: The grammar factory.
        tokens: The token sequence to test.
        mode: The tool choice mode.
        tools: The tools to pass to grammar generation.

    Returns:
        The index of the first rejected token.

    Raises:
        ValueError: If all tokens are accepted.
    """
    template = factory.select_jinja_template(reasoning=False)
    grammar = factory.get_lark_from_jinja(
        template=template, mode=ToolChoice(mode), tools=tools, json_schema=None, parallel_tool_calls=True
    )
    matcher = factory.get_matcher(grammar)
    for i, token in enumerate(tokens):
        if not matcher.consume_token(token):
            return i
    raise ValueError("All tokens were accepted — expected a rejection")


def _token_debug_repr(tokenizer: Tekkenizer, token_id: int) -> str:
    r"""Returns a debug representation of a token."""
    return repr(tokenizer.id_to_byte_piece(token_id, SpecialTokenPolicy.KEEP))


def _generate_general_cases(mistral_tokenizer: MistralTokenizer) -> list[TestCase]:
    instruct_tokenizer = mistral_tokenizer.instruct_tokenizer
    tokenizer = instruct_tokenizer.tokenizer
    assert isinstance(instruct_tokenizer, InstructTokenizerBase)

    cases: list[TestCase] = []
    # For programmatic tiny-vocab tokenizers, limit to ASCII-safe content
    is_full_vocab = tokenizer.n_words > 1000
    if is_full_vocab:
        items = {
            "newline": "\n",
            "blank": "_",
            "text": "Hello!",
            "text_with_newlines": "Hello!\n\nHow are you?\nI'm fine, thanks!",
            "emojis": "😃😂😊😍😘😗😙😚😋😛😜😝🤑🤗🤔🤐😐😑😶😬",
            "japanese": "こんにちは",
            "arabic": "مرحبا بكم في عالم الذكاء الاصطناعي",
        }
    else:
        items = {
            "text_a": "abc",
            "text_b": "hello",
            "text_de": "de",
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
        # v11: plain_text_think grammar requires <think>...</think> first, so plain text fails
        # v13+: think grammar has think? (optional), so plain text passes
        reasoning_fail = 0 if tokenizer.version < TokenizerVersion.v13 else None
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


def _generate_emoji_cases(mistral_tokenizer: MistralTokenizer) -> list[TestCase]:
    instruct_tokenizer = mistral_tokenizer.instruct_tokenizer
    tokenizer = instruct_tokenizer.tokenizer
    assert isinstance(instruct_tokenizer, InstructTokenizerBase)

    # Emoji grammar needs a full vocab tokenizer (tiny programmatic tokenizers lack emoji tokens)
    if tokenizer.n_words <= 1000:
        return []

    emoji_lark = EMOJI_LARK_PATH.read_text(encoding="utf-8")
    cases: list[TestCase] = []
    # Encode raw emoji text (no assistant message wrapping) so we get pure emoji tokens
    items: dict[str, tuple[str, int | None]] = {
        "emojis_valid_a": ("😃😂😊😍😘😗😙😚😋😛😜😝🤑🤗🤔🤐😐😑😶😬", None),
        "emojis_valid_b": ("😃😃😃", None),
        "emojis_invalid_text": ("😃smile", len(tokenizer.encode("😃", bos=False, eos=False))),
        "emojis_invalid_space": ("😃 ", len(tokenizer.encode("😃", bos=False, eos=False))),
    }
    for case_name, (text, should_fail_on) in items.items():
        # Use raw encode to avoid assistant message wrapping (BOS/EOS)
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
            {"auto": None, "any": None, "none": 0},
        ),
        (
            "multi_fcall",
            [
                ToolCall(function=FunctionCall(name="hello", arguments='{"arg1": "val1", "arg2": "val2"}')),
                ToolCall(function=FunctionCall(name="hello_1", arguments='{"arg1": "val1", "arg2": "val2"}')),
                ToolCall(function=FunctionCall(name="hello_2_3", arguments='{"arg1": "val1", "arg2": "val2"}')),
            ],
            {"auto": None, "any": None, "none": 0},
        ),
        (
            "emoji_fcall",
            [
                ToolCall(
                    function=FunctionCall(name="he🧦🧦o", arguments='{"arg1": "🐱", "arg2": "🐶", "arg🧦": "🧦"}'),
                )
            ],
            {"auto": None, "any": None, "none": 0},
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
            {"auto": None, "any": None, "none": 0},
        ),
        (
            "japanese_fcall",
            [ToolCall(function=FunctionCall(name="こんにちは", arguments='{"こん": "にちは"}'))],
            {"auto": None, "any": None, "none": 0},
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

        # v11: plain_text_think mandates <think> first, so bare tool calls fail at 0
        # v13+: think grammar has think? (optional), tool calls pass; fcalls allows content? prefix
        if tokenizer.version < TokenizerVersion.v13:
            reasoning_valid_for: dict[Mode, int | None] = {"auto": 0, "any": 0, "none": 0}
        else:
            reasoning_valid_for = {"auto": None, "any": None, "none": 0}
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

    # Broken / missing args edge cases (token-level construction)
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
            {"auto": -1, "any": -1, "none": 0},
        ),
        (
            "fcall_missing_args",
            [
                tokenizer.get_special_token("[TOOL_CALLS]"),
                *tokenizer.encode("hello", bos=False, eos=False),
                tokenizer.get_special_token("[ARGS]"),
                tokenizer.get_special_token("[TOOL_CALLS]"),
            ],
            {"auto": -1, "any": -1, "none": 0},
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

    # Non-reasoning uses base grammar where "any" mode is `body: fcalls` — no content allowed
    valid_for: dict[Mode, int | None] = {"auto": None, "any": 0, "none": text_len}

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

    # v11: plain_text_think mandates <think> first, so text+fcall without think fails at 0
    # v13+: think? optional, fcalls: content? fcall, so "any" mode accepts text before tool calls
    if tokenizer.version < TokenizerVersion.v13:
        reasoning_valid_for: dict[Mode, int | None] = {"auto": 0, "any": 0, "none": 0}
    else:
        reasoning_valid_for = {"auto": None, "any": None, "none": text_len}
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
            {"auto": 0, "any": 0, "none": 0},
        ),
        (
            "think_without_response",
            [TextChunk(text="<think>Hello!</think>")],
            {"auto": -1, "any": -1, "none": -1},
        ),
        (
            "unclosed_think",
            [TextChunk(text="<think>Hello!")],
            {"auto": -1, "any": -1, "none": -1},
        ),
        (
            "plain_think_with_response",
            [TextChunk(text="<think>Hello!</think>World!")],
            {
                "auto": None,
                "any": len(tokenizer.encode("<think>Hello!</think>", bos=False, eos=False)),
                "none": None,
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
            },
        ),
        (
            "think_with_text_and_tool_call",
            [
                TextChunk(text="<think>Hello!</think>Ho!"),
                ToolCall(function=FunctionCall(name="hello", arguments='{"arg1": "val1", "arg2": "val2"}')),
            ],
            {
                "auto": len(tokenizer.encode("<think>Hello!</think>Ho!", bos=False, eos=False)),
                "any": len(tokenizer.encode("<think>Hello!</think>", bos=False, eos=False)),
                "none": len(tokenizer.encode("<think>Hello!</think>Ho!", bos=False, eos=False)),
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
            {"auto": None, "any": -1, "none": None},
        ),
        (
            "plain_think",
            [ThinkChunk(thinking="Hello!")],
            {"auto": -1, "any": -1, "none": -1},
        ),
        (
            "plain_think_with_response",
            [ThinkChunk(thinking="Hello!"), TextChunk(text="World!")],
            {"auto": None, "any": -1, "none": None},
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
    for mode in _AUTO_ANY:
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

    # v11: plain_text_think mandates <think> first, so bare tool calls fail at 0
    # v13+: think? optional, tool calls pass
    if tokenizer.version < TokenizerVersion.v13:
        reasoning_valid_for: dict[Mode, int | None] = {"auto": 0, "any": 0, "none": 0}
    else:
        reasoning_valid_for = {"auto": None, "any": None, "none": 0}
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

    # Multi tool call should fail when parallel_tool_calls=False
    # Each tool call is a separate [TOOL_CALLS]...[ARGS]... sequence;
    # the second [TOOL_CALLS] is where it fails.
    multi_calls = [
        ToolCall(function=FunctionCall(name="fn1", arguments='{"arg1": "val1", "arg2": "val2"}')),
        ToolCall(function=FunctionCall(name="fn2", arguments='{"arg1": "val1", "arg2": "val2"}')),
    ]
    multi_tokens = _encode_content(instruct_tokenizer, multi_calls)
    single_tokens_with_eos = _encode_content(instruct_tokenizer, [multi_calls[0]])
    fail_idx = len(single_tokens_with_eos) - 1

    for mode in _AUTO_ANY:
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
    for mode in _AUTO_ANY:
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
    for mode in _AUTO_ANY:
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
    bogus_start = _find_first_rejection(factory, wrong_args_tokens, mode="auto", tools=tools_strict)
    for mode in _AUTO_ANY:
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
    fail_on_name = _find_first_rejection(factory, wrong_name_tokens, mode="auto", tools=tools_strict)
    for mode in _AUTO_ANY:
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
    for mode in _AUTO_ANY:
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
    # v11: plain_text_think mandates <think> first, so all bare tool calls fail at 0
    # v13+: think? optional, tool calls behave as without reasoning
    if tokenizer.version < TokenizerVersion.v13:
        for mode in _AUTO_ANY:
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
        for mode in _AUTO_ANY:
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
        for mode in _AUTO_ANY:
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


def _generate_json_schema(mistral_tokenizer: MistralTokenizer) -> list[TestCase]:
    instruct_tokenizer = mistral_tokenizer.instruct_tokenizer
    tokenizer = instruct_tokenizer.tokenizer
    assert isinstance(instruct_tokenizer, InstructTokenizerBase)

    # JSON schema validation requires a full vocab tokenizer
    if tokenizer.n_words <= 1000:
        return []

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
            "basic_person_invalid",
            '{"age": "John", "name": 30}',
            1,
            SchemaProvider.basic_person(),
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
    return cases


def _generate_cases(mistral_tokenizer: MistralTokenizer, factory: GrammarFactory) -> list[TestCase]:
    instruct_tokenizer = mistral_tokenizer.instruct_tokenizer
    assert isinstance(instruct_tokenizer, InstructTokenizerBase)
    tokenizer_version = instruct_tokenizer.tokenizer.version

    cases = _generate_general_cases(mistral_tokenizer)
    cases += _generate_emoji_cases(mistral_tokenizer)
    cases += _generate_json_schema(mistral_tokenizer)
    cases += _generate_cases_tool_calls(mistral_tokenizer)
    cases += _generate_single_tool_call(mistral_tokenizer)
    cases += _generate_strict_tool_calls(mistral_tokenizer, factory)
    cases += _generate_cases_text_and_tool_calls(mistral_tokenizer)

    if not tokenizer_version < TokenizerVersion.v13:
        cases += _generate_cases_thinking(mistral_tokenizer)
    else:
        cases += _generate_cases_thinking_v11(mistral_tokenizer)

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
            grammar = factory.get_lark_for_json_schema(json_schema=test_case.json_schema)
        else:
            template = factory.select_jinja_template(reasoning=test_case.reasoning)
            grammar = factory.get_lark_from_jinja(
                template=template,
                mode=ToolChoice(test_case.mode),
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

    @pytest.mark.parametrize(
        ("tokenizer", "expected"),
        [
            (MistralTokenizer.v1(), False),
            (MistralTokenizer.v3(is_tekken=True), False),
            (_build_tekken_mistral_tokenizer(TokenizerVersion.v11), True),
            (_build_tekken_mistral_tokenizer(TokenizerVersion.v13, add_think=True), True),
        ],
    )
    def test_grammar_factory_is_supported(self, tokenizer: MistralTokenizer, expected: bool) -> None:
        assert GrammarFactory.is_supported(tokenizer) is expected
