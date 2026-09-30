import base64

import llguidance as llg
import pytest

from mistral_common.guidance.grammar_factory import GrammarFactory
from mistral_common.protocol.instruct.normalize import get_normalizer
from mistral_common.protocol.instruct.request import ReasoningEffort
from mistral_common.protocol.instruct.validator import ValidationMode, get_validator
from mistral_common.tokens.tokenizers.base import SpecialTokens, TokenizerVersion
from mistral_common.tokens.tokenizers.instruct import (
    InstructTokenizerBase,
    InstructTokenizerV11,
    InstructTokenizerV13,
    InstructTokenizerV15,
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.model_settings_builder import EnumBuilder, ModelSettingsBuilder
from mistral_common.tokens.tokenizers.tekken import SpecialTokenInfo, Tekkenizer, TokenInfo

_NUM_SPECIAL_TOKENS = 100
_INSTRUCT_TOKENIZERS: dict[TokenizerVersion, type[InstructTokenizerBase]] = {
    TokenizerVersion.v11: InstructTokenizerV11,
    TokenizerVersion.v13: InstructTokenizerV13,
    TokenizerVersion.v15: InstructTokenizerV15,
}


def _special_tokens(version: TokenizerVersion) -> list[SpecialTokenInfo]:
    named = {32: SpecialTokens.args, 33: SpecialTokens.call_id}
    if version >= TokenizerVersion.v13:
        named |= {35: SpecialTokens.begin_think, 36: SpecialTokens.end_think}
    if version >= TokenizerVersion.v15:
        named |= {37: SpecialTokens.begin_model_settings, 38: SpecialTokens.end_model_settings}
    special_tokens = list(Tekkenizer.DEPRECATED_SPECIAL_TOKENS)
    special_tokens += [
        SpecialTokenInfo(rank=rank, token_str=named.get(rank, f"<SPECIAL_{rank}>"), is_control=True)
        for rank in range(len(special_tokens), max(named) + 1)
    ]
    return special_tokens


def _build_mistral_tokenizer(version: TokenizerVersion) -> MistralTokenizer:
    vocab = [TokenInfo(rank=i, token_bytes=base64.b64encode(bytes([i])).decode(), token_str=chr(i)) for i in range(256)]
    model_settings_builder = None
    if version.supports_model_settings:
        model_settings_builder = ModelSettingsBuilder(
            reasoning_effort=EnumBuilder[ReasoningEffort](values=list(ReasoningEffort), accepts_none=True, default=None)
        )
    tekkenizer = Tekkenizer(
        vocab,
        special_tokens=_special_tokens(version=version),
        pattern=r"(?s:.+)",
        vocab_size=len(vocab) + _NUM_SPECIAL_TOKENS,
        num_special_tokens=_NUM_SPECIAL_TOKENS,
        version=version,
        model_settings_builder=model_settings_builder,
    )
    return MistralTokenizer(
        _INSTRUCT_TOKENIZERS[version](tekkenizer),
        validator=get_validator(version, mode=ValidationMode.test),
        request_normalizer=get_normalizer(version, tekkenizer.model_settings_builder),
    )


def _accepts(mistral_tokenizer: MistralTokenizer, factory: GrammarFactory, grammar: str, text: str) -> bool:
    tokenizer = mistral_tokenizer.instruct_tokenizer.tokenizer
    matcher = llg.LLMatcher(factory.llg_tokenizer, grammar)
    tokens = [*tokenizer.encode(text, bos=False, eos=False), tokenizer.eos_id]
    return all(matcher.consume_token(token) for token in tokens) and not matcher.is_error()


@pytest.mark.parametrize("version", list(_INSTRUCT_TOKENIZERS), ids=lambda version: version.value)
def test_get_lark_for_json_schema_empty_schema_only_accepts_json(version: TokenizerVersion) -> None:
    mistral_tokenizer = _build_mistral_tokenizer(version=version)
    factory = GrammarFactory(mistral_tokenizer)
    grammar = factory.get_lark_for_json_schema(template=factory.select_jinja_template(), json_schema={})

    assert _accepts(mistral_tokenizer=mistral_tokenizer, factory=factory, grammar=grammar, text='{"a": 1}')
    assert _accepts(mistral_tokenizer=mistral_tokenizer, factory=factory, grammar=grammar, text="[1, 2]")
    assert not _accepts(mistral_tokenizer=mistral_tokenizer, factory=factory, grammar=grammar, text="hello")
