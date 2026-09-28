r"""Public chat workflow cases for pinned and synthetic v13 configurations."""

from collections.abc import Callable

import pytest

from mistral_common.tokens.tokenizers.base import Tokenized
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.integration.chat_cases import PublicChatSuccessCase
from tests.integration.chat_v13_cases import V13_SUCCESS_CASES
from tests.integration.expected_results import assert_public_success, load_expected_success
from tests.integration.tokenizer_configurations import TokenizerConfiguration
from tests.utils import decode_keep

_AUDIO_TEXT = "<s>[SYSTEM_PROMPT]hello[/SYSTEM_PROMPT][INST][BEGIN_AUDIO][AUDIO][AUDIO][/INST]"


def _encode_and_verify(
    case: PublicChatSuccessCase,
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> tuple[MistralTokenizer, Tokenized, str]:
    request = case.recipe.build()
    tokenizer = public_tokenizer(case.configuration)
    tokenized: Tokenized = tokenizer.encode_chat_completion(request)
    decoded_text = decode_keep(tokenizer=tokenizer, tokenized=tokenized)
    expected = load_expected_success(
        case_id=case.case_id,
        tokenizer_configuration_id=case.configuration.configuration_id,
    )
    assert_public_success(expected=expected, tokenized=tokenized, decoded_text=decoded_text)
    return tokenizer, tokenized, decoded_text


@pytest.mark.parametrize("case", V13_SUCCESS_CASES, ids=lambda case: case.case_id)
def test_public_chat_v13_success(
    case: PublicChatSuccessCase,
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    tokenizer, tokenized, decoded_text = _encode_and_verify(case=case, public_tokenizer=public_tokenizer)

    if case.case_id == "chat-v13-think-order":
        assert "[THINK]TS[/THINK]" in decoded_text
        assert "[THINK]T1[/THINK]" in decoded_text
        assert decoded_text.index("[TOOL_RESULTS]R1") < decoded_text.index("[TOOL_RESULTS]R2")
    elif case.case_id == "chat-v13-reversed-results":
        assert decoded_text.index("[TOOL_RESULTS]R1") < decoded_text.index("[TOOL_RESULTS]R2")
    elif case.case_id in {"chat-v13-call-id-x", "chat-v13-call-id-slash"}:
        tool_call_id = "x" if case.case_id.endswith("-x") else "call/id-1"
        assert tool_call_id not in decoded_text
        assert "[TOOL_CALLS]f[ARGS]{}" in decoded_text
        assert "[TOOL_RESULTS]b[/TOOL_RESULTS]" in decoded_text
    elif case.case_id == "chat-v13-prefixed-final":
        eos_id = tokenizer.instruct_tokenizer.tokenizer.eos_id
        assert tokenized.tokens[-1] != eos_id
    else:
        assert decoded_text == _AUDIO_TEXT
        assert len(tokenized.audios) == 1
