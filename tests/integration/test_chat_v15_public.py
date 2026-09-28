r"""Public chat workflow cases for pinned and synthetic v15 configurations."""

from collections.abc import Callable

import pytest

from mistral_common.tokens.tokenizers.base import Tokenized
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.integration.chat_cases import PublicChatErrorCase, PublicChatSuccessCase
from tests.integration.chat_v15_cases import V15_ERROR_CASES, V15_SUCCESS_CASES
from tests.integration.expected_results import assert_public_success, load_expected_success
from tests.integration.tokenizer_configurations import TokenizerConfiguration
from tests.utils import decode_keep

EXPECTED_TEXT_TOOL_AUDIO: str = (
    r"<s>"
    r'[AVAILABLE_TOOLS][{"type": "function", "function": {"name": "fn",'
    r' "description": "test", "parameters": {}}}]'
    r'[/AVAILABLE_TOOLS][MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]'
    r"[INST]Use the tool[/INST]"
    r"[TOOL_CALLS]fn[ARGS]{}</s>"
    r"[TOOL_RESULTS]result[BEGIN_AUDIO][AUDIO][AUDIO][/TOOL_RESULTS]"
)

EXPECTED_TEXT_SYSTEM_AUDIO: str = (
    r"<s>[SYSTEM_PROMPT]System with content[BEGIN_AUDIO][AUDIO][AUDIO][/SYSTEM_PROMPT]"
    r'[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]'
    r"[INST]Hello[/INST]"
)

EXPECTED_TEXT_USER_AUDIO: str = (
    r"<s>"
    r'[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]'
    r"[INST]Here is content[BEGIN_AUDIO][AUDIO][AUDIO][/INST]"
)

_SETTINGS_MARKERS: dict[str, str] = {
    "chat-v15-policy-absent-both": '[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]',
    "chat-v15-policy-none-both": '[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]',
    "chat-v15-policy-high-both": '[MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS]',
    "chat-v15-policy-none-only": '[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]',
    "chat-v15-default-absent": '[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]',
    "chat-v15-default-none": '[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]',
    "chat-v15-default-high": '[MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS]',
    "chat-v15-no-default-none": '[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]',
    "chat-v15-no-default-high": '[MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS]',
    "chat-v15-tool-audio": '[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]',
    "chat-v15-tool-audio-url": '[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]',
    "chat-v15-tool-image-url": '[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]',
    "chat-v15-system-audio": '[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]',
    "chat-v15-user-audio": '[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]',
    "chat-v15-user-audio-url": '[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]',
    "chat-v15-user-image-url": '[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]',
}
_NO_SETTINGS_MARKER = {"chat-v15-ignore-absent-empty", "chat-v15-ignore-none-no-builder", "chat-v15-no-default-absent"}
_AUDIO_TEXT: dict[str, str] = {
    "chat-v15-tool-audio": EXPECTED_TEXT_TOOL_AUDIO,
    "chat-v15-tool-audio-url": EXPECTED_TEXT_TOOL_AUDIO,
    "chat-v15-system-audio": EXPECTED_TEXT_SYSTEM_AUDIO,
    "chat-v15-user-audio": EXPECTED_TEXT_USER_AUDIO,
    "chat-v15-user-audio-url": EXPECTED_TEXT_USER_AUDIO,
}


def _encode_and_verify(
    *,
    case: PublicChatSuccessCase,
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> tuple[MistralTokenizer, Tokenized, str]:
    r"""Build, publicly encode and compare one v15 case to its full manifest.

    Args:
        case: Recipe and tokenizer configuration bound to this success.
        public_tokenizer: Session-cached loader for verified configurations.

    Returns:
        The tokenizer, tokenized output, and full decoded text.
    """
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


@pytest.mark.parametrize("case", V15_SUCCESS_CASES, ids=lambda case: case.case_id)
def test_public_chat_v15_success(
    case: PublicChatSuccessCase,
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    tokenizer, tokenized, decoded_text = _encode_and_verify(case=case, public_tokenizer=public_tokenizer)

    if case.case_id in {"chat-v15-call-id-x", "chat-v15-call-id-slash"}:
        tool_call_id = "x" if case.case_id.endswith("-x") else "call/id-1"
        assert tool_call_id not in decoded_text
        assert "[TOOL_CALLS]f[ARGS]{}" in decoded_text
        assert "[TOOL_RESULTS]b[/TOOL_RESULTS]" in decoded_text

    if case.case_id in _SETTINGS_MARKERS:
        assert _SETTINGS_MARKERS[case.case_id] in decoded_text
    if case.case_id in _NO_SETTINGS_MARKER:
        assert "[MODEL_SETTINGS]" not in decoded_text

    if case.case_id in {"chat-v15-tool-image-url", "chat-v15-user-image-url"}:
        assert len(tokenized.images) == 1
        assert tokenized.images[0].shape == (3, 28, 28)

    if case.case_id in _AUDIO_TEXT:
        assert decoded_text == _AUDIO_TEXT[case.case_id]
        assert len(tokenized.audios) == 1

    if case.case_id == "chat-v15-prefixed-final":
        eos_id = tokenizer.instruct_tokenizer.tokenizer.eos_id
        assert tokenized.tokens[-1] != eos_id


@pytest.mark.parametrize("case", V15_ERROR_CASES, ids=lambda case: case.case_id)
def test_public_chat_v15_error(
    case: PublicChatErrorCase,
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    request = case.recipe.build()
    tokenizer = public_tokenizer(case.configuration)

    with pytest.raises(case.expected_exception, match=case.message_pattern):
        tokenizer.encode_chat_completion(request)
