r"""Released-profile public chat behavior and full-result expectations."""

from collections.abc import Callable
from unittest.mock import patch

import pytest

from mistral_common.protocol.instruct.chunk import AudioURLType
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.fixtures.audio import get_dummy_audio_url_chunk
from tests.integration.chat_cases import (
    PublicChatErrorCase,
    PublicChatSuccessCase,
)
from tests.integration.chat_released_cases import (
    RELEASED_ERROR_CASES,
    RELEASED_SUCCESS_CASES,
    RELEASED_USER_AUDIO_URL_CASE,
)
from tests.integration.tokenizer_configurations import TokenizerConfiguration
from tests.integration.utils import encode_and_expect_error, encode_and_verify

_RELEASED_SUCCESS_CASES_WITHOUT_AUDIO_URL = tuple(
    case for case in RELEASED_SUCCESS_CASES if case is not RELEASED_USER_AUDIO_URL_CASE
)


@pytest.mark.parametrize(
    argnames="case",
    argvalues=_RELEASED_SUCCESS_CASES_WITHOUT_AUDIO_URL,
    ids=lambda case: case.case_id,
)
def test_released_profile_public_chat_success(
    case: PublicChatSuccessCase,
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    encode_and_verify(case=case, public_tokenizer=public_tokenizer)


def test_released_user_audio_url_uses_no_network(
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    assert get_dummy_audio_url_chunk().get_url_type() == AudioURLType.base64
    with patch(
        target="mistral_common.tokens.tokenizers.audio._requests_lib.get",
        side_effect=AssertionError("network used"),
    ):
        encode_and_verify(case=RELEASED_USER_AUDIO_URL_CASE, public_tokenizer=public_tokenizer)


@pytest.mark.parametrize(argnames="case", argvalues=RELEASED_ERROR_CASES, ids=lambda case: case.case_id)
def test_released_profile_public_chat_error(
    case: PublicChatErrorCase,
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    encode_and_expect_error(case=case, public_tokenizer=public_tokenizer)
