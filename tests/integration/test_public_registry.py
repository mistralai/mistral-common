r"""Cross-domain invariants for the public chat, transcription and speech registries.

Every public success case in every domain must own exactly one reviewed
manifest directory under ``tests/data/expected/``, and every semantic case id
must be globally unique across the chat, transcription and speech registries.
"""

from tests.integration.chat_cases import PublicChatErrorCase, PublicChatSuccessCase
from tests.integration.chat_registry import ALL_PUBLIC_CASES as CHAT_PUBLIC_CASES
from tests.integration.expected_results import _EXPECTED_ROOT
from tests.integration.speech_cases import (
    SPEECH_ERROR_CASES,
    SPEECH_SUCCESS_CASES,
    PublicSpeechErrorCase,
    PublicSpeechSuccessCase,
)
from tests.integration.transcription_cases import (
    TRANSCRIPTION_ERROR_CASES,
    TRANSCRIPTION_SUCCESS_CASES,
    PublicTranscriptionErrorCase,
    PublicTranscriptionSuccessCase,
)

_PublicCase = (
    PublicChatSuccessCase
    | PublicChatErrorCase
    | PublicTranscriptionSuccessCase
    | PublicTranscriptionErrorCase
    | PublicSpeechSuccessCase
    | PublicSpeechErrorCase
)

_ALL_PUBLIC_CASES: tuple[_PublicCase, ...] = (
    *CHAT_PUBLIC_CASES,
    *TRANSCRIPTION_SUCCESS_CASES,
    *TRANSCRIPTION_ERROR_CASES,
    *SPEECH_SUCCESS_CASES,
    *SPEECH_ERROR_CASES,
)
_ALL_SUCCESS_CASES: tuple[PublicChatSuccessCase | PublicTranscriptionSuccessCase | PublicSpeechSuccessCase, ...] = (
    *[case for case in CHAT_PUBLIC_CASES if isinstance(case, PublicChatSuccessCase)],
    *TRANSCRIPTION_SUCCESS_CASES,
    *SPEECH_SUCCESS_CASES,
)


def test_public_case_ids_are_globally_unique() -> None:
    case_ids = [case.case_id for case in _ALL_PUBLIC_CASES]

    assert len(case_ids) == len(set(case_ids))


def test_expected_manifest_directories_match_all_success_cases() -> None:
    manifest_directories = {path.name for path in _EXPECTED_ROOT.iterdir() if path.is_dir()}
    success_case_ids = {case.case_id for case in _ALL_SUCCESS_CASES}

    assert manifest_directories == success_case_ids
