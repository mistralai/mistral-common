r"""Central registry for all public chat success and error cases."""

from tests.integration.chat_cases import (
    SAMPLE_ERROR_CASES,
    SAMPLE_SUCCESS_CASES,
    PublicChatErrorCase,
    PublicChatSuccessCase,
)
from tests.integration.chat_released_cases import RELEASED_ERROR_CASES, RELEASED_SUCCESS_CASES
from tests.integration.chat_v3_cases import V3_SUCCESS_CASES
from tests.integration.chat_v7_cases import V7_SUCCESS_CASES
from tests.integration.chat_v13_cases import V13_SUCCESS_CASES
from tests.integration.chat_v15_cases import V15_ERROR_CASES, V15_SUCCESS_CASES

ALL_SUCCESS_CASES: tuple[PublicChatSuccessCase, ...] = (
    *SAMPLE_SUCCESS_CASES,
    *V3_SUCCESS_CASES,
    *V7_SUCCESS_CASES,
    *V13_SUCCESS_CASES,
    *V15_SUCCESS_CASES,
    *RELEASED_SUCCESS_CASES,
)

ALL_PUBLIC_CASES: tuple[PublicChatSuccessCase | PublicChatErrorCase, ...] = (
    *ALL_SUCCESS_CASES,
    *SAMPLE_ERROR_CASES,
    *V15_ERROR_CASES,
    *RELEASED_ERROR_CASES,
)
