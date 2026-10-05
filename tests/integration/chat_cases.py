r"""Case registry binding chat recipes to tokenizer configurations and outcomes.

One executable case pairs one reusable Python recipe with one identified
tokenizer configuration and exactly one expected result. Success cases carry
a reviewed manifest; error cases carry their expected project exception in
Python and have no manifest. Recipe/configuration reuse never attaches an
expectation map to a recipe.
"""

from dataclasses import dataclass

from mistral_common.exceptions import (
    InvalidMessageStructureException,
    MistralCommonException,
    TokenizerException,
    UnsupportedTokenizerFeatureException,
)
from tests.integration.chat_recipes import (
    MISMATCHED_TOOL_RESULTS,
    MISMATCHED_TOOL_RESULTS_FINETUNING,
    NO_TOOLS,
    PARALLEL_CALLS,
    PARALLEL_TOOL_RESULTS,
    SEVERAL_CALLS,
    WEATHER_FULL,
    WEATHER_NO_HISTORY,
    WEATHER_NO_SYSTEM,
    ChatRecipe,
)
from tests.integration.tokenizer_configurations import (
    BUNDLED_SPM_V1_TEST,
    BUNDLED_SPM_V2_FINETUNING,
    BUNDLED_SPM_V2_SERVING,
    BUNDLED_SPM_V2_TEST,
    BUNDLED_SPM_V3_TEST,
    TokenizerConfiguration,
)


@dataclass(frozen=True)
class PublicChatSuccessCase:
    """One executable public success case identified by its semantic id."""

    case_id: str
    recipe: ChatRecipe
    configuration: TokenizerConfiguration


@dataclass(frozen=True)
class PublicChatErrorCase:
    """One executable public encode-rejection case identified by its semantic id."""

    case_id: str
    recipe: ChatRecipe
    configuration: TokenizerConfiguration
    expected_exception: type[MistralCommonException]
    message_pattern: str


SAMPLE_SUCCESS_CASES: tuple[PublicChatSuccessCase, ...] = (
    PublicChatSuccessCase(case_id="chat-sample-no-tools-spm-v1", recipe=NO_TOOLS, configuration=BUNDLED_SPM_V1_TEST),
    PublicChatSuccessCase(case_id="chat-sample-no-tools-spm-v2", recipe=NO_TOOLS, configuration=BUNDLED_SPM_V2_TEST),
    PublicChatSuccessCase(case_id="chat-sample-no-tools-spm-v3", recipe=NO_TOOLS, configuration=BUNDLED_SPM_V3_TEST),
    PublicChatSuccessCase(
        case_id="chat-sample-weather-full-spm-v2", recipe=WEATHER_FULL, configuration=BUNDLED_SPM_V2_TEST
    ),
    PublicChatSuccessCase(
        case_id="chat-sample-weather-full-spm-v3", recipe=WEATHER_FULL, configuration=BUNDLED_SPM_V3_TEST
    ),
    PublicChatSuccessCase(
        case_id="chat-sample-weather-no-history-spm-v2",
        recipe=WEATHER_NO_HISTORY,
        configuration=BUNDLED_SPM_V2_TEST,
    ),
    PublicChatSuccessCase(
        case_id="chat-sample-weather-no-history-spm-v3",
        recipe=WEATHER_NO_HISTORY,
        configuration=BUNDLED_SPM_V3_TEST,
    ),
    PublicChatSuccessCase(
        case_id="chat-sample-several-calls-spm-v2", recipe=SEVERAL_CALLS, configuration=BUNDLED_SPM_V2_TEST
    ),
    PublicChatSuccessCase(
        case_id="chat-sample-several-calls-spm-v3", recipe=SEVERAL_CALLS, configuration=BUNDLED_SPM_V3_TEST
    ),
    PublicChatSuccessCase(
        case_id="chat-sample-weather-no-system-spm-v2",
        recipe=WEATHER_NO_SYSTEM,
        configuration=BUNDLED_SPM_V2_TEST,
    ),
    PublicChatSuccessCase(
        case_id="chat-sample-weather-no-system-spm-v3",
        recipe=WEATHER_NO_SYSTEM,
        configuration=BUNDLED_SPM_V3_TEST,
    ),
    PublicChatSuccessCase(
        case_id="chat-sample-parallel-calls-spm-v3", recipe=PARALLEL_CALLS, configuration=BUNDLED_SPM_V3_TEST
    ),
)

SAMPLE_ERROR_CASES: tuple[PublicChatErrorCase, ...] = (
    PublicChatErrorCase(
        case_id="chat-sample-v1-tools-rejected-test",
        recipe=WEATHER_FULL,
        configuration=BUNDLED_SPM_V1_TEST,
        expected_exception=TokenizerException,
        message_pattern=r"Tools not implemented for tokenizer V1",
    ),
    PublicChatErrorCase(
        case_id="chat-v2-parallel-results-rejected-test",
        recipe=PARALLEL_TOOL_RESULTS,
        configuration=BUNDLED_SPM_V2_TEST,
        expected_exception=UnsupportedTokenizerFeatureException,
        message_pattern=r"v2.*multiple tool results.*assistant turn",
    ),
    PublicChatErrorCase(
        case_id="chat-v2-extra-tool-result-structure-serving",
        recipe=MISMATCHED_TOOL_RESULTS,
        configuration=BUNDLED_SPM_V2_SERVING,
        expected_exception=InvalidMessageStructureException,
        message_pattern=r"Not the same number of function calls and responses",
    ),
    PublicChatErrorCase(
        case_id="chat-v2-extra-tool-result-structure-finetuning",
        recipe=MISMATCHED_TOOL_RESULTS_FINETUNING,
        configuration=BUNDLED_SPM_V2_FINETUNING,
        expected_exception=InvalidMessageStructureException,
        message_pattern=r"Not the same number of function calls and responses",
    ),
)

ALL_SUCCESS_CASES: tuple[PublicChatSuccessCase, ...] = SAMPLE_SUCCESS_CASES
