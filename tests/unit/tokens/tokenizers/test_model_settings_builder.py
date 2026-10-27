from pathlib import Path

import pytest
from pydantic import ValidationError

from mistral_common.exceptions import InvalidRequestException
from mistral_common.protocol.instruct.request import ChatCompletionRequest, ModelSettings, ReasoningEffort
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.model_settings_builder import ModelSettingsBuilder, ReasoningEffortBuilder
from tests.integrations.chat_templates.helpers import TestConfig, build_tekken_json


class TestReasoningEffortBuilder:
    def test_accepts_configured_values_and_builds_default(self) -> None:
        builder = ReasoningEffortBuilder(
            accepts_none=True,
            default=ReasoningEffort.medium,
            values=[ReasoningEffort.low, ReasoningEffort.medium, ReasoningEffort.max],
        )

        settings = ModelSettingsBuilder(reasoning_effort=builder).build_settings(
            ChatCompletionRequest(messages=[], reasoning_effort=None)
        )

        assert settings == ModelSettings(reasoning_effort=ReasoningEffort.medium)

    def test_accepts_none_is_required(self) -> None:
        with pytest.raises(ValidationError, match="accepts_none"):
            ReasoningEffortBuilder(
                accepts_none=False,
                default=None,
                values=[ReasoningEffort.low],
            )

    def test_default_is_required(self) -> None:
        with pytest.raises(ValidationError, match="must declare a default"):
            ReasoningEffortBuilder(
                accepts_none=True,
                default=None,
                values=[ReasoningEffort.low],
            )

    def test_rejects_effort_outside_configured_values(self) -> None:
        builder = ReasoningEffortBuilder(
            accepts_none=True,
            default=ReasoningEffort.low,
            values=[ReasoningEffort.low, ReasoningEffort.medium],
        )

        with pytest.raises(InvalidRequestException, match="should be one of"):
            builder.build_value(field_name="reasoning_effort", value=ReasoningEffort.max)

    def test_tokenizer_json_loads_reasoning_effort_builder(self, tmp_path: Path) -> None:
        path = build_tekken_json(config=TestConfig(version=TokenizerVersion.v15), output_dir=tmp_path)

        tokenizer = MistralTokenizer.from_file(str(path))

        assert isinstance(
            tokenizer.instruct_tokenizer.tokenizer.model_settings_builder.reasoning_effort, ReasoningEffortBuilder
        )
