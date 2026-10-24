r"""Integration tests for the reasoning effort remapper.

An `EnumBuilder` may carry a `mapping` from the values exposed on requests to
the model-side values encoded in the model settings. These tests cover the two
v15 releases: the previous one exposing `none` and `high`, both supported
as-is without any remapping, and a new one exposing `low`, `high` and `max`
while the model only supports `low`, `medium` and `high`.

For each release, `MistralTokenizer.encode_chat_completion` emits the
model-side value, the generated chat template renders the same value, and
both agree for every exposed value. An unset (`None`) effort resolves to the
release's default exposed value, remapped like any request value: `none`
as-is for the previous release, `high` to `medium` for the new one.
"""

from pathlib import Path

import pytest

from mistral_common.integrations.chat_templates.chat_templates import (
    convert_tokenizer_to_chat_template,
    generate_chat_template,
)
from mistral_common.integrations.chat_templates.template_generator import TemplateConfig
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest, ReasoningEffort
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.integrations.chat_templates.helpers import (
    TestConfig,
    build_render_args,
    build_tekken_json,
    encode_mistral_common,
    render_template,
)
from tests.utils import decode_keep

# Previous v15 release: exposes none and high, both supported as-is.
_NONE_HIGH_REASONING_EFFORT_VALUES: tuple[str, ...] = ("none", "high")
_NONE_HIGH_REASONING_EFFORT_DEFAULT = "none"
_NONE_HIGH_SETTINGS: dict[str, str] = {
    value: f'[MODEL_SETTINGS]{{"reasoning_effort": "{value}"}}[/MODEL_SETTINGS]'
    for value in _NONE_HIGH_REASONING_EFFORT_VALUES
}

# New v15 release: exposes low, high and max, remapped onto the low, medium and
# high the model supports. The default is the exposed "high": an unset effort is
# remapped to "medium".
_LOW_HIGH_MAX_REASONING_EFFORT_VALUES: tuple[str, ...] = ("low", "high", "max")
_LOW_HIGH_MAX_REASONING_EFFORT_MAPPING: dict[str, str] = {"low": "low", "high": "medium", "max": "high"}
_LOW_HIGH_MAX_REASONING_EFFORT_DEFAULT = "high"
_LOW_HIGH_MAX_DEFAULT_MODEL_SIDE_VALUE = _LOW_HIGH_MAX_REASONING_EFFORT_MAPPING[_LOW_HIGH_MAX_REASONING_EFFORT_DEFAULT]
_LOW_HIGH_MAX_MAPPED_DEFAULT_SETTINGS = (
    f'[MODEL_SETTINGS]{{"reasoning_effort": "{_LOW_HIGH_MAX_DEFAULT_MODEL_SIDE_VALUE}"}}[/MODEL_SETTINGS]'
)

_MESSAGES = [{"role": "user", "content": "Hello"}]


def _reasoning_effort_tekken_path(
    tmp_path: Path,
    values: tuple[str, ...],
    mapping: dict[str, str] | None,
    default: str | None,
) -> Path:
    r"""Write a v15 tekken.json whose reasoning effort builder exposes the given values."""
    config = TestConfig(
        version=TokenizerVersion.v15,
        model_settings_fields=frozenset({"reasoning_effort"}),
        reasoning_effort_values=values,
        reasoning_effort_mapping=mapping,
        reasoning_effort_default=default,
    )
    return build_tekken_json(config=config, output_dir=tmp_path)


def _none_high_reasoning_effort_tekken_path(tmp_path: Path) -> Path:
    r"""Write a v15 tekken.json for the previous release: none and high, no remapping."""
    return _reasoning_effort_tekken_path(
        tmp_path=tmp_path,
        values=_NONE_HIGH_REASONING_EFFORT_VALUES,
        mapping=None,
        default=_NONE_HIGH_REASONING_EFFORT_DEFAULT,
    )


def _low_high_max_reasoning_effort_tekken_path(tmp_path: Path) -> Path:
    r"""Write a v15 tekken.json for the new release: exposes low, high and max, remapped."""
    return _reasoning_effort_tekken_path(
        tmp_path=tmp_path,
        values=_LOW_HIGH_MAX_REASONING_EFFORT_VALUES,
        mapping=_LOW_HIGH_MAX_REASONING_EFFORT_MAPPING,
        default=_LOW_HIGH_MAX_REASONING_EFFORT_DEFAULT,
    )


def _generate_v15_template(
    reasoning_effort_values: tuple[str, ...] | None = None,
    reasoning_effort_mapping: dict[str, str] | None = None,
    reasoning_effort_default: str | None = None,
) -> str:
    r"""Generate the v15 template with reasoning effort model settings, optionally remapped."""
    return generate_chat_template(
        spm=False,
        tokenizer_version=TokenizerVersion.v15,
        image_support=False,
        audio_support=False,
        thinking_support=False,
        default_system_prompt=None,
        plain_thinking_support=False,
        use_special_token_variables=True,
        model_settings_fields=frozenset({"reasoning_effort"}),
        reasoning_effort_values=reasoning_effort_values,
        reasoning_effort_mapping=reasoning_effort_mapping,
        reasoning_effort_default=reasoning_effort_default,
    )


class TestTemplateConfigMapping:
    def test_values_require_reasoning_effort_field(self) -> None:
        with pytest.raises(ValueError, match="Reasoning effort values require the reasoning_effort"):
            TemplateConfig(
                version=TokenizerVersion.v15,
                model_settings_fields=frozenset({"json_schema"}),
                reasoning_effort_values=_NONE_HIGH_REASONING_EFFORT_VALUES,
            )

    def test_values_require_model_settings_support(self) -> None:
        with pytest.raises(ValueError, match="Reasoning effort values require the reasoning_effort"):
            TemplateConfig(version=TokenizerVersion.v13, reasoning_effort_values=_NONE_HIGH_REASONING_EFFORT_VALUES)

    def test_empty_values_rejected(self) -> None:
        with pytest.raises(ValueError, match="Reasoning effort values must be None or a non-empty tuple"):
            TemplateConfig(version=TokenizerVersion.v15, reasoning_effort_values=())

    def test_values_must_be_standard_efforts(self) -> None:
        with pytest.raises(ValueError, match="Unknown reasoning effort values: minimal"):
            TemplateConfig(version=TokenizerVersion.v15, reasoning_effort_values=("none", "minimal"))

    def test_values_must_match_mapping_keys(self) -> None:
        with pytest.raises(ValueError, match="Reasoning effort values must match the mapping keys"):
            TemplateConfig(
                version=TokenizerVersion.v15,
                reasoning_effort_values=("low", "high"),
                reasoning_effort_mapping=_LOW_HIGH_MAX_REASONING_EFFORT_MAPPING,
            )

    def test_mapping_requires_reasoning_effort_field(self) -> None:
        with pytest.raises(ValueError, match="Reasoning effort mapping requires the reasoning_effort"):
            TemplateConfig(
                version=TokenizerVersion.v15,
                model_settings_fields=frozenset({"json_schema"}),
                reasoning_effort_mapping=_LOW_HIGH_MAX_REASONING_EFFORT_MAPPING,
            )

    def test_mapping_requires_model_settings_support(self) -> None:
        with pytest.raises(ValueError, match="Reasoning effort mapping requires the reasoning_effort"):
            TemplateConfig(
                version=TokenizerVersion.v13, reasoning_effort_mapping=_LOW_HIGH_MAX_REASONING_EFFORT_MAPPING
            )

    def test_empty_mapping_rejected(self) -> None:
        with pytest.raises(ValueError, match="Reasoning effort mapping must be None or a non-empty mapping"):
            TemplateConfig(version=TokenizerVersion.v15, reasoning_effort_mapping={})

    def test_mapping_values_must_be_standard_efforts(self) -> None:
        with pytest.raises(ValueError, match="Unknown reasoning effort values in mapping: minimal"):
            TemplateConfig(version=TokenizerVersion.v15, reasoning_effort_mapping={"none": "minimal"})

    def test_mapping_must_preserve_reasoning_effort_order(self) -> None:
        with pytest.raises(ValueError, match="must preserve the order of reasoning efforts"):
            TemplateConfig(version=TokenizerVersion.v15, reasoning_effort_mapping={"none": "low", "low": "none"})

    def test_default_requires_reasoning_effort_field(self) -> None:
        with pytest.raises(ValueError, match="Reasoning effort default requires the reasoning_effort"):
            TemplateConfig(
                version=TokenizerVersion.v15,
                model_settings_fields=frozenset({"json_schema"}),
                reasoning_effort_default="none",
            )

    def test_default_must_be_an_exposed_value_with_mapping(self) -> None:
        with pytest.raises(ValueError, match="Reasoning effort default must be one of the exposed values"):
            TemplateConfig(
                version=TokenizerVersion.v15,
                reasoning_effort_mapping=_LOW_HIGH_MAX_REASONING_EFFORT_MAPPING,
                reasoning_effort_default="none",
            )

    def test_default_must_be_an_exposed_value_with_values(self) -> None:
        with pytest.raises(ValueError, match="Reasoning effort default must be one of the exposed values"):
            TemplateConfig(
                version=TokenizerVersion.v15,
                reasoning_effort_values=_NONE_HIGH_REASONING_EFFORT_VALUES,
                reasoning_effort_default="low",
            )

    def test_default_must_be_a_standard_value_without_mapping(self) -> None:
        with pytest.raises(ValueError, match="Reasoning effort default must be one of the exposed values"):
            TemplateConfig(version=TokenizerVersion.v15, reasoning_effort_default="minimal")


class TestGeneratedTemplateMapping:
    def test_mapping_is_baked_into_template(self) -> None:
        template = _generate_v15_template(reasoning_effort_mapping=_LOW_HIGH_MAX_REASONING_EFFORT_MAPPING)
        assert "{%- set reasoning_effort_mapping = {'low': 'low', 'high': 'medium', 'max': 'high'} %}" in template
        assert "{%- if reasoning_effort not in reasoning_effort_mapping %}" in template
        assert "{%- set reasoning_effort = reasoning_effort_mapping[reasoning_effort] %}" in template

    def test_template_without_mapping_validates_the_full_ladder(self) -> None:
        template = _generate_v15_template()
        assert "reasoning_effort_mapping" not in template
        assert "{#- The model was trained" not in template
        assert "{%- if reasoning_effort not in ['none', 'low', 'medium', 'high', 'xhigh', 'max'] %}" in template

    def test_template_without_mapping_validates_exposed_values(self) -> None:
        template = _generate_v15_template(reasoning_effort_values=_NONE_HIGH_REASONING_EFFORT_VALUES)
        assert "reasoning_effort_mapping" not in template
        assert "{#- The model was trained" not in template
        assert "{%- if reasoning_effort not in ['none', 'high'] %}" in template
        assert 'reasoning_effort must be either "none" or "high"' in template

    def test_mapping_comment_explains_the_remap(self) -> None:
        template = _generate_v15_template(reasoning_effort_mapping=_LOW_HIGH_MAX_REASONING_EFFORT_MAPPING)
        comment = (
            "{#- The model was trained on values that do not produce a standard number of tokens for "
            "the community, so exposed reasoning efforts are remapped onto the trained values to "
            "bring users to known territory. -#}"
        )
        assert comment in template
        assert template.index(comment) < template.index("{%- set reasoning_effort_mapping")

    def test_default_is_baked_into_template(self) -> None:
        template = _generate_v15_template(
            reasoning_effort_mapping=_LOW_HIGH_MAX_REASONING_EFFORT_MAPPING,
            reasoning_effort_default=_LOW_HIGH_MAX_REASONING_EFFORT_DEFAULT,
        )
        assert "{%- set reasoning_effort = 'high' %}" in template

    def test_template_without_default_falls_back_to_none(self) -> None:
        template = _generate_v15_template()
        assert "{%- set reasoning_effort = 'none' %}" in template

    @pytest.mark.parametrize("reasoning_effort", _NONE_HIGH_REASONING_EFFORT_VALUES)
    def test_render_without_mapping_keeps_exposed_value(self, reasoning_effort: str) -> None:
        template = _generate_v15_template(
            reasoning_effort_values=_NONE_HIGH_REASONING_EFFORT_VALUES,
            reasoning_effort_default=_NONE_HIGH_REASONING_EFFORT_DEFAULT,
        )
        rendered = render_template(template, messages=_MESSAGES, reasoning_effort=reasoning_effort)
        assert _NONE_HIGH_SETTINGS[reasoning_effort] in rendered

    def test_render_unset_effort_uses_default(self) -> None:
        template = _generate_v15_template(
            reasoning_effort_values=_NONE_HIGH_REASONING_EFFORT_VALUES,
            reasoning_effort_default=_NONE_HIGH_REASONING_EFFORT_DEFAULT,
        )
        rendered = render_template(template, messages=_MESSAGES)
        assert _NONE_HIGH_SETTINGS[_NONE_HIGH_REASONING_EFFORT_DEFAULT] in rendered

    def test_render_rejects_value_outside_exposed_values(self) -> None:
        template = _generate_v15_template(reasoning_effort_values=_NONE_HIGH_REASONING_EFFORT_VALUES)
        with pytest.raises(ValueError, match='reasoning_effort must be either "none" or "high"'):
            render_template(template, messages=_MESSAGES, reasoning_effort="low")


class TestConvertTokenizerToChatTemplateMapping:
    def test_exposed_values_and_default_are_detected_from_tokenizer_file(self, tmp_path: Path) -> None:
        tekken_path = _none_high_reasoning_effort_tekken_path(tmp_path)
        template = convert_tokenizer_to_chat_template(tokenizer_file=tekken_path)
        assert template == _generate_v15_template(
            reasoning_effort_values=_NONE_HIGH_REASONING_EFFORT_VALUES,
            reasoning_effort_default=_NONE_HIGH_REASONING_EFFORT_DEFAULT,
        )
        assert "reasoning_effort_mapping" not in template

    def test_full_ladder_values_are_detected_from_tokenizer_file(self, tmp_path: Path) -> None:
        tekken_path = _reasoning_effort_tekken_path(
            tmp_path=tmp_path,
            values=tuple(effort.value for effort in ReasoningEffort),
            mapping=None,
            default=None,
        )
        template = convert_tokenizer_to_chat_template(tokenizer_file=tekken_path)
        assert template == _generate_v15_template()
        assert "{%- if reasoning_effort not in ['none', 'low', 'medium', 'high', 'xhigh', 'max'] %}" in template

    def test_mapping_and_default_are_detected_from_tokenizer_file(self, tmp_path: Path) -> None:
        tekken_path = _low_high_max_reasoning_effort_tekken_path(tmp_path)
        template = convert_tokenizer_to_chat_template(tokenizer_file=tekken_path)
        assert template == _generate_v15_template(
            reasoning_effort_mapping=_LOW_HIGH_MAX_REASONING_EFFORT_MAPPING,
            reasoning_effort_default=_LOW_HIGH_MAX_REASONING_EFFORT_DEFAULT,
        )
        assert "{%- set reasoning_effort = 'high' %}" in template


class TestNoneHighReasoningEffortWithoutRemapping:
    r"""Previous v15 release: `none` and `high` exposed, both supported as-is."""

    @pytest.mark.parametrize("reasoning_effort", [ReasoningEffort.none, ReasoningEffort.high])
    def test_exposed_values_encode_as_is(self, tmp_path: Path, reasoning_effort: ReasoningEffort) -> None:
        tokenizer = MistralTokenizer.from_file(str(_none_high_reasoning_effort_tekken_path(tmp_path)))
        request = ChatCompletionRequest(messages=[UserMessage(content="Hello")], reasoning_effort=reasoning_effort)
        text = decode_keep(tokenizer, tokenizer.encode_chat_completion(request))
        assert _NONE_HIGH_SETTINGS[reasoning_effort.value] in text

    def test_unset_effort_encodes_default(self, tmp_path: Path) -> None:
        tokenizer = MistralTokenizer.from_file(str(_none_high_reasoning_effort_tekken_path(tmp_path)))
        request = ChatCompletionRequest(messages=[UserMessage(content="Hello")])
        text = decode_keep(tokenizer, tokenizer.encode_chat_completion(request))
        assert _NONE_HIGH_SETTINGS[_NONE_HIGH_REASONING_EFFORT_DEFAULT] in text

    @pytest.mark.parametrize(
        "reasoning_effort",
        [ReasoningEffort.none, ReasoningEffort.high, None],
        ids=["none", "high", "unset"],
    )
    def test_template_and_tokenizer_agree(self, tmp_path: Path, reasoning_effort: ReasoningEffort | None) -> None:
        tekken_path = _none_high_reasoning_effort_tekken_path(tmp_path)
        tokenizer = MistralTokenizer.from_file(str(tekken_path))
        template = convert_tokenizer_to_chat_template(tokenizer_file=tekken_path)
        request = ChatCompletionRequest(messages=[UserMessage(content="Hello")], reasoning_effort=reasoning_effort)

        tokenized = encode_mistral_common(mistral_tokenizer=tokenizer, chat_request=request, spm=False)
        rendered = render_template(template, **build_render_args(request))

        expected = _NONE_HIGH_SETTINGS[
            reasoning_effort.value if reasoning_effort is not None else _NONE_HIGH_REASONING_EFFORT_DEFAULT
        ]
        assert expected in tokenized
        assert tokenized == rendered


class TestLowHighMaxReasoningEffortRemapping:
    r"""New v15 release: `low`, `high` and `max` exposed, remapped to `low`, `medium` and `high`."""

    @pytest.mark.parametrize("reasoning_effort", ["low", "high", "max"])
    def test_render_maps_exposed_values(self, reasoning_effort: str) -> None:
        template = _generate_v15_template(
            reasoning_effort_mapping=_LOW_HIGH_MAX_REASONING_EFFORT_MAPPING,
            reasoning_effort_default=_LOW_HIGH_MAX_REASONING_EFFORT_DEFAULT,
        )
        rendered = render_template(template, messages=_MESSAGES, reasoning_effort=reasoning_effort)
        model_side_value = _LOW_HIGH_MAX_REASONING_EFFORT_MAPPING[reasoning_effort]
        assert f'[MODEL_SETTINGS]{{"reasoning_effort": "{model_side_value}"}}[/MODEL_SETTINGS]' in rendered

    def test_render_unset_effort_maps_default(self) -> None:
        template = _generate_v15_template(
            reasoning_effort_mapping=_LOW_HIGH_MAX_REASONING_EFFORT_MAPPING,
            reasoning_effort_default=_LOW_HIGH_MAX_REASONING_EFFORT_DEFAULT,
        )
        rendered = render_template(template, messages=_MESSAGES)
        assert _LOW_HIGH_MAX_MAPPED_DEFAULT_SETTINGS in rendered

    def test_render_rejects_value_outside_mapping(self) -> None:
        template = _generate_v15_template(
            reasoning_effort_mapping=_LOW_HIGH_MAX_REASONING_EFFORT_MAPPING,
            reasoning_effort_default=_LOW_HIGH_MAX_REASONING_EFFORT_DEFAULT,
        )
        with pytest.raises(ValueError, match='reasoning_effort must be one of "low", "high", "max"'):
            render_template(template, messages=_MESSAGES, reasoning_effort="none")

    @pytest.mark.parametrize("reasoning_effort", [ReasoningEffort.low, ReasoningEffort.high, ReasoningEffort.max])
    def test_exposed_values_encode_as_model_side_value(self, tmp_path: Path, reasoning_effort: ReasoningEffort) -> None:
        tokenizer = MistralTokenizer.from_file(str(_low_high_max_reasoning_effort_tekken_path(tmp_path)))
        request = ChatCompletionRequest(messages=[UserMessage(content="Hello")], reasoning_effort=reasoning_effort)
        text = decode_keep(tokenizer, tokenizer.encode_chat_completion(request))
        model_side_value = _LOW_HIGH_MAX_REASONING_EFFORT_MAPPING[reasoning_effort.value]
        assert f'[MODEL_SETTINGS]{{"reasoning_effort": "{model_side_value}"}}[/MODEL_SETTINGS]' in text

    def test_unset_effort_encodes_mapped_default(self, tmp_path: Path) -> None:
        # An unset effort resolves to the exposed default "high", remapped to "medium".
        tokenizer = MistralTokenizer.from_file(str(_low_high_max_reasoning_effort_tekken_path(tmp_path)))
        request = ChatCompletionRequest(messages=[UserMessage(content="Hello")])
        text = decode_keep(tokenizer, tokenizer.encode_chat_completion(request))
        assert _LOW_HIGH_MAX_MAPPED_DEFAULT_SETTINGS in text

    @pytest.mark.parametrize(
        "reasoning_effort",
        [ReasoningEffort.low, ReasoningEffort.high, ReasoningEffort.max, None],
        ids=["low", "high", "max", "unset"],
    )
    def test_template_and_tokenizer_agree(self, tmp_path: Path, reasoning_effort: ReasoningEffort | None) -> None:
        tekken_path = _low_high_max_reasoning_effort_tekken_path(tmp_path)
        tokenizer = MistralTokenizer.from_file(str(tekken_path))
        template = convert_tokenizer_to_chat_template(tokenizer_file=tekken_path)
        request = ChatCompletionRequest(messages=[UserMessage(content="Hello")], reasoning_effort=reasoning_effort)

        tokenized = encode_mistral_common(mistral_tokenizer=tokenizer, chat_request=request, spm=False)
        rendered = render_template(template, **build_render_args(request))

        if reasoning_effort is None:
            assert _LOW_HIGH_MAX_MAPPED_DEFAULT_SETTINGS in tokenized
        else:
            model_side_value = _LOW_HIGH_MAX_REASONING_EFFORT_MAPPING[reasoning_effort.value]
            assert f'[MODEL_SETTINGS]{{"reasoning_effort": "{model_side_value}"}}[/MODEL_SETTINGS]' in tokenized
        assert tokenized == rendered
