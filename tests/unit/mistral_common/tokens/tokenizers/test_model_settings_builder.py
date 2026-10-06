import pickle
from dataclasses import dataclass
from typing import Any

import pytest

from mistral_common.exceptions import InvalidRequestException
from mistral_common.protocol.instruct.request import (
    ChatCompletionRequest,
    JsonSchema,
    ModelSettings,
    ReasoningEffort,
    ResponseFormat,
    ResponseFormats,
)
from mistral_common.tokens.tokenizers.model_settings_builder import (
    EnumBuilder,
    JSONSchemaBuilder,
    ModelSettingsBuilder,
    ReasoningEffortEnumBuilder,
)

SCHEMA: dict[str, Any] = {"type": "object", "properties": {"value": {"type": "string"}}}
JSON_OBJECT_OR_ARRAY: dict[str, Any] = {"anyOf": [{"type": "object"}, {"type": "array"}]}


@dataclass
class StructuralRequest:
    reasoning_effort: ReasoningEffort | None
    response_format: ResponseFormat


@pytest.mark.parametrize(
    ("format_kind", "strict", "expected", "raises_error"),
    [
        ("text", False, None, False),
        ("json", False, JSON_OBJECT_OR_ARRAY, False),
        ("json_schema", False, SCHEMA, False),
        ("json_schema", True, SCHEMA, False),
        ("missing_schema", False, None, True),
    ],
)
def test_json_schema_builder_build_value_scenarios(
    format_kind: str,
    strict: bool,
    expected: dict[str, Any] | None,
    raises_error: bool,
) -> None:
    builder = JSONSchemaBuilder(accepts_none=False, default=None)
    if format_kind == "text":
        response_format = ResponseFormat(type=ResponseFormats.text)
    elif format_kind == "json":
        response_format = ResponseFormat(type=ResponseFormats.json)
    elif format_kind == "missing_schema":
        response_format = ResponseFormat(type=ResponseFormats.json_schema)
    else:
        response_format = ResponseFormat(
            type=ResponseFormats.json_schema,
            json_schema=JsonSchema(name="object", schema=SCHEMA, strict=strict),
        )

    if raises_error:
        with pytest.raises(InvalidRequestException, match="must define the schema"):
            builder.build_value(field_name="json_schema", value=response_format)
    else:
        assert builder.build_value(field_name="json_schema", value=response_format) == expected


@pytest.mark.parametrize(
    ("format_kind", "strict", "reasoning_effort", "with_json_schema_builder", "expected_schema"),
    [
        ("text", False, ReasoningEffort.high, False, None),
        ("text", False, None, True, None),
        ("json", False, None, True, JSON_OBJECT_OR_ARRAY),
        ("json_schema", False, None, True, SCHEMA),
        ("json_schema", True, None, True, SCHEMA),
        ("json_schema", False, ReasoningEffort.high, True, SCHEMA),
    ],
)
def test_build_settings_scenarios(
    format_kind: str,
    strict: bool,
    reasoning_effort: ReasoningEffort | None,
    with_json_schema_builder: bool,
    expected_schema: dict[str, Any] | None,
) -> None:
    if format_kind == "text":
        response_format = ResponseFormat(type=ResponseFormats.text)
    elif format_kind == "json":
        response_format = ResponseFormat(type=ResponseFormats.json)
    else:
        response_format = ResponseFormat(
            type=ResponseFormats.json_schema,
            json_schema=JsonSchema(name="object", schema=SCHEMA, strict=strict),
        )

    reasoning_effort_builder = None
    if reasoning_effort is not None:
        reasoning_effort_builder = EnumBuilder[ReasoningEffort](
            values=[ReasoningEffort.high], accepts_none=False, default=None
        )
    json_schema_builder = None
    if with_json_schema_builder:
        json_schema_builder = JSONSchemaBuilder(accepts_none=False, default=None)

    builder = ModelSettingsBuilder(
        reasoning_effort=reasoning_effort_builder,
        json_schema=json_schema_builder,
    )
    request: ChatCompletionRequest = ChatCompletionRequest(
        messages=[],
        reasoning_effort=reasoning_effort,
        response_format=response_format,
    )
    expected = ModelSettings(reasoning_effort=reasoning_effort, json_schema=expected_schema)

    assert builder.build_settings(request=request) == expected


def test_build_settings_accepts_structural_request() -> None:
    builder = ModelSettingsBuilder(
        reasoning_effort=EnumBuilder[ReasoningEffort](values=[ReasoningEffort.high], accepts_none=False, default=None)
    )
    request = StructuralRequest(
        reasoning_effort=ReasoningEffort.high,
        response_format=ResponseFormat(type=ResponseFormats.text),
    )

    assert builder.build_settings(request=request) == ModelSettings(reasoning_effort=ReasoningEffort.high)


def test_build_settings_preserves_generic_enum_builder_input() -> None:
    builder = ModelSettingsBuilder(
        reasoning_effort=EnumBuilder[ReasoningEffort](values=[ReasoningEffort.high], accepts_none=False, default=None)
    )
    request: ChatCompletionRequest = ChatCompletionRequest(messages=[], reasoning_effort=ReasoningEffort.high)

    assert builder.build_settings(request=request) == ModelSettings(reasoning_effort=ReasoningEffort.high)


def test_validate_settings_rejects_unsupported_json_schema() -> None:
    builder = ModelSettingsBuilder()
    settings = ModelSettings(json_schema={"type": "object"})

    with pytest.raises(InvalidRequestException, match="json_schema not supported for this model"):
        builder.validate_settings(settings=settings)


def test_reasoning_effort_enum_builder_pickle_round_trip() -> None:
    builder = ReasoningEffortEnumBuilder(values=[ReasoningEffort.high], accepts_none=False, default=None)

    assert pickle.loads(pickle.dumps(builder)) == builder
