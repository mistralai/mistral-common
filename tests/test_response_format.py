import pytest

from mistral_common.exceptions import InvalidRequestException
from mistral_common.protocol.instruct.request import (
    JsonSchema,
    ResponseFormat,
    ResponseFormats,
    SchemaRenderingMode,
)

SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}}


def test_text_get_schema_none() -> None:
    assert ResponseFormat(type=ResponseFormats.text).get_schema(SchemaRenderingMode.model_settings) is None


def test_json_get_schema_anyof() -> None:
    rf = ResponseFormat(type=ResponseFormats.json)
    assert rf.get_schema(SchemaRenderingMode.model_settings) == {"anyOf": [{"type": "object"}, {"type": "array"}]}


def test_json_schema_model_settings_returns_custom_schema() -> None:
    rf = ResponseFormat(type=ResponseFormats.json_schema, json_schema=JsonSchema(name="x", schema=SCHEMA))
    assert rf.get_schema(SchemaRenderingMode.model_settings) == SCHEMA


def test_json_schema_grammar_non_strict_is_generic_object() -> None:
    rf = ResponseFormat(type=ResponseFormats.json_schema, json_schema=JsonSchema(name="x", schema=SCHEMA))
    assert rf.get_schema(SchemaRenderingMode.grammar) == {"type": "object"}


def test_json_schema_grammar_strict_returns_custom_schema() -> None:
    rf = ResponseFormat(type=ResponseFormats.json_schema, json_schema=JsonSchema(name="x", schema=SCHEMA, strict=True))
    assert rf.get_schema(SchemaRenderingMode.grammar) == SCHEMA


def test_json_schema_missing_raises() -> None:
    with pytest.raises(InvalidRequestException, match="must define the schema"):
        ResponseFormat(type=ResponseFormats.json_schema).get_schema(SchemaRenderingMode.model_settings)


def test_invalid_schema_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid JSON Schema"):
        JsonSchema(name="x", schema={"type": 123})


def test_schema_alias_roundtrip() -> None:
    assert JsonSchema(name="x", schema=SCHEMA).custom_schema == SCHEMA
