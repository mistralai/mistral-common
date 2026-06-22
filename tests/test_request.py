from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import (
    ChatCompletionRequest,
    JsonSchema,
    ResponseFormat,
    ResponseFormats,
)

SCHEMA = {"type": "object"}


def test_to_openai_renames_custom_schema_to_schema() -> None:
    rf = ResponseFormat(type=ResponseFormats.json_schema, json_schema=JsonSchema(name="x", schema=SCHEMA))
    out = ChatCompletionRequest(messages=[UserMessage(content="hi")], response_format=rf).to_openai()
    assert out["response_format"]["json_schema"]["schema"] == SCHEMA
    assert "custom_schema" not in out["response_format"]["json_schema"]


def test_from_openai_accepts_schema_key() -> None:
    req = ChatCompletionRequest.from_openai(
        messages=[{"role": "user", "content": "hi"}],
        response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": SCHEMA}},
    )
    assert req.response_format.json_schema.custom_schema == SCHEMA
