import jsonschema
import jsonschema.exceptions


def validate_json_schema_by_draft7(value: dict) -> None:
    r"""Validate that a dict is a valid Draft 7 JSON Schema.

    Args:
        value: The candidate JSON schema.

    Raises:
        ValueError: If the value is not a valid Draft 7 JSON Schema.
    """
    try:
        jsonschema.Draft7Validator.check_schema(value)
    except jsonschema.exceptions.SchemaError as e:
        raise ValueError(f"Invalid JSON Schema: {e.message}") from e
