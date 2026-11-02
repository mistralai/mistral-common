from typing import Any

import jsonschema
import jsonschema.exceptions


def validate_json_schema_by_draft7(value: dict[str, Any]) -> None:
    r"""Validate that a dict is a valid Draft 7 JSON Schema.

    Callers add their own context to the error message.

    Args:
        value: The candidate JSON schema.

    Raises:
        ValueError: If the value is not a valid Draft 7 JSON Schema, with the schema error as message.
    """
    try:
        jsonschema.Draft7Validator.check_schema(value)
    except jsonschema.exceptions.SchemaError as e:
        raise ValueError(e.message) from e
