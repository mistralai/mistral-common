import jsonschema.exceptions
import pytest

from mistral_common.utils.json_utils import validate_json_schema_by_draft7


def test_valid_schema_passes() -> None:
    validate_json_schema_by_draft7(value={"type": "object"})


def test_invalid_schema_raises() -> None:
    with pytest.raises(ValueError, match="123 is not valid under any of the given schemas"):
        validate_json_schema_by_draft7(value={"type": 123})


def test_invalid_schema_error_chains_the_schema_error() -> None:
    with pytest.raises(ValueError, match="123 is not valid under any of the given schemas") as excinfo:
        validate_json_schema_by_draft7(value={"type": 123})

    assert isinstance(excinfo.value.__cause__, jsonschema.exceptions.SchemaError)
