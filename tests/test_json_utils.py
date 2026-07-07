import pytest

from mistral_common.utils.json_utils import validate_json_schema_by_draft7


def test_valid_schema_passes() -> None:
    validate_json_schema_by_draft7({"type": "object"})


def test_invalid_schema_raises() -> None:
    with pytest.raises(ValueError, match="Invalid JSON Schema"):
        validate_json_schema_by_draft7({"type": 123})
