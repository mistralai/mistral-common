from enum import Enum
from typing import Any

import pytest
from pydantic import ValidationError

from mistral_common.base import MistralBase


class LocalEnum(str, Enum):
    VALUE = "value"


class RequiredModel(MistralBase):
    value: str


class InvalidDefaultModel(MistralBase):
    value: int = "invalid"  # type: ignore[assignment]


class EnumModel(MistralBase):
    value: LocalEnum


@pytest.mark.parametrize(
    ("model_type", "data", "expected"),
    [
        pytest.param(MistralBase, {}, {}, id="empty"),
        pytest.param(RequiredModel, {"value": "known"}, {"value": "known"}, id="known"),
        pytest.param(RequiredModel, {"unknown": "extra"}, {}, id="unknown"),
        pytest.param(
            RequiredModel,
            {"value": "known", "unknown": "extra"},
            {"value": "known"},
            id="known-and-unknown",
        ),
    ],
)
def test_filter_cls_fields(model_type: type[MistralBase], data: dict[str, Any], expected: dict[str, Any]) -> None:
    assert model_type._filter_cls_fields(data) == expected


@pytest.mark.parametrize(
    ("model_type", "data", "expected"),
    [
        pytest.param(RequiredModel, {"value": "valid"}, RequiredModel(value="valid"), id="valid"),
        pytest.param(
            RequiredModel,
            {"value": "valid", "unknown": "extra"},
            RequiredModel(value="valid"),
            id="extra",
        ),
    ],
)
def test_model_validate_ignore_extra(
    model_type: type[MistralBase], data: dict[str, Any], expected: MistralBase
) -> None:
    assert model_type.model_validate_ignore_extra(data) == expected


def test_model_validate_ignore_extra_rejects_missing_required() -> None:
    with pytest.raises(ValidationError, match="value"):
        RequiredModel.model_validate_ignore_extra({})


def test_mistral_base_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError, match="extra"):
        RequiredModel.model_validate({"value": "valid", "extra": "forbidden"})


def test_mistral_base_validates_default_values() -> None:
    with pytest.raises(ValidationError, match="valid integer"):
        InvalidDefaultModel()


def test_mistral_base_uses_enum_values() -> None:
    model = EnumModel(value=LocalEnum.VALUE)
    assert model.value == "value"
    assert model.model_dump() == {"value": "value"}
