from enum import Enum

import pytest
from pydantic import ValidationError

from mistral_common.base import MistralBase
from mistral_common.protocol.instruct.chunk import TextChunk
from mistral_common.protocol.instruct.messages import UserMessage


class _RequiredParent(MistralBase):
    inherited_value: int


class _RequiredChild(_RequiredParent):
    child_value: str


class _Envelope(MistralBase):
    child: _RequiredChild


class _InvalidDefault(MistralBase):
    value: int = "not an integer"


class _Choice(str, Enum):
    first = "first"


class _EnumModel(MistralBase):
    choice: _Choice


def test_filter_cls_fields() -> None:
    assert MistralBase._filter_cls_fields({}) == {}

    filtered = UserMessage._filter_cls_fields({"role": "user", "content": "hi", "name": "u1"})
    assert filtered == {"role": "user", "content": "hi"}

    filtered = TextChunk._filter_cls_fields({"type": "text", "text": "hi", "annotations": []})
    assert filtered == {"type": "text", "text": "hi"}


def test_model_validate_ignore_extra_filters_and_validates() -> None:
    message = UserMessage.model_validate_ignore_extra({"role": "user", "content": "hi", "name": "u1"})
    assert message == UserMessage(content="hi")


def test_model_validate_ignore_extra_no_extra_keys() -> None:
    chunk = TextChunk.model_validate_ignore_extra({"type": "text", "text": "hello"})
    assert chunk == TextChunk(text="hello")


def test_model_validate_ignore_extra_raises_on_missing_required() -> None:
    with pytest.raises(ValidationError):
        UserMessage.model_validate_ignore_extra({"role": "user"})


def test_filter_cls_fields_includes_inherited_fields() -> None:
    data = {"inherited_value": 7, "child_value": "child", "unknown": "ignored"}

    assert _RequiredChild._filter_cls_fields(data) == {"inherited_value": 7, "child_value": "child"}
    assert _RequiredChild.model_validate_ignore_extra(data) == _RequiredChild(inherited_value=7, child_value="child")

    with pytest.raises(ValidationError):
        _RequiredChild.model_validate(data)


@pytest.mark.parametrize(
    "data",
    [
        {"child_value": "child", "unknown": "ignored"},
        {"inherited_value": "not an integer", "child_value": "child", "unknown": "ignored"},
    ],
    ids=["missing-inherited-field", "invalid-inherited-field"],
)
def test_model_validate_ignore_extra_validates_inherited_fields(data: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _RequiredChild.model_validate_ignore_extra(data)


def test_model_validate_ignore_extra_does_not_filter_nested_fields() -> None:
    valid_child = {"inherited_value": 7, "child_value": "child"}
    assert _Envelope.model_validate_ignore_extra({"child": valid_child, "unknown_outer": "ignored"}) == _Envelope(
        child=_RequiredChild(inherited_value=7, child_value="child")
    )

    invalid_nested_data = {
        "child": {"inherited_value": 7, "child_value": "child", "unknown_nested": "rejected"},
        "unknown_outer": "ignored",
    }
    with pytest.raises(ValidationError) as exc_info:
        _Envelope.model_validate_ignore_extra(invalid_nested_data)

    assert [error["loc"] for error in exc_info.value.errors()] == [("child", "unknown_nested")]


def test_model_validate_validates_defaults_on_instantiation() -> None:
    with pytest.raises(ValidationError):
        _InvalidDefault()


def test_model_validate_uses_enum_values() -> None:
    model = _EnumModel(choice=_Choice.first)

    assert model.choice == "first"
    assert type(model.choice) is str
