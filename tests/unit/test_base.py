import pytest
from pydantic import Field, ValidationError

from mistral_common.base import MistralBase
from mistral_common.protocol.instruct.chunk import ImageURLChunk, TextChunk
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ResponseFormat, ResponseFormats


class _RequiredParent(MistralBase):
    parent_id: int


class _RequiredChild(_RequiredParent):
    child_text: str


class _InvalidLabelDefault(MistralBase):
    label: str = Field(default="", min_length=1)


@pytest.mark.parametrize(
    "model_cls, data, expected",
    [
        pytest.param(MistralBase, {}, {}, id="empty-base"),
        pytest.param(
            UserMessage,
            {"role": "user", "content": "hi", "name": "u1"},
            {"role": "user", "content": "hi"},
            id="user-extra-name",
        ),
        pytest.param(
            TextChunk,
            {"type": "text", "text": "hi", "annotations": []},
            {"type": "text", "text": "hi"},
            id="text-extra-annotations",
        ),
    ],
)
def test_filter_cls_fields(model_cls: type[MistralBase], data: dict[str, object], expected: dict[str, object]) -> None:
    assert model_cls._filter_cls_fields(data) == expected


def test_model_validate_ignore_extra_filters_and_validates() -> None:
    message = UserMessage.model_validate_ignore_extra({"role": "user", "content": "hi", "name": "u1"})
    assert message == UserMessage(content="hi")


def test_model_validate_ignore_extra_raises_on_missing_required() -> None:
    with pytest.raises(ValidationError) as exc_info:
        UserMessage.model_validate_ignore_extra({"role": "user"})

    assert [error["loc"] for error in exc_info.value.errors()] == [("content",)]


def test_filter_cls_fields_includes_inherited_fields() -> None:
    data = {"parent_id": 7, "child_text": "child", "unknown": "ignored"}

    assert _RequiredChild._filter_cls_fields(data) == {"parent_id": 7, "child_text": "child"}
    assert _RequiredChild.model_validate_ignore_extra(data) == _RequiredChild(parent_id=7, child_text="child")


@pytest.mark.parametrize(
    "data",
    [
        pytest.param({"child_text": "child", "unknown": "ignored"}, id="missing-parent-id"),
        pytest.param(
            {"parent_id": "not an integer", "child_text": "child", "unknown": "ignored"},
            id="invalid-parent-id",
        ),
    ],
)
def test_model_validate_ignore_extra_validates_inherited_fields(data: dict[str, object]) -> None:
    with pytest.raises(ValidationError) as exc_info:
        _RequiredChild.model_validate_ignore_extra(data)

    assert [error["loc"] for error in exc_info.value.errors()] == [("parent_id",)]


def test_model_validate_ignore_extra_does_not_filter_nested_fields() -> None:
    data = {
        "image_url": {"url": "https://example.com/image.png", "unknown_nested": "rejected"},
        "unknown_outer": "ignored",
    }

    with pytest.raises(ValidationError) as exc_info:
        ImageURLChunk.model_validate_ignore_extra(data)

    assert ("image_url", "ImageURL", "unknown_nested") in [error["loc"] for error in exc_info.value.errors()]


def test_text_chunk_rejects_annotations() -> None:
    with pytest.raises(ValidationError) as exc_info:
        TextChunk(text="hello", annotations=[])

    assert [error["loc"] for error in exc_info.value.errors()] == [("annotations",)]


def test_model_validate_validates_defaults_on_instantiation() -> None:
    with pytest.raises(ValidationError) as exc_info:
        _InvalidLabelDefault()

    assert [(error["loc"], error["input"]) for error in exc_info.value.errors()] == [(("label",), "")]


def test_model_validate_uses_enum_values() -> None:
    default_model = ResponseFormat()
    explicit_model = ResponseFormat(type=ResponseFormats.json)

    assert default_model.type == ResponseFormats.text.value
    assert type(default_model.type) is str
    assert explicit_model.type == ResponseFormats.json.value
    assert type(explicit_model.type) is str
