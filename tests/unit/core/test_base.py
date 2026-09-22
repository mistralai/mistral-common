import pytest
from pydantic import ValidationError

from mistral_common.base import MistralBase
from mistral_common.protocol.instruct.chunk import TextChunk
from mistral_common.protocol.instruct.messages import UserMessage


def test_filter_cls_fields_filters_unknown_values() -> None:
    assert MistralBase._filter_cls_fields({}) == {}
    assert UserMessage._filter_cls_fields({"role": "user", "content": "hi", "name": "u1"}) == {
        "role": "user",
        "content": "hi",
    }
    assert TextChunk._filter_cls_fields({"type": "text", "text": "hi", "annotations": []}) == {
        "type": "text",
        "text": "hi",
    }


def test_model_validate_ignore_extra_filters_and_validates() -> None:
    assert UserMessage.model_validate_ignore_extra({"role": "user", "content": "hi", "name": "u1"}) == UserMessage(
        content="hi"
    )


def test_model_validate_ignore_extra_preserves_valid_input() -> None:
    assert TextChunk.model_validate_ignore_extra({"type": "text", "text": "hello"}) == TextChunk(text="hello")


def test_model_validate_ignore_extra_rejects_missing_required_fields() -> None:
    with pytest.raises(ValidationError):
        UserMessage.model_validate_ignore_extra({"role": "user"})
