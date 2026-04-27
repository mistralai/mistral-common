import pytest
from pydantic import ValidationError

from mistral_common.base import MistralBase
from mistral_common.protocol.instruct.chunk import TextChunk
from mistral_common.protocol.instruct.messages import UserMessage


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
