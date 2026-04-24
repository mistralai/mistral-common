from mistral_common.base import MistralBase
from mistral_common.protocol.instruct.chunk import TextChunk
from mistral_common.protocol.instruct.messages import UserMessage


def test_filter_cls_fields() -> None:
    assert MistralBase._filter_cls_fields({}) == {}

    filtered = UserMessage._filter_cls_fields({"role": "user", "content": "hi", "name": "u1"})
    assert filtered == {"role": "user", "content": "hi"}

    filtered = TextChunk._filter_cls_fields({"type": "text", "text": "hi", "annotations": []})
    assert filtered == {"type": "text", "text": "hi"}
