import json
from typing import List

import pytest
from mistral_common.exceptions import TokenizerException
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    ChatMessage,
    ImageChunk,
    SystemMessage,
    TextChunk,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import Function, FunctionCall, Tool, ToolCall
from mistral_common.tokens.tokenizers.base import InstructRequest, TokenizerVersion
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.multimodal import ImageEncoder
from mistral_common.tokens.tokenizers.sentencepiece import InstructTokenizerV7
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from PIL import Image

from tests.test_tekken import _quick_vocab


@pytest.fixture(scope="session")
def tekkenizer() -> InstructTokenizerV7:
    tokenizer = Tekkenizer(
        _quick_vocab([b"a", b"b", b"c", b"f", b"de"]),
        pattern=r".+",  # single token, whole string
        vocab_size=256 + 100,
        num_special_tokens=100,
        version=TokenizerVersion.v7,
    )
    return InstructTokenizerV7(tokenizer)


@pytest.fixture(scope="session")
def spm_tokenizer() -> InstructTokenizerV7:
    tokenizer = MistralTokenizer.v7(is_mm=True).instruct_tokenizer
    mm_encoder = tokenizer.mm_encoder
    assert isinstance(mm_encoder, ImageEncoder)
    # hardcode image_patch_size = 2 for easier checks
    mm_encoder.mm_config.image_patch_size = 2
    return tokenizer  # type: ignore


def test_tokenize_assistant_message(spm_tokenizer: InstructTokenizerV7) -> None:
    tokenized = spm_tokenizer.encode_instruct(
        InstructRequest(
            messages=[
                UserMessage(
                    content=[
                        TextChunk(
                            text="a",
                        ),
                        ImageChunk(image=Image.new("RGB", (4, 4), "red")),
                    ]
                ),
                AssistantMessage(content="b"),
                ToolMessage(tool_call_id="b", content="f"),
            ],
        )
    )
    _im = 10
    _im_break = 14
    _im_end = 15
    img_tokens = [_im, _im, _im_break, _im, _im, _im_end]
    assert tokenized.tokens == [
        1,  # bos
        3,  # begin_inst
        *img_tokens,
        1032,  # a
        4,  # end_inst
        1055,  # b
        2,  # eos
        8,  # [TOOL_RESULTS]
        1055,  # tool_call_id b
        18,  # [TOOL_CONTENT]
        1053,  # f
        9,  # [/TOOL_RESULTS]
    ]
    assert (
        tokenized.text
        == "<s>[INST][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]▁a[/INST]▁b</s>[TOOL_RESULTS]▁b[TOOL_CONTENT]▁f[/TOOL_RESULTS]"  # noqa
    )


@pytest.mark.parametrize(
    "messages, expected_text",
    [
        (
            [
                SystemMessage(content="a"),
                UserMessage(content="a"),
                AssistantMessage(
                    content="b",
                    tool_calls=[
                        ToolCall(
                            function=FunctionCall(
                                name="t",
                                arguments=json.dumps(
                                    {
                                        "g": "h",
                                    },
                                    ensure_ascii=False,
                                ),
                            ),
                        ),
                    ],
                ),
            ],
            '<s>[SYSTEM_PROMPT]▁a[/SYSTEM_PROMPT][AVAILABLE_TOOLS]▁[{"type":▁"function",▁"function":▁{"name":▁"t",▁"description":▁"",▁"parameters":▁{"type":▁"object",▁"properties":▁{"g":▁{"type":▁"string"},▁"h":▁{"type":▁"string"}}}}}][/AVAILABLE_TOOLS][INST]▁a[/INST]▁b[TOOL_CALLS]▁[{"name":▁"t",▁"arguments":▁{"g":▁"h"}}]</s>',  # noqa
        ),
        (
            [
                SystemMessage(content="a"),
                UserMessage(content="a"),
                UserMessage(content="c"),
                AssistantMessage(
                    content="b",
                    tool_calls=[
                        ToolCall(
                            function=FunctionCall(
                                name="t",
                                arguments=json.dumps(
                                    {
                                        "g": "h",
                                    },
                                    ensure_ascii=False,
                                ),
                            ),
                        ),
                    ],
                ),
                ToolMessage(content="b", tool_call_id="1234"),
            ],
            '<s>[SYSTEM_PROMPT]▁a[/SYSTEM_PROMPT][INST]▁a[/INST][AVAILABLE_TOOLS]▁[{"type":▁"function",▁"function":▁{"name":▁"t",▁"description":▁"",▁"parameters":▁{"type":▁"object",▁"properties":▁{"g":▁{"type":▁"string"},▁"h":▁{"type":▁"string"}}}}}][/AVAILABLE_TOOLS][INST]▁c[/INST]▁b[TOOL_CALLS]▁[{"name":▁"t",▁"arguments":▁{"g":▁"h"}}]</s>[TOOL_RESULTS]▁1234[TOOL_CONTENT]▁b[/TOOL_RESULTS]',  # noqa
        ),
    ],
)
def test_encode_spm(spm_tokenizer: InstructTokenizerV7, messages: List[ChatMessage], expected_text: str) -> None:
    tokenized = spm_tokenizer.encode_instruct(
        InstructRequest(
            available_tools=[
                Tool(
                    function=Function(
                        name="t",
                        parameters={
                            "type": "object",
                            "properties": {
                                "g": {"type": "string"},
                                "h": {"type": "string"},
                            },
                        },
                    )
                ),
            ],
            messages=messages,
        )
    )

    assert tokenized.text == expected_text, f"{tokenized.text} != {expected_text}"


def test_encode_chat_completion() -> None:
    tokenizer = MistralTokenizer.v7(is_mm=True)

    request: ChatCompletionRequest = ChatCompletionRequest(
        tools=[
            Tool(
                function=Function(
                    name="t",
                    parameters={
                        "type": "object",
                        "properties": {
                            "g": {"type": "string"},
                            "h": {"type": "string"},
                        },
                    },
                )
            ),
        ],
        messages=[
            SystemMessage(content="a"),
            UserMessage(
                content=[
                    TextChunk(
                        text="a",
                    ),
                    ImageChunk(image=Image.new("RGB", (4, 4), "red")),
                ]
            ),
            AssistantMessage(content="b"),
            ToolMessage(tool_call_id="123456789", content="f"),
        ],
    )

    encoded = tokenizer.encode_chat_completion(request)

    assert len(encoded.images) == 1
    assert encoded.images[0].shape == (3, 16, 16)
    assert (
        encoded.text
        == '<s>[SYSTEM_PROMPT]▁a[/SYSTEM_PROMPT][AVAILABLE_TOOLS]▁[{"type":▁"function",▁"function":▁{"name":▁"t",▁"description":▁"",▁"parameters":▁{"type":▁"object",▁"properties":▁{"g":▁{"type":▁"string"},▁"h":▁{"type":▁"string"}}}}}][/AVAILABLE_TOOLS][INST][IMG][IMG_END]▁a[/INST]▁b</s>[TOOL_RESULTS]▁123456789[TOOL_CONTENT]▁f[/TOOL_RESULTS]'  # noqa
    )


@pytest.mark.parametrize(
    "messages,truncated_text",
    [
        # max_tokens is always set to truncate at 15 tokens
        pytest.param(
            # with the system prompts, only one user message fits, keep the last one
            [
                SystemMessage(content="a"),
                UserMessage(content="c"),
                UserMessage(content="c"),
                SystemMessage(content="a"),
                UserMessage(content="bbbbbbb"),
            ],
            "<s>[SYSTEM_PROMPT]a[/SYSTEM_PROMPT][SYSTEM_PROMPT]a[/SYSTEM_PROMPT][INST]bbbbbbb[/INST]",
            id="keep_sys_and_last_message",
        ),
        pytest.param(
            # drop the first assistant message - everything else fits
            [
                AssistantMessage(content="c"),
                UserMessage(content="b"),
                UserMessage(content="a"),
                UserMessage(content="aaaaaaa"),
            ],
            "<s>[INST]b[/INST][INST]a[/INST][INST]aaaaaaa[/INST]",
        ),
        pytest.param(
            # the result can start with a non-user message because the input did too
            [
                AssistantMessage(content="c"),
                AssistantMessage(content="b"),
                UserMessage(content="a"),
                UserMessage(content="aaaaaaa"),
            ],
            "<s>b</s>[INST]a[/INST][INST]aaaaaaa[/INST]",
        ),
        pytest.param(
            # drop the first assistant message, then drop user+tool because the go together and both don't fit
            [
                AssistantMessage(content="c"),
                UserMessage(content="c"),
                ToolMessage(content="c", tool_call_id="1234"),
                UserMessage(content="a"),
                AssistantMessage(content="bbbbbbb"),
            ],
            "<s>[INST]a[/INST]bbbbbbb</s>",
            id="drop_by_chunk_1",
        ),
        pytest.param(
            # drop everything but the last message, because the first chunk (3 messages) is too big
            [
                UserMessage(content="c"),
                AssistantMessage(content="c"),
                AssistantMessage(content="c"),
                UserMessage(content="aaaaaaa"),
            ],
            "<s>[INST]aaaaaaa[/INST]",
            id="drop_by_chunk_2",
        ),
        pytest.param(
            [
                SystemMessage(content="a"),
                UserMessage(content="c"),
                AssistantMessage(content="c"),
                UserMessage(content="a"),
                AssistantMessage(content="a"),
                SystemMessage(content="b"),
                UserMessage(content="a"),
            ],
            "<s>[SYSTEM_PROMPT]a[/SYSTEM_PROMPT][INST]a[/INST]a</s>[SYSTEM_PROMPT]b[/SYSTEM_PROMPT][INST]a[/INST]",
            id="full_convo",
        ),
    ],
)
def test_truncation(tekkenizer: InstructTokenizerV7, messages: List[ChatMessage], truncated_text: str) -> None:
    tokenized = tekkenizer.encode_instruct(InstructRequest(messages=messages, truncate_at_max_tokens=15))
    assert tokenized.text == truncated_text, f"{tokenized.text} != {truncated_text}"


@pytest.mark.parametrize(
    "messages",
    [
        [
            # system prompt doesn't fit
            SystemMessage(content="a" * 10),
        ],
        [
            # last user msg doesn't fit
            UserMessage(content="a" * 10),
        ],
    ],
)
def test_truncation_failed(tekkenizer: InstructTokenizerV7, messages: List[ChatMessage]) -> None:
    with pytest.raises(TokenizerException):
        tekkenizer.encode_instruct(InstructRequest(messages=messages, truncate_at_max_tokens=9))


def test_from_model() -> None:
    tokenizer = MistralTokenizer.from_model("ministral-8b-2410", strict=True)
    assert tokenizer.instruct_tokenizer.tokenizer.version == TokenizerVersion.v3
    assert tokenizer.instruct_tokenizer.mm_encoder is None

    tokenizer = MistralTokenizer.from_model("mistral-small-2402", strict=True)
    assert tokenizer.instruct_tokenizer.tokenizer.version == TokenizerVersion.v2
    assert tokenizer.instruct_tokenizer.mm_encoder is None

    tokenizer = MistralTokenizer.from_model("mistral-small-2409", strict=True)
    assert tokenizer.instruct_tokenizer.tokenizer.version == TokenizerVersion.v3
    assert tokenizer.instruct_tokenizer.mm_encoder is None

    tokenizer = MistralTokenizer.from_model("mistral-large-2411", strict=True)
    assert tokenizer.instruct_tokenizer.tokenizer.version == TokenizerVersion.v7
    assert tokenizer.instruct_tokenizer.mm_encoder is None

    tokenizer = MistralTokenizer.from_model("pixtral-large-2411", strict=True)
    assert tokenizer.instruct_tokenizer.tokenizer.version == TokenizerVersion.v7
    assert tokenizer.instruct_tokenizer.mm_encoder is not None

    tokenizer = MistralTokenizer.from_model("pixtral-12b-2409", strict=True)
    assert tokenizer.instruct_tokenizer.tokenizer.version == TokenizerVersion.v3
    assert tokenizer.instruct_tokenizer.mm_encoder is not None

    with pytest.raises(TokenizerException):
        MistralTokenizer.from_model("unknown-model", strict=True)

    with pytest.warns(FutureWarning):
        tokenizer = MistralTokenizer.from_model("ministral-8b-2410", strict=False)
        assert tokenizer.instruct_tokenizer.tokenizer.version == TokenizerVersion.v3
        assert tokenizer.instruct_tokenizer.mm_encoder is None

    with pytest.warns(FutureWarning):
        tokenizer = MistralTokenizer.from_model("pixtral", strict=False)
        assert tokenizer.instruct_tokenizer.tokenizer.version == TokenizerVersion.v3
        assert tokenizer.instruct_tokenizer.mm_encoder is not None