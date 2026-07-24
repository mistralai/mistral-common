from collections.abc import Callable
from typing import cast

import pytest
from PIL import Image

from mistral_common.exceptions import (
    InvalidAssistantMessageException,
    InvalidMessageStructureException,
    TokenizerException,
)
from mistral_common.protocol.instruct.chunk import (
    ContentChunk,
    ImageChunk,
    TextChunk,
)
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    ChatMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import FunctionCall, ToolCall
from mistral_common.protocol.instruct.validator import (
    ValidationMode,
)
from mistral_common.tokens.tokenizers.base import (
    InstructRequest,
    InstructTokenizer,
    Tokenized,
    TokenizerVersion,
)
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV7
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.utils.requests.instruct import (
    abcd_single_turn_continue_request,
    single_user_message,
    system_image_tool_result_chat_request,
    system_two_users_tool_call_result_request,
    system_user_tool_call_request,
    text_image_user_message,
    v7_assistant_tool_call_and_content_request,
    v7_truncation_full_convo_request,
    v7_truncation_keep_sys_and_last_message_request,
)
from tests.utils.tokenizers import (
    build_mistral_tokenizer,
    image_token_ids,
    image_token_spans,
    special_image_ids,
)
from tests.utils.versions import TestConfig, config_id

_V7 = TestConfig(version=TokenizerVersion.v7)
_V7_AUDIO = TestConfig(version=TokenizerVersion.v7, audio=True)

v7_spm_mm = pytest.mark.parametrize("shipped_instruct_tokenizer", ["v7_spm_mm_small_patch"], indirect=True)
v7_any_instruct = pytest.mark.parametrize("instruct_tokenizer", [_V7, _V7_AUDIO], indirect=True, ids=config_id)


def _as_v7(instruct_tokenizer: InstructTokenizer) -> InstructTokenizerV7:
    assert isinstance(instruct_tokenizer, InstructTokenizerV7)
    return instruct_tokenizer


class TestInstructTokenizerV7:
    @v7_spm_mm
    def test_tokenize_assistant_message(
        self,
        shipped_instruct_tokenizer: InstructTokenizer,
        instruct_decoded_goldens: dict[str, dict[str, str]],
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
    ) -> None:
        instruct_v7_spm_mm = _as_v7(shipped_instruct_tokenizer)
        tokenized = instruct_v7_spm_mm.encode_instruct(
            InstructRequest(
                messages=[
                    text_image_user_message(),
                    AssistantMessage(content="b"),
                    ToolMessage(tool_call_id="b", content="f"),
                ],
            )
        )
        assert tokenized.tokens == instruct_token_id_goldens["v7_spm_mm_small_patch"]["image_tool_result"]
        assert tokenized.text == instruct_decoded_goldens["v7_spm_mm_small_patch"]["image_tool_result"]

    @v7_spm_mm
    def test_tokenize_empty_content_assistant_message(
        self,
        shipped_instruct_tokenizer: InstructTokenizer,
        instruct_decoded_goldens: dict[str, dict[str, str]],
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
    ) -> None:
        instruct_v7_spm_mm = _as_v7(shipped_instruct_tokenizer)
        golden_tokens = instruct_token_id_goldens["v7_spm_mm"]["assistant_prefix_tool_call"]
        # The request has a single assistant message, so `start()` contributes only the
        # leading bos token: `prefix_ids` is the rest of the golden tokens.
        expected = Tokenized(
            tokens=golden_tokens,
            text=instruct_decoded_goldens["v7_spm_mm"]["assistant_prefix_tool_call"],
            prefix_ids=golden_tokens[1:],
        )
        for content in [None, ""]:
            tool_calls: list[ToolCall] | None
            for tool_calls in [None, [], [ToolCall(function=FunctionCall(name="test_fn", arguments="{}"))]]:
                instruct_request: InstructRequest = InstructRequest(
                    messages=[AssistantMessage(content=content, tool_calls=tool_calls, prefix=True)]
                )
                if not content and not tool_calls:
                    with pytest.raises(TokenizerException, match="Invalid assistant message:"):
                        instruct_v7_spm_mm.encode_instruct(instruct_request)
                else:
                    assert instruct_v7_spm_mm.encode_instruct(instruct_request) == expected

    @v7_spm_mm
    def test_tokenize_assistant_message_continue_final_message(
        self,
        shipped_instruct_tokenizer: InstructTokenizer,
        instruct_decoded_goldens: dict[str, dict[str, str]],
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
    ) -> None:
        instruct_v7_spm_mm = _as_v7(shipped_instruct_tokenizer)
        tokenized = instruct_v7_spm_mm.encode_instruct(
            InstructRequest(
                messages=[
                    text_image_user_message(),
                    AssistantMessage(content="b"),
                ],
                continue_final_message=True,
            )
        )
        assert tokenized.tokens == instruct_token_id_goldens["v7_spm_mm_small_patch"]["image_continue_final_message"]
        assert tokenized.text == instruct_decoded_goldens["v7_spm_mm_small_patch"]["image_continue_final_message"]

        with pytest.raises(
            InvalidMessageStructureException, match="Cannot continue final message if it is not an assistant message"
        ):
            instruct_v7_spm_mm.encode_instruct(
                InstructRequest(
                    messages=[text_image_user_message()],
                    continue_final_message=True,
                )
            )

        with pytest.raises(
            InvalidAssistantMessageException,
            match="`continue_message` is only supported for assistant messages that have `prefix=False`.",
        ):
            instruct_v7_spm_mm.encode_assistant_message(
                AssistantMessage(
                    content='"blabla"',
                    prefix=True,
                ),
                is_before_last_user_message=False,
                continue_message=True,
            )

    @pytest.mark.parametrize(
        "request_name, build_request",
        [
            ("system_tool_call", system_user_tool_call_request),
            ("system_two_users_tool_result", system_two_users_tool_call_result_request),
        ],
    )
    @v7_spm_mm
    def test_encode_spm(
        self,
        shipped_instruct_tokenizer: InstructTokenizer,
        instruct_decoded_goldens: dict[str, dict[str, str]],
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
        request_name: str,
        build_request: Callable[[], InstructRequest],
    ) -> None:
        instruct_v7_spm_mm = _as_v7(shipped_instruct_tokenizer)
        tokenized = instruct_v7_spm_mm.encode_instruct(build_request())

        assert tokenized.tokens == instruct_token_id_goldens["v7_spm_mm"][request_name]
        assert tokenized.text == instruct_decoded_goldens["v7_spm_mm"][request_name]

    @pytest.mark.parametrize("shipped_mistral_tokenizer", ["v7_spm_mm"], indirect=True)
    def test_encode_chat_completion(
        self,
        shipped_mistral_tokenizer: MistralTokenizer,
        instruct_decoded_goldens: dict[str, dict[str, str]],
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
    ) -> None:
        tokenizer = shipped_mistral_tokenizer

        encoded = tokenizer.encode_chat_completion(system_image_tool_result_chat_request())

        assert len(encoded.images) == 1
        assert encoded.images[0].shape == (3, 16, 16)
        assert encoded.tokens == instruct_token_id_goldens["v7_spm_mm"]["image_tool_result_chat"]
        assert encoded.text == instruct_decoded_goldens["v7_spm_mm"]["image_tool_result_chat"]

    @pytest.mark.parametrize(
        "messages,truncated_text",
        [
            pytest.param(
                [
                    AssistantMessage(content="c"),
                    UserMessage(content="b"),
                    UserMessage(content="a"),
                    UserMessage(content="aaaaaaa"),
                ],
                "<s>[INST]b[/INST][INST]a[/INST][INST]aaaaaaa[/INST]",
            ),
            pytest.param(
                [
                    AssistantMessage(content="c"),
                    AssistantMessage(content="b"),
                    UserMessage(content="a"),
                    UserMessage(content="aaaaaaa"),
                ],
                "<s>b</s>[INST]a[/INST][INST]aaaaaaa[/INST]",
            ),
            pytest.param(
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
                [
                    UserMessage(content="c"),
                    AssistantMessage(content="c"),
                    AssistantMessage(content="c"),
                    UserMessage(content="aaaaaaa"),
                ],
                "<s>[INST]aaaaaaa[/INST]",
                id="drop_by_chunk_2",
            ),
        ],
    )
    @v7_any_instruct
    def test_truncation(
        self,
        instruct_tokenizer: InstructTokenizer,
        messages: list[ChatMessage],
        truncated_text: str,
    ) -> None:
        tokenizer = instruct_tokenizer

        tokenized = tokenizer.encode_instruct(InstructRequest(messages=messages, truncate_at_max_tokens=15))
        assert tokenized.text == truncated_text, f"{tokenized.text} != {truncated_text}"

    @pytest.mark.parametrize(
        "build_request,request_name",
        [
            (v7_truncation_keep_sys_and_last_message_request, "truncation_keep_sys_and_last_message"),
            (v7_truncation_full_convo_request, "truncation_full_convo"),
        ],
    )
    @pytest.mark.parametrize(
        ("instruct_tokenizer", "registry_key"),
        [(_V7, "v7_tekken"), (_V7_AUDIO, "v7_tekken_aud")],
        indirect=["instruct_tokenizer"],
        ids=[config_id(_V7), config_id(_V7_AUDIO)],
    )
    def test_truncation_matches_golden(
        self,
        instruct_tokenizer: InstructTokenizer,
        registry_key: str,
        build_request: Callable[[], InstructRequest],
        request_name: str,
        instruct_decoded_goldens: dict[str, dict[str, str]],
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
    ) -> None:
        tokenized = instruct_tokenizer.encode_instruct(build_request())
        assert tokenized.tokens == instruct_token_id_goldens[registry_key][request_name]
        assert tokenized.text == instruct_decoded_goldens[registry_key][request_name]

    @pytest.mark.parametrize(
        "messages",
        [
            [
                SystemMessage(content="a" * 10),
            ],
            [
                UserMessage(content="a" * 10),
            ],
        ],
    )
    @v7_any_instruct
    def test_truncation_failed(self, instruct_tokenizer: InstructTokenizer, messages: list[ChatMessage]) -> None:
        tokenizer = instruct_tokenizer
        with pytest.raises(TokenizerException):
            tokenizer.encode_instruct(InstructRequest(messages=messages, truncate_at_max_tokens=9))

    def test_from_model(self) -> None:
        with pytest.warns(FutureWarning, match="from_model.*deprecated"):
            tokenizer = MistralTokenizer.from_model("ministral-8b-2410")
            assert tokenizer.instruct_tokenizer.tokenizer.version == TokenizerVersion.v3
            assert tokenizer.instruct_tokenizer.image_encoder is None

            tokenizer = MistralTokenizer.from_model("mistral-small-2402")
            assert tokenizer.instruct_tokenizer.tokenizer.version == TokenizerVersion.v2
            assert tokenizer.instruct_tokenizer.image_encoder is None

            tokenizer = MistralTokenizer.from_model("mistral-small-2409")
            assert tokenizer.instruct_tokenizer.tokenizer.version == TokenizerVersion.v3
            assert tokenizer.instruct_tokenizer.image_encoder is None

            tokenizer = MistralTokenizer.from_model("mistral-large-2411")
            assert tokenizer.instruct_tokenizer.tokenizer.version == TokenizerVersion.v7
            assert tokenizer.instruct_tokenizer.image_encoder is None

            tokenizer = MistralTokenizer.from_model("pixtral-large-2411")
            assert tokenizer.instruct_tokenizer.tokenizer.version == TokenizerVersion.v7
            assert tokenizer.instruct_tokenizer.image_encoder is not None

            tokenizer = MistralTokenizer.from_model("pixtral-12b-2409")
            assert tokenizer.instruct_tokenizer.tokenizer.version == TokenizerVersion.v3
            assert tokenizer.instruct_tokenizer.image_encoder is not None

        with pytest.warns(FutureWarning), pytest.raises(TokenizerException):
            MistralTokenizer.from_model("unknown-model")

    @pytest.mark.parametrize(
        ("instruct_tokenizer", "registry_key"),
        [(_V7, "v7_tekken"), (_V7_AUDIO, "v7_tekken_aud")],
        indirect=["instruct_tokenizer"],
        ids=[config_id(_V7), config_id(_V7_AUDIO)],
    )
    def test_assistant_tool_call_and_content(
        self,
        instruct_tokenizer: InstructTokenizer,
        registry_key: str,
        instruct_decoded_goldens: dict[str, dict[str, str]],
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
    ) -> None:
        tokenizer = instruct_tokenizer
        instruct_request = v7_assistant_tool_call_and_content_request()
        tokenized = tokenizer.encode_instruct(instruct_request)
        tokens = tokenized.tokens
        text = tokenized.text

        assert tokens == instruct_token_id_goldens[registry_key]["assistant_tool_call_and_content"]
        assert text == instruct_decoded_goldens[registry_key]["assistant_tool_call_and_content"]

        tools = instruct_request.available_tools
        exclude = {"system_prompt", "truncate_at_max_tokens", "available_tools", "settings"}
        chat_completion_request = ChatCompletionRequest(**instruct_request.model_dump(exclude=exclude), tools=tools)
        mistral_tokenizer = build_mistral_tokenizer(tokenizer, TokenizerVersion.v7, mode=ValidationMode.finetuning)
        tokens_2 = mistral_tokenizer.encode_chat_completion(chat_completion_request)

        assert tokens == tokens_2.tokens

    @pytest.mark.parametrize("shipped_mistral_tokenizer", ["v7_spm_mm"], indirect=True)
    def test_encode_chat_completion_continue_final_message(
        self,
        shipped_mistral_tokenizer: MistralTokenizer,
        instruct_decoded_goldens: dict[str, dict[str, str]],
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
    ) -> None:
        tokenizer = shipped_mistral_tokenizer
        eos_id = tokenizer.instruct_tokenizer.tokenizer.eos_id

        encoded = tokenizer.encode_chat_completion(abcd_single_turn_continue_request())

        assert encoded.tokens == instruct_token_id_goldens["v7_spm_mm"]["abcd_single_turn_continue"]
        assert encoded.text == instruct_decoded_goldens["v7_spm_mm"]["abcd_single_turn_continue"]
        assert encoded.tokens[-1] != eos_id
        assert eos_id not in encoded.prefix_ids

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param(
                [
                    TextChunk(text=""),
                    ImageChunk(image=Image.new("RGB", (4, 4), "red")),
                    ImageChunk(image=Image.new("RGB", (6, 4), "blue")),
                ],
                id="empty-text-then-two-images",
            ),
            pytest.param(
                [
                    TextChunk(text="x"),
                    ImageChunk(image=Image.new("RGB", (4, 4), "red")),
                    ImageChunk(image=Image.new("RGB", (6, 4), "blue")),
                ],
                id="text-then-two-images",
            ),
            pytest.param(
                [
                    ImageChunk(image=Image.new("RGB", (4, 4), "red")),
                    ImageChunk(image=Image.new("RGB", (6, 4), "blue")),
                ],
                id="two-images",
            ),
        ],
    )
    @v7_spm_mm
    def test_multi_image_order_is_preserved(
        self, shipped_instruct_tokenizer: InstructTokenizer, content: list[ContentChunk]
    ) -> None:
        instruct_v7_spm_mm = _as_v7(shipped_instruct_tokenizer)
        tokenized = instruct_v7_spm_mm.encode_instruct(InstructRequest(messages=single_user_message(content=content)))
        special_ids = special_image_ids(cast(Tekkenizer, instruct_v7_spm_mm.tokenizer))
        assert image_token_spans(tokenized.tokens, special_ids) == [
            image_token_ids(2, 2, special_ids),
            image_token_ids(3, 2, special_ids),
        ]

    @v7_spm_mm
    def test_single_trailing_image_moves_first(self, shipped_instruct_tokenizer: InstructTokenizer) -> None:
        instruct_v7_spm_mm = _as_v7(shipped_instruct_tokenizer)
        tokenized = instruct_v7_spm_mm.encode_instruct(
            InstructRequest(
                messages=single_user_message(
                    content=[TextChunk(text="x"), ImageChunk(image=Image.new("RGB", (4, 4), "red"))]
                )
            )
        )
        special_ids = special_image_ids(cast(Tekkenizer, instruct_v7_spm_mm.tokenizer))
        assert image_token_spans(tokenized.tokens, special_ids) == [image_token_ids(2, 2, special_ids)]
        x_token = instruct_v7_spm_mm.tokenizer.encode("x", bos=False, eos=False)[0]
        assert tokenized.tokens.index(special_ids.img) < tokenized.tokens.index(x_token)

    @v7_spm_mm
    def test_single_leading_image_remains_first(self, shipped_instruct_tokenizer: InstructTokenizer) -> None:
        instruct_v7_spm_mm = _as_v7(shipped_instruct_tokenizer)
        tokenized = instruct_v7_spm_mm.encode_instruct(
            InstructRequest(
                messages=single_user_message(
                    content=[ImageChunk(image=Image.new("RGB", (4, 4), "red")), TextChunk(text="x")]
                )
            )
        )
        special_ids = special_image_ids(cast(Tekkenizer, instruct_v7_spm_mm.tokenizer))
        assert image_token_spans(tokenized.tokens, special_ids) == [image_token_ids(2, 2, special_ids)]
        x_token = instruct_v7_spm_mm.tokenizer.encode("x", bos=False, eos=False)[0]
        assert tokenized.tokens.index(special_ids.img) < tokenized.tokens.index(x_token)
