import pytest

from mistral_common.exceptions import InvalidAssistantMessageException, TokenizerException
from mistral_common.protocol.instruct.chunk import AudioChunk, AudioURLChunk, TextChunk, ThinkChunk
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    SystemMessage,
    ToolMessage,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import FunctionCall, ToolCall
from mistral_common.tokens.tokenizers.base import InstructTokenizer, TokenizerVersion
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV13
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.tekken import SpecialTokenPolicy
from tests.utils.requests.instruct import (
    abcd_messages,
    dummy_audio_chunk,
    dummy_audio_url_chunk,
    tool_call_chat_request,
    v13_system_user_audio_request,
)
from tests.utils.versions import TestConfig, config_id

_V13 = TestConfig(version=TokenizerVersion.v13)
_V13_THINK = TestConfig(version=TokenizerVersion.v13, think=True)
_V13_AUDIO = TestConfig(version=TokenizerVersion.v13, audio=True)

v13_instruct = pytest.mark.parametrize("instruct_tokenizer", [_V13], indirect=True, ids=config_id)
v13_think_instruct = pytest.mark.parametrize("instruct_tokenizer", [_V13_THINK], indirect=True, ids=config_id)
v13_mistral = pytest.mark.parametrize("mistral_tokenizer", [_V13], indirect=True, ids=config_id)
v13_think_mistral = pytest.mark.parametrize("mistral_tokenizer", [_V13_THINK], indirect=True, ids=config_id)
v13_audio_mistral = pytest.mark.parametrize("mistral_tokenizer", [_V13_AUDIO], indirect=True, ids=config_id)


def _as_v13(instruct_tokenizer: InstructTokenizer) -> InstructTokenizerV13:
    assert isinstance(instruct_tokenizer, InstructTokenizerV13)
    return instruct_tokenizer


@pytest.fixture
def audio_chunk() -> AudioChunk:
    return dummy_audio_chunk()


@pytest.fixture
def audio_url_chunk() -> AudioURLChunk:
    return dummy_audio_url_chunk()


class TestInstructTokenizerV13:
    @v13_think_mistral
    def test_encode_chat_completion_tools_and_think(
        self,
        mistral_tokenizer: MistralTokenizer,
        instruct_decoded_goldens: dict[str, dict[str, str]],
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
    ) -> None:
        tokenized = mistral_tokenizer.encode_chat_completion(tool_call_chat_request(think=True))
        assert tokenized.text == instruct_decoded_goldens["v13_think"]["tools_think"]
        assert tokenized.tokens == instruct_token_id_goldens["v13_think"]["tools_think"]

    @v13_mistral
    def test_encode_chat_completion_reorders_tool_results(
        self,
        mistral_tokenizer: MistralTokenizer,
        instruct_decoded_goldens: dict[str, dict[str, str]],
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
    ) -> None:
        tokenized = mistral_tokenizer.encode_chat_completion(tool_call_chat_request(swap_tool_results=True))
        assert tokenized.text == instruct_decoded_goldens["v13"]["tools_wrong_order"]
        assert tokenized.tokens == instruct_token_id_goldens["v13"]["tools_wrong_order"]

    @v13_mistral
    def test_encode_chat_completion_continue_final_message(self, mistral_tokenizer: MistralTokenizer) -> None:
        request: ChatCompletionRequest = ChatCompletionRequest(
            messages=abcd_messages(turns=1),
            continue_final_message=True,
        )
        encoded = mistral_tokenizer.encode_chat_completion(request)
        assert encoded.tokens[-1] != mistral_tokenizer.instruct_tokenizer.tokenizer.eos_id

    @v13_audio_mistral
    @pytest.mark.parametrize("audio_fixture", ["audio_chunk", "audio_url_chunk"])
    def test_encode_chat_completion_with_audio(
        self,
        mistral_tokenizer: MistralTokenizer,
        audio_fixture: str,
        request: pytest.FixtureRequest,
        instruct_decoded_goldens: dict[str, dict[str, str]],
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
    ) -> None:
        audio_chunk: AudioChunk | AudioURLChunk = request.getfixturevalue(audio_fixture)
        encoded = mistral_tokenizer.encode_chat_completion(v13_system_user_audio_request(audio_chunk))
        assert encoded.text == instruct_decoded_goldens["v13_aud"]["system_user_audio"]
        assert encoded.tokens == instruct_token_id_goldens["v13_aud"]["system_user_audio"]
        assert len(encoded.audios) == 1

    @v13_instruct
    def test_encode_tool_message_single_chunk(self, instruct_tokenizer: InstructTokenizer) -> None:
        tool_message = ToolMessage(content="R1", tool_call_id="123456789")
        encoded, images, audios = _as_v13(instruct_tokenizer).encode_tool_message(
            tool_message, is_before_last_user_message=False
        )
        assert encoded == [7, 182, 149, 8]
        assert images == []
        assert audios == []

    @v13_instruct
    def test_encode_tool_message_multiple_chunks(self, instruct_tokenizer: InstructTokenizer) -> None:
        tool_message = ToolMessage(content=[TextChunk(text="R1"), TextChunk(text="R2")], tool_call_id="123456789")
        encoded, images, audios = _as_v13(instruct_tokenizer).encode_tool_message(
            tool_message, is_before_last_user_message=False
        )
        assert encoded == [7, 182, 149, 182, 150, 8]
        assert images == []
        assert audios == []

    @v13_think_instruct
    @pytest.mark.parametrize(
        ("closed", "expected"),
        [(True, "[THINK]T1[/THINK]"), (False, "[THINK]T1")],
    )
    def test_encode_think_chunk(self, instruct_tokenizer: InstructTokenizer, closed: bool, expected: str) -> None:
        tokenizer = _as_v13(instruct_tokenizer)
        encoded = tokenizer.encode_think(ThinkChunk(thinking="T1", closed=closed))
        assert tokenizer.decode(encoded, special_token_policy=SpecialTokenPolicy.KEEP) == expected

    @v13_think_instruct
    @pytest.mark.parametrize(
        "message, expected",
        [
            (
                AssistantMessage(content="A1"),
                "A1",
            ),
            (
                AssistantMessage(content="A1", prefix=True),
                "A1",
            ),
            (
                AssistantMessage(content=[TextChunk(text="A1")]),
                "A1",
            ),
            (
                AssistantMessage(content=[ThinkChunk(thinking="T1"), TextChunk(text="A1")]),
                "[THINK]T1[/THINK]A1",
            ),
            (
                AssistantMessage(
                    content=[ThinkChunk(thinking="R1", closed=False), TextChunk(text="A1")],
                    tool_calls=[ToolCall(id="123456789", function=FunctionCall(name="F1", arguments="{'a': 1}"))],
                ),
                "[THINK]R1A1[TOOL_CALLS]F1[ARGS]\"{'a': 1}\"",
            ),
        ],
    )
    @pytest.mark.parametrize("continue_final_message", [True, False])
    def test_encode_assistant_message(
        self,
        instruct_tokenizer: InstructTokenizer,
        message: AssistantMessage,
        expected: str,
        continue_final_message: bool,
    ) -> None:
        tokenizer = _as_v13(instruct_tokenizer)
        if not continue_final_message:
            tokens = tokenizer.encode_assistant_message(
                message, is_before_last_user_message=False, continue_message=continue_final_message
            )
            if not message.prefix:
                expected += "</s>"
        else:
            if message.prefix:
                with pytest.raises(
                    InvalidAssistantMessageException,
                    match="`continue_message` is only supported for assistant messages that have `prefix=False`.",
                ):
                    tokenizer.encode_assistant_message(
                        message, is_before_last_user_message=False, continue_message=continue_final_message
                    )
                return
            tokens = tokenizer.encode_assistant_message(
                message, is_before_last_user_message=False, continue_message=continue_final_message
            )
        assert tokenizer.decode(tokens, special_token_policy=SpecialTokenPolicy.KEEP) == expected

    @v13_instruct
    def test_encode_assistant_message_invalid_raises(self, instruct_tokenizer: InstructTokenizer) -> None:
        tokenizer = _as_v13(instruct_tokenizer)
        with pytest.raises(TokenizerException, match=r"Invalid assistant message"):
            tokenizer.encode_assistant_message(
                AssistantMessage(content="", tool_calls=[]), is_before_last_user_message=False, continue_message=False
            )

        with pytest.raises(
            InvalidAssistantMessageException,
            match="`continue_message` is only supported for assistant messages that have `prefix=False`.",
        ):
            tokenizer.encode_assistant_message(
                AssistantMessage(content="z", tool_calls=[], prefix=True),
                is_before_last_user_message=False,
                continue_message=True,
            )

    @v13_think_instruct
    @pytest.mark.parametrize(
        "message, expected",
        [
            (
                SystemMessage(content="S1"),
                "[SYSTEM_PROMPT]S1[/SYSTEM_PROMPT]",
            ),
            (
                SystemMessage(content=[TextChunk(text="S1"), ThinkChunk(thinking="TS"), TextChunk(text="S2")]),
                "[SYSTEM_PROMPT]S1[THINK]TS[/THINK]S2[/SYSTEM_PROMPT]",
            ),
            (
                SystemMessage(
                    content=[
                        TextChunk(text="S1"),
                        TextChunk(text="S3"),
                        ThinkChunk(thinking="TS", closed=True),
                        ThinkChunk(thinking="TS", closed=True),
                        TextChunk(text="S2"),
                    ]
                ),
                "[SYSTEM_PROMPT]S1S3[THINK]TS[/THINK][THINK]TS[/THINK]S2[/SYSTEM_PROMPT]",
            ),
            (
                SystemMessage(
                    content=[
                        TextChunk(text="S1"),
                        TextChunk(text="S3"),
                        ThinkChunk(thinking="TS", closed=False),
                    ]
                ),
                "[SYSTEM_PROMPT]S1S3[THINK]TS[/SYSTEM_PROMPT]",
            ),
        ],
    )
    def test_encode_system_message(
        self,
        instruct_tokenizer: InstructTokenizer,
        message: SystemMessage,
        expected: str,
    ) -> None:
        tokenizer = _as_v13(instruct_tokenizer)
        encoded, audios = tokenizer.encode_system_message(message)
        assert tokenizer.decode(encoded, special_token_policy=SpecialTokenPolicy.KEEP) == expected
        assert audios == []
