import pytest

from mistral_common.protocol.instruct.chunk import AudioChunk, AudioURLChunk
from mistral_common.protocol.instruct.messages import AssistantMessage, SystemMessage, UserMessage
from mistral_common.protocol.instruct.normalize import InstructRequestNormalizerV13
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import FunctionCall, ToolCall
from mistral_common.protocol.instruct.validator import MistralRequestValidatorV11
from mistral_common.tokens.tokenizers.audio import AudioConfig, AudioEncoder, AudioSpectrogramConfig, SpecialAudioIDs
from mistral_common.tokens.tokenizers.base import SpecialTokens, TokenizerVersion
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV11
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.fixtures.audio import get_dummy_audio_chunk, get_dummy_audio_url_chunk
from tests.test_tekken import get_special_tokens, quick_vocab
from tests.utils import decode_keep


@pytest.fixture(scope="session")
def tekkenizer() -> InstructTokenizerV11:
    special_tokens = get_special_tokens(TokenizerVersion.v11)
    tokenizer = Tekkenizer(
        quick_vocab([b"a", b"b", b"c", b"f", b"de"]),
        special_tokens=special_tokens,
        pattern=r".+",  # single token, whole string
        vocab_size=256 + 100,
        num_special_tokens=100,
        version=TokenizerVersion.v11,
    )
    return InstructTokenizerV11(tokenizer)


def test_special_tokens(tekkenizer: InstructTokenizerV11) -> None:
    assert tekkenizer.ARGS == 32
    assert tekkenizer.CALL_ID == 33
    assert tekkenizer.TOOL_CALLS == 9


def test_tokenize_assistant_message(tekkenizer: InstructTokenizerV11) -> None:
    tokens = tekkenizer.encode_assistant_message(
        AssistantMessage(
            tool_calls=[ToolCall(function=FunctionCall(name="a_a_a", arguments="blabla"))],
        ),
        is_before_last_user_message=False,
    )
    assert tokens == [
        tekkenizer.TOOL_CALLS,
        197,
        195,
        197,
        195,
        197,
        tekkenizer.ARGS,
        134,
        198,
        208,
        197,
        198,
        208,
        197,
        134,
        2,
    ]
    assert tekkenizer.tokenizer._to_string(tokens) == ('[TOOL_CALLS]a_a_a[ARGS]"blabla"</s>')


def test_tokenize_prefixed_assistant_message(tekkenizer: InstructTokenizerV11) -> None:
    tokens = tekkenizer.encode_assistant_message(
        AssistantMessage(
            content='"blabla"',
            prefix=True,
        ),
        is_before_last_user_message=False,
    )
    assert tokens == [
        134,
        198,
        208,
        197,
        198,
        208,
        197,
        134,
    ]
    assert tekkenizer.tokenizer._to_string(tokens) == ('"blabla"')


def test_tokenize_assistant_messages(tekkenizer: InstructTokenizerV11) -> None:
    tokens = tekkenizer.encode_assistant_message(
        AssistantMessage(
            tool_calls=[
                ToolCall(function=FunctionCall(name="a_a_a", arguments="blabla")),
                ToolCall(function=FunctionCall(name="b", arguments="blu")),
            ],
        ),
        is_before_last_user_message=False,
    )
    assert tokens == [
        tekkenizer.TOOL_CALLS,
        197,
        195,
        197,
        195,
        197,
        tekkenizer.ARGS,
        134,
        198,
        208,
        197,
        198,
        208,
        197,
        134,
        tekkenizer.TOOL_CALLS,
        198,
        tekkenizer.ARGS,
        134,
        198,
        208,
        217,
        134,
        2,
    ]
    assert tekkenizer.tokenizer._to_string(tokens) == ('[TOOL_CALLS]a_a_a[ARGS]"blabla"[TOOL_CALLS]b[ARGS]"blu"</s>')


def test_tokenize_assistant_message_train(tekkenizer: InstructTokenizerV11) -> None:
    tokens = tekkenizer.encode_assistant_message(
        AssistantMessage(
            tool_calls=[ToolCall(function=FunctionCall(name="a_a_a", arguments="blabla"), id="ABC")],
        ),
        is_before_last_user_message=True,
    )
    assert tokens == [
        tekkenizer.TOOL_CALLS,
        197,
        195,
        197,
        195,
        197,
        tekkenizer.CALL_ID,
        165,
        166,
        167,
        tekkenizer.ARGS,
        134,
        198,
        208,
        197,
        198,
        208,
        197,
        134,
        2,
    ]
    assert tekkenizer.tokenizer._to_string(tokens) == ('[TOOL_CALLS]a_a_a[CALL_ID]ABC[ARGS]"blabla"</s>')


@pytest.fixture(scope="session")
def v11_tekkenizer_audio() -> InstructTokenizerV11:
    special_tokens = get_special_tokens(TokenizerVersion.v11, add_audio=True)
    tokenizer = Tekkenizer(
        quick_vocab([b"a", b"b", b"c", b"f", b"de"]),
        special_tokens=special_tokens,
        pattern=r".+",
        vocab_size=256 + 100,
        num_special_tokens=100,
        version=TokenizerVersion.v11,
    )
    audio_config = AudioConfig(
        sampling_rate=24_000,
        frame_rate=12.5,
        encoding_config=AudioSpectrogramConfig(
            num_mel_bins=128,
            window_size=400,
            hop_length=160,
        ),
    )
    special_audio_ids = SpecialAudioIDs(
        audio=tokenizer.get_special_token(SpecialTokens.audio.value),
        begin_audio=tokenizer.get_special_token(SpecialTokens.begin_audio.value),
        streaming_pad=None,
        text_to_audio=None,
        audio_to_text=None,
    )
    audio_encoder = AudioEncoder(audio_config, special_audio_ids)
    return InstructTokenizerV11(tokenizer, audio_encoder=audio_encoder)


@pytest.fixture(params=["audio_chunk", "audio_url_chunk"])
def audio_fixture(request: pytest.FixtureRequest) -> AudioChunk | AudioURLChunk:
    if request.param == "audio_chunk":
        return get_dummy_audio_chunk()
    return get_dummy_audio_url_chunk()


def test_v11_allows_system_message_with_audio(
    v11_tekkenizer_audio: InstructTokenizerV11, audio_fixture: AudioChunk | AudioURLChunk
) -> None:
    """V11 should accept system messages alongside audio (unlike V7)."""
    request_normalizer = InstructRequestNormalizerV13.normalizer()
    validator = MistralRequestValidatorV11()
    messages = [
        SystemMessage(content="hello"),
        UserMessage(content=[audio_fixture]),
    ]
    mistral_tokenizer = MistralTokenizer(
        instruct_tokenizer=v11_tekkenizer_audio, validator=validator, request_normalizer=request_normalizer
    )
    encoded = mistral_tokenizer.encode_chat_completion(ChatCompletionRequest(messages=messages))
    text = decode_keep(mistral_tokenizer, encoded)
    assert "[SYSTEM_PROMPT]hello[/SYSTEM_PROMPT]" in text
    assert len(encoded.audios) == 1
