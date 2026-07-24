from collections.abc import Callable

import pytest

from mistral_common.exceptions import InvalidRequestException, TokenizerException
from mistral_common.protocol.instruct.chunk import ContentChunk, ThinkChunk
from mistral_common.protocol.instruct.messages import SystemMessage
from mistral_common.protocol.instruct.request import (
    ChatCompletionRequest,
    InstructRequest,
    ModelSettings,
    ReasoningEffort,
)
from mistral_common.tokens.tokenizers.base import InstructTokenizer, TokenizerVersion
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.utils.requests.instruct import (
    abcd_messages,
    dummy_audio_chunk,
    dummy_audio_url_chunk,
    dummy_base64_image_url_chunk,
    math_interpreter_tools,
    multimodal_system_request,
    multimodal_tool_request,
    multimodal_user_request,
    tool_call_messages,
)
from tests.utils.versions import TestConfig

_V15_THINK = TestConfig(version=TokenizerVersion.v15, think=True)
_V15_AUDIO = TestConfig(version=TokenizerVersion.v15, audio=True)
_V15_THINK_IMAGE = TestConfig(version=TokenizerVersion.v15, think=True, image=True)
_ALL_EFFORTS = tuple(ReasoningEffort)

# Each multimodal case pairs an input chunk with the expected media counts, the tokenizer
# config able to encode it, and the (key, name) of its golden decoded text.
_TOOL_MULTIMODAL_PARAMS = [
    pytest.param(dummy_audio_chunk(), 1, 0, _V15_AUDIO, "v15_aud", "tool_audio", id="audio"),
    pytest.param(dummy_audio_url_chunk(), 1, 0, _V15_AUDIO, "v15_aud", "tool_audio", id="audio_url"),
    pytest.param(dummy_base64_image_url_chunk(), 0, 1, _V15_THINK_IMAGE, "v15_img_think", "tool_image", id="image_url"),
]
_SYSTEM_MULTIMODAL_PARAMS = [
    pytest.param(dummy_audio_chunk(), 1, 0, _V15_AUDIO, "v15_aud", "system_audio", id="audio"),
]
_USER_MULTIMODAL_PARAMS = [
    pytest.param(dummy_audio_chunk(), 1, 0, _V15_AUDIO, "v15_aud", "user_audio", id="audio"),
    pytest.param(dummy_audio_url_chunk(), 1, 0, _V15_AUDIO, "v15_aud", "user_audio", id="audio_url"),
    pytest.param(dummy_base64_image_url_chunk(), 0, 1, _V15_THINK_IMAGE, "v15_img_think", "user_image", id="image_url"),
]

_MULTIMODAL_ARGNAMES = ("content_chunk", "expected_audios", "expected_images", "config", "golden_key", "golden_name")


class TestInstructTokenizerV15:
    def test_encode_instruct_tools_and_reasoning_effort(
        self,
        instruct_tokenizer_factory: Callable[..., InstructTokenizer],
        instruct_decoded_goldens: dict[str, dict[str, str]],
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
    ) -> None:
        tokenizer = instruct_tokenizer_factory(_V15_THINK, _ALL_EFFORTS)
        request = InstructRequest(
            messages=tool_call_messages(),
            available_tools=math_interpreter_tools(),
            settings=ModelSettings(reasoning_effort=ReasoningEffort.high),
        )
        encoded = tokenizer.encode_instruct(request)
        assert encoded.text == instruct_decoded_goldens["v15_think"]["tools_reasoning_high"]
        assert encoded.tokens == instruct_token_id_goldens["v15_think"]["tools_reasoning_high"]

    def test_encode_instruct_no_tools_and_reasoning_effort(
        self,
        instruct_tokenizer_factory: Callable[..., InstructTokenizer],
        instruct_decoded_goldens: dict[str, dict[str, str]],
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
    ) -> None:
        tokenizer = instruct_tokenizer_factory(_V15_THINK, _ALL_EFFORTS)
        request: InstructRequest = InstructRequest(
            messages=tool_call_messages(),
            available_tools=None,
            settings=ModelSettings(reasoning_effort=ReasoningEffort.none),
        )
        encoded = tokenizer.encode_instruct(request)
        assert encoded.text == instruct_decoded_goldens["v15_think"]["no_tools_reasoning_none"]
        assert encoded.tokens == instruct_token_id_goldens["v15_think"]["no_tools_reasoning_none"]

    def test_encode_instruct_without_model_settings(
        self, instruct_tokenizer_factory: Callable[..., InstructTokenizer]
    ) -> None:
        tokenizer = instruct_tokenizer_factory(_V15_THINK, None)
        request: InstructRequest = InstructRequest(
            messages=tool_call_messages(), available_tools=None, settings=ModelSettings.none()
        )
        assert "[MODEL_SETTINGS]" not in (tokenizer.encode_instruct(request).text or "")

    def test_encode_instruct_system_think_chunk_raises(
        self, instruct_tokenizer_factory: Callable[..., InstructTokenizer]
    ) -> None:
        tokenizer = instruct_tokenizer_factory(_V15_THINK, _ALL_EFFORTS)
        request: InstructRequest = InstructRequest(
            messages=[SystemMessage(content=[ThinkChunk(thinking="Hi")])],
            settings=ModelSettings(reasoning_effort=ReasoningEffort.high),
        )
        with pytest.raises(TokenizerException, match="ThinkChunk in system message is not supported for this model"):
            tokenizer.encode_instruct(request)

    @pytest.mark.parametrize(
        ("reasoning_effort", "allowed_reasoning_effort", "raises", "match"),
        [
            (None, (ReasoningEffort.none, ReasoningEffort.high), None, None),
            ("none", (ReasoningEffort.none, ReasoningEffort.high), None, None),
            ("high", (ReasoningEffort.none, ReasoningEffort.high), None, None),
            ("high", (ReasoningEffort.none,), InvalidRequestException, "should be one of"),
            ("none", (ReasoningEffort.none,), None, None),
            ("none", (), InvalidRequestException, "not supported for this model"),
        ],
    )
    def test_encode_chat_completion_forbidden_reasoning_effort(
        self,
        mistral_tokenizer_factory: Callable[..., MistralTokenizer],
        reasoning_effort: str | None,
        allowed_reasoning_effort: tuple[ReasoningEffort, ...],
        raises: type[Exception] | None,
        match: str | None,
    ) -> None:
        tokenizer = mistral_tokenizer_factory(_V15_THINK, allowed_reasoning_effort)
        request = ChatCompletionRequest(
            messages=tool_call_messages(),
            tools=math_interpreter_tools(),
            reasoning_effort=reasoning_effort,  # type: ignore[arg-type]
        )
        if raises is not None:
            assert match is not None
            with pytest.raises(raises, match=match):
                tokenizer.encode_chat_completion(request)
        else:
            assert match is None
            tokenizer.encode_chat_completion(request)

    @pytest.mark.parametrize(("reasoning_effort", "allowed_reasoning_effort"), [(None, ()), ("none", None)])
    def test_encode_chat_completion_ignores_model_settings(
        self,
        mistral_tokenizer_factory: Callable[..., MistralTokenizer],
        reasoning_effort: str | None,
        allowed_reasoning_effort: tuple[ReasoningEffort, ...] | None,
    ) -> None:
        tokenizer = mistral_tokenizer_factory(_V15_THINK, allowed_reasoning_effort)
        request = ChatCompletionRequest(messages=tool_call_messages(), reasoning_effort=reasoning_effort)  # type: ignore[arg-type]
        assert "[MODEL_SETTINGS]" not in (tokenizer.encode_chat_completion(request).text or "")

    @pytest.mark.parametrize("reasoning_effort", [None, *list(ReasoningEffort)])
    def test_encode_chat_completion_applies_default_reasoning_effort(
        self,
        mistral_tokenizer_factory: Callable[..., MistralTokenizer],
        reasoning_effort: ReasoningEffort | None,
    ) -> None:
        tokenizer = mistral_tokenizer_factory(_V15_THINK, (ReasoningEffort.none, ReasoningEffort.high))
        request = ChatCompletionRequest(messages=tool_call_messages(), reasoning_effort=reasoning_effort)
        text = tokenizer.encode_chat_completion(request).text or ""
        if reasoning_effort == ReasoningEffort.high:
            assert '[MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS]' in text
        else:
            assert '[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]' in text

    @pytest.mark.parametrize("reasoning_effort", [None, *list(ReasoningEffort)])
    def test_encode_chat_completion_without_default_reasoning_effort(
        self,
        mistral_tokenizer_factory: Callable[..., MistralTokenizer],
        reasoning_effort: ReasoningEffort | None,
    ) -> None:
        tokenizer = mistral_tokenizer_factory(_V15_THINK, _ALL_EFFORTS, False)
        request = ChatCompletionRequest(messages=tool_call_messages(), reasoning_effort=reasoning_effort)
        text = tokenizer.encode_chat_completion(request).text or ""
        if reasoning_effort == ReasoningEffort.high:
            assert '[MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS]' in text
        elif reasoning_effort == ReasoningEffort.none:
            assert '[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]' in text
        else:
            assert "[MODEL_SETTINGS]" not in text

    def test_encode_chat_completion_continue_final_message(
        self, mistral_tokenizer_factory: Callable[..., MistralTokenizer]
    ) -> None:
        tokenizer = mistral_tokenizer_factory(_V15_THINK, (ReasoningEffort.none, ReasoningEffort.high))
        request: ChatCompletionRequest = ChatCompletionRequest(
            messages=abcd_messages(turns=1),
            continue_final_message=True,
        )
        encoded = tokenizer.encode_chat_completion(request)
        assert encoded.tokens[-1] != tokenizer.instruct_tokenizer.tokenizer.eos_id

    @pytest.mark.parametrize(_MULTIMODAL_ARGNAMES, _TOOL_MULTIMODAL_PARAMS)
    def test_encode_chat_completion_with_multimodal_tool(
        self,
        mistral_tokenizer_factory: Callable[..., MistralTokenizer],
        instruct_decoded_goldens: dict[str, dict[str, str]],
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
        content_chunk: ContentChunk,
        expected_audios: int,
        expected_images: int,
        config: TestConfig,
        golden_key: str,
        golden_name: str,
    ) -> None:
        tokenizer = mistral_tokenizer_factory(config, _ALL_EFFORTS)
        encoded = tokenizer.encode_chat_completion(multimodal_tool_request(content_chunk))
        assert encoded.text == instruct_decoded_goldens[golden_key][golden_name]
        assert encoded.tokens == instruct_token_id_goldens[golden_key][golden_name]
        assert len(encoded.audios) == expected_audios
        assert len(encoded.images) == expected_images

    @pytest.mark.parametrize(_MULTIMODAL_ARGNAMES, _SYSTEM_MULTIMODAL_PARAMS)
    def test_encode_chat_completion_with_multimodal_system(
        self,
        mistral_tokenizer_factory: Callable[..., MistralTokenizer],
        instruct_decoded_goldens: dict[str, dict[str, str]],
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
        content_chunk: ContentChunk,
        expected_audios: int,
        expected_images: int,
        config: TestConfig,
        golden_key: str,
        golden_name: str,
    ) -> None:
        tokenizer = mistral_tokenizer_factory(config, _ALL_EFFORTS)
        encoded = tokenizer.encode_chat_completion(multimodal_system_request(content_chunk))
        assert encoded.text == instruct_decoded_goldens[golden_key][golden_name]
        assert encoded.tokens == instruct_token_id_goldens[golden_key][golden_name]
        assert len(encoded.audios) == expected_audios
        assert len(encoded.images) == expected_images

    @pytest.mark.parametrize(_MULTIMODAL_ARGNAMES, _USER_MULTIMODAL_PARAMS)
    def test_encode_chat_completion_with_multimodal_user(
        self,
        mistral_tokenizer_factory: Callable[..., MistralTokenizer],
        instruct_decoded_goldens: dict[str, dict[str, str]],
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
        content_chunk: ContentChunk,
        expected_audios: int,
        expected_images: int,
        config: TestConfig,
        golden_key: str,
        golden_name: str,
    ) -> None:
        tokenizer = mistral_tokenizer_factory(config, _ALL_EFFORTS)
        encoded = tokenizer.encode_chat_completion(multimodal_user_request(content_chunk))
        assert encoded.text == instruct_decoded_goldens[golden_key][golden_name]
        assert encoded.tokens == instruct_token_id_goldens[golden_key][golden_name]
        assert len(encoded.audios) == expected_audios
        assert len(encoded.images) == expected_images
