from pathlib import Path
from typing import Any, Callable, Generic

from mistral_common.exceptions import (
    TokenizerException,
)
from mistral_common.protocol.fim.request import FIMRequest
from mistral_common.protocol.instruct.messages import (
    UATS,
    AssistantMessageType,
    SystemMessageType,
    ToolMessageType,
    UserMessageType,
)
from mistral_common.protocol.instruct.normalize import InstructRequestNormalizer, get_normalizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.validator import (
    MistralRequestValidator,
    ValidationMode,
    get_validator,
)
from mistral_common.protocol.speech.request import SpeechRequest
from mistral_common.protocol.transcription.request import TranscriptionRequest
from mistral_common.tokens.tokenizers.audio import AudioConfig, AudioEncoder, SpecialAudioIDs
from mistral_common.tokens.tokenizers.base import (
    InstructRequest,
    InstructRequestType,
    InstructTokenizer,
    SpecialTokenPolicy,
    SpecialTokens,
    TokenizedType,
    TokenizerVersion,
)
from mistral_common.tokens.tokenizers.image import (
    ImageConfig,
    ImageEncoder,
    SpecialImageIDs,
)
from mistral_common.tokens.tokenizers.instruct import (
    InstructTokenizerV1,
    InstructTokenizerV2,
    InstructTokenizerV3,
    InstructTokenizerV7,
    InstructTokenizerV11,
    InstructTokenizerV13,
    InstructTokenizerV15,
)
from mistral_common.tokens.tokenizers.sentencepiece import (
    SentencePieceTokenizer,
    get_image_config,
    is_sentencepiece,
)
from mistral_common.tokens.tokenizers.tekken import Tekkenizer, is_tekken
from mistral_common.tokens.tokenizers.utils import download_tokenizer_from_hf_hub


def load_image_encoder(image_config: ImageConfig, tokenizer: Tekkenizer | SentencePieceTokenizer) -> ImageEncoder:
    r"""Load a image encoder from a config and a tokenizer.

    Args:
        image_config: The image config.
        tokenizer: The tokenizer.

    Returns:
        The image encoder.
    """
    special_ids = SpecialImageIDs(
        img=tokenizer.get_special_token(SpecialTokens.img.value),
        img_break=tokenizer.get_special_token(SpecialTokens.img_break.value),
        img_end=tokenizer.get_special_token(SpecialTokens.img_end.value),
    )
    return ImageEncoder(image_config, special_ids)


def load_audio_encoder(audio_config: AudioConfig, tokenizer: Tekkenizer) -> AudioEncoder:
    r"""Load a audio encoder from a config and a tokenizer.

    Args:
        audio_config: The audio config.
        tokenizer: The tokenizer.

    Returns:
        The audio encoder.
    """

    def get_special_token_or_none(token: str) -> int | None:
        if not tokenizer.is_special(token):
            return None

        return tokenizer.get_special_token(token)

    special_ids = SpecialAudioIDs(
        audio=get_special_token_or_none(SpecialTokens.audio.value),
        begin_audio=get_special_token_or_none(SpecialTokens.begin_audio.value),
        streaming_pad=get_special_token_or_none(SpecialTokens.streaming_pad.value),
        text_to_audio=get_special_token_or_none(SpecialTokens.text_to_audio.value),
        audio_to_text=get_special_token_or_none(SpecialTokens.audio_to_text.value),
    )
    return AudioEncoder(audio_config, special_ids)


class MistralTokenizer(
    Generic[UserMessageType, AssistantMessageType, ToolMessageType, SystemMessageType, TokenizedType]
):
    r"""Mistral tokenizer.

    This class is a wrapper around a [InstructTokenizer][mistral_common.tokens.tokenizers.base.InstructTokenizer],
    a [MistralRequestValidator][mistral_common.protocol.instruct.validator.MistralRequestValidator] and a
    [InstructRequestNormalizer][mistral_common.protocol.instruct.normalize.InstructRequestNormalizer].

    It provides a convenient interface to tokenize, validate ad normalize Mistral requests.

    Attributes:
        instruct_tokenizer: The instruct tokenizer to use. See
            [InstructTokenizer][mistral_common.tokens.tokenizers.instruct.InstructTokenizer].
    """

    def __init__(
        self,
        instruct_tokenizer: InstructTokenizer[InstructRequest, FIMRequest, TokenizedType, AssistantMessageType],
        validator: MistralRequestValidator[UserMessageType, AssistantMessageType, ToolMessageType, SystemMessageType],
        request_normalizer: InstructRequestNormalizer[
            UserMessageType, AssistantMessageType, ToolMessageType, SystemMessageType, InstructRequestType
        ],
    ):
        r"""Initializes a `MistralTokenizer`.

        Args:
            instruct_tokenizer: The instruct tokenizer to use.
            validator: The request validator to use.
            request_normalizer: The request normalizer to use.
        """
        self._chat_completion_request_validator = validator
        self._instruct_request_normalizer = request_normalizer
        self.instruct_tokenizer: InstructTokenizer[InstructRequest, FIMRequest, TokenizedType, AssistantMessageType] = (
            instruct_tokenizer
        )

    def __reduce__(self) -> tuple[Callable, tuple[Any, ...]]:
        """
        Provides a recipe for pickling (serializing) this object, which is necessary for use with multiprocessing.

        Returns:
            A tuple of the factory function and the arguments to reconstruct the object from its source file.
        """
        return MistralTokenizer.from_file, (self.instruct_tokenizer.tokenizer.file_path, self.mode)

    @property
    def mode(self) -> ValidationMode:
        r"""The validation mode of the tokenizer."""
        return self._chat_completion_request_validator.mode

    @property
    def version(self) -> TokenizerVersion:
        r"""The version of the tokenizer."""
        return self.instruct_tokenizer.tokenizer.version

    @classmethod
    def _data_path(cls) -> Path:
        return Path(__file__).parents[2] / "data"

    @classmethod
    def v1(cls) -> "MistralTokenizer":
        r"""Get the Mistral tokenizer v1."""
        return cls.from_file(str(cls._data_path() / "tokenizer.model.v1"), mode=ValidationMode.test)

    @classmethod
    def v2(cls) -> "MistralTokenizer":
        r"""Get the Mistral tokenizer v2."""
        return cls.from_file(
            str(cls._data_path() / "mistral_instruct_tokenizer_240216.model.v2"), mode=ValidationMode.test
        )

    @classmethod
    def v3(cls, is_tekken: bool = False, is_mm: bool = False) -> "MistralTokenizer":
        r"""Get the Mistral tokenizer v3.

        Args:
            is_tekken: Whether the tokenizer is a tekken tokenizer. See
                [Tekkenizer][mistral_common.tokens.tokenizers.tekken.Tekkenizer].
            is_mm: Whether to load image tokenizer.

        Returns:
            The Mistral tokenizer v3.
        """
        if is_tekken and is_mm:
            tokenizer_name = "tekken_240911.json"
        elif is_tekken and not is_mm:
            tokenizer_name = "tekken_240718.json"
        elif not is_tekken and is_mm:
            raise ValueError("Multimodal tokenizer is currently only supported for tekken")
        else:
            tokenizer_name = "mistral_instruct_tokenizer_240323.model.v3"

        return cls.from_file(str(cls._data_path() / tokenizer_name), mode=ValidationMode.test)

    @classmethod
    def v7(cls, is_mm: bool = False) -> "MistralTokenizer":
        """Get the Mistral tokenizer v7.

        Args:
            is_mm: Whether to load the image tokenizer.

        Returns:
            The Mistral tokenizer v7.
        """
        if is_mm:
            return cls.from_file(
                str(cls._data_path() / "mistral_instruct_tokenizer_241114.model.v7m1"), mode=ValidationMode.test
            )
        else:
            return cls.from_file(
                str(cls._data_path() / "mistral_instruct_tokenizer_241114.model.v7"), mode=ValidationMode.test
            )

    @classmethod
    def from_model(cls, model: str, strict: bool = True) -> "MistralTokenizer":
        r"""Get the Mistral tokenizer for a given model.

        Args:
            model: The model name.
            strict: Has to be True, not used.

        Returns:
            The Mistral tokenizer for the given model.
        """

        if not strict:
            raise ValueError("strict has to be `True` since v1.10.0.")

        if model not in MODEL_NAME_TO_TOKENIZER_CLS:
            raise TokenizerException(f"Unrecognized model: {model}")

        return MODEL_NAME_TO_TOKENIZER_CLS[model]()

    @staticmethod
    def from_hf_hub(
        repo_id: str,
        token: bool | str | None = None,
        revision: str | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        mode: ValidationMode = ValidationMode.test,
    ) -> "MistralTokenizer":
        r"""Download the Mistral tokenizer for a given Hugging Face repository ID.

        See [here](https://huggingface.co/mistralai/models) for a list of our OSS models.

        Args:
            repo_id: The Hugging Face repo ID.
            token: The Hugging Face token to use to download the tokenizer.
            revision: The revision of the model to use. If `None`, the latest revision will be used.
            mode: The validation mode to use.
            force_download: Whether to force the download of the tokenizer. If `True`, the tokenizer will be downloaded
                even if it is already cached.
            local_files_only: Whether to only use local files. If `True`, the tokenizer will be downloaded only if it is
                already cached.

        Returns:
            The Mistral tokenizer for the given model.
        """
        tokenizer_path = download_tokenizer_from_hf_hub(
            repo_id=repo_id,
            token=token,
            revision=revision,
            force_download=force_download,
            local_files_only=local_files_only,
        )
        return MistralTokenizer.from_file(tokenizer_path, mode=mode)

    @classmethod
    def from_file(
        cls,
        tokenizer_filename: str | Path,
        mode: ValidationMode = ValidationMode.test,
    ) -> "MistralTokenizer":
        r"""Loads a tokenizer from a file.

        Args:
            tokenizer_filename: The path to the tokenizer file.
            mode: The validation mode to use.

        Returns:
            The loaded tokenizer.
        """
        tokenizer: SentencePieceTokenizer | Tekkenizer

        if is_tekken(tokenizer_filename):
            tokenizer = Tekkenizer.from_file(tokenizer_filename)
            image_config = tokenizer.image
            audio_config = tokenizer.audio
        elif is_sentencepiece(tokenizer_filename):
            tokenizer = SentencePieceTokenizer(tokenizer_filename)
            image_config = get_image_config(tokenizer_filename)
            # spm can't have audio
            audio_config = None
        else:
            raise TokenizerException(f"Unrecognized tokenizer file: {tokenizer_filename}")

        image_encoder = load_image_encoder(image_config, tokenizer) if image_config is not None else None

        audio_encoder = None
        if audio_config is not None:
            assert isinstance(tokenizer, Tekkenizer), "Audio is only supported for tekken tokenizers"
            audio_encoder = load_audio_encoder(audio_config, tokenizer)

        request_normalizer = get_normalizer(tokenizer.version, tokenizer.model_settings_builder)
        validator = get_validator(tokenizer.version, mode=mode)

        if tokenizer.version == TokenizerVersion.v1:
            assert image_encoder is None, "Tokenizer version needs to be >= v3"
            assert audio_encoder is None, "Tokenizer version needs to be >= v7"
            return MistralTokenizer(
                InstructTokenizerV1(tokenizer),
                validator=validator,
                request_normalizer=request_normalizer,
            )
        elif tokenizer.version == TokenizerVersion.v2:
            assert image_encoder is None, "Tokenizer version needs to be >= v3"
            assert audio_encoder is None, "Tokenizer version needs to be >= v7"
            return MistralTokenizer(
                InstructTokenizerV2(tokenizer),
                validator=validator,
                request_normalizer=request_normalizer,
            )
        elif tokenizer.version == TokenizerVersion.v3:
            assert audio_encoder is None, "Tokenizer version needs to be >= v7"
            return MistralTokenizer(
                InstructTokenizerV3(tokenizer, image_encoder=image_encoder),
                validator=validator,
                request_normalizer=request_normalizer,
            )
        elif tokenizer.version == TokenizerVersion.v7:
            return MistralTokenizer(
                InstructTokenizerV7(tokenizer, image_encoder=image_encoder, audio_encoder=audio_encoder),
                validator=validator,
                request_normalizer=request_normalizer,
            )
        elif tokenizer.version == TokenizerVersion.v11:
            return MistralTokenizer(
                InstructTokenizerV11(tokenizer, image_encoder=image_encoder, audio_encoder=audio_encoder),
                validator=validator,
                request_normalizer=request_normalizer,
            )
        elif tokenizer.version == TokenizerVersion.v13:
            return MistralTokenizer(
                InstructTokenizerV13(tokenizer, image_encoder=image_encoder, audio_encoder=audio_encoder),
                validator=validator,
                request_normalizer=request_normalizer,
            )
        elif tokenizer.version == TokenizerVersion.v15:
            return MistralTokenizer(
                InstructTokenizerV15(tokenizer, image_encoder=image_encoder, audio_encoder=audio_encoder),
                validator=validator,
                request_normalizer=request_normalizer,
            )

        raise TokenizerException(f"Unrecognized tokenizer filename: {tokenizer_filename}")

    def encode_chat_completion(
        self, request: ChatCompletionRequest[UATS], max_model_input_len: int | None = None
    ) -> TokenizedType:
        r"""Encodes a chat completion request.

        Args:
            request: The chat completion request to encode.
            max_model_input_len: The maximum length of the input to the model.
                If `None`, the input will not be truncated.

        Returns:
            The encoded chat completion request.
        """

        validated_request = self._chat_completion_request_validator.validate_request(request)

        if max_model_input_len is None and request.truncate_for_context_length:
            # the max_model_input_len arg should not be optional ;
            # but this function is used in many small scripts that have no use
            # for truncation, and don't provide the max model len
            raise TokenizerException(
                "encoding a chat completion request with truncation, but no max model len was provided",
            )

        instruct_request = self._instruct_request_normalizer.from_chat_completion_request(validated_request)

        if request.truncate_for_context_length:
            instruct_request.truncate_at_max_tokens = max_model_input_len

        return self.instruct_tokenizer.encode_instruct(instruct_request)

    def encode_transcription(self, request: TranscriptionRequest) -> TokenizedType:
        r"""Encodes a transcription request.

        Args:
            request: The transcription request to encode.

        Returns:
            The encoded transcription request.
        """
        return self.instruct_tokenizer.encode_transcription(request)

    def encode_speech_request(self, request: SpeechRequest) -> TokenizedType:
        r"""Encodes a speech synthesis request.

        Args:
            request: The speech request to encode.

        Returns:
            The encoded speech request.
        """
        return self.instruct_tokenizer.encode_speech_request(request)

    def encode_fim(self, request: FIMRequest) -> TokenizedType:
        r"""Encodes a fill in the middle request.

        Args:
            request: The fill in the middle request to encode.

        Returns:
            The encoded fill in the middle request.
        """
        return self.instruct_tokenizer.encode_fim(request)

    def decode(self, tokens: list[int], special_token_policy: SpecialTokenPolicy = SpecialTokenPolicy.IGNORE) -> str:
        r"""Decodes a list of tokens into a string.

        Args:
            tokens: The tokens to decode.
            special_token_policy: The policy to use for special tokens.

        Returns:
            The decoded string.
        """
        return self.instruct_tokenizer.decode(tokens, special_token_policy=special_token_policy)

    def _to_string(self, tokens: list[int]) -> str:
        return self.instruct_tokenizer._to_string(tokens)


MODEL_NAME_TO_TOKENIZER_CLS: dict[str, Callable[[], MistralTokenizer]] = {
    "ministral-8b-2410": lambda: MistralTokenizer.v3(is_tekken=True),
    "mistral-tiny-2312": MistralTokenizer.v2,
    "open-mistral-nemo-2407": lambda: MistralTokenizer.v3(is_tekken=True),
    "mistral-tiny-2407": MistralTokenizer.v3,
    "mistral-small-2312": MistralTokenizer.v2,
    "open-mixtral-8x22b-2404": MistralTokenizer.v3,
    "mistral-small-2402": MistralTokenizer.v2,
    "mistral-small-2409": lambda: MistralTokenizer.v3(is_tekken=True),
    "mistral-medium-2312": MistralTokenizer.v1,
    "mistral-large-2402": MistralTokenizer.v2,
    "mistral-large-2407": MistralTokenizer.v3,
    "mistral-large-2411": MistralTokenizer.v7,
    "pixtral-large-2411": lambda: MistralTokenizer.v7(is_mm=True),
    "codestral-2405": MistralTokenizer.v3,
    "codestral-mamba-2407": MistralTokenizer.v3,
    "pixtral-12b-2409": lambda: MistralTokenizer.v3(is_tekken=True, is_mm=True),
}
