import warnings
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
    r"""Build an ImageEncoder from a config and a tokenizer.

    Resolves the image special token IDs (img, `img_break`, `img_end`) from the
    tokenizer and combines them with the image config.

    Args:
        image_config: Image processing configuration (image size, patch size, etc.).
        tokenizer: Tokenizer providing the special token IDs for image markers.

    Returns:
        An ImageEncoder bound to the tokenizer's special token IDs.

    Raises:
        ValueError: If any of the required image special tokens is not defined
            in the tokenizer.
    """
    special_ids = SpecialImageIDs(
        img=tokenizer.get_special_token(SpecialTokens.img.value),
        img_break=tokenizer.get_special_token(SpecialTokens.img_break.value),
        img_end=tokenizer.get_special_token(SpecialTokens.img_end.value),
    )
    return ImageEncoder(image_config, special_ids)


def load_audio_encoder(audio_config: AudioConfig, tokenizer: Tekkenizer) -> AudioEncoder:
    r"""Build an AudioEncoder from a config and a tokenizer.

    Resolves the audio special token IDs (audio, `begin_audio`, `streaming_pad`,
    `text_to_audio`, `audio_to_text`) from the tokenizer. Tokens that are not
    defined in the tokenizer are set to `None`, making them optional.

    Args:
        audio_config: Audio processing configuration (encoding, sampling rate, etc.).
        tokenizer: Tekkenizer providing the special token IDs for audio markers.

    Returns:
        An AudioEncoder bound to the tokenizer's special token IDs. IDs for
        undefined tokens are `None`.
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
        instruct_tokenizer: InstructTokenizer[InstructRequest, FIMRequest, TokenizedType],
        validator: MistralRequestValidator[UserMessageType, AssistantMessageType, ToolMessageType, SystemMessageType],
        request_normalizer: InstructRequestNormalizer[
            UserMessageType, AssistantMessageType, ToolMessageType, SystemMessageType, InstructRequestType
        ],
    ):
        r"""Initialize a MistralTokenizer.

        Prefer classmethods like `from_file` or `from_hf_hub` over direct construction;
        they resolve the correct validator and normalizer for the tokenizer version.

        Args:
            instruct_tokenizer: Versioned tokenizer that performs the actual tokenization.
            validator: Validator applied to requests before normalization. Its mode
                (serving/finetuning/test) controls which constraints are enforced.
            request_normalizer: Normalizer converting validated ChatCompletionRequests
                into InstructRequests ready for tokenization.
        """
        self._chat_completion_request_validator = validator
        self._instruct_request_normalizer = request_normalizer
        self.instruct_tokenizer: InstructTokenizer[InstructRequest, FIMRequest, TokenizedType] = instruct_tokenizer

    def __reduce__(self) -> tuple[Callable, tuple[Any, ...]]:
        r"""Provide a pickling recipe so the tokenizer survives multiprocessing.

        The tokenizer is serialized as a (`from_file`, (path, mode)) pair, so the
        object is reconstructed from its source file rather than pickling internal state.

        Returns:
            A tuple of the `from_file` factory and its arguments (file path, validation mode).

        Raises:
            ValueError: If the tokenizer was not loaded from a file, in which case
                there is no path to reconstruct from.
        """
        return MistralTokenizer.from_file, (self.instruct_tokenizer.tokenizer.file_path, self.mode)

    @property
    def mode(self) -> ValidationMode:
        r"""The validation mode of the tokenizer.

        Returns:
            The ValidationMode (serving, finetuning, or test) used by this tokenizer's validator.
        """
        return self._chat_completion_request_validator.mode

    @property
    def version(self) -> TokenizerVersion:
        r"""The version of the tokenizer.

        Returns:
            The TokenizerVersion enum value determining supported features
            (tool calls, audio, model settings, etc.).
        """
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
            is_tekken: If `True`, loads the tekken (tiktoken-based) tokenizer instead
                of the sentencepiece one. Tekken is faster and used by recent models.
            is_mm: If `True`, loads the multimodal variant with image support.
                Only supported together with `is_tekken=True`.

        Returns:
            The Mistral tokenizer v3, in test validation mode.

        Raises:
            ValueError: If `is_mm` is `True` and `is_tekken` is `False` (multimodal requires tekken).
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
        r"""Get the Mistral tokenizer v7.

        Args:
            is_mm: If `True`, loads the multimodal variant with image support,
                otherwise loads the text-only variant.

        Returns:
            The Mistral tokenizer v7, in test validation mode.
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
        r"""Deprecated in favor of `from_hf_hub` or `from_file`, will be removed in 1.13.0.

        Args:
            model: The model name. Must be one of the known legacy model names
                (e.g., "mistral-small-2402", "codestral-2405"); newer models are
                not registered here.
            strict: Has to be `True`, not used.

        Returns:
            The Mistral tokenizer for the given model.

        Raises:
            ValueError: If strict is `False`.
            TokenizerException: If the model name is not recognized.
        """
        warnings.warn(
            "`MistralTokenizer.from_model` is deprecated and will be removed in 1.13.0. "
            "Use `MistralTokenizer.from_hf_hub(...)` or `MistralTokenizer.from_file(...)` instead.",
            FutureWarning,
        )

        if not strict:
            raise ValueError("strict has to be `True` since v1.10.0.")

        if model not in MODEL_NAME_TO_TOKENIZER_CLS:
            raise TokenizerException(
                f"Unrecognized model: {model}. Use `MistralTokenizer.from_hf_hub(...)` "
                f"or `MistralTokenizer.from_file(...)` to load newer or custom models."
            )

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
            repo_id: The Hugging Face repo ID, e.g. "mistralai/Mistral-Small-2411".
            token: Hugging Face access token for private repos. If `True`, uses the
                locally logged-in token. If `None`, uses no token.
            revision: Git branch, tag, or commit hash to download. If `None`, uses
                the latest revision of the default branch.
            mode: The validation mode to use for the loaded tokenizer.
            force_download: If `True`, re-downloads the tokenizer even if it is
                already present in the local Hugging Face cache.
            local_files_only: If `True`, only uses the local cache and never hits
                the network; fails if the tokenizer is not cached.

        Returns:
            The Mistral tokenizer for the given repository.
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
        r"""Load a tokenizer from a file.

        Detects the tokenizer type (tekken JSON or sentencepiece model) from the
        file, wires in the matching image/audio encoders, validator, and normalizer
        for the tokenizer version, and returns the fully assembled MistralTokenizer.

        Args:
            tokenizer_filename: Path to a tekken (.json containing "tekken" in the
                name) or sentencepiece (.model) tokenizer file.
            mode: The validation mode to use for the loaded tokenizer.

        Returns:
            The loaded tokenizer, configured with the version-appropriate
            InstructTokenizer, validator, and normalizer.

        Raises:
            TokenizerException: If the file is neither a tekken nor a sentencepiece
                tokenizer, or the tokenizer version is unrecognized.
            AssertionError: If the file declares image/audio support that its
                tokenizer version does not support.
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
        r"""Encode a chat completion request into tokens.

        Validates the request, normalizes it into an InstructRequest, and tokenizes
        it. This is the main entry point for chat tokenization.

        Args:
            request: The chat completion request to encode.
            max_model_input_len: Maximum number of input tokens the model accepts.
                Used only when `request.truncate_for_context_length` is `True` to
                truncate the conversation from the start. If `None`, no truncation
                is applied.

        Returns:
            The tokenized request (tokens, message boundaries, etc.), specific to
            the tokenizer version.

        Raises:
            TokenizerException: If `request.truncate_for_context_length` is `True` but
                `max_model_input_len` is `None`.
            MistralCommonException: If request validation fails.
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
        r"""Encode a transcription request into tokens.

        Args:
            request: The transcription request containing the audio to transcribe.

        Returns:
            The tokenized transcription request.
        """
        return self.instruct_tokenizer.encode_transcription(request)

    def encode_speech_request(self, request: SpeechRequest) -> TokenizedType:
        r"""Encode a speech synthesis request into tokens.

        Args:
            request: The speech request containing the text and optional audio.

        Returns:
            The tokenized speech request.
        """
        return self.instruct_tokenizer.encode_speech_request(request)

    def encode_fim(self, request: FIMRequest) -> TokenizedType:
        r"""Encode a fill-in-the-middle request into tokens.

        Wraps the prompt with prefix/middle/suffix markers so the model
        can complete the middle section.

        Args:
            request: The fill-in-the-middle request containing the prompt parts.

        Returns:
            The tokenized fill-in-the-middle request.
        """
        return self.instruct_tokenizer.encode_fim(request)

    def decode(self, tokens: list[int], special_token_policy: SpecialTokenPolicy = SpecialTokenPolicy.IGNORE) -> str:
        r"""Decode a list of tokens into a string.

        Args:
            tokens: List of token IDs to decode.
            special_token_policy: Policy for handling special tokens:
                - IGNORE: Skip special tokens (default)
                - KEEP: Include special token strings in the output
                - RAISE: Raise ValueError if special tokens are present

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
