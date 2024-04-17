from __future__ import annotations

from pathlib import Path
from typing import Type

from mistral_common.exceptions import (
    TokenizerException,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.validator import (
    MistralRequestValidator,
    MistralRequestValidatorV3,
    ValidationMode,
)
from mistral_common.tokens.instruct.normalize import InstructRequestNormalizer
from mistral_common.tokens.tokenizers.base import InstructTokenizer, Tokenized
from mistral_common.tokens.tokenizers.sentencepiece import (
    SentencePieceInstructTokenizerV1,
    SentencePieceInstructTokenizerV2,
    SentencePieceInstructTokenizerV3,
)


class MistralTokenizer:
    def __init__(
        self,
        instruct_tokenizer: InstructTokenizer,
        validator: Type[MistralRequestValidator] = MistralRequestValidator,
        request_normalizer: Type[InstructRequestNormalizer] = InstructRequestNormalizer,
        mode: ValidationMode = ValidationMode.test,
    ):
        self._chat_completion_request_validator = validator(mode)
        self._instruct_request_normalizer = request_normalizer()
        self.instruct_tokenizer = instruct_tokenizer


    @classmethod
    def _data_path(cls) -> Path:
        return Path(__file__).parents[2] / "data"

    @classmethod
    def v1(cls) -> MistralTokenizer:
        """open-mistral-7B // open-mixtral-8x7B // mistral-embed"""
        return cls.from_file(str(cls._data_path() / "tokenizer.model.v1"), mode=ValidationMode.test)

    @classmethod
    def v2(cls) -> MistralTokenizer:
        """mistral-small // mistral-large"""
        return cls.from_file(
            str(cls._data_path() / "mistral_instruct_tokenizer_240216.model.v2"), mode=ValidationMode.test
        )

    @classmethod
    def v3(cls) -> MistralTokenizer:
        """open-mixtral-8x22B"""
        return cls.from_file(
            str(cls._data_path() / "mistral_instruct_tokenizer_240216.model.v3"), mode=ValidationMode.test
        )

    @classmethod
    def from_model(cls, model: str) -> MistralTokenizer:
        model_name_to_tokenizer_cls = {
            "open-mistral-7b": MistralTokenizer.v1,
            "open-mixtral-8x7b": MistralTokenizer.v1,
            "mistral-embed": MistralTokenizer.v1,
            "mistral-small": MistralTokenizer.v2,
            "mistral-large": MistralTokenizer.v2,
            "open-mixtral-8x22b": MistralTokenizer.v3,
        }

        # Prefix search the model name mapping
        for model_name, tokenizer_cls in model_name_to_tokenizer_cls.items():
            if model_name in model:
                return tokenizer_cls()

        raise TokenizerException(f"Unrecognized model: {model}")

    @classmethod
    def from_file(cls, tokenizer_filename: str, mode: ValidationMode = ValidationMode.test) -> MistralTokenizer:
        """
        Depending on which model we are loading, tokenization and validation might be different. 💩
        """
        if tokenizer_filename.endswith(".model.v1"):
            return cls(
                SentencePieceInstructTokenizerV1(tokenizer_filename),
                validator=MistralRequestValidator,
                request_normalizer=InstructRequestNormalizer,
                mode=mode,
            )
        elif tokenizer_filename.endswith(".model.v2"):
            return cls(
                SentencePieceInstructTokenizerV2(tokenizer_filename),
                validator=MistralRequestValidator,
                request_normalizer=InstructRequestNormalizer,
                mode=mode,
            )
        elif tokenizer_filename.endswith(".model.v3"):
            return cls(
                SentencePieceInstructTokenizerV3(tokenizer_filename),
                validator=MistralRequestValidatorV3,
                request_normalizer=InstructRequestNormalizer,
                mode=mode,
            )
        elif tokenizer_filename.endswith(".model"):
            return cls(
                SentencePieceInstructTokenizerV1(tokenizer_filename),
                validator=MistralRequestValidator,
                request_normalizer=InstructRequestNormalizer,
                mode=mode,
            )
        else:
            raise TokenizerException(f"Unrecognized tokenizer filename: {tokenizer_filename}")

    def encode_chat_completion(self, request: ChatCompletionRequest) -> Tokenized:
        validated_request = self._chat_completion_request_validator.validate_request(request)
        instruct_request = self._instruct_request_normalizer.from_chat_completion_request(validated_request)
        return self.instruct_tokenizer.encode_instruct(instruct_request)
