from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Generic, List, Union

from mistral_common.exceptions import (
    TokenizerException,
)
from mistral_common.protocol.instruct.messages import (
    UATS,
    AssistantMessageType,
    SystemMessageType,
    ToolMessageType,
    UserMessageType,
)
from mistral_common.protocol.instruct.normalize import InstructRequestNormalizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.validator import (
    MistralRequestValidator,
    MistralRequestValidatorV3,
    ValidationMode,
)
from mistral_common.tokens.instruct.request import FIMRequest
from mistral_common.tokens.tokenizers.base import (
    InstructRequest,
    InstructRequestType,
    InstructTokenizer,
    TokenizedType,
    TokenizerVersion,
)
from mistral_common.tokens.tokenizers.sentencepiece import (
    InstructTokenizerV1,
    InstructTokenizerV2,
    InstructTokenizerV3,
    SentencePieceTokenizer,
    is_sentencepiece,
)
from mistral_common.tokens.tokenizers.tekken import Tekkenizer, is_tekken


class MistralTokenizer(
    Generic[UserMessageType, AssistantMessageType, ToolMessageType, SystemMessageType, TokenizedType]
):
    def __init__(
        self,
        instruct_tokenizer: InstructTokenizer[InstructRequest, FIMRequest, TokenizedType, AssistantMessageType],
        validator: MistralRequestValidator[UserMessageType, AssistantMessageType, ToolMessageType, SystemMessageType],
        request_normalizer: InstructRequestNormalizer[
            UserMessageType, AssistantMessageType, ToolMessageType, SystemMessageType, InstructRequestType
        ],
    ):
        self._chat_completion_request_validator = validator
        self._instruct_request_normalizer = request_normalizer
        self.instruct_tokenizer = instruct_tokenizer

    @classmethod
    def _data_path(cls) -> Path:
        return Path(__file__).parents[2] / "data"

    @classmethod
    def v1(cls) -> "MistralTokenizer":
        """open 7B x 8x7B + embed"""
        return cls.from_file(str(cls._data_path() / "tokenizer.model.v1"), mode=ValidationMode.test)

    @classmethod
    def v2(cls) -> "MistralTokenizer":
        """mistral-small // mistral-large"""
        return cls.from_file(
            str(cls._data_path() / "mistral_instruct_tokenizer_240216.model.v2"), mode=ValidationMode.test
        )

    @classmethod
    def v3(cls, is_tekken: bool = False) -> "MistralTokenizer":
        """open-mixtral-8x22B"""
        tokenizer_name = "mistral_instruct_tokenizer_240323.model.v3" if not is_tekken else "tekken_240718.json"
        return cls.from_file(str(cls._data_path() / tokenizer_name), mode=ValidationMode.test)

    @classmethod
    def from_model(cls, model: str) -> "MistralTokenizer":
        model_name_to_tokenizer_cls: Dict[str, Callable[[], MistralTokenizer]] = {
            "open-mistral-7b": MistralTokenizer.v1,
            "open-mixtral-8x7b": MistralTokenizer.v1,
            "mistral-embed": MistralTokenizer.v1,
            "mistral-small": MistralTokenizer.v2,
            "mistral-large": MistralTokenizer.v2,
            "open-mixtral-8x22b": MistralTokenizer.v3,
            "codestral-22b": MistralTokenizer.v3,
            "mistral-nemo": lambda: MistralTokenizer.v3(is_tekken=True),
        }

        # Prefix search the model name mapping
        for model_name, tokenizer_cls in model_name_to_tokenizer_cls.items():
            if model_name in model:
                return tokenizer_cls()

        raise TokenizerException(f"Unrecognized model: {model}")

    @classmethod
    def from_file(
        cls,
        tokenizer_filename: str,
        mode: ValidationMode = ValidationMode.test,
    ) -> "MistralTokenizer":
        """
        Depending on which model we are loading, tokenization and validation might be different. 💩
        """
        tokenizer: Union[SentencePieceTokenizer, Tekkenizer]

        if is_tekken(tokenizer_filename):
            tokenizer = Tekkenizer.from_file(tokenizer_filename)
        elif is_sentencepiece(tokenizer_filename):
            tokenizer = SentencePieceTokenizer(tokenizer_filename)
        else:
            raise TokenizerException(f"Unrecognized tokenizer file: {tokenizer_filename}")

        request_normalizer = InstructRequestNormalizer.normalizer()

        if tokenizer.version == TokenizerVersion.v1:
            return MistralTokenizer(
                InstructTokenizerV1(tokenizer),
                validator=MistralRequestValidator(mode=mode),
                request_normalizer=request_normalizer,
            )
        elif tokenizer.version == TokenizerVersion.v2:
            return MistralTokenizer(
                InstructTokenizerV2(tokenizer),
                validator=MistralRequestValidator(mode=mode),
                request_normalizer=request_normalizer,
            )
        elif tokenizer.version == TokenizerVersion.v3:
            return MistralTokenizer(
                InstructTokenizerV3(tokenizer),
                validator=MistralRequestValidatorV3(mode=mode),
                request_normalizer=request_normalizer,
            )

        raise TokenizerException(f"Unrecognized tokenizer version: {tokenizer.version}")

    def encode_chat_completion(self, request: ChatCompletionRequest[UATS]) -> TokenizedType:
        validated_request = self._chat_completion_request_validator.validate_request(request)
        instruct_request = self._instruct_request_normalizer.from_chat_completion_request(validated_request)
        return self.instruct_tokenizer.encode_instruct(instruct_request)

    def encode_fim(self, request: FIMRequest) -> TokenizedType:
        return self.instruct_tokenizer.encode_fim(request)

    def decode(self, tokens: List[int]) -> str:
        return self.instruct_tokenizer.decode(tokens)
