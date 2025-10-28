import json
from abc import abstractmethod
from typing import Any, Generic, Sequence, overload

import numpy as np

from mistral_common.audio import Audio
from mistral_common.exceptions import (
    InvalidAssistantMessageException,
    InvalidMessageStructureException,
    TokenizerException,
)
from mistral_common.protocol.fim.request import FIMRequest
from mistral_common.protocol.instruct.chunk import (
    AudioChunk,
    AudioURLChunk,
    ContentChunk,
    ImageChunk,
    ImageURLChunk,
    TextChunk,
    ThinkChunk,
    UserContentChunk,
)
from mistral_common.protocol.instruct.messages import (
    UATS,
    AssistantMessage,
    AssistantMessageType,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.request import InstructRequest
from mistral_common.protocol.instruct.tool_calls import Tool, ToolCall
from mistral_common.protocol.transcription.request import TranscriptionRequest
from mistral_common.tokens.tokenizers.audio import AudioEncoder
from mistral_common.tokens.tokenizers.base import (
    FIMRequestType,
    InstructRequestType,
    InstructTokenizer,
    SpecialTokenPolicy,
    SpecialTokens,
    Tokenized,
    TokenizedType,
    Tokenizer,
    UserMessagePosition,
)
from mistral_common.tokens.tokenizers.image import ImageEncoder
from mistral_common.tokens.tokenizers.tekken import Tekkenizer


class InstructTokenizerBase(
    InstructTokenizer, Generic[InstructRequestType, FIMRequestType, TokenizedType, AssistantMessageType]
):
    r"""Base instruct tokenizer."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        image_encoder: ImageEncoder | None = None,
        audio_encoder: AudioEncoder | None = None,
    ):
        r"""Initialize the instruct tokenizer.

        Args:
            tokenizer: The tokenizer to use.
            image_encoder: The image encoder to use if any.
            audio_encoder: The audio encoder to use.
        """
        self.tokenizer = tokenizer
        self.image_encoder = image_encoder
        self.audio_encoder = audio_encoder
        super().__init__(tokenizer, image_encoder, audio_encoder)

    @property
    def mm_encoder(self) -> ImageEncoder | None:
        # this funtion is deprecated, use image_encoder instead
        # TODO(Patrick) - throw a deprecation warning once
        # changes applied to vllm and transformers
        return self.image_encoder

    def start(self) -> list[int]:
        r"""Return the start tokens."""
        return [self.tokenizer.bos_id]

    @staticmethod
    def find_first_last_user(request: InstructRequest) -> tuple[int, int]:
        r"""Find the first and last user message in the request.

        Args:
            request: The request to search for user messages.

        Returns:
            The index of the first and last user message.
        """
        last_user_idx = -1
        first_user_idx = -1
        for i, msg in list(enumerate(request.messages)):
            if isinstance(msg, UserMessage):
                if first_user_idx == -1:
                    first_user_idx = i
                last_user_idx = i
        return first_user_idx, last_user_idx

    @abstractmethod
    def encode_tool_message(self, message: ToolMessage, is_before_last_user_message: bool) -> list[int]:
        r"""Encode a tool message.

        Raises:
            NotImplementedError: The tool message is not implemented for the base tokenizer.
        """
        raise NotImplementedError("Tool message not implemented")

    @abstractmethod
    def encode_assistant_message(
        self, message: AssistantMessageType, is_before_last_user_message: bool, continue_message: bool
    ) -> list[int]:
        r"""Encode an assistant message.

        Raises:
            NotImplementedError: The assistant message is not implemented for the base tokenizer.
        """
        raise NotImplementedError("Assistant message not implemented")

    @abstractmethod
    def encode_think(self, chunk: ThinkChunk) -> list[int]:
        r"""Encode a think chunk.

        Raises:
            NotImplementedError: The think chunk is not implemented for the base tokenizer.
        """
        raise NotImplementedError("Think chunk not implemented")

    def _truncate_for_max_tokens(
        self,
        tokenized: list[list[int] | None],
        messages: list[AssistantMessageType],
        max_tokens: int,
        last_user_message_index: int,
    ) -> None:
        # Tokenizer ⩽ V3 does not support truncation
        return

    @classmethod
    def validate_messages(cls, messages: list[UATS]) -> None:
        # for v7 we start validates messages
        ...

    def encode_instruct(
        self,
        request: InstructRequest[AssistantMessageType, Tool],
    ) -> Tokenized:
        r"""Encode an instruct request.

        Args:
            request: The request to encode.

        Returns:
            The encoded tokens.
        """
        # init at bos
        images: list[np.ndarray] = []
        audios: list[Audio] = []
        prefix_ids: list[int] | None = None
        tokens_list: list[list[int] | None] = []

        # validate messages
        self.validate_messages(request.messages)

        # find last user message
        first_user_idx, last_user_idx = self.find_first_last_user(request)
        for msg_idx, msg in enumerate(request.messages):
            if (
                request.continue_final_message
                and (msg_idx == len(request.messages) - 1)
                and not isinstance(msg, AssistantMessage)
            ):
                raise InvalidMessageStructureException(
                    "Cannot continue final message if it is not an assistant message"
                )
            if isinstance(msg, UserMessage):
                new_tokens, new_images, new_audios = self.encode_user_message(
                    msg,
                    request.available_tools,
                    msg_idx == last_user_idx,
                    msg_idx == first_user_idx,
                    system_prompt=request.system_prompt,
                    force_img_first=True,  # img is always first when providing text/img chunk pair
                )
                images.extend(new_images)
                audios.extend(new_audios)
            elif isinstance(msg, ToolMessage):
                new_tokens = self.encode_tool_message(msg, msg_idx < last_user_idx)
            elif isinstance(msg, AssistantMessage):
                continue_message = request.continue_final_message and (msg_idx == len(request.messages) - 1)

                new_tokens = self.encode_assistant_message(
                    msg, msg_idx < last_user_idx, continue_message=continue_message
                )
                if msg_idx == len(request.messages) - 1:
                    prefix_ids = new_tokens
            elif isinstance(msg, SystemMessage):
                new_tokens = self.encode_system_message(msg)
            else:
                raise TokenizerException(f"Unknown message type {type(msg)}")

            tokens_list.append(new_tokens)

        if request.truncate_at_max_tokens is not None:
            self._truncate_for_max_tokens(
                tokens_list,
                request.messages,
                request.truncate_at_max_tokens,
                last_user_idx,
            )
        tokens = self.start()

        for tok in tokens_list:
            if tok is not None:
                tokens.extend(tok)

        return Tokenized(
            tokens=tokens,
            text=self.decode(tokens, special_token_policy=SpecialTokenPolicy.KEEP),
            prefix_ids=prefix_ids,
            images=images,
            audios=audios,
        )

    def decode(self, tokens: list[int], special_token_policy: SpecialTokenPolicy | None = None) -> str:
        r"""Decode tokens to a string.

        Args:
            tokens: The tokens to decode.
            special_token_policy: The policy to use for special tokens.
                Passing `None` will default to `self._special_token_policy` for
                [Tekkenizer][mistral_common.tokens.tokenizers.tekken.Tekkenizer] and `SpecialTokenPolicy.IGNORE`
                for [SentencePieceTokenizer][mistral_common.tokens.tokenizers.sentencepiece.SentencePieceTokenizer].
                Note that passing `None` will be deprecated and `special_token_policy` will default to
                `SpecialTokenPolicy.IGNORE` in `mistral_common=1.10.0`.

        Returns:
            The decoded string.
        """
        return self.tokenizer.decode(tokens, special_token_policy=special_token_policy)

    def _to_string(self, tokens: list[int]) -> str:
        return self.tokenizer._to_string(tokens)


class InstructTokenizerV1(
    InstructTokenizerBase, Generic[InstructRequestType, FIMRequestType, TokenizedType, AssistantMessageType]
):
    r"""Instruct tokenizer V1.

    This tokenizer has basic for messages. It does not support tools or image inputs.
    """

    def encode_user_message(
        self,
        message: UserMessage,
        available_tools: list[Tool] | None,
        is_last: bool,
        is_first: bool,
        system_prompt: str | None = None,
        force_img_first: bool = False,
    ) -> tuple[list[int], list[np.ndarray], list[Audio]]:
        r"""Encode a user message.

        Args:
            message: The message to encode.
            available_tools: Not used.
            is_last: Not used.
            is_first: Whether the message is the first one.
            system_prompt: The system prompt.
            force_img_first: Not used.

        Returns:
            The encoded tokens and empty list.
        """
        assert isinstance(message.content, str), "Message content must be normalized"
        assert self.image_encoder is None, "InstructTokenizerV1 cannot encode images"

        content = ""
        if is_first and system_prompt:
            content = system_prompt + "\n\n" + message.content
        else:
            content = message.content

        message_txt = f"[INST] {content} [/INST]"
        curr_tokens, image, audio = self.encode_user_content(content=message_txt, is_last=False, system_prompt=None)
        return curr_tokens, image, audio

    def encode_system_message(self, message: SystemMessage) -> list[int]:
        raise NotImplementedError(f"System message encoding not implemented for {self.__class__.__name__}")

    def encode_user_content(
        self,
        content: str | list[UserContentChunk],
        is_last: bool,
        system_prompt: str | None = None,
        force_img_first: bool = False,
    ) -> tuple[list[int], list[np.ndarray], list[Audio]]:
        r"""Encode a user content.

        Args:
            content: The content to encode.
            is_last: Whether the message is the last one.
            system_prompt: The system prompt.
            force_img_first: Not used.

        Returns:
            The encoded tokens and empty list.
        """
        assert isinstance(content, str)

        if is_last and system_prompt:
            content = system_prompt + "\n\n" + content

        tokens = self.tokenizer.encode(content, bos=False, eos=False)
        return tokens, [], []

    def encode_tool_message(self, message: ToolMessage, is_before_last_user_message: bool) -> list[int]:
        r"""Encode a tool message.

        Raises:
            TokenizerException: The tool message is not implemented for this version.
        """
        raise TokenizerException("Tools not implemented for tokenizer V1")

    def encode_assistant_message(
        self, message: AssistantMessageType, is_before_last_user_message: bool, continue_message: bool
    ) -> list[int]:
        r"""Encode an assistant message.

        Args:
            message: The message to encode.
            is_before_last_user_message: Not used.
            continue_message: Whether to continue the message generation.
                Only use this if the assistant message is the last message.

        Returns:
            The encoded tokens.
        """
        assert isinstance(message, AssistantMessage), message
        if message.tool_calls is not None and len(message.tool_calls) > 0:
            raise TokenizerException("Tools not implemented for tokenizer V1")
        if continue_message and message.prefix:
            raise InvalidAssistantMessageException(
                "`continue_message` is only supported for assistant messages that have `prefix=False`."
            )
        elif message.content:
            assert isinstance(message.content, str), "Message content must be a string for tokenizer < V13"
            curr_tokens = self.tokenizer.encode(message.content, bos=False, eos=False)
        else:
            raise TokenizerException(f"{message.content} // {message.tool_calls}")
        if not message.prefix and not continue_message:
            curr_tokens.append(self.tokenizer.eos_id)
        return curr_tokens

    def encode_think(self, chunk: ThinkChunk) -> list[int]:
        r"""Encode a think chunk.

        Raises:
            TokenizerException: The think chunk is not implemented for this version.
        """
        raise TokenizerException("Think not implemented for tokenizer < V13.")

    def encode_fim(self, request: FIMRequest) -> Tokenized:
        r"""Encode a FIM request.

        Raises:
           TokenizerException: The FIM request is not implemented for this version.
        """
        raise TokenizerException(f"FIM not available for {self.tokenizer.version}")

    def encode_transcription(self, request: TranscriptionRequest) -> Tokenized:
        raise TokenizerException(f"Transcription not available for {self.tokenizer.version}")


class InstructTokenizerV2(
    InstructTokenizerV1, Generic[InstructRequestType, FIMRequestType, TokenizedType, AssistantMessageType]
):
    r"""Instruct tokenizer V2.

    This tokenizer adds supports to images, tools and FIM requests.
    """

    _user_message_position_to_encode_tools = UserMessagePosition.last

    def __init__(
        self,
        tokenizer: Tokenizer,
        image_encoder: ImageEncoder | None = None,
        audio_encoder: AudioEncoder | None = None,
    ):
        r"""Initialize the tokenizer.

        Args:
            tokenizer: The tokenizer to use.
            image_encoder: The image encoder to use.
            audio_encoder: The audio encoder to use.
        """
        super().__init__(tokenizer, image_encoder, audio_encoder)
        self.BEGIN_INST = self.tokenizer.get_control_token(SpecialTokens.begin_inst.value)
        self.END_INST = self.tokenizer.get_control_token(SpecialTokens.end_inst.value)
        self.BEGIN_AVAILABLE_TOOLS = self.tokenizer.get_control_token(SpecialTokens.begin_tools.value)
        self.END_AVAILABLE_TOOLS = self.tokenizer.get_control_token(SpecialTokens.end_tools.value)
        self.BEGIN_TOOL_RESULTS = self.tokenizer.get_control_token(SpecialTokens.begin_tool_results.value)
        self.END_TOOL_RESULTS = self.tokenizer.get_control_token(SpecialTokens.end_tool_results.value)
        self.TOOL_CALLS = self.tokenizer.get_control_token(SpecialTokens.tool_calls.value)
        self.BOS = self.tokenizer.get_control_token(SpecialTokens.bos.value)
        self.PREFIX = self.tokenizer.get_control_token(SpecialTokens.prefix.value)
        self.SUFFIX = self.tokenizer.get_control_token(SpecialTokens.suffix.value)

    def encode_user_message(
        self,
        message: UserMessage,
        available_tools: list[Tool] | None,
        is_last: bool,
        is_first: bool,
        system_prompt: str | None = None,
        force_img_first: bool = False,
    ) -> tuple[list[int], list[np.ndarray], list[Audio]]:
        r"""Encode a user message.

        Args:
            message: The message to encode.
            available_tools: The list of available tools if any.
            is_last: Whether the message is the last one.
            is_first: Not used.
            system_prompt: The system prompt.
            force_img_first: Whether to force the image to be first.

        Returns:
            The encoded tokens and the list of images.
        """
        do_encode_tools = False
        do_encode_tools |= is_first and (self._user_message_position_to_encode_tools == UserMessagePosition.first)
        do_encode_tools |= is_last and (self._user_message_position_to_encode_tools == UserMessagePosition.last)
        tools_tokens: list[int] = []

        if do_encode_tools and available_tools:
            tools = [tool.model_dump() for tool in available_tools]
            tools_json_tokens = self.tokenizer.encode(json.dumps(tools, ensure_ascii=False), bos=False, eos=False)
            tools_tokens = [
                self.BEGIN_AVAILABLE_TOOLS,
                *tools_json_tokens,
                self.END_AVAILABLE_TOOLS,
            ]

        tokens, image, audio = self.encode_user_content(
            content=message.content,
            is_last=is_last,
            system_prompt=system_prompt,
            force_img_first=force_img_first,
        )

        prefix_tokens = [*tools_tokens, self.BEGIN_INST]
        suffix_tokens = [self.END_INST]

        curr_tokens = prefix_tokens + tokens + suffix_tokens

        return curr_tokens, image, audio

    def _parse_json_content(self, content: str) -> Any:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content

    def _prepare_tool_result(self, tool_message: ToolMessage) -> dict[str, Any]:
        r"""Bit of a hack due to the way tool results are tokenized."""
        return {
            "name": tool_message.name,
            "content": self._parse_json_content(tool_message.content),
        }

    def encode_tool_message(self, message: ToolMessage, is_before_last_user_message: bool) -> list[int]:
        r"""Encode a tool message.

        Args:
            message: The message to encode.
            is_before_last_user_message: Whether the message is before the last user message. If true, the message is
                not encoded.

        Returns:
            The encoded tokens.
        """
        if is_before_last_user_message:
            # don't tokenize last tool response before last user msg
            return []

        # Currently only supports single tool results
        tool_result_str = json.dumps([self._prepare_tool_result(message)], ensure_ascii=False)
        curr_tokens = [
            self.BEGIN_TOOL_RESULTS,
            *self.tokenizer.encode(tool_result_str, bos=False, eos=False),
            self.END_TOOL_RESULTS,
        ]
        return curr_tokens

    def _prepare_function_call(self, tool_call: ToolCall) -> dict[str, Any]:
        r"""Bit of a hack due to the way function calls are tokenized."""
        return {
            "name": tool_call.function.name,
            "arguments": self._parse_json_content(tool_call.function.arguments),
        }

    def _encode_normal_content_assistant_message(self, message: AssistantMessageType) -> list[int]:
        assert message.content, f"Assistant message must have content. Got {message}"
        assert isinstance(message.content, str), "Message content must be a string for tokenizer < V7"
        return self.tokenizer.encode(message.content.rstrip(" "), bos=False, eos=False)

    def _encode_tool_calls_in_assistant_message(self, message: AssistantMessageType) -> list[int]:
        assert message.tool_calls, f"Assistant message must have tool calls. Got {message}"
        prepared_tool_calls = []
        for tool_call in message.tool_calls:
            prepared_tool_calls.append(self._prepare_function_call(tool_call))
        tool_call_str = json.dumps(prepared_tool_calls, ensure_ascii=False)
        curr_tokens = [
            self.TOOL_CALLS,
            *self.tokenizer.encode(tool_call_str, bos=False, eos=False),
        ]
        return curr_tokens

    def encode_assistant_message(
        self, message: AssistantMessageType, is_before_last_user_message: bool, continue_message: bool
    ) -> list[int]:
        r"""Encode an assistant message.

        Args:
            message: The message to encode.
            is_before_last_user_message: Whether the message is before the last user message. If has tools and true, the
                message is not encoded.
            continue_message: Whether to continue the message generation.
                Only use this if the assistant message is the last message.

        Returns:
            The encoded tokens.
        """
        if message.tool_calls and message.content:
            raise ValueError(f"Cannot have tool calls and content defined in the same assistant message {message}")
        if continue_message and message.prefix:
            raise InvalidAssistantMessageException(
                "`continue_message` is only supported for assistant messages that have `prefix=False`."
            )

        if message.tool_calls:
            if is_before_last_user_message:
                # don't tokenize tool call before last user message
                return []
            curr_tokens = self._encode_tool_calls_in_assistant_message(message)
        elif message.content:
            assert isinstance(message.content, str), "Message content must be a string for tokenizer < V7"
            curr_tokens = self._encode_normal_content_assistant_message(message)
        else:
            raise TokenizerException(f"Invalid assistant message: {message.content}")
        if not message.prefix and not continue_message:
            curr_tokens.append(self.tokenizer.eos_id)
        return curr_tokens

    def _encode_infilling(self, text: str) -> list[int]:
        r"""Remove prefix space in the case of SentencePieceTokenizers."""

        return self.tokenizer.encode("☺" + text, bos=False, eos=False)[2:]

    def encode_fim(self, request: FIMRequest) -> Tokenized:
        r"""Encode a FIM request.

        Args:
            request: The request to encode.

        Returns:
            The encoded tokens.
        """
        prefix_tokens = self.tokenizer.encode(request.prompt, bos=False, eos=False)
        suffix_tokens = self._encode_infilling(request.suffix) if request.suffix else []
        tokens = [
            self.BOS,
            self.SUFFIX,
            *suffix_tokens,
            self.PREFIX,
            *prefix_tokens,
        ]
        return Tokenized(tokens=tokens, text=self.decode(tokens, special_token_policy=SpecialTokenPolicy.KEEP))


class InstructTokenizerV3(
    InstructTokenizerV2, Generic[InstructRequestType, FIMRequestType, TokenizedType, AssistantMessageType]
):
    r"""Instruct tokenizer V3.

    The only difference with V2 tokenizer is that it encodes the tool messages differently.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        image_encoder: ImageEncoder | None = None,
        audio_encoder: AudioEncoder | None = None,
    ):
        r"""Initialize the tokenizer.

        Args:
            tokenizer: The tokenizer to use.
            image_encoder: The image encoder to use.
            audio_encoder: The audio encoder to use.
        """
        super().__init__(tokenizer, image_encoder=image_encoder, audio_encoder=audio_encoder)

    def _prepare_function_call(self, tool_call: ToolCall) -> dict[str, Any]:
        function_call = {
            "name": tool_call.function.name,
            "arguments": self._parse_json_content(tool_call.function.arguments),
        }

        if tool_call.id and tool_call.id != "null":
            function_call["id"] = tool_call.id

        return function_call

    def _prepare_tool_result(self, tool_message: ToolMessage) -> dict[str, Any]:
        assert tool_message.tool_call_id is not None, "Tool message has to have the tool call id defined in v3"

        return {
            "content": self._parse_json_content(tool_message.content),
            "call_id": tool_message.tool_call_id,
        }

    def encode_tool_message(self, message: ToolMessage, is_before_last_user_message: bool) -> list[int]:
        r"""Encode a tool message.

        Note:
            Same as [V2][mistral_common.tokens.tokenizers.instruct.InstructTokenizerV2.encode_tool_message] but tools
            are not wrapped in a list and the history is also tokenized.

        Args:
            message: The message to encode.
            is_before_last_user_message: Whether the message is before the last user message. If true, the message is
                not encoded.

        Returns:
            The encoded tokens.
        """
        tool_result_str = json.dumps(self._prepare_tool_result(message), ensure_ascii=False)
        curr_tokens = [
            self.BEGIN_TOOL_RESULTS,
            *self.tokenizer.encode(tool_result_str, bos=False, eos=False),
            self.END_TOOL_RESULTS,
        ]
        return curr_tokens

    def encode_assistant_message(
        self, message: AssistantMessageType, is_before_last_user_message: bool, continue_message: bool
    ) -> list[int]:
        r"""Encode an assistant message.

        Note:
            Same as [V2][mistral_common.tokens.tokenizers.instruct.InstructTokenizerV2.encode_assistant_message] but
            always encode the tool history.
            continue_message: Whether to continue the message generation.
                Only use this if the assistant message is the last message.

        Args:
            message: The message to encode.
            is_before_last_user_message: Not used.

        Returns:
            The encoded tokens.
        """
        return super().encode_assistant_message(message, False, continue_message)

    @overload
    def _encode_content_chunk(self, chunk: str | TextChunk | ThinkChunk) -> tuple[list[int], None, None]: ...
    @overload
    def _encode_content_chunk(self, chunk: ImageChunk | ImageURLChunk) -> tuple[list[int], np.ndarray, None]: ...
    @overload
    def _encode_content_chunk(self, chunk: AudioChunk | AudioURLChunk) -> tuple[list[int], None, Audio]: ...
    def _encode_content_chunk(self, chunk: str | ContentChunk) -> tuple[list[int], np.ndarray | None, Audio | None]:
        if isinstance(chunk, str):
            return self.tokenizer.encode(chunk, bos=False, eos=False), None, None
        elif isinstance(chunk, TextChunk):
            return self.tokenizer.encode(chunk.text, bos=False, eos=False), None, None
        elif isinstance(chunk, ThinkChunk):
            return self.encode_think(chunk), None, None
        elif isinstance(chunk, (ImageChunk, ImageURLChunk)):
            assert self.image_encoder is not None, "Make sure to define a image encoder at init"
            img_encoding = self.image_encoder(chunk)

            return img_encoding.tokens, img_encoding.image, None

        elif isinstance(chunk, (AudioChunk, AudioURLChunk)):
            # the following is only possible for >= v7
            assert self.audio_encoder is not None, "Make sure to define a audio encoder at init"
            audio_encoding = self.audio_encoder(chunk)

            return audio_encoding.tokens, None, audio_encoding.audio
        else:
            raise ValueError(f"Unknown chunk type: {chunk}")

    def _encode_content_chunks(
        self, content: Sequence[ContentChunk]
    ) -> tuple[list[int], list[np.ndarray], list[Audio]]:
        tokens: list[int] = []
        images: list[np.ndarray] = []
        audio: list[Audio] = []

        for chunk in content:
            chunk_tokens, maybe_image, maybe_audio = self._encode_content_chunk(chunk)
            tokens.extend(chunk_tokens)
            if maybe_image is not None:
                images.append(maybe_image)
            if maybe_audio is not None:
                audio.append(maybe_audio)
        return tokens, images, audio

    def encode_user_content(
        self,
        content: str | list[UserContentChunk],
        is_last: bool,
        system_prompt: str | None = None,
        force_img_first: bool = False,
    ) -> tuple[list[int], list[np.ndarray], list[Audio]]:
        r"""Encode a user content.

        Args:
            content: The content to encode.
            is_last: Whether the message is the last one.
            system_prompt: The system prompt.
            force_img_first: Whether to force the image to be first.

        Returns:
            The encoded tokens and the images.
        """
        if isinstance(content, str):
            return super().encode_user_content(content, is_last, system_prompt)

        tokens: list[int] = []
        images: list[np.ndarray] = []
        audio: list[Audio] = []

        has_one_img_one_text_first = len(content) == 2 and isinstance(content[1], (ImageChunk, ImageURLChunk))
        if force_img_first and has_one_img_one_text_first:
            # make sure that if exactly one image and text chunk are passed we force the image chunk to be first
            content = [content[1], content[0]]

        first_chunk = True
        for chunk in content:
            content_str = ""
            if first_chunk and is_last and system_prompt:
                first_chunk = False
                content_str = system_prompt + "\n\n"
                tokens += self.tokenizer.encode(content_str, bos=False, eos=False)

            if isinstance(chunk, (AudioChunk, AudioURLChunk)):
                assert not content_str, (
                    f"It is not possible that `content` is non-empty when chunk is of type {type(chunk)}."
                )
                chunk_tokens, _, chunk_audio = self._encode_content_chunk(chunk)
                audio.append(chunk_audio)
            elif isinstance(chunk, (ImageChunk, ImageURLChunk)):
                chunk_tokens, chunk_image, _ = self._encode_content_chunk(chunk)
                images.append(chunk_image)
            else:
                chunk_tokens = self._encode_content_chunk(chunk)[0]
            tokens.extend(chunk_tokens)

        return tokens, images, audio


class InstructTokenizerV7(InstructTokenizerV3):
    r"""Instruct tokenizer V7.

    The difference with V3 tokenizer is that it encodes the system prompts differently:
    - in V7 the system prompts are treated as separate SystemMessages
    - they are no longer prepended to the last user message
    - they are printed between special tokens

    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        image_encoder: ImageEncoder | None = None,
        audio_encoder: AudioEncoder | None = None,
    ) -> None:
        r"""Initialize the tokenizer.

        Args:
            tokenizer: The tokenizer to use.
            image_encoder: The image encoder to use.
            audio_encoder: The audio encoder to use.
        """

        super().__init__(tokenizer, image_encoder, audio_encoder)
        self.BEGIN_SYSTEM = self.tokenizer.get_control_token(SpecialTokens.begin_system.value)
        self.END_SYSTEM = self.tokenizer.get_control_token(SpecialTokens.end_system.value)
        self.BEGIN_TOOL_CONTENT = self.tokenizer.get_control_token(SpecialTokens.begin_tool_content.value)

        self.TRANSCRIBE = None
        if audio_encoder is not None:
            self.TRANSCRIBE = self.tokenizer.get_control_token(SpecialTokens.transcribe.value)

    def _truncate_for_max_tokens(
        self,
        tokenized_messages: list[list[int] | None],
        messages: list[AssistantMessageType],
        max_tokens: int,
        last_user_message_index: int,
    ) -> None:
        # drop some messages to fit in max_tokens. Rules:
        # - don't drop any system messages
        # - when a user message is dropped, all following assistant|tool message should be dropped until the next
        #   user message
        # - we never drop the last message
        to_drop = sum(len(t) for t in tokenized_messages if t is not None) - max_tokens

        def drop(idx: int) -> None:
            nonlocal to_drop
            if isinstance(messages[idx], SystemMessage):
                # never drop system messages
                return
            if idx == last_user_message_index:
                # never drop the last user message
                return
            tok = tokenized_messages[idx]
            assert tok is not None
            to_drop -= len(tok)
            tokenized_messages[idx] = None

        current_idx = 0
        while to_drop > 0 and current_idx < len(messages):
            drop(current_idx)
            current_idx += 1
            if isinstance(messages[current_idx - 1], UserMessage):
                # if we just dropped a UserMessage,
                # also drop everything until the next user message
                while current_idx < len(messages) and not isinstance(messages[current_idx], UserMessage):
                    drop(current_idx)
                    current_idx += 1

        if to_drop > 0:
            raise TokenizerException("Input couldn't fit in truncate_at_max_token")

    def encode_system_message(self, message: SystemMessage) -> list[int]:
        r"""Encode a system message.

        Args:
            message: The message to encode.

        Returns:
            The encoded tokens.
        """

        tokens = [self.BEGIN_SYSTEM]
        if isinstance(content := message.content, str):
            content = [TextChunk(text=content)]
        tokens += self._encode_content_chunks(content)[0]
        tokens.append(self.END_SYSTEM)
        return tokens

    def encode_user_content(
        self,
        content: str | list[UserContentChunk],
        is_last: bool,
        system_prompt: str | None = None,
        force_img_first: bool = False,
    ) -> tuple[list[int], list[np.ndarray], list[Audio]]:
        r"""Encode a user content.

        Args:
            content: The content to encode.
            is_last: Whether the message is the last one.
            system_prompt: The system prompt.
            force_img_first: Whether to force the image to be first.

        Returns:
            The encoded tokens and the images.
        """
        assert system_prompt is None, "in Tokenizer V7 we don't encode system prompts in user messages"

        if isinstance(content, str):
            return super().encode_user_content(content, is_last, system_prompt)

        has_one_img_one_text_first = len(content) == 2 and isinstance(content[1], (ImageChunk, ImageURLChunk))
        if force_img_first and has_one_img_one_text_first:
            # make sure that if exactly one image and text chunk are passed we force the image chunk to be first
            content = [content[1], content[0]]

        tokens, images, audio = self._encode_content_chunks(content)
        return tokens, images, audio

    def encode_user_message(
        self,
        message: UserMessage,
        available_tools: list[Tool] | None,
        is_last: bool,
        is_first: bool,
        system_prompt: str | None = None,
        force_img_first: bool = False,
    ) -> tuple[list[int], list[np.ndarray], list[Audio]]:
        r"""Encode a user message.

        Args:
            message: The message to encode.
            available_tools: The list of available tools if any.
            is_last: Whether the message is the last one.
            is_first: Whether the message is the first one.
            system_prompt: Not used.
            force_img_first: Whether to force the image to be first.

        Returns:
            The encoded tokens and the list of images.
        """
        assert system_prompt is None, "in Tokenizer V7 we don't encode system prompts in user messages"

        tokens, images, audio = super().encode_user_message(
            message,
            available_tools,
            is_last=is_last,
            is_first=is_first,
            system_prompt=None,
            force_img_first=force_img_first,
        )

        return tokens, images, audio

    def encode_transcription(self, request: TranscriptionRequest) -> Tokenized:
        r"""
        Encodes an audio transcription request into a tokenized format.

        This method processes a transcription request containing audio data,
        encodes the user message, and returns the tokenized output.

        Args:
            request: The transcription request object containing
                the audio data to be encoded.

        Returns:
            Tokenized: The tokenized representation of the audio data, including processed audio and tokens
        """

        assert self.TRANSCRIBE is not None, f"{self.__class__.__name__} needs to have a TRANSCRIBE token"
        prefix = self.start()
        tokens, _, audio = self.encode_user_message(
            UserMessage(content=[AudioChunk(input_audio=request.audio)]),
            available_tools=[],
            is_last=True,
            is_first=True,
            system_prompt=None,
        )

        tokens = [*prefix, *tokens]
        if request.language is not None:
            language_string = f"lang:{request.language}"  # no space.
            tokens += self.tokenizer.encode(language_string, bos=False, eos=False)

        tokens.append(self.TRANSCRIBE)
        return Tokenized(tokens=tokens, text=self.tokenizer._to_string(tokens), audios=audio)

    @classmethod
    def validate_messages(cls, messages: list[UATS]) -> None:
        if cls._has_audio(messages):
            if any(isinstance(message, SystemMessage) for message in messages):
                raise ValueError("System messages are not yet allowed when audio is present")

    @staticmethod
    def _has_audio(messages: list[UATS]) -> bool:
        return any(
            isinstance(message, UserMessage)
            and isinstance(message.content, list)
            and any(isinstance(chunk, AudioChunk) for chunk in message.content)
            for message in messages
        )

    def encode_tool_message(self, message: ToolMessage, is_before_last_user_message: bool) -> list[int]:
        r"""Encode a tool message.

        Note:
            Same as [V3][mistral_common.tokens.tokenizers.instruct.InstructTokenizerV3.encode_tool_message]
            but tools are not wrapped in a list and history is also tokenized

        Args:
            message: The message to encode.
            is_before_last_user_message: Not used.

        Returns:
            The encoded tokens.
        """
        assert message.tool_call_id is not None
        assert isinstance(message.content, str), "Message content must be normalized"
        tool_call_id_tokens = self.tokenizer.encode(message.tool_call_id, bos=False, eos=False)
        tokens = self.tokenizer.encode(message.content, bos=False, eos=False)

        prefix_tokens = [
            self.BEGIN_TOOL_RESULTS,
            *tool_call_id_tokens,
            self.BEGIN_TOOL_CONTENT,
        ]
        curr_tokens = [
            *prefix_tokens,
            *tokens,
            self.END_TOOL_RESULTS,
        ]
        return curr_tokens

    def encode_assistant_message(
        self, message: AssistantMessageType, is_before_last_user_message: bool, continue_message: bool
    ) -> list[int]:
        r"""Encode an assistant message.

        Args:
            message: The message to encode.
            is_before_last_user_message: Not used.
            continue_message: Whether to continue the message generation.
                Only use this if the assistant message is the last message.

        Returns:
            The encoded tokens.
        """
        if not message.content and not message.tool_calls:
            raise TokenizerException(f"Invalid assistant message: {message}")
        if continue_message and message.prefix:
            raise InvalidAssistantMessageException(
                "`continue_message` is only supported for assistant messages that have `prefix=False`."
            )

        curr_tokens: list = []
        if message.content:
            if isinstance(message.content, str):
                curr_tokens = self._encode_normal_content_assistant_message(message)
            elif isinstance(message.content, list):
                curr_tokens += self._encode_content_chunks(message.content)[0]
        if message.tool_calls:
            curr_tokens += self._encode_tool_calls_in_assistant_message(message)
        if not message.prefix and not continue_message:
            curr_tokens.append(self.tokenizer.eos_id)

        return curr_tokens


class InstructTokenizerV11(InstructTokenizerV7):
    r"""Instruct tokenizer V11.

    The difference with V7 tokenizer is that it encodes tool calls differently:
    Tool call results are encoded as :
    - [begin tool call] call_name_tokens [call id] call_id_tokens [args] content tokens
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        image_encoder: ImageEncoder | None = None,
        audio_encoder: AudioEncoder | None = None,
    ) -> None:
        super().__init__(tokenizer, image_encoder, audio_encoder)
        self.ARGS = self.tokenizer.get_control_token(SpecialTokens.args.value)
        self.CALL_ID = self.tokenizer.get_control_token(SpecialTokens.call_id.value)

    def _encode_tool_calls_in_assistant_message(self, message: AssistantMessageType) -> list[int]:
        assert message.tool_calls, f"Assistant message must have tool calls. Got {message}"
        curr_tokens = []
        for tool_call in message.tool_calls:
            prepared = self._prepare_function_call(tool_call)

            ids = []
            if "id" in prepared:
                ids = [self.CALL_ID, *self.tokenizer.encode(prepared["id"], bos=False, eos=False)]

            curr_tokens += [
                self.TOOL_CALLS,
                *self.tokenizer.encode(prepared["name"], bos=False, eos=False),
                *ids,
                self.ARGS,
                *self.tokenizer.encode(json.dumps(prepared["arguments"], ensure_ascii=False), bos=False, eos=False),
            ]
        return curr_tokens


class InstructTokenizerV13(InstructTokenizerV11):
    r"""Instruct tokenizer V13.

    The difference with V11 tokenizer is that it encodes tool calls differently:
        - available tools are tokenized at the first user message.
        - call id is no longer tokenized for tool calls or results.
    """

    _user_message_position_to_encode_tools = UserMessagePosition.first

    def __init__(
        self,
        tokenizer: Tokenizer,
        image_encoder: ImageEncoder | None = None,
        audio_encoder: AudioEncoder | None = None,
    ) -> None:
        super().__init__(tokenizer, image_encoder, audio_encoder)
        assert isinstance(tokenizer, Tekkenizer), f"Tokenizer must be a Tekkenizer. Got {type(tokenizer)}"
        if (
            SpecialTokens.begin_think.value in tokenizer._special_tokens_reverse_vocab
            and SpecialTokens.end_think.value in tokenizer._special_tokens_reverse_vocab
        ):
            self.BEGIN_THINK: int | None = tokenizer.get_control_token(SpecialTokens.begin_think.value)
            self.END_THINK: int | None = tokenizer.get_control_token(SpecialTokens.end_think.value)
        else:
            self.BEGIN_THINK = None
            self.END_THINK = None

    def _encode_tool_calls_in_assistant_message(self, message: AssistantMessageType) -> list[int]:
        assert message.tool_calls, f"Assistant message must have tool calls. Got {message}"
        curr_tokens = []
        for tool_call in message.tool_calls:
            assert tool_call.id and tool_call.id != "null"
            prepared = self._prepare_function_call(tool_call)

            curr_tokens += [
                self.TOOL_CALLS,
                *self.tokenizer.encode(prepared["name"], bos=False, eos=False),
                self.ARGS,
                *self.tokenizer.encode(json.dumps(prepared["arguments"], ensure_ascii=False), bos=False, eos=False),
            ]
        return curr_tokens

    def encode_tool_message(self, message: ToolMessage, is_before_last_user_message: bool) -> list[int]:
        r"""Encode a tool message.

        Args:
            message: The message to encode.
            is_before_last_user_message: Not used.
        Returns:
            The encoded tokens.
        """
        assert message.tool_call_id is not None, "Tool call id must be provided for tokenizer >= v13"

        tokens = self.tokenizer.encode(message.content, bos=False, eos=False)
        curr_tokens = [
            self.BEGIN_TOOL_RESULTS,
            *tokens,
            self.END_TOOL_RESULTS,
        ]
        return curr_tokens

    def encode_think(self, chunk: ThinkChunk) -> list[int]:
        r"""Encode a thinking chunk.

        Args:
            chunk: The thinking chunk to encode.
        Returns:
            The encoded tokens.
        """
        assert self.BEGIN_THINK is not None, "think tokens are not available for this tokenizer."
        assert self.END_THINK is not None, "think tokens are not available for this tokenizer."
        tokens = self.tokenizer.encode(chunk.thinking, bos=False, eos=False)
        think_tokens = [self.BEGIN_THINK, *tokens]
        if chunk.closed:
            think_tokens.append(self.END_THINK)
        return think_tokens
