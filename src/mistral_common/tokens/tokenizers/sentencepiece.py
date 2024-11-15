import json
import logging
import os
from abc import abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Set, Tuple, Union

import numpy as np
from sentencepiece import SentencePieceProcessor

from mistral_common.exceptions import TokenizerException
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    AssistantMessageType,
    ContentChunk,
    SystemMessage,
    TextChunk,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.tool_calls import Tool, ToolCall
from mistral_common.tokens.instruct.request import FIMRequest, InstructRequest
from mistral_common.tokens.tokenizers.base import (
    FIMRequestType,
    InstructRequestType,
    InstructTokenizer,
    SpecialTokens,
    Tokenized,
    TokenizedType,
    Tokenizer,
    TokenizerVersion,
)
from mistral_common.tokens.tokenizers.multimodal import MultimodalConfig, MultiModalEncoder, MultiModalVersion


def is_sentencepiece(path: Union[str, Path]) -> bool:
    if isinstance(path, str):
        path = Path(path)

    instruct_versions = list(TokenizerVersion.__members__)
    mm_versions = list(MultiModalVersion.__members__) + [""]  # allow no mm version
    suffixes = [f".model.{v}{m}" for v in instruct_versions for m in mm_versions] + [".model"]

    return path.is_file() and any(path.name.endswith(suffix) for suffix in suffixes)


def get_spm_version(tokenizer_filename: str, raise_deprecated: bool = False) -> TokenizerVersion:
    _version_str = tokenizer_filename.split(".")[-1].split("m")[0]
    if _version_str == "model":
        if raise_deprecated:
            raise TokenizerException(f"Make sure to rename your tokenizer file to end with {tokenizer_filename}.v1.")

        # tokenizer.model => tokenizer.model.v1
        return TokenizerVersion("v1")

    if _version_str not in TokenizerVersion.__members__:
        raise TokenizerException(f"Unrecognized tokenizer filename: {tokenizer_filename}")

    return TokenizerVersion(_version_str)


def get_mm_config(tokenizer_filename: str) -> Optional[MultimodalConfig]:
    _version_str = tokenizer_filename.split(".")[-1]
    if "m" not in _version_str:
        return None

    _mm_version_str = "m" + _version_str.split("m")[-1]

    if _mm_version_str not in MultiModalVersion.__members__:
        raise TokenizerException(f"Unrecognized tokenizer filename: {tokenizer_filename}")

    return MultiModalVersion(_mm_version_str).config


class SentencePieceTokenizer(Tokenizer):
    def __init__(self, model_path: str, tokenizer_version: Optional[TokenizerVersion] = None) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self._model = SentencePieceProcessor(model_file=model_path)

        assert self._model.vocab_size() == self._model.get_piece_size()
        self._vocab = [self._model.id_to_piece(i) for i in range(self.n_words)]

        self._version: TokenizerVersion = tokenizer_version or get_spm_version(model_path, raise_deprecated=False)

        super().__init__()

    @property
    def version(self) -> TokenizerVersion:
        return self._version

    def get_control_token(self, s: str) -> int:
        return self._model.piece_to_id(s)  # type: ignore

    @property
    def n_words(self) -> int:
        return self._model.vocab_size()  # type: ignore

    def vocab(self) -> List[str]:
        return self._vocab

    @property
    def bos_id(self) -> int:
        return self._model.bos_id()  # type: ignore

    @property
    def eos_id(self) -> int:
        return self._model.eos_id()  # type: ignore

    @cached_property
    def _control_tokens(self) -> Set[int]:
        return {tok for tok in range(self.n_words) if self._model.IsControl(tok)}

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert isinstance(s, str)
        t: List[int] = self._model.encode(s)
        if bos:
            t = [self.bos_id, *t]
        if eos:
            t = [*t, self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self._model.decode(t)  # type: ignore

    def id_to_piece(self, token_id: int) -> str:
        return self._model.id_to_piece(token_id)  # type: ignore

    def to_string(self, tokens: List[int]) -> str:
        """
        Converts tokens into a string for debugging purposes
        """
        text = ""
        curr_tokens: List[int] = []
        for tok in tokens:
            if tok in self._control_tokens:
                if curr_tokens:
                    text += "".join([self.id_to_piece(tok) for tok in curr_tokens])
                    curr_tokens = []

                text += self.id_to_piece(tok)

            else:
                curr_tokens.append(tok)

        if curr_tokens:
            text += "".join([self.id_to_piece(tok) for tok in curr_tokens])

        return text

    @property
    def pad_id(self) -> int:
        return self._model.pad_id()  # type: ignore

    @property
    def unk_id(self) -> int:
        return self._model.unk_id()  # type: ignore


class InstructTokenizerBase(
    InstructTokenizer, Generic[InstructRequestType, FIMRequestType, TokenizedType, AssistantMessageType]
):
    def __init__(self, tokenizer: Tokenizer, mm_encoder: Optional[MultiModalEncoder] = None):
        self.tokenizer = tokenizer
        self.mm_encoder = mm_encoder
        super().__init__(tokenizer, mm_encoder)

    def start(self) -> List[int]:
        return [self.tokenizer.bos_id]

    @staticmethod
    def find_first_last_user(request: InstructRequest) -> Tuple[int, int]:
        # find last user message
        last_user_idx = -1
        first_user_idx = -1
        for i, msg in list(enumerate(request.messages)):
            if isinstance(msg, UserMessage):
                if first_user_idx == -1:
                    first_user_idx = i
                last_user_idx = i
        return first_user_idx, last_user_idx

    @abstractmethod
    def encode_tool_message(self, message: ToolMessage, is_before_last_user_message: bool) -> List[int]:
        raise NotImplementedError("Tool message not implemented")

    @abstractmethod
    def encode_assistant_message(self, message: AssistantMessageType, is_before_last_user_message: bool) -> List[int]:
        raise NotImplementedError("Assistant message not implemented")

    def _truncate_for_max_tokens(
        self,
        tokenized: List[Optional[List[int]]],
        messages: List[AssistantMessageType],
        max_tokens: int,
        last_user_message_index: int,
    ) -> None:
        # Tokenizer ⩽ V3 does not support truncation
        return

    def encode_instruct(
        self,
        request: InstructRequest[AssistantMessageType, Tool],
    ) -> Tokenized:
        # init at bos
        images: List[np.ndarray] = []
        prefix_ids: Optional[List[int]] = None
        tokens_list: List[Optional[List[int]]] = []

        # find last user message
        first_user_idx, last_user_idx = self.find_first_last_user(request)
        for msg_idx, msg in enumerate(request.messages):
            if isinstance(msg, UserMessage):
                new_tokens, new_images = self.encode_user_message(
                    msg,
                    request.available_tools,
                    msg_idx == last_user_idx,
                    msg_idx == first_user_idx,
                    system_prompt=request.system_prompt,
                    force_img_first=True,  # img is always first when providing text/img chunk pair
                )
                images.extend(new_images)
            elif isinstance(msg, ToolMessage):
                new_tokens = self.encode_tool_message(msg, msg_idx < last_user_idx)
            elif isinstance(msg, AssistantMessage):
                new_tokens = self.encode_assistant_message(msg, msg_idx < last_user_idx)
                if msg_idx == len(request.messages) - 1:
                    prefix_ids = new_tokens
            elif isinstance(msg, SystemMessage):
                new_tokens = self.encode_system_message(msg)

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
            text=self.tokenizer.to_string(tokens),
            prefix_ids=prefix_ids,
            images=images,
        )

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)


class InstructTokenizerV1(
    InstructTokenizerBase, Generic[InstructRequestType, FIMRequestType, TokenizedType, AssistantMessageType]
):
    def encode_user_message(
        self,
        message: UserMessage,
        available_tools: Optional[List[Tool]],
        is_last: bool,
        is_first: bool,
        system_prompt: Optional[str] = None,
        force_img_first: bool = False,
    ) -> Tuple[List[int], List[np.ndarray]]:
        assert message.content is not None
        assert isinstance(message.content, str), "Message content must be normalized"
        assert self.mm_encoder is None, "InstructTokenizerV1 cannot encode images"

        content = ""
        if is_first and system_prompt:
            content = system_prompt + "\n\n" + message.content
        else:
            content = message.content

        message_txt = f"[INST] {content} [/INST]"
        curr_tokens, image_tokens = self.encode_user_content(content=message_txt, is_last=False, system_prompt=None)
        return curr_tokens, image_tokens

    def encode_user_content(
        self,
        content: Union[str, List[ContentChunk]],
        is_last: bool,
        system_prompt: Optional[str] = None,
        force_img_first: bool = False,
    ) -> Tuple[List[int], List[np.ndarray]]:
        assert isinstance(content, str)

        if is_last and system_prompt:
            content = system_prompt + "\n\n" + content

        tokens = self.tokenizer.encode(content, bos=False, eos=False)
        return tokens, []

    def encode_tool_message(self, message: ToolMessage, is_before_last_user_message: bool) -> List[int]:
        raise TokenizerException("Tools not implemented for tokenizer V1")

    def encode_assistant_message(self, message: AssistantMessageType, is_before_last_user_message: bool) -> List[int]:
        assert isinstance(message, AssistantMessage), message
        if message.tool_calls is not None and len(message.tool_calls) > 0:
            raise TokenizerException("Tools not implemented for tokenizer V1")
        elif message.content:
            curr_tokens = self.tokenizer.encode(message.content, bos=False, eos=False)
        else:
            raise TokenizerException(f"{message.content} // {message.tool_calls}")
        if not message.prefix:
            curr_tokens.append(self.tokenizer.eos_id)
        return curr_tokens

    def encode_fim(self, request: FIMRequest) -> Tokenized:
        raise TokenizerException("FIM not available for tokenizer V1")


class InstructTokenizerV2(
    InstructTokenizerV1, Generic[InstructRequestType, FIMRequestType, TokenizedType, AssistantMessageType]
):
    def __init__(self, tokenizer: Tokenizer, mm_encoder: Optional[MultiModalEncoder] = None):
        super().__init__(tokenizer, mm_encoder)
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
        available_tools: Optional[List[Tool]],
        is_last: bool,
        is_first: bool,
        system_prompt: Optional[str] = None,
        force_img_first: bool = False,
    ) -> Tuple[List[int], List[np.ndarray]]:
        assert message.content is not None
        tools_tokens: List[int] = []
        if is_last and available_tools:
            tools = [tool.model_dump() for tool in available_tools]
            tools_json_tokens = self.tokenizer.encode(json.dumps(tools, ensure_ascii=False), bos=False, eos=False)
            tools_tokens = [
                self.BEGIN_AVAILABLE_TOOLS,
                *tools_json_tokens,
                self.END_AVAILABLE_TOOLS,
            ]

        tokens, image_tokens = self.encode_user_content(
            content=message.content,
            is_last=is_last,
            system_prompt=system_prompt,
            force_img_first=force_img_first,
        )

        prefix_tokens = [*tools_tokens, self.BEGIN_INST]
        suffix_tokens = [self.END_INST]

        curr_tokens = prefix_tokens + tokens + suffix_tokens

        return curr_tokens, image_tokens

    def _parse_json_content(self, content: str) -> Any:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content

    def _prepare_tool_result(self, tool_message: ToolMessage) -> Dict[str, Any]:
        """
        Bit of a hack due to the way tool results are tokenized
        """
        assert tool_message.content is not None, "Tool message content cannot be None"
        return {
            "name": tool_message.name,
            "content": self._parse_json_content(tool_message.content),
        }

    def encode_tool_message(self, message: ToolMessage, is_before_last_user_message: bool) -> List[int]:
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

    def _prepare_function_call(self, tool_call: ToolCall) -> Dict[str, Any]:
        """
        Bit of a hack due to the way function calls are tokenized
        """
        return {
            "name": tool_call.function.name,
            "arguments": self._parse_json_content(tool_call.function.arguments),
        }

    def _encode_normal_content_assistant_message(self, message: AssistantMessageType) -> List[int]:
        assert message.content, f"Assistant message must have content. Got {message}"
        return self.tokenizer.encode(message.content.rstrip(" "), bos=False, eos=False)

    def _encode_tool_calls_in_assistant_message(self, message: AssistantMessageType) -> List[int]:
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

    def encode_assistant_message(self, message: AssistantMessageType, is_before_last_user_message: bool) -> List[int]:
        if message.tool_calls:
            if is_before_last_user_message:
                # don't tokenize tool call before last user message
                return []
            curr_tokens = self._encode_tool_calls_in_assistant_message(message)
        elif message.content:
            curr_tokens = self._encode_normal_content_assistant_message(message)
        else:
            raise TokenizerException(f"Invalid assistant message: {message.content}")
        if not message.prefix:
            curr_tokens.append(self.tokenizer.eos_id)
        return curr_tokens

    def _encode_infilling(self, text: str) -> List[int]:
        """
        Remove prefix space in the case of SentencePieceTokenizers
        Thanks Fabian !
        """

        return self.tokenizer.encode("☺" + text, bos=False, eos=False)[2:]

    def encode_fim(self, request: FIMRequest) -> Tokenized:
        prefix_tokens = self.tokenizer.encode(request.prompt, bos=False, eos=False)
        suffix_tokens = self._encode_infilling(request.suffix) if request.suffix else []
        tokens = [
            self.BOS,
            self.SUFFIX,
            *suffix_tokens,
            self.PREFIX,
            *prefix_tokens,
        ]
        return Tokenized(tokens=tokens, text=self.tokenizer.to_string(tokens))


class InstructTokenizerV3(
    InstructTokenizerV2, Generic[InstructRequestType, FIMRequestType, TokenizedType, AssistantMessageType]
):
    """
    The only difference with V3 tokenizer is that it encodes the tool messages differently
    """

    def __init__(self, tokenizer: Tokenizer, mm_encoder: Optional[MultiModalEncoder] = None) -> None:
        super().__init__(tokenizer, mm_encoder=mm_encoder)

    def _prepare_function_call(self, tool_call: ToolCall) -> Dict[str, Any]:
        function_call = {
            "name": tool_call.function.name,
            "arguments": self._parse_json_content(tool_call.function.arguments),
        }

        if tool_call.id and tool_call.id != "null":
            function_call["id"] = tool_call.id

        return function_call

    def _prepare_tool_result(self, tool_message: ToolMessage) -> Dict[str, Any]:
        assert tool_message.content is not None, "Tool message content cannot be None"
        assert tool_message.tool_call_id is not None, "Tool message has to have the tool call id defined in v3"

        return {
            "content": self._parse_json_content(tool_message.content),
            "call_id": tool_message.tool_call_id,
        }

    def encode_tool_message(self, message: ToolMessage, is_before_last_user_message: bool) -> List[int]:
        """
        Same as V2 but tools not wrapped in a list and history is tokenized also
        """
        tool_result_str = json.dumps(self._prepare_tool_result(message), ensure_ascii=False)
        curr_tokens = [
            self.BEGIN_TOOL_RESULTS,
            *self.tokenizer.encode(tool_result_str, bos=False, eos=False),
            self.END_TOOL_RESULTS,
        ]
        return curr_tokens

    def encode_assistant_message(self, message: AssistantMessageType, is_before_last_user_message: bool) -> List[int]:
        """
        Same as V2 but always encode tool history
        """
        return super().encode_assistant_message(message, False)

    def encode_user_content(
        self,
        content: Union[str, List[ContentChunk]],
        is_last: bool,
        system_prompt: Optional[str] = None,
        force_img_first: bool = False,
    ) -> Tuple[List[int], List[np.ndarray]]:
        if isinstance(content, str):
            return super().encode_user_content(content, is_last, system_prompt)

        tokens: List[int] = []
        images: List[np.ndarray] = []

        has_one_img_one_text_first = (
            len(content) == 2 and isinstance(content[0], TextChunk) and not isinstance(content[1], TextChunk)
        )
        if force_img_first and has_one_img_one_text_first:
            # make sure that if exactly one image and text chunk are passed we force the image chunk to be first
            content = [content[1], content[0]]

        first_chunk = True
        for chunk in content:
            content = ""
            if first_chunk and is_last and system_prompt:
                first_chunk = False
                content = system_prompt + "\n\n"
            if isinstance(chunk, TextChunk):
                content += chunk.text
                tokens.extend(self.tokenizer.encode(content, bos=False, eos=False))
            else:
                assert self.mm_encoder is not None, "Make sure to define a multi-modal encoder at init"
                if content:
                    tokens.extend(self.tokenizer.encode(content, bos=False, eos=False))

                img_encoding = self.mm_encoder(chunk)

                tokens.extend(img_encoding.tokens)
                images.append(img_encoding.image)

        return tokens, images


class InstructTokenizerV7(InstructTokenizerV3):
    """
    The difference with V3 tokenizer is that it encodes the system prompts differently:
    - in V7 the system prompts are treated as separate SystemMessages
    - they are no longer prepended to the last user message
    - they are printed between special tokens
    Tool call results are encoded as :
    - [begin tool call] call_id_tokens [tool_content]  content tokens [end tool call]
    """

    def __init__(self, tokenizer: Tokenizer, mm_encoder: Optional[MultiModalEncoder] = None) -> None:
        super().__init__(tokenizer, mm_encoder)
        self.BEGIN_SYSTEM = self.tokenizer.get_control_token(SpecialTokens.begin_system.value)
        self.END_SYSTEM = self.tokenizer.get_control_token(SpecialTokens.end_system.value)
        self.BEGIN_TOOL_CONTENT = self.tokenizer.get_control_token(SpecialTokens.begin_tool_content.value)

    def _truncate_for_max_tokens(
        self,
        tokenized_messages: List[Optional[List[int]]],
        messages: List[AssistantMessageType],
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

    def encode_system_message(self, message: SystemMessage) -> List[int]:
        assert message.content is not None
        assert isinstance(message.content, str), "Message content must be normalized"
        tokens = [
            self.BEGIN_SYSTEM,
            *self.tokenizer.encode(message.content, bos=False, eos=False),
            self.END_SYSTEM,
        ]
        return tokens

    def encode_user_message(
        self,
        message: UserMessage,
        available_tools: Optional[List[Tool]],
        is_last: bool,
        is_first: bool,
        system_prompt: Optional[str] = None,
        force_img_first: bool = False,
    ) -> Tuple[List[int], List[np.ndarray]]:
        assert system_prompt is None, "in Tokenizer V7 we don't encode system prompts in user messages"
        return super().encode_user_message(
            message,
            available_tools,
            is_last=is_last,
            is_first=is_first,
            system_prompt=None,
            force_img_first=force_img_first,
        )

    def encode_tool_message(self, message: ToolMessage, is_before_last_user_message: bool) -> List[int]:
        """
        Same as V3 but tools not wrapped in a list and history is tokenized also
        """
        assert message.tool_call_id is not None
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

    def encode_assistant_message(self, message: AssistantMessageType, is_before_last_user_message: bool) -> List[int]:
        if not message.content and not message.tool_calls:
            raise TokenizerException(f"Invalid assistant message: {message}")
        curr_tokens: list = []
        if message.content:
            if isinstance(message.content, str):
                curr_tokens += self._encode_normal_content_assistant_message(message)
            elif isinstance(message.content, list):
                curr_tokens += self.encode_content_chunks(
                    message.content, is_last=False, system_prompt=None, force_img_first=True
                ).tokens
        if message.tool_calls:
            curr_tokens += self._encode_tool_calls_in_assistant_message(message)
        if not message.prefix:
            curr_tokens.append(self.tokenizer.eos_id)

        return curr_tokens
