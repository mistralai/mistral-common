import json
import logging
import os
from abc import abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Set, Tuple, Union

from mistral_common.exceptions import TokenizerException
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    AssistantMessageType,
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
from sentencepiece import SentencePieceProcessor


def is_sentencepiece(path: Union[str, Path]) -> bool:
    if isinstance(path, str):
        path = Path(path)

    suffixes = [f".model.{v}" for v in list(TokenizerVersion.__members__)] + [".model"]
    return path.is_file() and any(path.name.endswith(suffix) for suffix in suffixes)


def get_spm_version(tokenizer_filename: str, raise_deprecated: bool = False) -> TokenizerVersion:
    _version_str = tokenizer_filename.split(".")[-1]
    if _version_str == "model":
        if raise_deprecated:
            raise TokenizerException(f"Make sure to rename your tokenizer file to end with {tokenizer_filename}.v1.")

        # tokenizer.model => tokenizer.model.v1
        return TokenizerVersion("v1")

    if _version_str not in TokenizerVersion.__members__:
        raise TokenizerException(f"Unrecognized tokenizer filename: {tokenizer_filename}")

    return TokenizerVersion(_version_str)


class SentencePieceTokenizer(Tokenizer):
    def __init__(self, model_path: str) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self._model = SentencePieceProcessor(model_file=model_path)

        assert self._model.vocab_size() == self._model.get_piece_size()
        self._vocab = [self._model.id_to_piece(i) for i in range(self.n_words)]

        self._version: TokenizerVersion = get_spm_version(model_path, raise_deprecated=False)

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


class InstructTokenizerBase(
    InstructTokenizer, Generic[InstructRequestType, FIMRequestType, TokenizedType, AssistantMessageType]
):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        super().__init__(tokenizer)

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
    def encode_user_message(
        self,
        message: UserMessage,
        available_tools: Optional[List[Tool]],
        is_last: bool,
        is_first: bool,
        system_prompt: Optional[str] = None,
    ) -> List[int]:
        ...

    @abstractmethod
    def encode_tool_message(self, message: ToolMessage, is_before_last_user_message: bool) -> List[int]:
        raise NotImplementedError("Tool message not implemented")

    @abstractmethod
    def encode_assistant_message(self, message: AssistantMessageType, is_before_last_user_message: bool) -> List[int]:
        raise NotImplementedError("Assistant message not implemented")

    def encode_instruct(self, request: InstructRequest[AssistantMessageType, Tool]) -> Tokenized:
        # init at bos
        tokens = self.start()
        prefix_ids: Optional[List[int]] = None
        # find last user message
        first_user_idx, last_user_idx = self.find_first_last_user(request)
        for msg_idx, msg in enumerate(request.messages):
            if isinstance(msg, UserMessage):
                new_tokens = self.encode_user_message(
                    msg,
                    request.available_tools,
                    msg_idx == last_user_idx,
                    msg_idx == first_user_idx,
                    system_prompt=request.system_prompt,
                )
            elif isinstance(msg, ToolMessage):
                new_tokens = self.encode_tool_message(msg, msg_idx < last_user_idx)
            elif isinstance(msg, AssistantMessage):
                new_tokens = self.encode_assistant_message(msg, msg_idx < last_user_idx)
                if msg_idx == len(request.messages) - 1:
                    prefix_ids = new_tokens

            tokens.extend(new_tokens)

        return Tokenized(tokens=tokens, text=self.tokenizer.to_string(tokens), prefix_ids=prefix_ids)

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
    ) -> List[int]:
        assert message.content is not None
        assert isinstance(message.content, str), "Message content must be normalized"
        content = ""
        if is_first and system_prompt:
            content = system_prompt + "\n\n" + message.content
        else:
            content = message.content

        message_txt = f"[INST] {content} [/INST]"
        curr_tokens = self.tokenizer.encode(message_txt, bos=False, eos=False)
        return curr_tokens

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
    def __init__(self, tokenizer: Tokenizer):
        super().__init__(tokenizer)
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
    ) -> List[int]:
        assert message.content is not None
        assert isinstance(message.content, str), "Message content must be nornmalized"
        content = ""
        tools_tokens: List[int] = []
        if is_last and available_tools:
            tools = [tool.model_dump() for tool in available_tools]
            tools_json_tokens = self.tokenizer.encode(json.dumps(tools, ensure_ascii=False), bos=False, eos=False)
            tools_tokens = [
                self.BEGIN_AVAILABLE_TOOLS,
                *tools_json_tokens,
                self.END_AVAILABLE_TOOLS,
            ]

        if is_last and system_prompt:
            content = system_prompt + "\n\n" + message.content
        else:
            content = message.content

        curr_tokens = [
            *tools_tokens,
            self.BEGIN_INST,
            *self.tokenizer.encode(content, bos=False, eos=False),
            self.END_INST,
        ]
        return curr_tokens

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

    def encode_assistant_message(self, message: AssistantMessageType, is_before_last_user_message: bool) -> List[int]:
        if message.tool_calls is not None and len(message.tool_calls) > 0:
            if is_before_last_user_message:
                # don't tokenize tool call before last user message
                return []

            prepared_tool_calls = []
            for tool_call in message.tool_calls:
                prepared_tool_calls.append(self._prepare_function_call(tool_call))

            tool_call_str = json.dumps(prepared_tool_calls, ensure_ascii=False)
            curr_tokens = [
                self.TOOL_CALLS,
                *self.tokenizer.encode(tool_call_str, bos=False, eos=False),
            ]
        elif message.content:
            curr_tokens = self.tokenizer.encode(message.content, bos=False, eos=False)
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

    def __init__(self, tokenizer: Tokenizer):
        super().__init__(tokenizer)

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
