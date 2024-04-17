import json
import logging
import os
from abc import abstractmethod
from functools import cached_property
from typing import Any, Dict, List, Optional, Set

from mistral_common.exceptions import TokenizerException
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.tool_calls import Tool, ToolCall
from mistral_common.tokens.instruct.request import InstructRequest
from mistral_common.tokens.tokenizers.base import (
    InstructTokenizer,
    SpecialTokens,
    Tokenized,
    Tokenizer,
)
from sentencepiece import SentencePieceProcessor


class SentencePieceTokenizer(Tokenizer):
    def __init__(self, model_path: str) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self._model = SentencePieceProcessor(model_file=model_path)

        assert self._model.vocab_size() == self._model.get_piece_size()
        self._vocab = [self._model.id_to_piece(i) for i in range(self.n_words)]

        super().__init__()

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


class SentencePieceInstructTokenizer(InstructTokenizer):
    def __init__(self, model_path: str):
        self.tokenizer = SentencePieceTokenizer(model_path)

    def start(self) -> List[int]:
        return [self.tokenizer.bos_id]

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
        ...

    @abstractmethod
    def encode_assistant_message(self, message: AssistantMessage, is_before_last_user_message: bool) -> List[int]:
        ...

    def encode_instruct(self, request: InstructRequest) -> Tokenized:
        # init at bos
        tokens = self.start()
        # find last user message
        last_user_idx = -1
        first_user_idx = -1
        for i, msg in list(enumerate(request.messages)):
            if isinstance(msg, UserMessage):
                if first_user_idx == -1:
                    first_user_idx = i
                last_user_idx = i
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

            tokens.extend(new_tokens)

        return Tokenized(tokens=tokens, text=self.tokenizer.to_string(tokens))


class SentencePieceInstructTokenizerV1(SentencePieceInstructTokenizer):
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

    def encode_assistant_message(self, message: AssistantMessage, is_before_last_user_message: bool) -> List[int]:
        assert isinstance(message, AssistantMessage), message
        if message.tool_calls is not None and len(message.tool_calls) > 0:
            raise TokenizerException("Tools not implemented for tokenizer V1")
        elif message.content:
            curr_tokens = self.tokenizer.encode(message.content, bos=False, eos=False)
        else:
            raise TokenizerException(f"{message.content} // {message.tool_calls}")

        curr_tokens.append(self.tokenizer.eos_id)
        return curr_tokens


class SentencePieceInstructTokenizerV2(SentencePieceInstructTokenizer):
    def __init__(self, model_path: str):
        super().__init__(model_path)

        self.BEGIN_INST = self.get_control_token(SpecialTokens.begin_inst.value)
        self.END_INST = self.get_control_token(SpecialTokens.end_inst.value)
        self.BEGIN_AVAILABLE_TOOLS = self.get_control_token(SpecialTokens.begin_tools.value)
        self.END_AVAILABLE_TOOLS = self.get_control_token(SpecialTokens.end_tools.value)
        self.BEGIN_TOOL_RESULTS = self.get_control_token(SpecialTokens.begin_tool_results.value)
        self.END_TOOL_RESULTS = self.get_control_token(SpecialTokens.end_tool_results.value)
        self.TOOL_CALLS = self.get_control_token(SpecialTokens.tool_calls.value)

    def get_control_token(self, s: str) -> int:
        return self.tokenizer._model.piece_to_id(s)  # type: ignore

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

    def encode_assistant_message(self, message: AssistantMessage, is_before_last_user_message: bool) -> List[int]:
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

        curr_tokens.append(self.tokenizer.eos_id)
        return curr_tokens


class SentencePieceInstructTokenizerV3(SentencePieceInstructTokenizerV2):
    """
    The only difference with V3 tokenizer is that it encodes the tool messages differently
    """

    def _prepare_function_call(self, tool_call: ToolCall) -> Dict[str, Any]:
        return {
            "name": tool_call.function.name,
            "arguments": self._parse_json_content(tool_call.function.arguments),
            "id": tool_call.id,
        }

    def _prepare_tool_result(self, tool_message: ToolMessage) -> Dict[str, Any]:
        assert tool_message.content is not None, "Tool message content cannot be None"
        return {
            "call_id": tool_message.tool_call_id,
            "content": self._parse_json_content(tool_message.content),
        }

    def encode_tool_message(self, message: ToolMessage, is_before_last_user_message: bool) -> List[int]:
        """
        Same as V2 but tools not wrapped in a list and history is tokenized also
        """
        tool_result_str = json.dumps(self._prepare_tool_result(message))
        curr_tokens = [
            self.BEGIN_TOOL_RESULTS,
            *self.tokenizer.encode(tool_result_str, bos=False, eos=False),
            self.END_TOOL_RESULTS,
        ]
        return curr_tokens

    def encode_assistant_message(self, message: AssistantMessage, is_before_last_user_message: bool) -> List[int]:
        """
        Same as V2 but always encode tool history
        """
        return super().encode_assistant_message(message, False)
