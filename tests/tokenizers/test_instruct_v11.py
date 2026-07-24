import pytest

from mistral_common.exceptions import InvalidAssistantMessageException
from mistral_common.protocol.instruct.messages import AssistantMessage
from mistral_common.protocol.instruct.tool_calls import FunctionCall, ToolCall
from mistral_common.tokens.tokenizers.base import InstructTokenizer, TokenizerVersion
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV11
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.utils.requests.instruct import v11_plain_text_think_request
from tests.utils.versions import TestConfig, config_id

_V11 = TestConfig(version=TokenizerVersion.v11)

v11_tokenizer = pytest.mark.parametrize("instruct_tokenizer", [_V11], indirect=True, ids=config_id)
v11_mistral = pytest.mark.parametrize("mistral_tokenizer", [_V11], indirect=True, ids=config_id)


def _as_v11(instruct_tokenizer: InstructTokenizer) -> InstructTokenizerV11:
    assert isinstance(instruct_tokenizer, InstructTokenizerV11)
    return instruct_tokenizer


class TestInstructTokenizerV11:
    @v11_tokenizer
    def test_special_tokens(self, instruct_tokenizer: InstructTokenizer) -> None:
        instruct_v11 = _as_v11(instruct_tokenizer)
        assert instruct_v11.ARGS == 32
        assert instruct_v11.CALL_ID == 33
        assert instruct_v11.TOOL_CALLS == 9

    @v11_tokenizer
    def test_tokenize_assistant_message(self, instruct_tokenizer: InstructTokenizer) -> None:
        instruct_v11 = _as_v11(instruct_tokenizer)
        tokens = instruct_v11.encode_assistant_message(
            AssistantMessage(
                tool_calls=[ToolCall(function=FunctionCall(name="a_a_a", arguments="blabla"))],
            ),
            is_before_last_user_message=False,
            continue_message=False,
        )
        assert tokens == [
            instruct_v11.TOOL_CALLS,
            197,
            195,
            197,
            195,
            197,
            instruct_v11.ARGS,
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
        assert instruct_v11.tokenizer._to_string(tokens) == ('[TOOL_CALLS]a_a_a[ARGS]"blabla"</s>')

    @v11_tokenizer
    def test_tokenize_assistant_message_continue_message(self, instruct_tokenizer: InstructTokenizer) -> None:
        instruct_v11 = _as_v11(instruct_tokenizer)
        tokens = instruct_v11.encode_assistant_message(
            AssistantMessage(
                content='"blabla"',
            ),
            is_before_last_user_message=False,
            continue_message=True,
        )
        assert tokens == [134, 198, 208, 197, 198, 208, 197, 134]
        assert instruct_v11.tokenizer._to_string(tokens) == ('"blabla"')

        with pytest.raises(
            InvalidAssistantMessageException,
            match="`continue_message` is only supported for assistant messages that have `prefix=False`.",
        ):
            instruct_v11.encode_assistant_message(
                AssistantMessage(
                    content='"blabla"',
                    prefix=True,
                ),
                is_before_last_user_message=False,
                continue_message=True,
            )

    @v11_tokenizer
    def test_tokenize_assistant_messages(self, instruct_tokenizer: InstructTokenizer) -> None:
        instruct_v11 = _as_v11(instruct_tokenizer)
        tokens = instruct_v11.encode_assistant_message(
            AssistantMessage(
                tool_calls=[
                    ToolCall(function=FunctionCall(name="a_a_a", arguments="blabla")),
                    ToolCall(function=FunctionCall(name="b", arguments="blu")),
                ],
            ),
            is_before_last_user_message=False,
            continue_message=False,
        )
        assert tokens == [
            instruct_v11.TOOL_CALLS,
            197,
            195,
            197,
            195,
            197,
            instruct_v11.ARGS,
            134,
            198,
            208,
            197,
            198,
            208,
            197,
            134,
            instruct_v11.TOOL_CALLS,
            198,
            instruct_v11.ARGS,
            134,
            198,
            208,
            217,
            134,
            2,
        ]
        assert instruct_v11.tokenizer._to_string(tokens) == (
            '[TOOL_CALLS]a_a_a[ARGS]"blabla"[TOOL_CALLS]b[ARGS]"blu"</s>'
        )

    @v11_tokenizer
    def test_tokenize_assistant_message_train(self, instruct_tokenizer: InstructTokenizer) -> None:
        instruct_v11 = _as_v11(instruct_tokenizer)
        tokens = instruct_v11.encode_assistant_message(
            AssistantMessage(
                tool_calls=[ToolCall(function=FunctionCall(name="a_a_a", arguments="blabla"), id="ABC")],
            ),
            is_before_last_user_message=True,
            continue_message=False,
        )
        assert tokens == [
            instruct_v11.TOOL_CALLS,
            197,
            195,
            197,
            195,
            197,
            instruct_v11.CALL_ID,
            165,
            166,
            167,
            instruct_v11.ARGS,
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
        assert instruct_v11.tokenizer._to_string(tokens) == ('[TOOL_CALLS]a_a_a[CALL_ID]ABC[ARGS]"blabla"</s>')

    @v11_mistral
    def test_encode_chat_completion_plain_text_think(
        self,
        mistral_tokenizer: MistralTokenizer,
        instruct_decoded_goldens: dict[str, dict[str, str]],
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
    ) -> None:
        encoded = mistral_tokenizer.encode_chat_completion(v11_plain_text_think_request())
        assert encoded.text == instruct_decoded_goldens["v11"]["plain_text_think"]
        assert encoded.tokens == instruct_token_id_goldens["v11"]["plain_text_think"]
        assert "<think>" in (encoded.text or "")
        assert "[THINK]" not in (encoded.text or "")
