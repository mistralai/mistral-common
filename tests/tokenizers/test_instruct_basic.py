import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from mistral_common.exceptions import InvalidAssistantMessageException, InvalidMessageStructureException
from mistral_common.protocol.instruct.messages import AssistantMessage, UserMessage
from mistral_common.protocol.instruct.request import InstructRequest
from mistral_common.protocol.instruct.tool_calls import FunctionCall, ToolCall
from mistral_common.tokens.tokenizers.base import InstructTokenizer, TokenizerVersion
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.sentencepiece import (
    SentencePieceTokenizer,
    is_sentencepiece,
)
from tests.utils.requests.instruct import (
    REQUEST_MULTI_TURN_WITH_TOOLS_TEST,
    abcd_messages,
    abcd_multi_turn_tools_request,
    abcd_system_single_turn_request,
    abcd_system_tools_multi_turn_request,
    abcd_trailing_user_messages,
    simple_tool,
    single_turn_tool_request,
    tool_call_no_id_request,
    tool_call_null_id_request,
    tool_message_chunks_request,
    tool_message_json_request,
    tool_message_multiple_shots_with_history_request,
    tool_message_multiple_shots_without_history_request,
    tool_message_plain_request,
    tool_multiple_calls_request,
    tool_response_chunks_request,
    tool_response_json_request,
    tool_response_plain_request,
)

# Constants for v3 parametrization
TEKKEN_BEGIN_TOOL_ID = 5
TEKKEN_END_TOOL_ID = 6

SPM_BEGIN_TOOL_ID = 6
SPM_END_TOOL_ID = 7


v1_tokenizer = pytest.mark.parametrize("shipped_instruct_tokenizer", ["v1_spm"], indirect=True)
v2_tokenizer = pytest.mark.parametrize("shipped_instruct_tokenizer", ["v2_spm"], indirect=True)
v3_tokenizer = pytest.mark.parametrize("shipped_instruct_tokenizer", ["v3_spm"], indirect=True)


@v1_tokenizer
class TestInstructTokenizerV1:
    def test_encode_instruct_multi_turn(
        self,
        shipped_instruct_tokenizer: InstructTokenizer,
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
        instruct_decoded_goldens: dict[str, dict[str, str]],
    ) -> None:
        tokenizer = shipped_instruct_tokenizer
        tokenized = tokenizer.encode_instruct(InstructRequest(messages=abcd_messages()))
        tokens, text = tokenized.tokens, tokenized.text
        assert text == instruct_decoded_goldens["v1_spm"]["abcd_multi_turn"]
        assert tokens == instruct_token_id_goldens["v1_spm"]["abcd_multi_turn"]

    def test_encode_instruct_system_single_turn(
        self,
        shipped_instruct_tokenizer: InstructTokenizer,
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
        instruct_decoded_goldens: dict[str, dict[str, str]],
    ) -> None:
        tokenizer = shipped_instruct_tokenizer
        tokenized = tokenizer.encode_instruct(abcd_system_single_turn_request())
        tokens, text = tokenized.tokens, tokenized.text
        assert text == instruct_decoded_goldens["v1_spm"]["abcd_system_single_turn"]
        assert tokens == instruct_token_id_goldens["v1_spm"]["abcd_system_single_turn"]
        assert tokenizer.tokenizer.decode(tokens) == "[INST] SYSTEM\n\na [/INST]"

    def test_encode_instruct_system_multi_turn(
        self,
        shipped_instruct_tokenizer: InstructTokenizer,
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
        instruct_decoded_goldens: dict[str, dict[str, str]],
    ) -> None:
        tokenizer = shipped_instruct_tokenizer
        tokenized = tokenizer.encode_instruct(
            InstructRequest(
                messages=abcd_messages(),
                system_prompt="SYSTEM",
            )
        )
        tokens, text = tokenized.tokens, tokenized.text
        assert text == instruct_decoded_goldens["v1_spm"]["abcd_system_multi_turn"]
        assert tokens == instruct_token_id_goldens["v1_spm"]["abcd_system_multi_turn"]
        first_eos = tokens.index(2)
        assert tokenizer.tokenizer.decode(tokens[first_eos:]) == "[INST] c [/INST] d"

    def test_encode_instruct_continue_final_message(
        self,
        shipped_instruct_tokenizer: InstructTokenizer,
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
        instruct_decoded_goldens: dict[str, dict[str, str]],
    ) -> None:
        tokenizer = shipped_instruct_tokenizer
        tokenized = tokenizer.encode_instruct(
            InstructRequest(
                messages=abcd_messages(),
                system_prompt="SYSTEM",
                continue_final_message=True,
            )
        )
        tokens, text = tokenized.tokens, tokenized.text
        assert text == instruct_decoded_goldens["v1_spm"]["abcd_system_multi_turn_continue"]
        assert tokens == instruct_token_id_goldens["v1_spm"]["abcd_system_multi_turn_continue"]

        with pytest.raises(
            InvalidMessageStructureException,
            match="Cannot continue final message if it is not an assistant message",
        ):
            tokenizer.encode_instruct(
                InstructRequest(
                    messages=abcd_trailing_user_messages(),
                    system_prompt="SYSTEM",
                    continue_final_message=True,
                )
            )

        with pytest.raises(
            InvalidAssistantMessageException,
            match="`continue_message` is only supported for assistant messages that have `prefix=False`.",
        ):
            tokenizer.encode_assistant_message(  # type: ignore[attr-defined]
                AssistantMessage(
                    content='"blabla"',
                    prefix=True,
                ),
                is_before_last_user_message=False,
                continue_message=True,
            )


class TestV1SilentlyDropsTools:
    @pytest.mark.parametrize("shipped_mistral_tokenizer", ["v1_spm"], indirect=True)
    def test_declared_tools_are_not_rendered(self, shipped_mistral_tokenizer: MistralTokenizer) -> None:
        # v1 neither raises nor emits an `[AVAILABLE_TOOLS]` schema for a request that only
        # declares `tools` (never calls one): the schema is silently dropped, so the
        # rendering is byte-identical to the same conversation without any tools declared.
        # This is why `multi_turn_tools` has no v1_spm golden -- see `registry.SCENARIOS`.
        with_tools = REQUEST_MULTI_TURN_WITH_TOOLS_TEST.model_copy(deep=True)
        without_tools = REQUEST_MULTI_TURN_WITH_TOOLS_TEST.model_copy(deep=True, update={"tools": None})
        with_tools_text = shipped_mistral_tokenizer.encode_chat_completion(with_tools).text
        without_tools_text = shipped_mistral_tokenizer.encode_chat_completion(without_tools).text
        assert with_tools_text == without_tools_text
        assert "[AVAILABLE_TOOLS]" not in with_tools_text


@v2_tokenizer
class TestInstructTokenizerV2:
    def test_encode_instruct_multi_turn(
        self,
        shipped_instruct_tokenizer: InstructTokenizer,
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
        instruct_decoded_goldens: dict[str, dict[str, str]],
    ) -> None:
        tokenizer = shipped_instruct_tokenizer
        tokenized = tokenizer.encode_instruct(InstructRequest(messages=abcd_messages()))
        tokens, text = tokenized.tokens, tokenized.text
        assert text == instruct_decoded_goldens["v2_spm"]["abcd_multi_turn"]
        assert tokens == instruct_token_id_goldens["v2_spm"]["abcd_multi_turn"]

    def test_encode_instruct_single_turn_tool(
        self,
        shipped_instruct_tokenizer: InstructTokenizer,
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
        instruct_decoded_goldens: dict[str, dict[str, str]],
    ) -> None:
        tokenizer = shipped_instruct_tokenizer
        tokenized = tokenizer.encode_instruct(single_turn_tool_request())
        tokens, text = tokenized.tokens, tokenized.text
        assert text == instruct_decoded_goldens["v2_spm"]["single_turn_tool"]
        assert tokens == instruct_token_id_goldens["v2_spm"]["single_turn_tool"]
        begin_tool, end_tool = tokens.index(6), tokens.index(7)
        assert tokens[:begin_tool] + tokens[end_tool + 1 :] == [1, 3, 1032, 4]
        json.loads(tokenizer.tokenizer.decode(tokens[begin_tool : end_tool + 1]))

    def test_encode_instruct_multi_turn_tools(
        self,
        shipped_instruct_tokenizer: InstructTokenizer,
        instruct_decoded_goldens: dict[str, dict[str, str]],
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
    ) -> None:
        tokenizer = shipped_instruct_tokenizer
        tokenized = tokenizer.encode_instruct(
            InstructRequest(
                messages=abcd_messages(),
                available_tools=[
                    simple_tool(),
                    simple_tool(name="tool2", description="2"),
                ],
            )
        )
        tokens, text = tokenized.tokens, tokenized.text
        assert text == instruct_decoded_goldens["v2_spm"]["abcd_multi_turn_tools"]
        assert tokens == instruct_token_id_goldens["v2_spm"]["abcd_multi_turn_tools"]
        begin_tool, end_tool = tokens.index(6), tokens.index(7)
        json.loads(tokenizer.tokenizer.decode(tokens[begin_tool : end_tool + 1]))

    def test_encode_instruct_system_single_turn(
        self,
        shipped_instruct_tokenizer: InstructTokenizer,
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
        instruct_decoded_goldens: dict[str, dict[str, str]],
    ) -> None:
        tokenizer = shipped_instruct_tokenizer
        tokenized = tokenizer.encode_instruct(abcd_system_single_turn_request())
        tokens, text = tokenized.tokens, tokenized.text
        assert text == instruct_decoded_goldens["v2_spm"]["abcd_system_single_turn"]
        assert tokens == instruct_token_id_goldens["v2_spm"]["abcd_system_single_turn"]
        assert tokenizer.tokenizer.decode(tokens) == "SYSTEM\n\na"

    def test_encode_instruct_system_multi_turn(
        self,
        shipped_instruct_tokenizer: InstructTokenizer,
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
        instruct_decoded_goldens: dict[str, dict[str, str]],
    ) -> None:
        tokenizer = shipped_instruct_tokenizer
        tokenized = tokenizer.encode_instruct(
            InstructRequest(
                messages=abcd_messages(),
                system_prompt="SYSTEM",
            )
        )
        tokens, text = tokenized.tokens, tokenized.text
        assert text == instruct_decoded_goldens["v2_spm"]["abcd_system_multi_turn"]
        assert tokens == instruct_token_id_goldens["v2_spm"]["abcd_system_multi_turn"]
        first_eos = tokens.index(2)
        assert tokenizer.tokenizer.decode(tokens[first_eos:]) == "SYSTEM\n\nc d"

    def test_encode_instruct_continue_final_message(
        self,
        shipped_instruct_tokenizer: InstructTokenizer,
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
        instruct_decoded_goldens: dict[str, dict[str, str]],
    ) -> None:
        tokenizer = shipped_instruct_tokenizer
        tokenized = tokenizer.encode_instruct(
            InstructRequest(
                messages=abcd_messages(),
                system_prompt="SYSTEM",
                continue_final_message=True,
            )
        )
        tokens, text = tokenized.tokens, tokenized.text
        assert text == instruct_decoded_goldens["v2_spm"]["abcd_system_multi_turn_continue"]
        assert tokens == instruct_token_id_goldens["v2_spm"]["abcd_system_multi_turn_continue"]

        with pytest.raises(
            InvalidMessageStructureException,
            match="Cannot continue final message if it is not an assistant message",
        ):
            tokenizer.encode_instruct(
                InstructRequest(
                    messages=abcd_trailing_user_messages(),
                    system_prompt="SYSTEM",
                    continue_final_message=True,
                )
            )

        with pytest.raises(
            InvalidAssistantMessageException,
            match="`continue_message` is only supported for assistant messages that have `prefix=False`.",
        ):
            tokenizer.encode_assistant_message(  # type: ignore[attr-defined]
                AssistantMessage(
                    content='"blabla"',
                    prefix=True,
                ),
                is_before_last_user_message=False,
                continue_message=True,
            )

    def test_encode_instruct_system_tools_multi_turn(
        self,
        shipped_instruct_tokenizer: InstructTokenizer,
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
        instruct_decoded_goldens: dict[str, dict[str, str]],
    ) -> None:
        tokenizer = shipped_instruct_tokenizer
        tokenized = tokenizer.encode_instruct(abcd_system_tools_multi_turn_request())
        tokens, text = tokenized.tokens, tokenized.text
        assert text == instruct_decoded_goldens["v2_spm"]["abcd_system_tools_multi_turn"]
        assert tokens == instruct_token_id_goldens["v2_spm"]["abcd_system_tools_multi_turn"]

        begin_tool, end_tool = tokens.index(6), tokens.index(7)
        assert tokens[end_tool + 1 :].index(3) == 0  # begin_inst follows end_tool
        assert tokenizer.tokenizer.decode(tokens[:begin_tool]) == "a b"
        assert tokenizer.tokenizer.decode(tokens[end_tool + 1 :]) == "SYSTEM\n\nc d"

    def test_encode_instruct_tool_response(
        self,
        shipped_instruct_tokenizer: InstructTokenizer,
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
        instruct_decoded_goldens: dict[str, dict[str, str]],
    ) -> None:
        tokenizer = shipped_instruct_tokenizer
        tokenized = tokenizer.encode_instruct(tool_response_plain_request())
        tokens, text = tokenized.tokens, tokenized.text
        assert text == instruct_decoded_goldens["v2_spm"]["tool_response_plain"]
        assert tokens == instruct_token_id_goldens["v2_spm"]["tool_response_plain"]

        tokenized = tokenizer.encode_instruct(tool_response_json_request())
        tokens, text = tokenized.tokens, tokenized.text
        assert text == instruct_decoded_goldens["v2_spm"]["tool_response_json"]
        assert tokens == instruct_token_id_goldens["v2_spm"]["tool_response_json"]

        tokenized = tokenizer.encode_instruct(tool_response_chunks_request())
        tokens, text = tokenized.tokens, tokenized.text
        assert text == instruct_decoded_goldens["v2_spm"]["tool_response_chunks"]
        assert tokens == instruct_token_id_goldens["v2_spm"]["tool_response_chunks"]

    def test_encode_instruct_tool_message_multiple_shots_without_history(
        self,
        shipped_instruct_tokenizer: InstructTokenizer,
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
        instruct_decoded_goldens: dict[str, dict[str, str]],
    ) -> None:
        tokenizer = shipped_instruct_tokenizer
        tokenized = tokenizer.encode_instruct(tool_message_multiple_shots_without_history_request())
        tokens, text = tokenized.tokens, tokenized.text
        assert text == instruct_decoded_goldens["v2_spm"]["tool_message_multiple_shots_without_history"]
        assert tokens == instruct_token_id_goldens["v2_spm"]["tool_message_multiple_shots_without_history"]


class TestInstructTokenizerV3:
    def test_is_spm(self) -> None:
        # this is valid
        for suffix in list(TokenizerVersion.__members__) + ["v3m1"]:
            with NamedTemporaryFile(suffix=".model." + suffix) as f:
                assert is_sentencepiece(f.name)

        with NamedTemporaryFile(suffix=".model") as f:
            assert is_sentencepiece(f.name)

        # this is not valid
        with NamedTemporaryFile(suffix=".model.vx") as f:
            assert not is_sentencepiece(f.name)

    def test_spm_version(self) -> None:
        directory = Path(__file__).parent.parent.parent / "src" / "mistral_common" / "data"

        for file in directory.iterdir():
            if not file.is_file() or str(file).endswith(".json"):
                continue
            suffix = file.suffix[1:].split("m")[0]
            if suffix == "model":
                assert SentencePieceTokenizer(str(file)).version == TokenizerVersion.v1
            else:
                assert SentencePieceTokenizer(str(file)).version == TokenizerVersion(suffix)

    @pytest.mark.parametrize(
        "shipped_instruct_tokenizer, golden_key, begin_tool_index, end_tool_index, expected_tokens_before_tool",
        [
            ("v3_spm", "v3_spm", SPM_BEGIN_TOOL_ID, SPM_END_TOOL_ID, [1, 3, 1032, 4]),
            ("v3_tekken", "v3_tekken", TEKKEN_BEGIN_TOOL_ID, TEKKEN_END_TOOL_ID, [1, 3, 1097, 4]),
        ],
        indirect=["shipped_instruct_tokenizer"],
        ids=["spm", "tekken"],
    )
    def test_encode_instruct_single_turn_tool(
        self,
        shipped_instruct_tokenizer: InstructTokenizer,
        golden_key: str,
        begin_tool_index: int,
        end_tool_index: int,
        expected_tokens_before_tool: list[int],
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
        instruct_decoded_goldens: dict[str, dict[str, str]],
    ) -> None:
        tokenizer = shipped_instruct_tokenizer
        tokenized = tokenizer.encode_instruct(single_turn_tool_request())
        tokens, text = tokenized.tokens, tokenized.text
        assert text == instruct_decoded_goldens[golden_key]["single_turn_tool"]
        assert tokens == instruct_token_id_goldens[golden_key]["single_turn_tool"]

        begin_tool, end_tool = tokens.index(begin_tool_index), tokens.index(end_tool_index)
        assert tokens[:begin_tool] + tokens[end_tool + 1 :] == expected_tokens_before_tool + []
        json.loads(tokenizer.tokenizer.decode(tokens[begin_tool : end_tool + 1]))

    @pytest.mark.parametrize(
        "shipped_instruct_tokenizer, golden_key, begin_tool_index, end_tool_index, "
        "expected_tokens_before_tool, expected_tokens_after_tool",
        [
            (
                "v3_spm",
                "v3_spm",
                SPM_BEGIN_TOOL_ID,
                SPM_END_TOOL_ID,
                [1, 3, 1032, 4, 1055],
                [2, 3, 1045, 4, 1049, 2],
            ),
            (
                "v3_tekken",
                "v3_tekken",
                TEKKEN_BEGIN_TOOL_ID,
                TEKKEN_END_TOOL_ID,
                [1, 3, 1097, 4, 1098],
                [2, 3, 1099, 4, 1100, 2],
            ),
        ],
        indirect=["shipped_instruct_tokenizer"],
        ids=["spm", "tekken"],
    )
    def test_encode_instruct_multi_turn_tools(
        self,
        shipped_instruct_tokenizer: InstructTokenizer,
        golden_key: str,
        begin_tool_index: int,
        end_tool_index: int,
        expected_tokens_before_tool: list[int],
        expected_tokens_after_tool: list[int],
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
        instruct_decoded_goldens: dict[str, dict[str, str]],
    ) -> None:
        tokenizer = shipped_instruct_tokenizer
        tokenized = tokenizer.encode_instruct(abcd_multi_turn_tools_request())
        tokens, text = tokenized.tokens, tokenized.text
        assert text == instruct_decoded_goldens[golden_key]["abcd_multi_turn_tools"]
        assert tokens == instruct_token_id_goldens[golden_key]["abcd_multi_turn_tools"]

        begin_tool, end_tool = tokens.index(begin_tool_index), tokens.index(end_tool_index)
        assert tokens[:begin_tool] + tokens[end_tool + 1 :] == expected_tokens_before_tool + expected_tokens_after_tool
        json.loads(tokenizer.tokenizer.decode(tokens[begin_tool : end_tool + 1]))

    @pytest.mark.parametrize(
        "shipped_instruct_tokenizer, golden_key, begin_tool_index, end_tool_index, "
        "decoded_before_tool, decoded_after_tool",
        [
            ("v3_spm", "v3_spm", SPM_BEGIN_TOOL_ID, SPM_END_TOOL_ID, "a b", "SYSTEM\n\nc d"),
            ("v3_tekken", "v3_tekken", TEKKEN_BEGIN_TOOL_ID, TEKKEN_END_TOOL_ID, "ab", "SYSTEM\n\ncd"),
        ],
        indirect=["shipped_instruct_tokenizer"],
        ids=["spm", "tekken"],
    )
    def test_encode_instruct_system_tools_multi_turn(
        self,
        shipped_instruct_tokenizer: InstructTokenizer,
        golden_key: str,
        begin_tool_index: int,
        end_tool_index: int,
        decoded_before_tool: str,
        decoded_after_tool: str,
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
        instruct_decoded_goldens: dict[str, dict[str, str]],
    ) -> None:
        tokenizer = shipped_instruct_tokenizer
        tokenized = tokenizer.encode_instruct(abcd_system_tools_multi_turn_request())
        tokens, text = tokenized.tokens, tokenized.text
        assert text == instruct_decoded_goldens[golden_key]["abcd_system_tools_multi_turn"]
        assert tokens == instruct_token_id_goldens[golden_key]["abcd_system_tools_multi_turn"]

        begin_tool, end_tool = tokens.index(begin_tool_index), tokens.index(end_tool_index)
        assert tokens[end_tool + 1 :].index(3) == 0  # begin_inst follows end_tool
        assert tokenizer.tokenizer.decode(tokens[:begin_tool]) == decoded_before_tool
        assert tokenizer.tokenizer.decode(tokens[end_tool + 1 :]) == decoded_after_tool

    @pytest.mark.parametrize(
        "shipped_instruct_tokenizer, golden_key",
        [
            ("v3_spm", "v3_spm"),
            ("v3_tekken", "v3_tekken"),
        ],
        indirect=["shipped_instruct_tokenizer"],
        ids=["spm", "tekken"],
    )
    def test_encode_instruct_continue_final_message(
        self,
        shipped_instruct_tokenizer: InstructTokenizer,
        golden_key: str,
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
        instruct_decoded_goldens: dict[str, dict[str, str]],
    ) -> None:
        tokenizer = shipped_instruct_tokenizer
        tokenized = tokenizer.encode_instruct(
            InstructRequest(
                messages=abcd_messages(),
                system_prompt="SYSTEM",
                continue_final_message=True,
            )
        )
        assert tokenized.text == instruct_decoded_goldens[golden_key]["abcd_system_multi_turn_continue"]
        assert tokenized.tokens == instruct_token_id_goldens[golden_key]["abcd_system_multi_turn_continue"]

        with pytest.raises(
            InvalidMessageStructureException,
            match="Cannot continue final message if it is not an assistant message",
        ):
            tokenizer.encode_instruct(
                InstructRequest(
                    messages=abcd_trailing_user_messages(),
                    system_prompt="SYSTEM",
                    continue_final_message=True,
                )
            )

        with pytest.raises(
            InvalidAssistantMessageException,
            match="`continue_message` is only supported for assistant messages that have `prefix=False`.",
        ):
            tokenizer.encode_assistant_message(  # type: ignore[attr-defined]
                AssistantMessage(
                    content='"blabla"',
                    prefix=True,
                ),
                is_before_last_user_message=False,
                continue_message=True,
            )

    @pytest.mark.parametrize(
        "shipped_instruct_tokenizer, golden_key",
        [
            ("v3_spm", "v3_spm"),
            ("v3_tekken", "v3_tekken"),
        ],
        indirect=["shipped_instruct_tokenizer"],
        ids=["spm", "tekken"],
    )
    def test_encode_instruct_tool_message(
        self,
        shipped_instruct_tokenizer: InstructTokenizer,
        golden_key: str,
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
        instruct_decoded_goldens: dict[str, dict[str, str]],
    ) -> None:
        tokenizer = shipped_instruct_tokenizer
        tokenized = tokenizer.encode_instruct(tool_message_plain_request())
        tokens, text = tokenized.tokens, tokenized.text
        assert text == instruct_decoded_goldens[golden_key]["tool_message_plain"]
        assert tokens == instruct_token_id_goldens[golden_key]["tool_message_plain"]

        tokenized = tokenizer.encode_instruct(tool_message_json_request())
        tokens, text = tokenized.tokens, tokenized.text
        assert text == instruct_decoded_goldens[golden_key]["tool_message_json"]
        assert tokens == instruct_token_id_goldens[golden_key]["tool_message_json"]

        tokenized = tokenizer.encode_instruct(tool_message_chunks_request())
        tokens, text = tokenized.tokens, tokenized.text
        assert text == instruct_decoded_goldens[golden_key]["tool_message_chunks"]
        assert tokens == instruct_token_id_goldens[golden_key]["tool_message_chunks"]

    @pytest.mark.parametrize(
        "shipped_instruct_tokenizer, golden_key",
        [
            ("v3_spm", "v3_spm"),
            ("v3_tekken", "v3_tekken"),
        ],
        indirect=["shipped_instruct_tokenizer"],
        ids=["spm", "tekken"],
    )
    def test_encode_instruct_tool_message_no_id_fine_tuning_ok(
        self,
        shipped_instruct_tokenizer: InstructTokenizer,
        golden_key: str,
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
        instruct_decoded_goldens: dict[str, dict[str, str]],
    ) -> None:
        tokenizer = shipped_instruct_tokenizer
        # In fine-tuning we allow passing a tool call as the last message.
        # We need to make sure to not parse this empty id as "null"
        for request, golden_name in (
            (tool_call_null_id_request(), "tool_call_null_id"),
            (tool_call_no_id_request(), "tool_call_no_id"),
        ):
            tokenized = tokenizer.encode_instruct(request)
            assert tokenized.text == instruct_decoded_goldens[golden_key][golden_name]
            assert tokenized.tokens == instruct_token_id_goldens[golden_key][golden_name]

    @pytest.mark.parametrize(
        "shipped_instruct_tokenizer, golden_key",
        [
            ("v3_spm", "v3_spm"),
            ("v3_tekken", "v3_tekken"),
        ],
        indirect=["shipped_instruct_tokenizer"],
        ids=["spm", "tekken"],
    )
    def test_encode_instruct_tool_message_multiple_shots_with_history(
        self,
        shipped_instruct_tokenizer: InstructTokenizer,
        golden_key: str,
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
        instruct_decoded_goldens: dict[str, dict[str, str]],
    ) -> None:
        tokenizer = shipped_instruct_tokenizer
        tokenized = tokenizer.encode_instruct(tool_message_multiple_shots_with_history_request())
        tokens, text = tokenized.tokens, tokenized.text
        assert text == instruct_decoded_goldens[golden_key]["tool_message_multiple_shots_with_history"]
        assert tokens == instruct_token_id_goldens[golden_key]["tool_message_multiple_shots_with_history"]

    @pytest.mark.parametrize(
        "shipped_instruct_tokenizer, golden_key",
        [
            ("v3_spm", "v3_spm"),
            ("v3_tekken", "v3_tekken"),
        ],
        indirect=["shipped_instruct_tokenizer"],
        ids=["spm", "tekken"],
    )
    def test_encode_instruct_tool_message_multiple_calls(
        self,
        shipped_instruct_tokenizer: InstructTokenizer,
        golden_key: str,
        instruct_token_id_goldens: dict[str, dict[str, list[int]]],
        instruct_decoded_goldens: dict[str, dict[str, str]],
    ) -> None:
        tokenizer = shipped_instruct_tokenizer
        tokenized = tokenizer.encode_instruct(tool_multiple_calls_request())
        tokens, text = tokenized.tokens, tokenized.text
        assert text == instruct_decoded_goldens[golden_key]["tool_multiple_calls"]
        assert tokens == instruct_token_id_goldens[golden_key]["tool_multiple_calls"]

    @pytest.mark.parametrize(
        "shipped_instruct_tokenizer", ["v3_spm", "v3_tekken"], indirect=True, ids=["spm", "tekken"]
    )
    def test_encode_instruct_assistant_tool_call_and_content(
        self, shipped_instruct_tokenizer: InstructTokenizer
    ) -> None:
        tokenizer = shipped_instruct_tokenizer
        req: InstructRequest = InstructRequest(
            messages=[
                UserMessage(content="a"),
                AssistantMessage(
                    content="b",
                    tool_calls=[
                        ToolCall(id="0", function=FunctionCall(name="b", arguments="{}")),
                    ],
                ),
            ],
        )

        with pytest.raises(
            ValueError, match="Cannot have tool calls and content defined in the same assistant message"
        ):
            tokenizer.encode_instruct(req)
