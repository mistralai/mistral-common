import pytest

from mistral_common.exceptions import UnsupportedTokenizerFeatureException
from mistral_common.protocol.instruct.messages import AssistantMessage, ChatMessage, ToolMessage, UserMessage
from mistral_common.protocol.instruct.request import InstructRequest
from mistral_common.protocol.instruct.tool_calls import FunctionCall, Tool, ToolCall
from mistral_common.tokens.tokenizers.base import InstructTokenizer, SpecialTokenPolicy, SpecialTokens, Tokenized
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

_PARALLEL_RESULTS_MESSAGE = r"v2.*multiple tool results.*assistant turn"


@pytest.fixture()
def v2_tokenizer() -> InstructTokenizer:
    return MistralTokenizer.v2().instruct_tokenizer


def build_instruct_request(*, call_count: int, result_count: int) -> InstructRequest[ChatMessage, Tool]:
    tool_calls = [
        ToolCall(id=f"call0000{index}", function=FunctionCall(name=f"tool_{index}", arguments="{}"))
        for index in range(1, call_count + 1)
    ]
    messages: list[ChatMessage] = [
        UserMessage(content="Run these tools."),
        AssistantMessage(content=None, tool_calls=tool_calls),
    ]
    messages.extend(
        ToolMessage(
            name=f"tool_{index}",
            content=f"result {index}",
            tool_call_id=f"call0000{index}",
        )
        for index in range(1, result_count + 1)
    )
    return InstructRequest[ChatMessage, Tool](messages=messages)


def tool_result_block_count(tokenizer: InstructTokenizer, tokenized: Tokenized) -> int:
    r"""Count encoded v2 or v3 tool-result blocks."""
    marker = tokenizer.tokenizer.get_special_token(SpecialTokens.begin_tool_results.value)
    return tokenized.tokens.count(marker)


@pytest.mark.parametrize(
    "call_count",
    [2, 1],
    ids=["two-calls-two-results", "one-call-two-results"],
)
def test_rejects_multiple_results_for_one_v2_turn(v2_tokenizer: InstructTokenizer, call_count: int) -> None:
    request = build_instruct_request(call_count=call_count, result_count=2)

    with pytest.raises(UnsupportedTokenizerFeatureException, match=_PARALLEL_RESULTS_MESSAGE):
        v2_tokenizer.encode_instruct(request)


def test_allows_multiple_calls_with_one_result(v2_tokenizer: InstructTokenizer) -> None:
    request = build_instruct_request(call_count=2, result_count=1)

    tokenized = v2_tokenizer.encode_instruct(request)

    assert tool_result_block_count(tokenizer=v2_tokenizer, tokenized=tokenized) == 1


def test_allows_sequential_single_result_turns(v2_tokenizer: InstructTokenizer) -> None:
    request = InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="First request."),
            AssistantMessage(
                content=None,
                tool_calls=[ToolCall(id="call00001", function=FunctionCall(name="tool_1", arguments="{}"))],
            ),
            ToolMessage(name="tool_1", content="first result", tool_call_id="call00001"),
            AssistantMessage(content="First turn is complete."),
            UserMessage(content="Second request."),
            AssistantMessage(
                content=None,
                tool_calls=[ToolCall(id="call00002", function=FunctionCall(name="tool_2", arguments="{}"))],
            ),
            ToolMessage(name="tool_2", content="second result", tool_call_id="call00002"),
        ]
    )

    tokenized = v2_tokenizer.encode_instruct(request)

    assert tool_result_block_count(tokenizer=v2_tokenizer, tokenized=tokenized) == 1
    assert "second▁result" in v2_tokenizer.decode(tokens=tokenized.tokens, special_token_policy=SpecialTokenPolicy.KEEP)


def test_ignores_multiple_results_before_latest_user(v2_tokenizer: InstructTokenizer) -> None:
    request = InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="Earlier request."),
            AssistantMessage(
                content=None,
                tool_calls=[
                    ToolCall(id="call00001", function=FunctionCall(name="tool_1", arguments="{}")),
                    ToolCall(id="call00002", function=FunctionCall(name="tool_2", arguments="{}")),
                ],
            ),
            ToolMessage(name="tool_1", content="earlier result one", tool_call_id="call00001"),
            ToolMessage(name="tool_2", content="earlier result two", tool_call_id="call00002"),
            UserMessage(content="Latest request."),
            AssistantMessage(content="Latest answer."),
        ]
    )

    tokenized = v2_tokenizer.encode_instruct(request)

    assert tool_result_block_count(tokenizer=v2_tokenizer, tokenized=tokenized) == 0
    assert "Latest▁answer." in v2_tokenizer.decode(
        tokens=tokenized.tokens, special_token_policy=SpecialTokenPolicy.KEEP
    )


def test_v3_still_encodes_multiple_tool_results() -> None:
    tokenizer = MistralTokenizer.v3().instruct_tokenizer
    request = build_instruct_request(call_count=2, result_count=2)

    tokenized = tokenizer.encode_instruct(request)

    assert tool_result_block_count(tokenizer=tokenizer, tokenized=tokenized) == 2
