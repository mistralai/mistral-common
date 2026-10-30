from mistral_common.protocol.instruct.request import ChatCompletionRequest, ReasoningEffort
from mistral_common.protocol.instruct.tool_calls import ToolChoiceEnum


def test_reasoning_effort_order() -> None:
    assert list(ReasoningEffort) == [
        ReasoningEffort.none,
        ReasoningEffort.minimal,
        ReasoningEffort.low,
        ReasoningEffort.medium,
        ReasoningEffort.high,
        ReasoningEffort.xhigh,
        ReasoningEffort.max,
    ]


def test_from_openai_null_tool_choice_means_default() -> None:
    r"""OpenAI clients send `tool_choice: null` to mean the default (`auto`)."""
    request = ChatCompletionRequest.from_openai([{"role": "user", "content": "hi"}], tool_choice=None)

    assert request.tool_choice == ToolChoiceEnum.auto


def test_from_openai_null_parallel_tool_calls_means_default() -> None:
    request = ChatCompletionRequest.from_openai([{"role": "user", "content": "hi"}], parallel_tool_calls=None)

    assert request.parallel_tool_calls is True
