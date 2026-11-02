from mistral_common.protocol.instruct.request import ReasoningEffort


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
