import pytest

from mistral_common.exceptions import InvalidAssistantMessageException, InvalidMessageStructureException
from mistral_common.protocol.instruct.messages import AssistantMessage, UserMessage
from mistral_common.tokens.instruct.request import InstructRequest
from mistral_common.tokens.tokenizers.base import InstructTokenizer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer


@pytest.fixture()
def tokenizer() -> InstructTokenizer:
    return MistralTokenizer.v1().instruct_tokenizer


def test_normal(tokenizer: InstructTokenizer) -> None:
    tokenized = tokenizer.encode_instruct(
        InstructRequest(
            messages=[
                UserMessage(content="a"),
                AssistantMessage(content="b"),
                UserMessage(content="c"),
                AssistantMessage(content="d"),
            ]
        )
    )
    tokens, text = tokenized.tokens, tokenized.text
    assert text == "<s>▁[INST]▁a▁[/INST]▁b</s>▁[INST]▁c▁[/INST]▁d</s>"
    assert tokens == [
        1,
        733,
        16289,
        28793,
        264,
        733,
        28748,
        16289,
        28793,
        287,
        2,
        733,
        16289,
        28793,
        277,
        733,
        28748,
        16289,
        28793,
        281,
        2,
    ]


def test_system_singleturn(tokenizer: InstructTokenizer) -> None:
    tokenized = tokenizer.encode_instruct(InstructRequest(messages=[UserMessage(content="a")], system_prompt="SYSTEM"))
    tokens, text = tokenized.tokens, tokenized.text
    assert text == "<s>▁[INST]▁SYSTEM<0x0A><0x0A>a▁[/INST]"
    assert tokens == [1, 733, 16289, 28793, 17121, 22526, 13, 13, 28708, 733, 28748, 16289, 28793]
    assert tokenizer.tokenizer.decode(tokens) == "[INST] SYSTEM\n\na [/INST]"


def test_system_multiturn(tokenizer: InstructTokenizer) -> None:
    tokenized = tokenizer.encode_instruct(
        InstructRequest(
            messages=[
                UserMessage(content="a"),
                AssistantMessage(content="b"),
                UserMessage(content="c"),
                AssistantMessage(content="d"),
            ],
            system_prompt="SYSTEM",
        )
    )
    tokens, text = tokenized.tokens, tokenized.text
    assert text == "<s>▁[INST]▁SYSTEM<0x0A><0x0A>a▁[/INST]▁b</s>▁[INST]▁c▁[/INST]▁d</s>"
    assert tokens == [
        1,
        733,
        16289,
        28793,
        17121,
        22526,
        13,
        13,
        28708,
        733,
        28748,
        16289,
        28793,
        287,
        2,
        733,
        16289,
        28793,
        277,
        733,
        28748,
        16289,
        28793,
        281,
        2,
    ]
    first_eos = tokens.index(2)
    assert tokenizer.tokenizer.decode(tokens[first_eos:]) == "[INST] c [/INST] d"


def test_continue_final_message(tokenizer: InstructTokenizer) -> None:
    tokenized = tokenizer.encode_instruct(
        InstructRequest(
            messages=[
                UserMessage(content="a"),
                AssistantMessage(content="b"),
                UserMessage(content="c"),
                AssistantMessage(content="d"),
            ],
            system_prompt="SYSTEM",
            continue_final_message=True,
        )
    )
    tokens, text = tokenized.tokens, tokenized.text
    assert text == "<s>▁[INST]▁SYSTEM<0x0A><0x0A>a▁[/INST]▁b</s>▁[INST]▁c▁[/INST]▁d"
    assert tokens == [
        1,
        733,
        16289,
        28793,
        17121,
        22526,
        13,
        13,
        28708,
        733,
        28748,
        16289,
        28793,
        287,
        2,
        733,
        16289,
        28793,
        277,
        733,
        28748,
        16289,
        28793,
        281,
    ]

    with pytest.raises(
        InvalidMessageStructureException, match="Cannot continue final message if it is not an assistant message"
    ):
        tokenizer.encode_instruct(
            InstructRequest(
                messages=[
                    UserMessage(content="a"),
                    AssistantMessage(content="b"),
                    UserMessage(content="c"),
                ],
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
