import pytest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage, AssistantMessage
from mistral_common.protocol.instruct.request import InstructRequest
from mistral_common.exceptions import TokenizerException

def test_truncation_preserves_final_assistant_message():
    tokenizer = MistralTokenizer.v7()
    messages = [
        UserMessage(content="Turn 1 User"),
        AssistantMessage(content="Turn 1 Assistant"),
        UserMessage(content="Turn 2 User"),
        AssistantMessage(content="Turn 2 Assistant"),
    ]
    
    # Tronchiamo per far spazio solo al turno 2
    res = tokenizer.instruct_tokenizer.encode_instruct(
        InstructRequest(messages=messages, truncate_at_max_tokens=20)
    )
    decoded = tokenizer.instruct_tokenizer.decode(res.tokens)
    
    assert "Turn 2 Assistant" in decoded
    assert "Turn 2 User" in decoded

def test_truncation_raises_if_last_turn_exceeds_max_tokens():
    tokenizer = MistralTokenizer.v7()
    messages = [
        UserMessage(content="Hello, who are you?"),
        AssistantMessage(content="I am a helpful AI assistant created by Mistral AI."),
    ]
    
    # Se il limite e' inferiore al solo turno finale, deve sollevare TokenizerException
    with pytest.raises(TokenizerException):
        tokenizer.instruct_tokenizer.encode_instruct(
            InstructRequest(messages=messages, truncate_at_max_tokens=5)
        )
