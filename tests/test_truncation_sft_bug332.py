import pytest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage, AssistantMessage
from mistral_common.protocol.instruct.request import InstructRequest
from mistral_common.exceptions import TokenizerException

def test_truncation_preserves_final_assistant_message():
    tokenizer = MistralTokenizer.v7()
    messages = [
        UserMessage(content="Hello, who are you?"),
        AssistantMessage(content="I am a helpful AI assistant created by Mistral AI.")
    ]

    # Non c'è abbastanza spazio per l'intera conversazione + risposta finale,
    # ci si aspetta TokenizerException anziché la cancellazione silenziosa dell'assistente.
    with pytest.raises(TokenizerException):
        tokenizer.instruct_tokenizer.encode_instruct(
            InstructRequest(messages=messages, truncate_at_max_tokens=10)
        )

def test_truncation_drops_earlier_turns_and_keeps_final_assistant_message():
    tokenizer = MistralTokenizer.v7()
    messages = [
        UserMessage(content="Old question 1"),
        AssistantMessage(content="Old answer 1"),
        UserMessage(content="Old question 2"),
        AssistantMessage(content="Old answer 2"),
        UserMessage(content="Current question"),
        AssistantMessage(content="Current answer"),
    ]

    # Calcoliamo i token totali per impostare max_tokens sufficiente a contenere il turno finale
    full_res = tokenizer.instruct_tokenizer.encode_instruct(
        InstructRequest(messages=messages)
    )
    
    # Inseriamo un valore di max_tokens che costringe a cancellare i turni vecchi ma preserva quello finale
    truncated_res = tokenizer.instruct_tokenizer.encode_instruct(
        InstructRequest(messages=messages, truncate_at_max_tokens=25)
    )

    decoded = tokenizer.instruct_tokenizer.decode(truncated_res.tokens)
    assert "Current question" in decoded
    assert "Current answer" in decoded
