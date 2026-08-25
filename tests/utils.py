from mistral_common.tokens.tokenizers.base import InstructTokenizer, SpecialTokenPolicy, Tokenized
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer


def decode_keep(tokenizer: InstructTokenizer | MistralTokenizer, tokenized: Tokenized) -> str:
    r"""Decode `tokenized.tokens` back to text, keeping special tokens.

    Args:
        tokenizer: The tokenizer that produced `tokenized`.
        tokenized: The tokenized result to decode.

    Returns:
        The decoded string, including special tokens.
    """
    return tokenizer.decode(tokens=tokenized.tokens, special_token_policy=SpecialTokenPolicy.KEEP)
