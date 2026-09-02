import base64
from typing import Sequence

from mistral_common.tokens.tokenizers.base import SpecialTokens, TokenizerVersion
from mistral_common.tokens.tokenizers.tekken import SpecialTokenInfo, Tekkenizer, TokenInfo


def quick_vocab(extra_toks: Sequence[bytes] = ()) -> list[TokenInfo]:
    vocab = [TokenInfo(rank=i, token_bytes=base64.b64encode(bytes([i])).decode(), token_str=chr(i)) for i in range(256)]
    for i, tok in enumerate(extra_toks):
        vocab.append(
            TokenInfo(
                rank=256 + i,
                token_bytes=base64.b64encode(tok).decode(),
                token_str=tok.decode(),
            )
        )
    return vocab


def get_special_tokens(
    tokenizer_version: TokenizerVersion, add_audio: bool = False, add_think: bool = False
) -> list[SpecialTokenInfo]:
    special_tokens = list(Tekkenizer.DEPRECATED_SPECIAL_TOKENS)
    if tokenizer_version < TokenizerVersion.v7 and add_audio:
        raise ValueError("Audio tokens are only supported in v7 and above")

    if tokenizer_version <= TokenizerVersion.v7 and not add_audio:
        return special_tokens

    # fill special tokens until 24
    special_tokens += [
        SpecialTokenInfo(rank=i, token_str=f"<SPECIAL_{i}>", is_control=True) for i in range(len(special_tokens), 24)
    ]

    if add_audio:
        # add audio tokes
        special_tokens += [
            SpecialTokenInfo(rank=24, token_str=SpecialTokens.audio, is_control=True),
            SpecialTokenInfo(rank=25, token_str=SpecialTokens.begin_audio, is_control=True),
        ]

    # fill special tokens until 32
    special_tokens += [
        SpecialTokenInfo(rank=i, token_str=f"<SPCECIAL_{i}>", is_control=True) for i in range(len(special_tokens), 32)
    ]

    if tokenizer_version > TokenizerVersion.v7:
        special_tokens += [
            SpecialTokenInfo(rank=32, token_str=SpecialTokens.args, is_control=True),
            SpecialTokenInfo(rank=33, token_str=SpecialTokens.call_id, is_control=True),
        ]

    # fill special tokens until 34
    special_tokens += [
        SpecialTokenInfo(rank=i, token_str=f"<SPCECIAL_{i}>", is_control=True) for i in range(len(special_tokens), 34)
    ]

    if add_audio:
        assert not add_think, f"Audio and think tokens are mutually exclusive, got {add_audio} and {add_think}"
        special_tokens += [
            SpecialTokenInfo(rank=34, token_str=SpecialTokens.transcribe, is_control=True),
            SpecialTokenInfo(rank=35, token_str=SpecialTokens.text_to_audio, is_control=True),
            SpecialTokenInfo(rank=36, token_str=SpecialTokens.audio_to_text, is_control=True),
        ]

    if tokenizer_version < TokenizerVersion.v13:
        return special_tokens

    if not add_audio and add_think:
        special_tokens += [SpecialTokenInfo(rank=34, token_str=f"<SPCECIAL_{34}>", is_control=True)]

    if add_think:
        special_tokens += [
            SpecialTokenInfo(rank=35, token_str="[THINK]", is_control=True),
            SpecialTokenInfo(rank=36, token_str="[/THINK]", is_control=True),
        ]

    if tokenizer_version >= TokenizerVersion.v15:
        # fill until rank 37
        special_tokens += [
            SpecialTokenInfo(rank=i, token_str=f"<SPCECIAL_{i}>", is_control=True)
            for i in range(len(special_tokens), 37)
        ]
        special_tokens += [
            SpecialTokenInfo(rank=37, token_str=SpecialTokens.begin_model_settings, is_control=True),
            SpecialTokenInfo(rank=38, token_str=SpecialTokens.end_model_settings, is_control=True),
        ]

    return special_tokens
