import base64
import json
import re
from pathlib import Path
from typing import List, Optional, Sequence

import pytest
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from mistral_common.tokens.tokenizers.tekken import (
    ModelData,
    TekkenConfig,
    Tekkenizer,
    TokenInfo,
    is_tekken,
)


def _quick_vocab(extra_toks: Sequence[bytes] = ()) -> List[TokenInfo]:
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


def _write_tekkenizer_model(
    tmp_path: Path,
    vocab: Optional[List[TokenInfo]] = None,
    pattern: str = ".",
    num_special_tokens: int = 100,
    version: Optional[str] = "v3",
) -> None:
    # Create the vocab.json file
    if vocab is None:
        vocab = _quick_vocab()

    config = {
        "pattern": pattern,
        "default_num_special_tokens": num_special_tokens,
        "default_vocab_size": 256 + 3 + num_special_tokens,
    }

    if version is not None:
        config["version"] = version

    model = ModelData(
        vocab=vocab if vocab else _quick_vocab(),
        config=TekkenConfig(**config),  # type: ignore
        version=1,
        type="Tekken",
    )
    with open(tmp_path, "w") as f:
        json.dump(model, f)


def test_roundtrip() -> None:
    tekkenizer = Tekkenizer(
        _quick_vocab(extra_toks=[b"beau", b"My", b"unused"]),
        pattern=".",
        vocab_size=256 + 3 + 100,
        num_special_tokens=100,
        version=TokenizerVersion.v3,
    )
    inputs = "My very beautiful string"
    encoded = tekkenizer.encode(inputs, False, False)
    decoded = tekkenizer.decode(encoded)
    assert inputs == decoded


def test_version(tmp_path: Path) -> None:
    tokpath = tmp_path / "tekken.json"

    vocab = _quick_vocab(extra_toks=[b"beau", b"My", b"unused"])
    pattern = "."
    num_special_tokens = 100

    # test all versions can be loaded
    assert len(TokenizerVersion.__members__) > 0
    for version in TokenizerVersion.__members__:
        _write_tekkenizer_model(tokpath, vocab, pattern, num_special_tokens, version=str(version))
        tekkenizer_loaded = Tekkenizer.from_file(tokpath)

        assert tekkenizer_loaded.version == TokenizerVersion(version)

    # test `None` and other version cannot be loaded
    for version in [None, "dummy-v"]:  # type: ignore
        _write_tekkenizer_model(tokpath, vocab, pattern, num_special_tokens, version=version)
        with pytest.raises(ValueError, match=re.compile("Unknown version:*")):
            tekkenizer_loaded = Tekkenizer.from_file(tokpath)


def test_read_from_file(tmp_path: Path) -> None:
    inputs = "My very beatuiful string"
    tokpath = tmp_path / "tekken.json"
    vocab = _quick_vocab(extra_toks=[b"beau", b"My", b"unused"])
    pattern = "."
    num_special_tokens = 100
    _write_tekkenizer_model(tokpath, vocab, pattern, num_special_tokens)

    tekkenizer_loaded = Tekkenizer.from_file(tokpath)
    tekkenizer = Tekkenizer(
        vocab,
        pattern,
        vocab_size=256 + 3 + num_special_tokens,
        num_special_tokens=100,
        version=TokenizerVersion.v3,
    )
    encoded = tekkenizer.encode(inputs, False, False)
    encoded_from_loaded = tekkenizer_loaded.encode(inputs, False, False)

    assert encoded == encoded_from_loaded


def test_istekken(tmp_path: Path) -> None:
    # Initialize the Tekkenizer with a path
    common_case = tmp_path / "tekken.json"
    _write_tekkenizer_model(common_case)
    assert is_tekken(common_case)
    common_case.unlink()

    fancy_name = tmp_path / "tekken_the_destroyer.json"
    _write_tekkenizer_model(fancy_name)
    assert is_tekken(fancy_name)
    fancy_name.unlink()

    bad_name = tmp_path / "sentencepiece.json"
    _write_tekkenizer_model(bad_name)
    assert not is_tekken(bad_name)
    bad_name.unlink()

    not_json = tmp_path / "tekken.model"
    _write_tekkenizer_model(not_json)
    assert not is_tekken(not_json)
    not_json.unlink()

    assert not is_tekken(tmp_path / "nonexistent.json")


def test_isbyte() -> None:
    tekkenizer = Tekkenizer(
        _quick_vocab([b"hello"]),
        pattern=r".+",  # single token, whole string
        vocab_size=256 + 1 + 100,
        num_special_tokens=100,
        version=TokenizerVersion.v3,
    )
    tok = tekkenizer.encode("hello", False, False)
    assert len(tok) == 1
    assert not tekkenizer.is_byte(tok[0])

    byte_tok = tekkenizer.encode(chr(0), False, False)
    assert len(byte_tok) == 1
    assert tekkenizer.is_byte(byte_tok[0]), byte_tok
    assert byte_tok[0] < 256 + tekkenizer.num_special_tokens <= tok[0]
