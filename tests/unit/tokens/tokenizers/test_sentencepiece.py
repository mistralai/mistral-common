from pathlib import Path

import numpy as np
import pytest

import mistral_common
from mistral_common.exceptions import TokenizerException
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy, Tokenizer, TokenizerVersion
from mistral_common.tokens.tokenizers.sentencepiece import (
    SentencePieceTokenizer,
    get_spm_version,
    is_sentencepiece,
)

_DATA_DIR = Path(mistral_common.__file__).parent / "data"
_V7_MODEL = _DATA_DIR / "mistral_instruct_tokenizer_241114.model.v7"

# Bundled SentencePiece models and the version their filename encodes; the
# multimodal suffix (v7m1) selects the image config, not the instruct version.
_BUNDLED_SPM_MODELS = [
    pytest.param("tokenizer.model.v1", TokenizerVersion.v1, id="v1"),
    pytest.param("mistral_instruct_tokenizer_240216.model.v2", TokenizerVersion.v2, id="v2"),
    pytest.param("mistral_instruct_tokenizer_240323.model.v3", TokenizerVersion.v3, id="v3"),
    pytest.param("mistral_instruct_tokenizer_241114.model.v7", TokenizerVersion.v7, id="v7"),
    pytest.param("mistral_instruct_tokenizer_241114.model.v7m1", TokenizerVersion.v7, id="v7m1"),
]


@pytest.fixture(scope="module")
def tokenizer_v7() -> SentencePieceTokenizer:
    r"""Provide the bundled v7 SentencePiece model."""
    return SentencePieceTokenizer(model_path=_V7_MODEL)


def test_bundled_v7_matches_tokenizer_contract(tokenizer_v7: SentencePieceTokenizer) -> None:
    r"""Expose the bundled v7 vocabulary and control-token IDs through the base `Tokenizer` interface."""
    tokenizer: Tokenizer = tokenizer_v7
    vocab = tokenizer.vocab()

    assert tokenizer.n_words == len(vocab)
    assert vocab[tokenizer.bos_id] == "<s>"
    assert vocab[tokenizer.eos_id] == "</s>"
    assert tokenizer.get_special_token("<s>") == tokenizer.bos_id
    assert tokenizer.get_special_token("</s>") == tokenizer.eos_id
    assert tokenizer.id_to_piece(tokenizer.unk_id) == "<unk>"
    assert tokenizer.pad_id == -1


@pytest.mark.parametrize(
    ("token", "is_control"),
    [
        pytest.param("</s>", True, id="string-eos"),
        pytest.param("a", False, id="string-piece"),
        pytest.param(1, True, id="int-eos"),
        pytest.param(230, True, id="int-control"),
        pytest.param(1001, False, id="int-piece"),
        pytest.param(np.int64(1), True, id="numpy-int-eos"),
        pytest.param(np.int64(1001), False, id="numpy-int-piece"),
    ],
)
def test_is_special_classifies_strings_ints_and_numpy_ints(
    tokenizer_v7: SentencePieceTokenizer, token: str | int, is_control: bool
) -> None:
    r"""Classify control tokens given as strings, ints and NumPy integers."""
    assert tokenizer_v7.is_special(token) is is_control


def test_special_ids_property(tokenizer_v7: SentencePieceTokenizer) -> None:
    r"""Expose the control-token IDs as a non-empty set."""
    special_ids = tokenizer_v7.special_ids
    assert isinstance(special_ids, set)
    assert len(special_ids) > 0


def test_num_special_tokens_property(tokenizer_v7: SentencePieceTokenizer) -> None:
    r"""Report the bundled v7 control-token count, consistent with `special_ids`."""
    num_special = tokenizer_v7.num_special_tokens
    assert isinstance(num_special, int)
    assert num_special == 748
    assert num_special == len(tokenizer_v7.special_ids)


def test_decode_raise_policy_decodes_normal_tokens(tokenizer_v7: SentencePieceTokenizer) -> None:
    r"""Decode ordinary tokens under RAISE, matching IGNORE when no special token is present."""
    text = "Hello world, how are you?"
    ids = tokenizer_v7.encode(text, bos=False, eos=False)
    assert tokenizer_v7.decode(ids, SpecialTokenPolicy.RAISE) == text
    assert tokenizer_v7.decode(ids, SpecialTokenPolicy.RAISE) == tokenizer_v7.decode(ids, SpecialTokenPolicy.IGNORE)


def test_decode_raise_policy_raises_on_special_tokens(tokenizer_v7: SentencePieceTokenizer) -> None:
    r"""Reject decoding a sequence that contains a control token under RAISE."""
    ids = tokenizer_v7.encode("Hello world", bos=True, eos=True)
    assert any(tokenizer_v7.is_special(tok) for tok in ids)
    with pytest.raises(ValueError, match="special_token_policy=RAISE"):
        tokenizer_v7.decode(ids, SpecialTokenPolicy.RAISE)


@pytest.mark.parametrize("policy", list(SpecialTokenPolicy))
def test_decode_string_special_token_policy_matches_enum(
    tokenizer_v7: SentencePieceTokenizer, policy: SpecialTokenPolicy
) -> None:
    r"""Accept each policy as its string value, decoding identically to the enum member."""
    include_special = policy is not SpecialTokenPolicy.RAISE
    ids = tokenizer_v7.encode("Hello world", bos=include_special, eos=include_special)
    assert tokenizer_v7.decode(ids, policy.value) == tokenizer_v7.decode(ids, policy)  # type: ignore[arg-type]


def test_decode_invalid_special_token_policy_string_raises(tokenizer_v7: SentencePieceTokenizer) -> None:
    r"""Reject an unknown policy string with the project-owned diagnostic."""
    ids = tokenizer_v7.encode("Hello world", bos=True, eos=True)
    with pytest.raises(ValueError, match=r"Invalid `special_token_policy`"):
        tokenizer_v7.decode(ids, "keeep")  # type: ignore[arg-type]


_KNOWN_SUFFIXES = [
    pytest.param(f".model.{version.value}{mm}", id=f"model-{version.value}{mm}")
    for version in TokenizerVersion
    for mm in ("", "m1")
] + [pytest.param(".model", id="model-bare")]


@pytest.mark.parametrize("suffix", _KNOWN_SUFFIXES)
def test_is_sentencepiece_recognizes_known_suffixes(tmp_path: Path, suffix: str) -> None:
    r"""Recognize every instruct version with and without a multimodal suffix, plus the bare `.model`."""
    path = tmp_path / f"tokenizer{suffix}"
    path.touch()
    assert is_sentencepiece(path) is True


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("tokenizer.model.vx", id="unknown-version"),
        pytest.param("tokenizer.model.v16", id="future-version"),
        pytest.param("tokenizer.json", id="non-model-suffix"),
    ],
)
def test_is_sentencepiece_rejects_unknown_suffixes(tmp_path: Path, name: str) -> None:
    r"""Reject existing files whose suffix is not a known SentencePiece suffix."""
    path = tmp_path / name
    path.touch()
    assert is_sentencepiece(path) is False


def test_is_sentencepiece_rejects_missing_file(tmp_path: Path) -> None:
    r"""Detection requires the file to exist, even with a recognized suffix."""
    assert is_sentencepiece(tmp_path / "tokenizer.model.v3") is False


@pytest.mark.parametrize(("filename", "expected_version"), _BUNDLED_SPM_MODELS)
def test_bundled_spm_model_loads_with_filename_version(filename: str, expected_version: TokenizerVersion) -> None:
    r"""Infer each bundled model's instruct version from its filename when loading."""
    tokenizer = SentencePieceTokenizer(model_path=_DATA_DIR / filename)
    assert tokenizer.version == expected_version


@pytest.mark.parametrize(("filename", "expected_version"), _BUNDLED_SPM_MODELS)
def test_get_spm_version_parses_bundled_filenames(filename: str, expected_version: TokenizerVersion) -> None:
    r"""Parse each bundled filename to its instruct version without loading the model."""
    assert get_spm_version(_DATA_DIR / filename) == expected_version


def test_get_spm_version_treats_bare_model_as_v1() -> None:
    r"""Map the deprecated unversioned `tokenizer.model` filename to v1."""
    assert get_spm_version("tokenizer.model") == TokenizerVersion.v1


def test_get_spm_version_rejects_unknown_suffix() -> None:
    r"""Raise `TokenizerException` for an unrecognized version suffix."""
    with pytest.raises(TokenizerException, match="Unrecognized tokenizer filename"):
        get_spm_version("tokenizer.model.vx")


def test_get_spm_version_raise_deprecated_rejects_bare_model() -> None:
    r"""Raise `TokenizerException` for the bare `.model` filename when deprecation is requested."""
    with pytest.raises(TokenizerException, match="Make sure to rename your tokenizer file"):
        get_spm_version("tokenizer.model", raise_deprecated=True)


def test_missing_model_file_constructor_asserts(tmp_path: Path) -> None:
    r"""Distinguish construction from detection: a missing path fails the constructor assertion."""
    with pytest.raises(AssertionError):
        SentencePieceTokenizer(model_path=tmp_path / "missing.model.v3")
