r"""Reviewed expected-result manifests for public chat workflow cases.

Each success case stores ``expected.json`` plus optional NumPy sidecars under
``tests/data/expected/<case-id>/``. Loading verifies the manifest's case and
configuration associations, keeps sidecar paths inside the case directory and
never allows pickle-based array loading. One suite-wide ``(atol, rtol)``
pair governs every public image/audio array comparison.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.testing import assert_allclose

from mistral_common.tokens.tokenizers.base import Tokenized

_EXPECTED_ROOT = Path(__file__).resolve().parents[1] / "data" / "expected"

# Suite-wide numeric comparison policy for every public image and audio array
# in this child and the public transcription/speech child. The pair was
# measured across Ubuntu CI Python 3.10-3.14 (seven public media profiles,
# zero same-input differences, all changed inputs still failing at this
# tolerance); see the public-chat child plan for the recorded evidence.
# Per-case or per-array overrides are not allowed.
ATOL: float = 1e-4
RTOL: float = 0.0


@dataclass(frozen=True)
class ExpectedImage:
    """One expected returned image sidecar."""

    path: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class ExpectedAudio:
    """One expected returned audio waveform sidecar."""

    path: str
    shape: tuple[int, ...]
    dtype: str
    sampling_rate: int
    format: str


@dataclass(frozen=True)
class ExpectedSuccess:
    """The reviewed full public result of one success case."""

    case_id: str
    tokenizer_configuration_id: str
    token_ids: list[int]
    decoded_text: str
    images: tuple[ExpectedImage, ...]
    audios: tuple[ExpectedAudio, ...]
    root: Path


def case_directory(expected_root: Path, case_id: str) -> Path:
    """Directory holding one case's manifest and sidecars."""
    return expected_root / case_id


def resolve_sidecar(case_dir: Path, relative_path: str) -> Path:
    r"""Resolve one manifest media reference, rejecting escapes.

    Args:
        case_dir: The case's own manifest directory.
        relative_path: The relative reference recorded in the manifest.

    Returns:
        The contained, absolute sidecar path.

    Raises:
        ValueError: The reference is absolute, is not a ``.npy`` file, or
            resolves outside the case directory.
    """
    candidate = Path(relative_path)
    if candidate.is_absolute() or candidate.drive or candidate.root:
        raise ValueError(f"Sidecar path must be relative, got {relative_path!r}")
    if candidate.suffix != ".npy":
        raise ValueError(f"Sidecar path must reference a .npy file, got {relative_path!r}")
    resolved = (case_dir / candidate).resolve()
    if not resolved.is_relative_to(case_dir.resolve()):
        raise ValueError(f"Sidecar path escapes the case directory: {relative_path!r}")
    return resolved


def load_sidecar(sidecar_path: Path) -> np.ndarray:
    """Load one expected array sidecar without pickle support."""
    loaded = np.load(file=sidecar_path, allow_pickle=False)
    return np.asarray(loaded)


def _parse_image(entry: dict[str, Any]) -> ExpectedImage:
    return ExpectedImage(
        path=entry["path"],
        shape=tuple(int(dimension) for dimension in entry["shape"]),
        dtype=entry["dtype"],
    )


def _parse_audio(entry: dict[str, Any]) -> ExpectedAudio:
    return ExpectedAudio(
        path=entry["path"],
        shape=tuple(int(dimension) for dimension in entry["shape"]),
        dtype=entry["dtype"],
        sampling_rate=int(entry["sampling_rate"]),
        format=entry["format"],
    )


def load_expected_success(
    *, case_id: str, tokenizer_configuration_id: str, expected_root: Path | None = None
) -> ExpectedSuccess:
    r"""Load and verify one reviewed success manifest.

    Args:
        case_id: The globally unique semantic case id; must equal the manifest's.
        tokenizer_configuration_id: The bound configuration id; must equal the manifest's.
        expected_root: Root of expected manifests; defaults to the repository's
            ``tests/data/expected``.

    Returns:
        The verified expected success.

    Raises:
        ValueError: The manifest is missing, its ids disagree with the
            requested case, or a media reference is invalid.
    """
    root = expected_root if expected_root is not None else _EXPECTED_ROOT
    case_dir = case_directory(expected_root=root, case_id=case_id)
    manifest_path = case_dir / "expected.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing expected manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())

    if manifest.get("case_id") != case_id:
        raise ValueError(f"Manifest case_id {manifest.get('case_id')!r} does not match the requested case {case_id!r}")
    if manifest.get("tokenizer_configuration_id") != tokenizer_configuration_id:
        raise ValueError(
            f"Manifest tokenizer_configuration_id {manifest.get('tokenizer_configuration_id')!r} "
            f"does not match the requested configuration {tokenizer_configuration_id!r}"
        )

    images = tuple(_parse_image(entry) for entry in manifest.get("images", []))
    audios = tuple(_parse_audio(entry) for entry in manifest.get("audios", []))
    return ExpectedSuccess(
        case_id=case_id,
        tokenizer_configuration_id=tokenizer_configuration_id,
        token_ids=[int(token_id) for token_id in manifest["token_ids"]],
        decoded_text=manifest["decoded_text"],
        images=images,
        audios=audios,
        root=root,
    )


def assert_public_success(*, expected: ExpectedSuccess, tokenized: Tokenized, decoded_text: str) -> None:
    r"""Assert one public encode result equals its reviewed expectation.

    Compares the complete ordered token ids and the complete decoded text
    exactly, then checks returned media metadata and, after metadata passes,
    all array values under the suite-wide tolerance pair.

    Args:
        expected: The reviewed expectation for this case.
        tokenized: The public encode result.
        decoded_text: The result decoded with special tokens kept.
    """
    assert tokenized.tokens == expected.token_ids, (
        f"Token ids differ from the reviewed manifest: "
        f"expected {len(expected.token_ids)} ids, got {len(tokenized.tokens)}"
    )
    assert decoded_text == expected.decoded_text, "Decoded text differs from the reviewed manifest"

    case_dir = case_directory(expected_root=expected.root, case_id=expected.case_id)
    assert len(tokenized.images) == len(expected.images), (
        f"Expected {len(expected.images)} returned images, got {len(tokenized.images)}"
    )
    for actual_image, expected_image in zip(tokenized.images, expected.images, strict=True):
        assert actual_image.shape == expected_image.shape, (
            f"Image shape {actual_image.shape} differs from reviewed {expected_image.shape}"
        )
        assert actual_image.dtype == np.dtype(expected_image.dtype), (
            f"Image dtype {actual_image.dtype} differs from reviewed {expected_image.dtype}"
        )
        assert_allclose(
            actual=actual_image.astype(dtype=np.float64),
            desired=load_sidecar(resolve_sidecar(case_dir, expected_image.path)),
            atol=ATOL,
            rtol=RTOL,
            err_msg="Image array differs from the reviewed sidecar",
        )

    assert len(tokenized.audios) == len(expected.audios), (
        f"Expected {len(expected.audios)} returned audios, got {len(tokenized.audios)}"
    )
    for actual_audio, expected_audio in zip(tokenized.audios, expected.audios, strict=True):
        assert actual_audio.sampling_rate == expected_audio.sampling_rate, (
            f"Audio sampling rate {actual_audio.sampling_rate} differs from reviewed {expected_audio.sampling_rate}"
        )
        assert actual_audio.format == expected_audio.format, (
            f"Audio format {actual_audio.format!r} differs from reviewed {expected_audio.format!r}"
        )
        assert actual_audio.audio_array.shape == expected_audio.shape, (
            f"Audio shape {actual_audio.audio_array.shape} differs from reviewed {expected_audio.shape}"
        )
        assert actual_audio.audio_array.dtype == np.dtype(expected_audio.dtype), (
            f"Audio dtype {actual_audio.audio_array.dtype} differs from reviewed {expected_audio.dtype}"
        )
        assert_allclose(
            actual=actual_audio.audio_array.astype(dtype=np.float64),
            desired=load_sidecar(resolve_sidecar(case_dir, expected_audio.path)),
            atol=ATOL,
            rtol=RTOL,
            err_msg="Audio array differs from the reviewed sidecar",
        )
