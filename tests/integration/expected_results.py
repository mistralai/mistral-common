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
    array: np.ndarray


@dataclass(frozen=True)
class ExpectedAudio:
    """One expected returned audio waveform sidecar."""

    path: str
    shape: tuple[int, ...]
    dtype: str
    sampling_rate: int
    format: str
    array: np.ndarray


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


@dataclass(frozen=True)
class _ImageManifestEntry:
    path: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class _AudioManifestEntry:
    path: str
    shape: tuple[int, ...]
    dtype: str
    sampling_rate: int
    format: str


@dataclass(frozen=True)
class _ValidatedManifest:
    case_id: str
    tokenizer_configuration_id: str
    token_ids: list[int]
    decoded_text: str
    images: list[_ImageManifestEntry]
    audios: list[_AudioManifestEntry]


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
    if not isinstance(loaded, np.ndarray):
        if isinstance(loaded, np.lib.npyio.NpzFile):
            loaded.close()
        raise ValueError(f"Expected a NumPy array sidecar, got {type(loaded).__name__}")
    return loaded


def _load_manifest_sidecar(
    *,
    case_dir: Path,
    case_id: str,
    media_kind: str,
    entry_index: int,
    path: str,
    shape: tuple[int, ...],
    dtype: str,
) -> np.ndarray:
    entry_context = f"Case {case_id!r} {media_kind} entry {entry_index}"
    try:
        sidecar_path = resolve_sidecar(case_dir=case_dir, relative_path=path)
    except ValueError as error:
        raise ValueError(f"{entry_context}: {error}") from error
    if not sidecar_path.is_file():
        raise ValueError(f"{entry_context} references missing sidecar {path!r}")

    try:
        array = load_sidecar(sidecar_path=sidecar_path)
    except (OSError, ValueError) as error:
        raise ValueError(f"{entry_context} cannot load sidecar {path!r} without pickle: {error}") from error

    if array.shape != shape:
        raise ValueError(f"{entry_context} sidecar shape {array.shape} does not match manifest declared shape {shape}")
    try:
        declared_dtype = np.dtype(dtype)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{entry_context} has invalid manifest dtype {dtype!r} ({type(dtype).__name__})") from error
    if array.dtype != declared_dtype:
        raise ValueError(
            f"{entry_context} sidecar dtype {array.dtype} does not match manifest declared dtype {declared_dtype}"
        )
    return array


def _require_exact_integer(*, value: Any, context: str, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{context} field {field!r} must be an integer, got {value!r} ({type(value).__name__})")
    return value


def _require_object(*, value: Any, context: str, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} field {field!r} must be an object, got {value!r} ({type(value).__name__})")
    return value


def _require_array(*, value: Any, context: str, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context} field {field!r} must be an array, got {value!r} ({type(value).__name__})")
    return value


def _require_string(*, value: Any, context: str, field: str, non_empty: bool) -> str:
    if not isinstance(value, str) or (non_empty and not value):
        requirement = "a non-empty string" if non_empty else "a string"
        raise ValueError(f"{context} field {field!r} must be {requirement}, got {value!r} ({type(value).__name__})")
    return value


def _required_field(*, record: dict[str, Any], field: str, context: str) -> Any:
    if field not in record:
        raise ValueError(f"{context} is missing required field {field!r}")
    return record[field]


def _validate_shape(*, value: Any, context: str) -> tuple[int, ...]:
    dimensions = _require_array(value=value, context=context, field="shape")
    return tuple(
        _require_exact_integer(value=dimension, context=context, field=f"shape[{index}]")
        for index, dimension in enumerate(dimensions)
    )


def _validate_image_entry(*, value: Any, case_id: str, entry_index: int) -> _ImageManifestEntry:
    context = f"Case {case_id!r} image entry {entry_index}"
    entry = _require_object(value=value, context=context, field="entry")
    path = _require_string(
        value=_required_field(record=entry, field="path", context=context),
        context=context,
        field="path",
        non_empty=False,
    )
    shape = _validate_shape(value=_required_field(record=entry, field="shape", context=context), context=context)
    dtype = _require_string(
        value=_required_field(record=entry, field="dtype", context=context),
        context=context,
        field="dtype",
        non_empty=True,
    )
    return _ImageManifestEntry(path=path, shape=shape, dtype=dtype)


def _validate_audio_entry(*, value: Any, case_id: str, entry_index: int) -> _AudioManifestEntry:
    context = f"Case {case_id!r} audio entry {entry_index}"
    entry = _require_object(value=value, context=context, field="entry")
    path = _require_string(
        value=_required_field(record=entry, field="path", context=context),
        context=context,
        field="path",
        non_empty=False,
    )
    shape = _validate_shape(value=_required_field(record=entry, field="shape", context=context), context=context)
    dtype = _require_string(
        value=_required_field(record=entry, field="dtype", context=context),
        context=context,
        field="dtype",
        non_empty=True,
    )
    sampling_rate = _require_exact_integer(
        value=_required_field(record=entry, field="sampling_rate", context=context),
        context=context,
        field="sampling_rate",
    )
    audio_format = _require_string(
        value=_required_field(record=entry, field="format", context=context),
        context=context,
        field="format",
        non_empty=False,
    )
    return _AudioManifestEntry(
        path=path,
        shape=shape,
        dtype=dtype,
        sampling_rate=sampling_rate,
        format=audio_format,
    )


def _validate_manifest_schema(*, value: Any, case_id: str) -> _ValidatedManifest:
    context = f"Case {case_id!r} manifest"
    manifest = _require_object(value=value, context=context, field="manifest")
    required_fields = (
        "case_id",
        "tokenizer_configuration_id",
        "token_ids",
        "decoded_text",
        "images",
        "audios",
    )
    for field in required_fields:
        _required_field(record=manifest, field=field, context=context)

    manifest_case_id = _require_string(value=manifest["case_id"], context=context, field="case_id", non_empty=False)
    configuration_id = _require_string(
        value=manifest["tokenizer_configuration_id"],
        context=context,
        field="tokenizer_configuration_id",
        non_empty=False,
    )
    token_ids = [
        _require_exact_integer(value=token_id, context=context, field=f"token_ids[{index}]")
        for index, token_id in enumerate(
            _require_array(value=manifest["token_ids"], context=context, field="token_ids")
        )
    ]
    decoded_text = _require_string(
        value=manifest["decoded_text"], context=context, field="decoded_text", non_empty=False
    )
    image_values = _require_array(value=manifest["images"], context=context, field="images")
    audio_values = _require_array(value=manifest["audios"], context=context, field="audios")
    images = [
        _validate_image_entry(value=entry, case_id=case_id, entry_index=index)
        for index, entry in enumerate(image_values)
    ]
    audios = [
        _validate_audio_entry(value=entry, case_id=case_id, entry_index=index)
        for index, entry in enumerate(audio_values)
    ]
    return _ValidatedManifest(
        case_id=manifest_case_id,
        tokenizer_configuration_id=configuration_id,
        token_ids=token_ids,
        decoded_text=decoded_text,
        images=images,
        audios=audios,
    )


def _parse_image(entry: _ImageManifestEntry, *, case_dir: Path, case_id: str, entry_index: int) -> ExpectedImage:
    return ExpectedImage(
        path=entry.path,
        shape=entry.shape,
        dtype=entry.dtype,
        array=_load_manifest_sidecar(
            case_dir=case_dir,
            case_id=case_id,
            media_kind="image",
            entry_index=entry_index,
            path=entry.path,
            shape=entry.shape,
            dtype=entry.dtype,
        ),
    )


def _parse_audio(entry: _AudioManifestEntry, *, case_dir: Path, case_id: str, entry_index: int) -> ExpectedAudio:
    return ExpectedAudio(
        path=entry.path,
        shape=entry.shape,
        dtype=entry.dtype,
        sampling_rate=entry.sampling_rate,
        format=entry.format,
        array=_load_manifest_sidecar(
            case_dir=case_dir,
            case_id=case_id,
            media_kind="audio",
            entry_index=entry_index,
            path=entry.path,
            shape=entry.shape,
            dtype=entry.dtype,
        ),
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
    manifest = _validate_manifest_schema(value=json.loads(manifest_path.read_text()), case_id=case_id)

    if manifest.case_id != case_id:
        raise ValueError(
            f"Manifest case_id {manifest.case_id!r} ({type(manifest.case_id).__name__}) "
            f"does not match the requested case {case_id!r}"
        )
    if manifest.tokenizer_configuration_id != tokenizer_configuration_id:
        raise ValueError(
            f"Manifest tokenizer_configuration_id {manifest.tokenizer_configuration_id!r} "
            f"({type(manifest.tokenizer_configuration_id).__name__}) "
            f"does not match the requested configuration {tokenizer_configuration_id!r}"
        )

    images = tuple(
        _parse_image(entry, case_dir=case_dir, case_id=case_id, entry_index=index)
        for index, entry in enumerate(manifest.images)
    )
    audios = tuple(
        _parse_audio(entry, case_dir=case_dir, case_id=case_id, entry_index=index)
        for index, entry in enumerate(manifest.audios)
    )
    return ExpectedSuccess(
        case_id=manifest.case_id,
        tokenizer_configuration_id=manifest.tokenizer_configuration_id,
        token_ids=manifest.token_ids,
        decoded_text=manifest.decoded_text,
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
            desired=expected_image.array,
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
            desired=expected_audio.array,
            atol=ATOL,
            rtol=RTOL,
            err_msg="Audio array differs from the reviewed sidecar",
        )
