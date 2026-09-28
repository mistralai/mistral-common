r"""Integrity of the reviewed expected-result manifests.

These cases protect the golden comparison itself: a wrong association, an
escaping or missing sidecar, a pickle-based array, or a changed token/text
value must fail loudly instead of silently comparing an unrelated golden.
"""

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from mistral_common.tokens.tokenizers.audio import Audio
from mistral_common.tokens.tokenizers.base import Tokenized
from tests.integration.expected_results import (
    ExpectedSuccess,
    assert_public_success,
    load_expected_success,
    load_sidecar,
    resolve_sidecar,
)


@pytest.fixture()
def manifest_dir(tmp_path: Path) -> Path:
    """One writable case directory with a valid text-only manifest."""
    case_dir = tmp_path / "integrity-case"
    case_dir.mkdir()
    (case_dir / "expected.json").write_text(
        json.dumps(
            {
                "case_id": "integrity-case",
                "tokenizer_configuration_id": "integrity-configuration",
                "token_ids": [1, 2, 3],
                "decoded_text": "<s>hello</s>",
                "images": [],
                "audios": [],
            }
        )
    )
    return case_dir


def _load(manifest_dir: Path) -> ExpectedSuccess:
    return load_expected_success(
        case_id="integrity-case",
        tokenizer_configuration_id="integrity-configuration",
        expected_root=manifest_dir.parent,
    )


def _assert_full_path(manifest_dir: Path, tokenized: Tokenized) -> None:
    expected = _load(manifest_dir)
    assert_public_success(expected=expected, tokenized=tokenized, decoded_text="<s>hello</s>")


def _write_image_entry(manifest_dir: Path, *, path: str, shape: list[int], dtype: str) -> None:
    manifest_path = manifest_dir / "expected.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["images"] = [{"path": path, "shape": shape, "dtype": dtype}]
    manifest_path.write_text(json.dumps(manifest))


def _write_manifest(manifest_dir: Path, manifest: dict[str, object]) -> None:
    (manifest_dir / "expected.json").write_text(json.dumps(manifest))


def _read_manifest(manifest_dir: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads((manifest_dir / "expected.json").read_text()))


def test_missing_manifest_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Missing expected manifest"):
        load_expected_success(
            case_id="absent-case",
            tokenizer_configuration_id="integrity-configuration",
            expected_root=tmp_path,
        )


def test_manifest_case_id_mismatch_is_rejected(manifest_dir: Path) -> None:
    copied_dir = manifest_dir.parent / "other-case"
    copied_dir.mkdir()
    (copied_dir / "expected.json").write_text((manifest_dir / "expected.json").read_text())
    with pytest.raises(ValueError, match="does not match the requested case"):
        load_expected_success(
            case_id="other-case",
            tokenizer_configuration_id="integrity-configuration",
            expected_root=manifest_dir.parent,
        )


def test_manifest_configuration_mismatch_is_rejected(manifest_dir: Path) -> None:
    with pytest.raises(ValueError, match="does not match the requested configuration"):
        load_expected_success(
            case_id="integrity-case",
            tokenizer_configuration_id="other-configuration",
            expected_root=manifest_dir.parent,
        )


@pytest.mark.parametrize("missing_field", ["images", "audios"])
def test_manifest_requires_ordered_media_fields(manifest_dir: Path, missing_field: str) -> None:
    manifest_path = manifest_dir / "expected.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest[missing_field]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=f"missing required.*{missing_field}"):
        _assert_full_path(manifest_dir, tokenized=Tokenized(tokens=[1, 2, 3]))


def test_escaping_sidecar_path_is_rejected(manifest_dir: Path) -> None:
    with pytest.raises(ValueError, match="escapes the case directory"):
        resolve_sidecar(case_dir=manifest_dir, relative_path="../outside.npy")


def test_manifest_escaping_sidecar_is_rejected(manifest_dir: Path) -> None:
    (manifest_dir.parent / "outside.npy").write_bytes(b"outside the case")
    _write_image_entry(manifest_dir, path="../outside.npy", shape=[1, 2], dtype="float32")
    tokenized = Tokenized(tokens=[1, 2, 3], images=[np.zeros(shape=(1, 2), dtype=np.float32)])

    with pytest.raises(ValueError, match="Case 'integrity-case' image entry 0:.*escapes the case directory"):
        _assert_full_path(manifest_dir, tokenized=tokenized)


def test_manifest_missing_sidecar_is_rejected(manifest_dir: Path) -> None:
    _write_image_entry(manifest_dir, path="images/missing.npy", shape=[1, 2], dtype="float32")
    tokenized = Tokenized(tokens=[1, 2, 3], images=[np.zeros(shape=(1, 2), dtype=np.float32)])

    with pytest.raises(ValueError, match="Case 'integrity-case' image entry 0 references missing sidecar"):
        _assert_full_path(manifest_dir, tokenized=tokenized)


def test_absolute_sidecar_path_is_rejected(manifest_dir: Path) -> None:
    with pytest.raises(ValueError, match="must be relative"):
        resolve_sidecar(case_dir=manifest_dir, relative_path=str(manifest_dir / "absolute.npy"))


def test_non_npy_sidecar_path_is_rejected(manifest_dir: Path) -> None:
    with pytest.raises(ValueError, match="must reference a .npy file"):
        resolve_sidecar(case_dir=manifest_dir, relative_path="payload.txt")


def test_pickle_sidecar_is_rejected(manifest_dir: Path) -> None:
    pickled_path = manifest_dir / "pickled.npy"
    np.save(file=pickled_path, arr=np.array([{"surprise": "pickle"}], dtype=object))
    with pytest.raises(ValueError, match="pick"):
        load_sidecar(sidecar_path=pickled_path)


def test_missing_sidecar_file_is_rejected(manifest_dir: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_sidecar(sidecar_path=manifest_dir / "absent.npy")


def test_changed_token_ids_fail_the_comparison(manifest_dir: Path) -> None:
    expected = _load(manifest_dir)
    tokenized = Tokenized(tokens=[1, 2, 4])
    with pytest.raises(AssertionError, match="Token ids differ"):
        assert_public_success(expected=expected, tokenized=tokenized, decoded_text="<s>hello</s>")


def test_non_integer_token_id_is_rejected_before_public_comparison(manifest_dir: Path) -> None:
    manifest_path = manifest_dir / "expected.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["token_ids"] = [1.9, 2, 3]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=r"Case 'integrity-case'.*token_ids\[0\].*1\.9"):
        _assert_full_path(manifest_dir, tokenized=Tokenized(tokens=[1, 2, 3]))


@pytest.mark.parametrize(
    ("token_ids", "message"),
    [
        ({}, "field 'token_ids' must be an array"),
        ("123", "field 'token_ids' must be an array"),
        ([True, 2, 3], r"token_ids\[0\].*must be an integer"),
    ],
    ids=["object", "string", "boolean-item"],
)
def test_token_ids_must_be_an_array_of_exact_integers(manifest_dir: Path, token_ids: object, message: str) -> None:
    manifest = _read_manifest(manifest_dir)
    manifest["token_ids"] = token_ids
    _write_manifest(manifest_dir, manifest)

    with pytest.raises(ValueError, match=message):
        _assert_full_path(manifest_dir, tokenized=Tokenized(tokens=[]))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("case_id", 42, "field 'case_id' must be a string"),
        (
            "tokenizer_configuration_id",
            None,
            "field 'tokenizer_configuration_id' must be a string",
        ),
        ("decoded_text", 17, "field 'decoded_text' must be a string"),
    ],
    ids=["case-id", "configuration-id", "decoded-text"],
)
def test_manifest_string_fields_reject_other_json_types(
    manifest_dir: Path, field: str, value: object, message: str
) -> None:
    manifest = _read_manifest(manifest_dir)
    manifest[field] = value
    _write_manifest(manifest_dir, manifest)

    with pytest.raises(ValueError, match=message):
        _assert_full_path(manifest_dir, tokenized=Tokenized(tokens=[1, 2, 3]))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("images", {}, "field 'images' must be an array"),
        ("images", "none", "field 'images' must be an array"),
        ("audios", {}, "field 'audios' must be an array"),
        ("audios", "none", "field 'audios' must be an array"),
    ],
    ids=["images-object", "images-string", "audios-object", "audios-string"],
)
def test_manifest_media_fields_must_be_arrays(manifest_dir: Path, field: str, value: object, message: str) -> None:
    manifest = _read_manifest(manifest_dir)
    manifest[field] = value
    _write_manifest(manifest_dir, manifest)

    with pytest.raises(ValueError, match=message):
        _assert_full_path(manifest_dir, tokenized=Tokenized(tokens=[1, 2, 3]))


@pytest.mark.parametrize("field", ["images", "audios"], ids=["image", "audio"])
def test_media_entries_must_be_objects(manifest_dir: Path, field: str) -> None:
    manifest = _read_manifest(manifest_dir)
    manifest[field] = ["not-an-object"]
    _write_manifest(manifest_dir, manifest)

    media_kind = "image" if field == "images" else "audio"
    with pytest.raises(ValueError, match=rf"{media_kind} entry 0.*must be an object"):
        _assert_full_path(manifest_dir, tokenized=Tokenized(tokens=[1, 2, 3]))


@pytest.mark.parametrize(
    ("field", "entry", "message"),
    [
        (
            "images",
            {"path": 5, "shape": [1], "dtype": "float32"},
            r"image entry 0.*field 'path'.*must be a string",
        ),
        (
            "audios",
            {"path": None, "shape": [1], "dtype": "float32", "sampling_rate": 16000, "format": "wav"},
            r"audio entry 0.*field 'path'.*must be a string",
        ),
        (
            "images",
            {"path": "images/0.npy", "shape": 1, "dtype": "float32"},
            r"image entry 0.*field 'shape'.*must be an array",
        ),
        (
            "audios",
            {"path": "audios/0.npy", "shape": "4", "dtype": "float32", "sampling_rate": 16000, "format": "wav"},
            r"audio entry 0.*field 'shape'.*must be an array",
        ),
        (
            "images",
            {"path": "images/0.npy", "shape": [True], "dtype": "float32"},
            r"image entry 0.*shape\[0\].*must be an integer",
        ),
        (
            "audios",
            {
                "path": "audios/0.npy",
                "shape": [4],
                "dtype": "float32",
                "sampling_rate": True,
                "format": "wav",
            },
            r"audio entry 0.*sampling_rate.*must be an integer",
        ),
    ],
    ids=[
        "image-path",
        "audio-path",
        "image-shape",
        "audio-shape",
        "boolean-shape",
        "boolean-sampling-rate",
    ],
)
def test_media_entry_fields_have_the_declared_types(
    manifest_dir: Path, field: str, entry: dict[str, object], message: str
) -> None:
    manifest = _read_manifest(manifest_dir)
    manifest[field] = [entry]
    _write_manifest(manifest_dir, manifest)

    with pytest.raises(ValueError, match=message):
        _assert_full_path(manifest_dir, tokenized=Tokenized(tokens=[1, 2, 3]))


@pytest.mark.parametrize(
    ("media_kind", "declared_dtype"),
    [
        ("image", None),
        ("image", 7),
        ("image", ""),
        ("audio", None),
        ("audio", 7),
        ("audio", ""),
    ],
    ids=["image-null", "image-number", "image-empty", "audio-null", "audio-number", "audio-empty"],
)
def test_media_dtype_must_be_a_non_empty_string(manifest_dir: Path, media_kind: str, declared_dtype: object) -> None:
    array = np.zeros(shape=(4,), dtype=np.float64)
    if media_kind == "image":
        media_dir = manifest_dir / "images"
        field = "images"
        entry = {"path": "images/0.npy", "shape": [4], "dtype": declared_dtype}
    else:
        media_dir = manifest_dir / "audios"
        field = "audios"
        entry = {
            "path": "audios/0.npy",
            "shape": [4],
            "dtype": declared_dtype,
            "sampling_rate": 16000,
            "format": "wav",
        }
    media_dir.mkdir()
    np.save(file=media_dir / "0.npy", arr=array)
    manifest = _read_manifest(manifest_dir)
    manifest[field] = [entry]
    _write_manifest(manifest_dir, manifest)
    if media_kind == "image":
        tokenized = Tokenized(tokens=[1, 2, 3], images=[array])
    else:
        tokenized = Tokenized(tokens=[1, 2, 3], audios=[Audio(audio_array=array, sampling_rate=16000, format="wav")])

    with pytest.raises(ValueError, match=rf"{media_kind} entry 0.*field 'dtype'.*must be a non-empty string"):
        _assert_full_path(manifest_dir, tokenized=tokenized)


@pytest.mark.parametrize(
    ("field", "missing_field"),
    [
        ("images", "path"),
        ("images", "shape"),
        ("images", "dtype"),
        ("audios", "path"),
        ("audios", "shape"),
        ("audios", "dtype"),
        ("audios", "sampling_rate"),
        ("audios", "format"),
    ],
)
def test_media_entries_require_all_schema_fields(manifest_dir: Path, field: str, missing_field: str) -> None:
    entry: dict[str, object] = {"path": f"{field}/0.npy", "shape": [1], "dtype": "float32"}
    if field == "audios":
        entry.update({"sampling_rate": 16000, "format": "wav"})
    del entry[missing_field]
    manifest = _read_manifest(manifest_dir)
    manifest[field] = [entry]
    _write_manifest(manifest_dir, manifest)

    media_kind = "image" if field == "images" else "audio"
    with pytest.raises(ValueError, match=rf"{media_kind} entry 0.*missing required field '{missing_field}'"):
        _assert_full_path(manifest_dir, tokenized=Tokenized(tokens=[1, 2, 3]))


def test_audio_format_must_be_a_string(manifest_dir: Path) -> None:
    audio_array = np.zeros(shape=(4,), dtype=np.float32)
    (manifest_dir / "audios").mkdir()
    np.save(file=manifest_dir / "audios" / "0.npy", arr=audio_array)
    manifest = _read_manifest(manifest_dir)
    manifest["audios"] = [
        {
            "path": "audios/0.npy",
            "shape": [4],
            "dtype": "float32",
            "sampling_rate": 16000,
            "format": 7,
        }
    ]
    _write_manifest(manifest_dir, manifest)
    tokenized = Tokenized(tokens=[1, 2, 3], audios=[Audio(audio_array=audio_array, sampling_rate=16000, format="wav")])

    with pytest.raises(ValueError, match=r"audio entry 0.*field 'format'.*must be a string"):
        _assert_full_path(manifest_dir, tokenized=tokenized)


@pytest.mark.parametrize(
    "field",
    ["case_id", "tokenizer_configuration_id", "token_ids", "decoded_text", "images", "audios"],
)
def test_manifest_requires_all_top_level_fields(manifest_dir: Path, field: str) -> None:
    manifest = _read_manifest(manifest_dir)
    del manifest[field]
    _write_manifest(manifest_dir, manifest)

    with pytest.raises(ValueError, match=rf"Case 'integrity-case'.*missing required field '{field}'"):
        _assert_full_path(manifest_dir, tokenized=Tokenized(tokens=[1, 2, 3]))


def test_manifest_root_must_be_an_object(manifest_dir: Path) -> None:
    (manifest_dir / "expected.json").write_text(json.dumps(["not", "an", "object"]))

    with pytest.raises(ValueError, match=r"Case 'integrity-case'.*manifest.*must be an object"):
        _assert_full_path(manifest_dir, tokenized=Tokenized(tokens=[1, 2, 3]))


def test_non_integer_image_shape_dimension_is_rejected_before_comparison(manifest_dir: Path) -> None:
    (manifest_dir / "images").mkdir()
    np.save(file=manifest_dir / "images" / "0.npy", arr=np.zeros(shape=(1, 2), dtype=np.float32))
    _write_image_entry(manifest_dir, path="images/0.npy", shape=[1, 2], dtype="float32")
    manifest_path = manifest_dir / "expected.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["images"][0]["shape"] = [1.9, 2]
    manifest_path.write_text(json.dumps(manifest))
    tokenized = Tokenized(tokens=[1, 2, 3], images=[np.zeros(shape=(1, 2), dtype=np.float32)])

    with pytest.raises(ValueError, match=r"Case 'integrity-case' image entry 0.*shape\[0\].*1\.9"):
        _assert_full_path(manifest_dir, tokenized=tokenized)


def test_non_integer_audio_shape_dimension_is_rejected_before_comparison(manifest_dir: Path) -> None:
    (manifest_dir / "audios").mkdir()
    np.save(file=manifest_dir / "audios" / "0.npy", arr=np.zeros(shape=(4,), dtype=np.float32))
    manifest_path = manifest_dir / "expected.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["audios"] = [
        {"path": "audios/0.npy", "shape": [4.9], "dtype": "float32", "sampling_rate": 16000, "format": "wav"}
    ]
    manifest_path.write_text(json.dumps(manifest))
    tokenized = Tokenized(
        tokens=[1, 2, 3],
        audios=[Audio(audio_array=np.zeros(shape=(4,), dtype=np.float32), sampling_rate=16000, format="wav")],
    )

    with pytest.raises(ValueError, match=r"Case 'integrity-case' audio entry 0.*shape\[0\].*4\.9"):
        _assert_full_path(manifest_dir, tokenized=tokenized)


def test_non_integer_audio_sampling_rate_is_rejected_before_comparison(manifest_dir: Path) -> None:
    (manifest_dir / "audios").mkdir()
    np.save(file=manifest_dir / "audios" / "0.npy", arr=np.zeros(shape=(4,), dtype=np.float32))
    manifest_path = manifest_dir / "expected.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["audios"] = [
        {"path": "audios/0.npy", "shape": [4], "dtype": "float32", "sampling_rate": 16000.9, "format": "wav"}
    ]
    manifest_path.write_text(json.dumps(manifest))
    tokenized = Tokenized(
        tokens=[1, 2, 3],
        audios=[Audio(audio_array=np.zeros(shape=(4,), dtype=np.float32), sampling_rate=16000, format="wav")],
    )

    with pytest.raises(ValueError, match=r"Case 'integrity-case' audio entry 0.*sampling_rate.*16000\.9"):
        _assert_full_path(manifest_dir, tokenized=tokenized)


def test_changed_decoded_text_fails_the_comparison(manifest_dir: Path) -> None:
    expected = _load(manifest_dir)
    tokenized = Tokenized(tokens=[1, 2, 3])
    with pytest.raises(AssertionError, match="Decoded text differs"):
        assert_public_success(expected=expected, tokenized=tokenized, decoded_text="<s>hello there</s>")


def test_changed_media_values_fail_under_the_shared_tolerance(tmp_path: Path) -> None:
    case_dir = tmp_path / "integrity-media-case"
    case_dir.mkdir()
    (case_dir / "images").mkdir()
    np.save(file=case_dir / "images" / "0.npy", arr=np.zeros(shape=(1, 2), dtype=np.float32))
    (case_dir / "audios").mkdir()
    np.save(file=case_dir / "audios" / "0.npy", arr=np.zeros(shape=(4,), dtype=np.float32))
    (case_dir / "expected.json").write_text(
        json.dumps(
            {
                "case_id": "integrity-media-case",
                "tokenizer_configuration_id": "integrity-configuration",
                "token_ids": [1],
                "decoded_text": "<s></s>",
                "images": [{"path": "images/0.npy", "shape": [1, 2], "dtype": "float32"}],
                "audios": [
                    {
                        "path": "audios/0.npy",
                        "shape": [4],
                        "dtype": "float32",
                        "sampling_rate": 16000,
                        "format": "wav",
                    }
                ],
            }
        )
    )
    expected = load_expected_success(
        case_id="integrity-media-case",
        tokenizer_configuration_id="integrity-configuration",
        expected_root=tmp_path,
    )

    changed_image = np.full(shape=(1, 2), fill_value=1.0, dtype=np.float32)
    changed_audio = Audio(
        audio_array=np.full(shape=(4,), fill_value=1.0, dtype=np.float32), sampling_rate=16000, format="wav"
    )
    tokenized = Tokenized(tokens=[1], images=[changed_image], audios=[changed_audio])
    with pytest.raises(AssertionError, match="Image array differs"):
        assert_public_success(expected=expected, tokenized=tokenized, decoded_text="<s></s>")

    tokenized = Tokenized(
        tokens=[1],
        images=[np.zeros(shape=(1, 2), dtype=np.float32)],
        audios=[changed_audio],
    )
    with pytest.raises(AssertionError, match="Audio array differs"):
        assert_public_success(expected=expected, tokenized=tokenized, decoded_text="<s></s>")


def test_media_metadata_mismatches_fail_before_values(tmp_path: Path) -> None:
    case_dir = tmp_path / "integrity-metadata-case"
    case_dir.mkdir()
    (case_dir / "images").mkdir()
    np.save(file=case_dir / "images" / "0.npy", arr=np.zeros(shape=(2, 1), dtype=np.float32))
    (case_dir / "expected.json").write_text(
        json.dumps(
            {
                "case_id": "integrity-metadata-case",
                "tokenizer_configuration_id": "integrity-configuration",
                "token_ids": [1],
                "decoded_text": "<s></s>",
                "images": [{"path": "images/0.npy", "shape": [2, 1], "dtype": "float32"}],
                "audios": [],
            }
        )
    )
    expected = load_expected_success(
        case_id="integrity-metadata-case",
        tokenizer_configuration_id="integrity-configuration",
        expected_root=tmp_path,
    )
    tokenized = Tokenized(tokens=[1], images=[np.zeros(shape=(1, 2), dtype=np.float32)])
    with pytest.raises(AssertionError, match="Image shape"):
        assert_public_success(expected=expected, tokenized=tokenized, decoded_text="<s></s>")


def test_sidecar_dtype_mismatch_is_rejected_before_values(manifest_dir: Path) -> None:
    (manifest_dir / "images").mkdir()
    np.save(file=manifest_dir / "images" / "0.npy", arr=np.zeros(shape=(1, 2), dtype=np.float64))
    _write_image_entry(manifest_dir, path="images/0.npy", shape=[1, 2], dtype="float32")
    tokenized = Tokenized(tokens=[1, 2, 3], images=[np.zeros(shape=(1, 2), dtype=np.float32)])

    with pytest.raises(ValueError, match="sidecar dtype.*does not match manifest declared dtype float32"):
        _assert_full_path(manifest_dir, tokenized=tokenized)


def test_sidecar_shape_mismatch_is_rejected_before_values(manifest_dir: Path) -> None:
    (manifest_dir / "images").mkdir()
    np.save(file=manifest_dir / "images" / "0.npy", arr=np.zeros(shape=(2, 1), dtype=np.float32))
    _write_image_entry(manifest_dir, path="images/0.npy", shape=[1, 2], dtype="float32")
    tokenized = Tokenized(tokens=[1, 2, 3], images=[np.zeros(shape=(1, 2), dtype=np.float32)])

    with pytest.raises(ValueError, match=r"sidecar shape.*does not match manifest declared shape \(1, 2\)"):
        _assert_full_path(manifest_dir, tokenized=tokenized)
