r"""Integrity of the reviewed expected-result manifests.

These cases protect the golden comparison itself: a wrong association, an
escaping or missing sidecar, a pickle-based array, or a changed token/text
value must fail loudly instead of silently comparing an unrelated golden.
"""

import json
from pathlib import Path

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
