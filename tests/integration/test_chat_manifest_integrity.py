r"""Integrity of the reviewed expected-result manifests.

These cases protect the golden comparison itself: a wrong association, an
escaping or missing sidecar, a pickle-based array, or a changed token/text
value must fail loudly instead of silently comparing an unrelated golden.
"""

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from shutil import copytree
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

import tests.integration.expected_results as expected_results
from mistral_common.exceptions import InvalidMessageStructureException, TokenizerException
from mistral_common.protocol.instruct.chunk import AudioURLType
from mistral_common.protocol.instruct.messages import ChatMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.audio import Audio
from mistral_common.tokens.tokenizers.base import Tokenized
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.fixtures.audio import get_dummy_audio_url_chunk
from tests.integration.chat_cases import (
    SAMPLE_V1_TOOLS_REJECTED_CASE,
    PublicChatErrorCase,
    PublicChatSuccessCase,
)
from tests.integration.chat_recipes import NO_TOOLS, build_image_url_chunk, build_user_media_request
from tests.integration.chat_released_cases import RELEASED_USER_AUDIO_URL_CASE
from tests.integration.chat_v15_cases import V15_USER_IMAGE_URL_CASE
from tests.integration.expected_results import (
    _EXPECTED_ROOT,
    ATOL,
    RTOL,
    ExpectedAudio,
    ExpectedImage,
    ExpectedSuccess,
    assert_public_success,
    load_expected_success,
    load_sidecar,
    resolve_sidecar,
)
from tests.integration.tokenizer_configurations import TokenizerConfiguration
from tests.integration.utils import encode_and_expect_error, encode_and_verify


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
    manifest = _read_manifest(manifest_dir)
    manifest["images"] = [{"path": path, "shape": shape, "dtype": dtype}]
    _write_manifest(manifest_dir=manifest_dir, manifest=manifest)


def _write_manifest(manifest_dir: Path, manifest: object) -> None:
    (manifest_dir / "expected.json").write_text(json.dumps(manifest))


def _read_manifest(manifest_dir: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads((manifest_dir / "expected.json").read_text())
    return manifest


def test_missing_manifest_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(expected_exception=ValueError, match="Missing expected manifest"):
        load_expected_success(
            case_id="absent-case",
            tokenizer_configuration_id="integrity-configuration",
            expected_root=tmp_path,
        )


def test_manifest_case_id_mismatch_is_rejected(manifest_dir: Path) -> None:
    copied_dir = manifest_dir.parent / "other-case"
    copied_dir.mkdir()
    (copied_dir / "expected.json").write_text((manifest_dir / "expected.json").read_text())
    with pytest.raises(expected_exception=ValueError, match="does not match the requested case"):
        load_expected_success(
            case_id="other-case",
            tokenizer_configuration_id="integrity-configuration",
            expected_root=manifest_dir.parent,
        )


def test_manifest_configuration_mismatch_is_rejected(manifest_dir: Path) -> None:
    with pytest.raises(expected_exception=ValueError, match="does not match the requested configuration"):
        load_expected_success(
            case_id="integrity-case",
            tokenizer_configuration_id="other-configuration",
            expected_root=manifest_dir.parent,
        )


def test_manifest_escaping_sidecar_is_rejected(manifest_dir: Path) -> None:
    (manifest_dir.parent / "outside.npy").write_bytes(b"outside the case")
    _write_image_entry(manifest_dir=manifest_dir, path="../outside.npy", shape=[1, 2], dtype="float32")
    tokenized = Tokenized(tokens=[1, 2, 3], images=[np.zeros(shape=(1, 2), dtype=np.float32)])

    with pytest.raises(
        expected_exception=ValueError, match="Case 'integrity-case' image entry 0:.*escapes the case directory"
    ):
        _assert_full_path(manifest_dir=manifest_dir, tokenized=tokenized)


def test_manifest_missing_sidecar_is_rejected(manifest_dir: Path) -> None:
    _write_image_entry(manifest_dir=manifest_dir, path="images/missing.npy", shape=[1, 2], dtype="float32")
    tokenized = Tokenized(tokens=[1, 2, 3], images=[np.zeros(shape=(1, 2), dtype=np.float32)])

    with pytest.raises(
        expected_exception=ValueError, match="Case 'integrity-case' image entry 0 references missing sidecar"
    ):
        _assert_full_path(manifest_dir=manifest_dir, tokenized=tokenized)


def test_absolute_sidecar_path_is_rejected(manifest_dir: Path) -> None:
    with pytest.raises(expected_exception=ValueError, match="must be relative"):
        resolve_sidecar(case_dir=manifest_dir, relative_path=str(manifest_dir / "absolute.npy"))


def test_non_npy_sidecar_path_is_rejected(manifest_dir: Path) -> None:
    with pytest.raises(expected_exception=ValueError, match="must reference a .npy file"):
        resolve_sidecar(case_dir=manifest_dir, relative_path="payload.txt")


def test_pickle_sidecar_is_rejected(manifest_dir: Path) -> None:
    pickled_path = manifest_dir / "pickled.npy"
    np.save(file=pickled_path, arr=np.array(object=[{"surprise": "pickle"}], dtype=object))
    with pytest.raises(expected_exception=ValueError, match="pick"):
        load_sidecar(sidecar_path=pickled_path)


def test_npz_archive_renamed_as_npy_is_rejected(manifest_dir: Path) -> None:
    images_dir = manifest_dir / "images"
    images_dir.mkdir()
    archive_path = images_dir / "0.npz"
    np.savez(file=archive_path, p=np.array(object=["p"], dtype="<U1"))
    archive_path.rename(images_dir / "0.npy")
    _write_image_entry(manifest_dir=manifest_dir, path="images/0.npy", shape=[1], dtype="<U1")
    tokenized = Tokenized(tokens=[1, 2, 3], images=[np.array(object=["p"], dtype="<U1")])

    with pytest.raises(
        expected_exception=ValueError,
        match=r"Case 'integrity-case' image entry 0 cannot load sidecar 'images/0.npy'.*array sidecar",
    ):
        _assert_full_path(manifest_dir=manifest_dir, tokenized=tokenized)


@pytest.mark.parametrize(
    argnames=("payload", "message"),
    argvalues=[
        pytest.param({"token_ids": {}}, "field 'token_ids' must be an array", id="token-ids-object"),
        pytest.param({"token_ids": "123"}, "field 'token_ids' must be an array", id="token-ids-string"),
        pytest.param(
            {"token_ids": [True, 2, 3]},
            r"token_ids\[0\].*must be an integer",
            id="token-ids-boolean-item",
        ),
        pytest.param(
            {"token_ids": [1.9, 2, 3]},
            r"Case 'integrity-case'.*token_ids\[0\].*1\.9",
            id="token-ids-float-item",
        ),
        pytest.param({"case_id": 42}, "field 'case_id' must be a string", id="field-case-id"),
        pytest.param(
            {"tokenizer_configuration_id": None},
            "field 'tokenizer_configuration_id' must be a string",
            id="field-configuration-id",
        ),
        pytest.param({"decoded_text": 17}, "field 'decoded_text' must be a string", id="field-decoded-text"),
        pytest.param({"images": {}}, "field 'images' must be an array", id="field-images-object"),
        pytest.param({"images": "none"}, "field 'images' must be an array", id="field-images-string"),
        pytest.param({"audios": {}}, "field 'audios' must be an array", id="field-audios-object"),
        pytest.param({"audios": "none"}, "field 'audios' must be an array", id="field-audios-string"),
        pytest.param({"images": ["not-an-object"]}, r"image entry 0.*must be an object", id="entry-object-image"),
        pytest.param({"audios": ["not-an-object"]}, r"audio entry 0.*must be an object", id="entry-object-audio"),
        pytest.param(
            {"images": [{"shape": [1], "dtype": "float32"}]},
            r"image entry 0.*missing required field 'path'",
            id="entry-missing-image-path",
        ),
        pytest.param(
            {"images": [{"path": "images/0.npy", "dtype": "float32"}]},
            r"image entry 0.*missing required field 'shape'",
            id="entry-missing-image-shape",
        ),
        pytest.param(
            {"images": [{"path": "images/0.npy", "shape": [1]}]},
            r"image entry 0.*missing required field 'dtype'",
            id="entry-missing-image-dtype",
        ),
        pytest.param(
            {"audios": [{"shape": [1], "dtype": "float32", "sampling_rate": 16000, "format": "wav"}]},
            r"audio entry 0.*missing required field 'path'",
            id="entry-missing-audio-path",
        ),
        pytest.param(
            {"audios": [{"path": "audios/0.npy", "dtype": "float32", "sampling_rate": 16000, "format": "wav"}]},
            r"audio entry 0.*missing required field 'shape'",
            id="entry-missing-audio-shape",
        ),
        pytest.param(
            {"audios": [{"path": "audios/0.npy", "shape": [1], "sampling_rate": 16000, "format": "wav"}]},
            r"audio entry 0.*missing required field 'dtype'",
            id="entry-missing-audio-dtype",
        ),
        pytest.param(
            {"audios": [{"path": "audios/0.npy", "shape": [1], "dtype": "float32", "format": "wav"}]},
            r"audio entry 0.*missing required field 'sampling_rate'",
            id="entry-missing-audio-sampling-rate",
        ),
        pytest.param(
            {"audios": [{"path": "audios/0.npy", "shape": [1], "dtype": "float32", "sampling_rate": 16000}]},
            r"audio entry 0.*missing required field 'format'",
            id="entry-missing-audio-format",
        ),
        pytest.param(
            {"images": [{"path": 5, "shape": [1], "dtype": "float32"}]},
            r"image entry 0.*field 'path'.*must be a string",
            id="entry-type-image-path",
        ),
        pytest.param(
            {"audios": [{"path": None, "shape": [1], "dtype": "float32", "sampling_rate": 16000, "format": "wav"}]},
            r"audio entry 0.*field 'path'.*must be a string",
            id="entry-type-audio-path",
        ),
        pytest.param(
            {"images": [{"path": "images/0.npy", "shape": 1, "dtype": "float32"}]},
            r"image entry 0.*field 'shape'.*must be an array",
            id="entry-type-image-shape",
        ),
        pytest.param(
            {
                "audios": [
                    {"path": "audios/0.npy", "shape": "4", "dtype": "float32", "sampling_rate": 16000, "format": "wav"}
                ]
            },
            r"audio entry 0.*field 'shape'.*must be an array",
            id="entry-type-audio-shape",
        ),
        pytest.param(
            {"images": [{"path": "images/0.npy", "shape": [True], "dtype": "float32"}]},
            r"image entry 0.*shape\[0\].*must be an integer",
            id="entry-type-boolean-image-shape",
        ),
        pytest.param(
            {
                "audios": [
                    {"path": "audios/0.npy", "shape": [4], "dtype": "float32", "sampling_rate": True, "format": "wav"}
                ]
            },
            r"audio entry 0.*sampling_rate.*must be an integer",
            id="entry-type-boolean-audio-sampling-rate",
        ),
        pytest.param(
            {"images": [{"path": "images/0.npy", "shape": [1.9, 2], "dtype": "float32"}]},
            r"image entry 0.*shape\[0\].*1\.9",
            id="entry-type-non-integer-image-shape",
        ),
        pytest.param(
            {
                "audios": [
                    {
                        "path": "audios/0.npy",
                        "shape": [4.9],
                        "dtype": "float32",
                        "sampling_rate": 16000,
                        "format": "wav",
                    }
                ]
            },
            r"audio entry 0.*shape\[0\].*4\.9",
            id="entry-type-non-integer-audio-shape",
        ),
        pytest.param(
            {
                "audios": [
                    {
                        "path": "audios/0.npy",
                        "shape": [4],
                        "dtype": "float32",
                        "sampling_rate": 16000.9,
                        "format": "wav",
                    }
                ]
            },
            r"audio entry 0.*sampling_rate.*16000\.9",
            id="entry-type-non-integer-audio-sampling-rate",
        ),
        pytest.param(
            {
                "audios": [
                    {"path": "audios/0.npy", "shape": [4], "dtype": "float32", "sampling_rate": 16000, "format": 7}
                ]
            },
            r"audio entry 0.*field 'format'.*must be a string",
            id="entry-type-audio-format",
        ),
        pytest.param(
            {"images": [{"path": "images/0.npy", "shape": [4], "dtype": None}]},
            r"image entry 0.*field 'dtype'.*must be a non-empty string",
            id="dtype-image-null",
        ),
        pytest.param(
            {"images": [{"path": "images/0.npy", "shape": [4], "dtype": 7}]},
            r"image entry 0.*field 'dtype'.*must be a non-empty string",
            id="dtype-image-number",
        ),
        pytest.param(
            {"images": [{"path": "images/0.npy", "shape": [4], "dtype": ""}]},
            r"image entry 0.*field 'dtype'.*must be a non-empty string",
            id="dtype-image-empty",
        ),
        pytest.param(
            {
                "audios": [
                    {"path": "audios/0.npy", "shape": [4], "dtype": None, "sampling_rate": 16000, "format": "wav"}
                ]
            },
            r"audio entry 0.*field 'dtype'.*must be a non-empty string",
            id="dtype-audio-null",
        ),
        pytest.param(
            {"audios": [{"path": "audios/0.npy", "shape": [4], "dtype": 7, "sampling_rate": 16000, "format": "wav"}]},
            r"audio entry 0.*field 'dtype'.*must be a non-empty string",
            id="dtype-audio-number",
        ),
        pytest.param(
            {"audios": [{"path": "audios/0.npy", "shape": [4], "dtype": "", "sampling_rate": 16000, "format": "wav"}]},
            r"audio entry 0.*field 'dtype'.*must be a non-empty string",
            id="dtype-audio-empty",
        ),
    ],
)
def test_manifest_rejects_invalid_payload(manifest_dir: Path, payload: dict[str, Any], message: str) -> None:
    manifest = _read_manifest(manifest_dir)
    manifest.update(payload)
    _write_manifest(manifest_dir=manifest_dir, manifest=manifest)

    with pytest.raises(expected_exception=ValueError, match=message):
        _assert_full_path(manifest_dir=manifest_dir, tokenized=Tokenized(tokens=[1, 2, 3]))


@pytest.mark.parametrize(
    argnames="field",
    argvalues=["case_id", "tokenizer_configuration_id", "token_ids", "decoded_text", "images", "audios"],
)
def test_manifest_requires_all_top_level_fields(manifest_dir: Path, field: str) -> None:
    manifest = _read_manifest(manifest_dir)
    del manifest[field]
    _write_manifest(manifest_dir=manifest_dir, manifest=manifest)

    with pytest.raises(
        expected_exception=ValueError, match=rf"Case 'integrity-case'.*missing required field '{field}'"
    ):
        _assert_full_path(manifest_dir=manifest_dir, tokenized=Tokenized(tokens=[1, 2, 3]))


def test_manifest_root_must_be_an_object(manifest_dir: Path) -> None:
    _write_manifest(manifest_dir=manifest_dir, manifest=["not", "an", "object"])

    with pytest.raises(expected_exception=ValueError, match=r"Case 'integrity-case'.*manifest.*must be an object"):
        _assert_full_path(manifest_dir=manifest_dir, tokenized=Tokenized(tokens=[1, 2, 3]))


@pytest.mark.parametrize(
    argnames=("token_ids", "decoded_text", "message"),
    argvalues=[
        ([1, 2, 4], "<s>hello</s>", "Token ids differ"),
        ([1, 2, 3], "<s>hello there</s>", "Decoded text differs"),
    ],
    ids=["token-ids", "decoded-text"],
)
def test_changed_public_result_fails_the_comparison(
    manifest_dir: Path, token_ids: list[int], decoded_text: str, message: str
) -> None:
    expected = _load(manifest_dir)
    tokenized = Tokenized(tokens=token_ids)

    with pytest.raises(expected_exception=AssertionError, match=message):
        assert_public_success(expected=expected, tokenized=tokenized, decoded_text=decoded_text)


@pytest.mark.parametrize(
    argnames=("mutation", "message"),
    argvalues=[
        pytest.param(
            lambda tokenized: tokenized.model_copy(update={"images": []}),
            "Expected 1 returned images",
            id="image-count",
        ),
        pytest.param(
            lambda tokenized: tokenized.model_copy(update={"images": [np.zeros(shape=(2, 1), dtype=np.float32)]}),
            "Image shape",
            id="image-shape",
        ),
        pytest.param(
            lambda tokenized: tokenized.model_copy(update={"images": [np.zeros(shape=(1, 2), dtype=np.float64)]}),
            "Image dtype",
            id="image-dtype",
        ),
        pytest.param(
            lambda tokenized: tokenized.model_copy(update={"audios": []}),
            "Expected 1 returned audios",
            id="audio-count",
        ),
        pytest.param(
            lambda tokenized: tokenized.model_copy(
                update={
                    "audios": [
                        Audio(audio_array=np.zeros(shape=(4,), dtype=np.float32), sampling_rate=8000, format="wav")
                    ]
                }
            ),
            "Audio sampling rate",
            id="audio-sampling-rate",
        ),
        pytest.param(
            lambda tokenized: tokenized.model_copy(
                update={
                    "audios": [
                        Audio(audio_array=np.zeros(shape=(4,), dtype=np.float32), sampling_rate=16000, format="mp3")
                    ]
                }
            ),
            "Audio format",
            id="audio-format",
        ),
        pytest.param(
            lambda tokenized: tokenized.model_copy(
                update={
                    "audios": [
                        Audio(audio_array=np.zeros(shape=(3,), dtype=np.float32), sampling_rate=16000, format="wav")
                    ]
                }
            ),
            "Audio shape",
            id="audio-shape",
        ),
        pytest.param(
            lambda tokenized: tokenized.model_copy(
                update={
                    "audios": [
                        Audio(audio_array=np.zeros(shape=(4,), dtype=np.float64), sampling_rate=16000, format="wav")
                    ]
                }
            ),
            "Audio dtype",
            id="audio-dtype",
        ),
    ],
)
def test_media_metadata_mismatch_fails_the_comparison(mutation: Callable[[Tokenized], Tokenized], message: str) -> None:
    expected_image_array = np.zeros(shape=(1, 2), dtype=np.float32)
    expected_audio_array = np.zeros(shape=(4,), dtype=np.float32)
    expected = ExpectedSuccess(
        token_ids=[1],
        decoded_text="<s></s>",
        images=(ExpectedImage(shape=(1, 2), dtype="float32", array=expected_image_array),),
        audios=(
            ExpectedAudio(
                shape=(4,),
                dtype="float32",
                sampling_rate=16000,
                format="wav",
                array=expected_audio_array,
            ),
        ),
    )
    tokenized = Tokenized(
        tokens=[1],
        images=[expected_image_array],
        audios=[Audio(audio_array=expected_audio_array, sampling_rate=16000, format="wav")],
    )
    changed_tokenized = mutation(tokenized)

    with pytest.raises(expected_exception=AssertionError, match=message):
        assert_public_success(expected=expected, tokenized=changed_tokenized, decoded_text="<s></s>")


@pytest.mark.parametrize(
    argnames=("image_offset", "audio_offset", "expect_failure"),
    argvalues=[
        pytest.param(0.9e-4, 0.0, False, id="image-within-tolerance"),
        pytest.param(1.1e-4, 0.0, True, id="image-beyond-tolerance"),
        pytest.param(0.0, 0.9e-4, False, id="audio-within-tolerance"),
        pytest.param(0.0, 1.1e-4, True, id="audio-beyond-tolerance"),
    ],
)
def test_shared_tolerance_boundary(image_offset: float, audio_offset: float, expect_failure: bool) -> None:
    expected_array = np.full(shape=(2,), fill_value=1e4, dtype=np.float64)
    expected = ExpectedSuccess(
        token_ids=[1],
        decoded_text="<s></s>",
        images=(ExpectedImage(shape=(2,), dtype="float64", array=expected_array),),
        audios=(
            ExpectedAudio(
                shape=(2,),
                dtype="float64",
                sampling_rate=16000,
                format="wav",
                array=expected_array,
            ),
        ),
    )
    tokenized = Tokenized(
        tokens=[1],
        images=[np.full(shape=(2,), fill_value=1e4 + image_offset, dtype=np.float64)],
        audios=[
            Audio(
                audio_array=np.full(shape=(2,), fill_value=1e4 + audio_offset, dtype=np.float64),
                sampling_rate=16000,
                format="wav",
            )
        ],
    )

    if expect_failure:
        with pytest.raises(expected_exception=AssertionError, match="array differs"):
            assert_public_success(expected=expected, tokenized=tokenized, decoded_text="<s></s>")
    else:
        assert_public_success(expected=expected, tokenized=tokenized, decoded_text="<s></s>")


def test_suite_tolerance_pair_is_the_approved_pair() -> None:
    assert (ATOL, RTOL) == (1e-4, 0.0)


@pytest.mark.parametrize(
    argnames=("atol", "rtol", "image_offset", "audio_offset"),
    argvalues=[
        pytest.param(2e-4, 0.0, 1.5e-4, 1.5e-4, id="patched-atol"),
        pytest.param(1e-4, 1e-6, 1e-3, 1e-3, id="patched-rtol"),
    ],
)
def test_comparator_uses_the_shared_pair(
    atol: float,
    rtol: float,
    image_offset: float,
    audio_offset: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(expected_results, "ATOL", atol)
    monkeypatch.setattr(expected_results, "RTOL", rtol)
    expected_array = np.full(shape=(2,), fill_value=1e4, dtype=np.float64)
    expected = ExpectedSuccess(
        token_ids=[1],
        decoded_text="<s></s>",
        images=(ExpectedImage(shape=(2,), dtype="float64", array=expected_array),),
        audios=(
            ExpectedAudio(
                shape=(2,),
                dtype="float64",
                sampling_rate=16000,
                format="wav",
                array=expected_array,
            ),
        ),
    )
    tokenized = Tokenized(
        tokens=[1],
        images=[np.full(shape=(2,), fill_value=1e4 + image_offset, dtype=np.float64)],
        audios=[
            Audio(
                audio_array=np.full(shape=(2,), fill_value=1e4 + audio_offset, dtype=np.float64),
                sampling_rate=16000,
                format="wav",
            )
        ],
    )

    assert_public_success(expected=expected, tokenized=tokenized, decoded_text="<s></s>")


@pytest.mark.parametrize(
    argnames=("sidecar_shape", "sidecar_dtype", "declared_dtype", "message"),
    argvalues=[
        pytest.param(
            (1, 2),
            "float64",
            "float32",
            "sidecar dtype.*does not match manifest declared dtype float32",
            id="dtype",
        ),
        pytest.param(
            (2, 1),
            "float32",
            "float32",
            r"sidecar shape.*does not match manifest declared shape \(1, 2\)",
            id="shape",
        ),
        pytest.param((1, 2), "float32", "float33", "invalid manifest dtype", id="invalid-dtype"),
    ],
)
def test_sidecar_metadata_mismatch_is_rejected_before_values(
    manifest_dir: Path, sidecar_shape: tuple[int, ...], sidecar_dtype: str, declared_dtype: str, message: str
) -> None:
    (manifest_dir / "images").mkdir()
    np.save(file=manifest_dir / "images" / "0.npy", arr=np.zeros(shape=sidecar_shape, dtype=sidecar_dtype))
    _write_image_entry(manifest_dir=manifest_dir, path="images/0.npy", shape=[1, 2], dtype=declared_dtype)
    tokenized = Tokenized(tokens=[1, 2, 3], images=[np.zeros(shape=(1, 2), dtype=np.float32)])

    with pytest.raises(expected_exception=ValueError, match=message):
        _assert_full_path(manifest_dir=manifest_dir, tokenized=tokenized)


@pytest.mark.parametrize(
    argnames=("case", "change_manifest", "message"),
    argvalues=[
        pytest.param(
            V15_USER_IMAGE_URL_CASE,
            lambda *, manifest: manifest.update(token_ids=[*manifest["token_ids"][:-1], manifest["token_ids"][-1] + 1]),
            "Token ids differ",
            id="token-ids",
        ),
        pytest.param(
            V15_USER_IMAGE_URL_CASE,
            lambda *, manifest: manifest.update(decoded_text=manifest["decoded_text"] + "!"),
            "Decoded text differs",
            id="decoded-text",
        ),
        pytest.param(
            V15_USER_IMAGE_URL_CASE,
            lambda *, manifest: np.save(
                file=expected_results._EXPECTED_ROOT / manifest["case_id"] / manifest["images"][0]["path"],
                arr=np.load(
                    file=expected_results._EXPECTED_ROOT / manifest["case_id"] / manifest["images"][0]["path"],
                    allow_pickle=False,
                )
                + 1.0,
            ),
            "Image array differs",
            id="image-values",
        ),
        pytest.param(
            RELEASED_USER_AUDIO_URL_CASE,
            lambda *, manifest: np.save(
                file=expected_results._EXPECTED_ROOT / manifest["case_id"] / manifest["audios"][0]["path"],
                arr=np.load(
                    file=expected_results._EXPECTED_ROOT / manifest["case_id"] / manifest["audios"][0]["path"],
                    allow_pickle=False,
                )
                + 1.0,
            ),
            "Audio array differs",
            id="audio-values",
        ),
    ],
)
def test_encode_and_verify_rejects_a_changed_manifest(
    case: PublicChatSuccessCase,
    change_manifest: Callable[..., None],
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    case_dir = tmp_path / case.case_id
    copytree(src=_EXPECTED_ROOT / case.case_id, dst=case_dir)
    manifest_path = case_dir / "expected.json"
    manifest = json.loads(manifest_path.read_text())
    monkeypatch.setattr(expected_results, "_EXPECTED_ROOT", tmp_path)
    change_manifest(manifest=manifest)
    manifest_path.write_text(json.dumps(obj=manifest, ensure_ascii=False))

    assert get_dummy_audio_url_chunk().get_url_type() == AudioURLType.base64
    with patch(
        target="mistral_common.tokens.tokenizers.audio._requests_lib.get",
        side_effect=AssertionError("network used"),
    ):
        with pytest.raises(expected_exception=AssertionError, match=message):
            encode_and_verify(case=case, public_tokenizer=public_tokenizer)


def test_changed_media_input_fails_public_encode_comparison(
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    changed_recipe = replace(
        V15_USER_IMAGE_URL_CASE.recipe,
        build=lambda: build_user_media_request(
            prompt="Here is content",
            content_chunk=build_image_url_chunk(
                image=Image.new(mode="RGB", size=(4, 4), color="blue"),
            ),
        ),
    )
    changed_case = replace(V15_USER_IMAGE_URL_CASE, recipe=changed_recipe)

    with pytest.raises(expected_exception=AssertionError, match="Image array differs"):
        encode_and_verify(case=changed_case, public_tokenizer=public_tokenizer)


@pytest.mark.parametrize(
    argnames=("case", "outer_exception", "outer_match"),
    argvalues=[
        pytest.param(
            replace(SAMPLE_V1_TOOLS_REJECTED_CASE, message_pattern=r"not the v1 error"),
            AssertionError,
            "Regex pattern did not match",
            id="wrong-message",
        ),
        pytest.param(
            replace(SAMPLE_V1_TOOLS_REJECTED_CASE, expected_exception=InvalidMessageStructureException),
            TokenizerException,
            "Tools not implemented for tokenizer V1",
            id="wrong-exception-type",
        ),
        pytest.param(
            replace(SAMPLE_V1_TOOLS_REJECTED_CASE, recipe=NO_TOOLS),
            pytest.fail.Exception,
            "DID NOT RAISE",
            id="encode-succeeds",
        ),
    ],
)
def test_encode_and_expect_error_rejects_wrong_outcomes(
    case: PublicChatErrorCase,
    outer_exception: type[BaseException],
    outer_match: str,
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    with pytest.raises(
        expected_exception=outer_exception,
        match=outer_match,
    ):
        encode_and_expect_error(case=case, public_tokenizer=public_tokenizer)


def test_construction_failure_fails_expected_error_case(
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    case = SAMPLE_V1_TOOLS_REJECTED_CASE

    def fail_request_construction() -> ChatCompletionRequest[ChatMessage]:
        raise case.expected_exception("request construction: Tools not implemented for tokenizer V1")

    sabotaged_recipe = replace(case.recipe, build=fail_request_construction)
    sabotaged_case = replace(case, recipe=sabotaged_recipe)

    with pytest.raises(expected_exception=case.expected_exception, match=case.message_pattern) as error_info:
        encode_and_expect_error(case=sabotaged_case, public_tokenizer=public_tokenizer)
    assert str(error_info.value).startswith("request construction:")
