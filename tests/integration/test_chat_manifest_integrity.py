r"""Integrity of the reviewed expected-result manifests.

These cases protect the golden comparison itself: a wrong association, an
escaping or missing sidecar, a pickle-based array, or a changed token/text
value must fail loudly instead of silently comparing an unrelated golden.
"""

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from mistral_common.exceptions import InvalidMessageStructureException, TokenizerException
from mistral_common.protocol.instruct.messages import ChatMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.audio import Audio
from mistral_common.tokens.tokenizers.base import Tokenized
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.integration.chat_cases import SAMPLE_V1_TOOLS_REJECTED_CASE, PublicChatErrorCase
from tests.integration.chat_recipes import build_image_url_chunk, build_user_media_request
from tests.integration.chat_v15_cases import V15_USER_IMAGE_URL_CASE
from tests.integration.expected_results import (
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
    argnames=("token_ids", "message"),
    argvalues=[
        ({}, "field 'token_ids' must be an array"),
        ("123", "field 'token_ids' must be an array"),
        ([True, 2, 3], r"token_ids\[0\].*must be an integer"),
        ([1.9, 2, 3], r"Case 'integrity-case'.*token_ids\[0\].*1\.9"),
    ],
    ids=["object", "string", "boolean-item", "float-item"],
)
def test_token_ids_must_be_an_array_of_exact_integers(manifest_dir: Path, token_ids: object, message: str) -> None:
    manifest = _read_manifest(manifest_dir)
    manifest["token_ids"] = token_ids
    _write_manifest(manifest_dir=manifest_dir, manifest=manifest)

    with pytest.raises(expected_exception=ValueError, match=message):
        _assert_full_path(manifest_dir=manifest_dir, tokenized=Tokenized(tokens=[]))


@pytest.mark.parametrize(
    argnames=("field", "value", "message"),
    argvalues=[
        pytest.param("case_id", 42, "field 'case_id' must be a string", id="case-id"),
        pytest.param(
            "tokenizer_configuration_id",
            None,
            "field 'tokenizer_configuration_id' must be a string",
            id="configuration-id",
        ),
        pytest.param("decoded_text", 17, "field 'decoded_text' must be a string", id="decoded-text"),
        pytest.param("images", {}, "field 'images' must be an array", id="images-object"),
        pytest.param("images", "none", "field 'images' must be an array", id="images-string"),
        pytest.param("audios", {}, "field 'audios' must be an array", id="audios-object"),
        pytest.param("audios", "none", "field 'audios' must be an array", id="audios-string"),
    ],
)
def test_manifest_fields_reject_other_json_types(manifest_dir: Path, field: str, value: object, message: str) -> None:
    manifest = _read_manifest(manifest_dir)
    manifest[field] = value
    _write_manifest(manifest_dir=manifest_dir, manifest=manifest)

    with pytest.raises(expected_exception=ValueError, match=message):
        _assert_full_path(manifest_dir=manifest_dir, tokenized=Tokenized(tokens=[1, 2, 3]))


@pytest.mark.parametrize(argnames="field", argvalues=["images", "audios"], ids=["image", "audio"])
def test_media_entries_must_be_objects(manifest_dir: Path, field: str) -> None:
    manifest = _read_manifest(manifest_dir)
    manifest[field] = ["not-an-object"]
    _write_manifest(manifest_dir=manifest_dir, manifest=manifest)

    media_kind = "image" if field == "images" else "audio"
    with pytest.raises(expected_exception=ValueError, match=rf"{media_kind} entry 0.*must be an object"):
        _assert_full_path(manifest_dir=manifest_dir, tokenized=Tokenized(tokens=[1, 2, 3]))


@pytest.mark.parametrize(
    argnames=("field", "entry", "message"),
    argvalues=[
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
        (
            "images",
            {"path": "images/0.npy", "shape": [1.9, 2], "dtype": "float32"},
            r"image entry 0.*shape\[0\].*1\.9",
        ),
        (
            "audios",
            {"path": "audios/0.npy", "shape": [4.9], "dtype": "float32", "sampling_rate": 16000, "format": "wav"},
            r"audio entry 0.*shape\[0\].*4\.9",
        ),
        (
            "audios",
            {"path": "audios/0.npy", "shape": [4], "dtype": "float32", "sampling_rate": 16000.9, "format": "wav"},
            r"audio entry 0.*sampling_rate.*16000\.9",
        ),
        (
            "audios",
            {"path": "audios/0.npy", "shape": [4], "dtype": "float32", "sampling_rate": 16000, "format": 7},
            r"audio entry 0.*field 'format'.*must be a string",
        ),
    ],
    ids=[
        "image-path",
        "audio-path",
        "image-shape",
        "audio-shape",
        "boolean-shape",
        "boolean-sampling-rate",
        "non-integer-image-shape",
        "non-integer-audio-shape",
        "non-integer-audio-sampling-rate",
        "numeric-audio-format",
    ],
)
def test_media_entry_fields_have_the_declared_types(
    manifest_dir: Path, field: str, entry: dict[str, object], message: str
) -> None:
    manifest = _read_manifest(manifest_dir)
    manifest[field] = [entry]
    _write_manifest(manifest_dir=manifest_dir, manifest=manifest)

    with pytest.raises(expected_exception=ValueError, match=message):
        _assert_full_path(manifest_dir=manifest_dir, tokenized=Tokenized(tokens=[1, 2, 3]))


@pytest.mark.parametrize(
    argnames=("media_kind", "declared_dtype"),
    argvalues=[
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
    _write_manifest(manifest_dir=manifest_dir, manifest=manifest)
    if media_kind == "image":
        tokenized = Tokenized(tokens=[1, 2, 3], images=[array])
    else:
        tokenized = Tokenized(tokens=[1, 2, 3], audios=[Audio(audio_array=array, sampling_rate=16000, format="wav")])

    with pytest.raises(
        expected_exception=ValueError, match=rf"{media_kind} entry 0.*field 'dtype'.*must be a non-empty string"
    ):
        _assert_full_path(manifest_dir=manifest_dir, tokenized=tokenized)


@pytest.mark.parametrize(
    argnames=("field", "missing_field"),
    argvalues=[
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
    _write_manifest(manifest_dir=manifest_dir, manifest=manifest)

    media_kind = "image" if field == "images" else "audio"
    with pytest.raises(
        expected_exception=ValueError, match=rf"{media_kind} entry 0.*missing required field '{missing_field}'"
    ):
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
    argnames=("media_kind", "offset", "expect_failure"),
    argvalues=[
        ("image", 0.9e-4, False),
        ("image", 1.1e-4, True),
        ("audio", 0.9e-4, False),
        ("audio", 1.1e-4, True),
    ],
    ids=["image-within-tolerance", "image-beyond-tolerance", "audio-within-tolerance", "audio-beyond-tolerance"],
)
def test_shared_tolerance_boundary(media_kind: str, offset: float, expect_failure: bool) -> None:
    assert (ATOL, RTOL) == (1e-4, 0.0)

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
    changed_array = np.full(shape=(2,), fill_value=1e4 + offset, dtype=np.float64)
    image_array = changed_array if media_kind == "image" else expected_array
    audio_array = changed_array if media_kind == "audio" else expected_array
    tokenized = Tokenized(
        tokens=[1],
        images=[image_array],
        audios=[Audio(audio_array=audio_array, sampling_rate=16000, format="wav")],
    )

    if expect_failure:
        with pytest.raises(expected_exception=AssertionError, match="array differs"):
            assert_public_success(expected=expected, tokenized=tokenized, decoded_text="<s></s>")
    else:
        assert_public_success(expected=expected, tokenized=tokenized, decoded_text="<s></s>")


@pytest.mark.parametrize(
    argnames=("sidecar_shape", "sidecar_dtype", "message"),
    argvalues=[
        ((1, 2), "float64", "sidecar dtype.*does not match manifest declared dtype float32"),
        ((2, 1), "float32", r"sidecar shape.*does not match manifest declared shape \(1, 2\)"),
    ],
    ids=["dtype", "shape"],
)
def test_sidecar_metadata_mismatch_is_rejected_before_values(
    manifest_dir: Path, sidecar_shape: tuple[int, ...], sidecar_dtype: str, message: str
) -> None:
    (manifest_dir / "images").mkdir()
    np.save(file=manifest_dir / "images" / "0.npy", arr=np.zeros(shape=sidecar_shape, dtype=sidecar_dtype))
    _write_image_entry(manifest_dir=manifest_dir, path="images/0.npy", shape=[1, 2], dtype="float32")
    tokenized = Tokenized(tokens=[1, 2, 3], images=[np.zeros(shape=(1, 2), dtype=np.float32)])

    with pytest.raises(expected_exception=ValueError, match=message):
        _assert_full_path(manifest_dir=manifest_dir, tokenized=tokenized)


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
    argnames="case",
    argvalues=[
        pytest.param(replace(SAMPLE_V1_TOOLS_REJECTED_CASE, message_pattern=r"not the v1 error"), id="wrong-message"),
        pytest.param(
            replace(SAMPLE_V1_TOOLS_REJECTED_CASE, expected_exception=InvalidMessageStructureException),
            id="wrong-exception-type",
        ),
    ],
)
def test_encode_and_expect_error_requires_type_and_message(
    case: PublicChatErrorCase,
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    with pytest.raises(
        expected_exception=(pytest.fail.Exception, AssertionError, TokenizerException),
        match=r"(Tools not implemented for tokenizer V1|Regex pattern did not match)",
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
