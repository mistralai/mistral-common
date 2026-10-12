r"""Released-profile public chat behavior and full-result expectations."""

from collections.abc import Callable
from dataclasses import replace
from enum import Enum
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image
from pydantic import BaseModel

from mistral_common.protocol.instruct.chunk import AudioURLType, ImageChunk
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.fixtures.audio import get_dummy_audio_url_chunk
from tests.integration.chat_cases import (
    PublicChatErrorCase,
    PublicChatSuccessCase,
)
from tests.integration.chat_recipes import ChatRecipe
from tests.integration.chat_registry import ALL_PUBLIC_CASES, ALL_SUCCESS_CASES
from tests.integration.chat_released_cases import (
    RELEASED_ERROR_CASES,
    RELEASED_SUCCESS_CASES,
    RELEASED_USER_AUDIO_URL_CASE,
)
from tests.integration.chat_v7_cases import V7_DIRECT_EQUALITY_CASES
from tests.integration.tokenizer_configurations import (
    BUNDLED_SPM_V1_TEST,
    BUNDLED_TEKKEN_V3_MM_PATCH2_TEST,
    BUNDLED_TEKKEN_V3_MM_TEST,
    PINNED_V7_IMAGE_TEST,
    SYNTHETIC_V13_AUDIO_TEST,
    TokenizerConfiguration,
)
from tests.integration.utils import encode_and_expect_error, encode_and_verify

_REGISTERED_RECIPES = tuple({case.recipe.recipe_id: case.recipe for case in ALL_PUBLIC_CASES}.values())
_REGISTERED_CONFIGURATIONS = tuple(
    {case.configuration.configuration_id: case.configuration for case in ALL_PUBLIC_CASES}.values()
)
_V7_DIRECT_BUILDERS = tuple(dict.fromkeys(case.build_direct_request for case in V7_DIRECT_EQUALITY_CASES))
_RELEASED_SUCCESS_CASES_WITHOUT_AUDIO_URL = tuple(
    case for case in RELEASED_SUCCESS_CASES if case is not RELEASED_USER_AUDIO_URL_CASE
)


def _mutable_containers(
    value: object, *, path: tuple[str | int, ...]
) -> list[tuple[tuple[str | int, ...], dict[Any, Any] | list[Any]]]:
    containers: list[tuple[tuple[str | int, ...], dict[Any, Any] | list[Any]]] = []
    if isinstance(value, BaseModel):
        for field_name in type(value).model_fields:
            containers.extend(_mutable_containers(value=getattr(value, field_name), path=(*path, field_name)))
    elif isinstance(value, dict):
        containers.append((path, value))
        for key, item in value.items():
            containers.extend(_mutable_containers(value=item, path=(*path, key)))
    elif isinstance(value, list):
        containers.append((path, value))
        for index, item in enumerate(value):
            containers.extend(_mutable_containers(value=item, path=(*path, index)))
    return containers


def _mutable_media_object_ids(value: object) -> set[int]:
    if isinstance(value, (np.ndarray, Image.Image)):
        return {id(value)}
    if isinstance(value, BaseModel):
        return {
            object_id
            for field_name in type(value).model_fields
            for object_id in _mutable_media_object_ids(getattr(value, field_name))
        }
    if isinstance(value, dict):
        return {object_id for item in value.values() for object_id in _mutable_media_object_ids(item)}
    if isinstance(value, (list, tuple)):
        return {object_id for item in value for object_id in _mutable_media_object_ids(item)}
    return set()


def _structure(value: object) -> object:
    if isinstance(value, BaseModel):
        return (
            type(value),
            tuple((field_name, _structure(getattr(value, field_name))) for field_name in type(value).model_fields),
        )
    if isinstance(value, np.ndarray):
        return (type(value), value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, Image.Image):
        return (type(value), value.mode, value.size, value.tobytes())
    if isinstance(value, dict):
        return tuple((key, _structure(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_structure(item) for item in value)
    if isinstance(value, Enum):
        return (type(value), value.value)
    return value


@pytest.mark.parametrize(argnames="recipe", argvalues=_REGISTERED_RECIPES, ids=lambda recipe: recipe.recipe_id)
def test_registered_chat_recipes_build_deeply_fresh_requests(recipe: ChatRecipe) -> None:
    _assert_fresh_request_graph(recipe_id=recipe.recipe_id, build=recipe.build)


@pytest.mark.parametrize(
    argnames="build_direct_request",
    argvalues=_V7_DIRECT_BUILDERS,
    ids=lambda build_direct_request: build_direct_request.__name__,
)
def test_registered_v7_direct_recipes_build_deeply_fresh_requests(
    build_direct_request: Callable[[], BaseModel],
) -> None:
    _assert_fresh_request_graph(recipe_id=build_direct_request.__name__, build=build_direct_request)


def test_freshness_check_rejects_shared_image() -> None:
    shared_image = Image.new(mode="RGB", size=(1, 1))

    def build_shared_image_request() -> BaseModel:
        return UserMessage(content=[ImageChunk(image=shared_image)])

    with pytest.raises(expected_exception=AssertionError, match="reuses mutable image or array objects"):
        _assert_fresh_request_graph(recipe_id="shared-image", build=build_shared_image_request)


def _assert_fresh_request_graph(*, recipe_id: str, build: Callable[[], BaseModel]) -> None:
    first = build()
    second = build()
    expected_structure = _structure(second)
    assert _structure(first) == expected_structure
    shared_media_ids = _mutable_media_object_ids(first) & _mutable_media_object_ids(second)
    assert not shared_media_ids, f"Recipe {recipe_id} reuses mutable image or array objects across builds"

    first_containers = _mutable_containers(value=first, path=())
    second_container_ids = {id(container) for _, container in _mutable_containers(value=second, path=())}
    shared_containers = [
        (len(path), container) for path, container in first_containers if id(container) in second_container_ids
    ]
    candidates = shared_containers or [(len(path), container) for path, container in first_containers]
    _, target = max(candidates, key=lambda item: item[0])
    if isinstance(target, dict):
        target["__freshness_probe__"] = recipe_id
    else:
        target.append(("freshness-probe", recipe_id))

    third = build()
    assert (_structure(second), _structure(third)) == (expected_structure, expected_structure)


@pytest.mark.parametrize(
    argnames="configuration",
    argvalues=_REGISTERED_CONFIGURATIONS,
    ids=lambda configuration: configuration.configuration_id,
)
def test_registered_configurations_load_in_their_mode(
    configuration: TokenizerConfiguration,
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    assert public_tokenizer(configuration).mode == configuration.mode


def test_tokenizer_artifact_digest_mismatch_is_rejected() -> None:
    configuration = replace(BUNDLED_SPM_V1_TEST, sha256="0" * 64)

    with pytest.raises(expected_exception=ValueError, match="Tokenizer artifact mismatch"):
        configuration.load()


@pytest.mark.parametrize(
    argnames="invalid_configuration",
    argvalues=["synthetic-path", "file-missing-digest", "file-synthetic-loader"],
    ids=["synthetic-with-path", "file-backed-without-digest", "file-backed-with-synthetic-loader"],
)
def test_incomplete_tokenizer_configuration_is_rejected(invalid_configuration: str) -> None:
    with pytest.raises(expected_exception=ValueError, match="incomplete"):
        match invalid_configuration:
            case "synthetic-path":
                replace(
                    BUNDLED_SPM_V1_TEST,
                    provenance="synthetic",
                    synthetic_loader=SYNTHETIC_V13_AUDIO_TEST.synthetic_loader,
                )
            case "file-missing-digest":
                replace(BUNDLED_SPM_V1_TEST, sha256=None)
            case "file-synthetic-loader":
                replace(BUNDLED_SPM_V1_TEST, synthetic_loader=SYNTHETIC_V13_AUDIO_TEST.synthetic_loader)


def test_public_tokenizer_cache_isolates_configurations(
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    patch2_tokenizer = public_tokenizer(BUNDLED_TEKKEN_V3_MM_PATCH2_TEST)
    unmodified_tokenizer = public_tokenizer(BUNDLED_TEKKEN_V3_MM_TEST)
    patch2_encoder = patch2_tokenizer.instruct_tokenizer.image_encoder
    unmodified_encoder = unmodified_tokenizer.instruct_tokenizer.image_encoder

    assert patch2_tokenizer is not unmodified_tokenizer
    assert patch2_encoder is not None
    assert unmodified_encoder is not None
    assert patch2_encoder.image_config is not unmodified_encoder.image_config
    assert patch2_encoder.image_config.image_patch_size == 2
    assert unmodified_encoder.image_config.image_patch_size == 16
    assert public_tokenizer(BUNDLED_TEKKEN_V3_MM_PATCH2_TEST) is patch2_tokenizer
    assert unmodified_encoder.image_config.image_patch_size == 16


def test_registry_identities_are_unique() -> None:
    configurations: dict[str, TokenizerConfiguration] = {}
    recipes: dict[str, ChatRecipe] = {}
    for case in ALL_PUBLIC_CASES:
        configuration = configurations.get(case.configuration.configuration_id)
        if configuration is None:
            configurations[case.configuration.configuration_id] = case.configuration
        else:
            assert configuration is case.configuration

        recipe = recipes.get(case.recipe.recipe_id)
        if recipe is None:
            recipes[case.recipe.recipe_id] = case.recipe
        else:
            assert recipe is case.recipe


def test_expected_manifest_directories_match_success_cases() -> None:
    expected_root = Path(__file__).resolve().parents[1] / "data" / "expected"
    manifest_directories = {path.name for path in expected_root.iterdir() if path.is_dir()}
    success_case_ids = {case.case_id for case in ALL_SUCCESS_CASES}

    assert manifest_directories == success_case_ids


@pytest.mark.parametrize(
    argnames="case",
    argvalues=_RELEASED_SUCCESS_CASES_WITHOUT_AUDIO_URL,
    ids=lambda case: case.case_id,
)
def test_released_profile_public_chat_success(
    case: PublicChatSuccessCase,
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    encode_and_verify(case=case, public_tokenizer=public_tokenizer)


def test_released_user_audio_url_uses_no_network(
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    assert get_dummy_audio_url_chunk().get_url_type() == AudioURLType.base64
    with patch(
        target="mistral_common.tokens.tokenizers.audio._requests_lib.get",
        side_effect=AssertionError("network used"),
    ):
        encode_and_verify(case=RELEASED_USER_AUDIO_URL_CASE, public_tokenizer=public_tokenizer)


@pytest.mark.parametrize(argnames="case", argvalues=RELEASED_ERROR_CASES, ids=lambda case: case.case_id)
def test_released_profile_public_chat_error(
    case: PublicChatErrorCase,
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    encode_and_expect_error(case=case, public_tokenizer=public_tokenizer)


def test_missing_pinned_tokenizer_names_provisioning_script(tmp_path: Path) -> None:
    configuration = replace(
        PINNED_V7_IMAGE_TEST,
        tokenizer_path=tmp_path / "v7-image.tekken.json",
    )

    with pytest.raises(expected_exception=FileNotFoundError, match=r"scripts/provision_test_tokenizers\.py"):
        configuration.load()


def test_public_chat_case_ids_are_globally_unique() -> None:
    case_ids = [case.case_id for case in ALL_PUBLIC_CASES]

    assert len(case_ids) == len(set(case_ids))
