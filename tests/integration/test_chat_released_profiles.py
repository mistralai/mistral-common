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

from mistral_common.protocol.instruct.chunk import AudioURLType
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.fixtures.audio import get_dummy_audio_url_chunk
from tests.integration.chat_cases import (
    PublicChatErrorCase,
    PublicChatSuccessCase,
)
from tests.integration.chat_recipes import ChatRecipe
from tests.integration.chat_registry import ALL_PUBLIC_CASES
from tests.integration.chat_released_cases import RELEASED_ERROR_CASES, RELEASED_SUCCESS_CASES
from tests.integration.chat_v7_cases import V7_DIRECT_EQUALITY_CASES, V7DirectEqualityCase
from tests.integration.tokenizer_configurations import PINNED_V7_IMAGE_TEST, TokenizerConfiguration
from tests.integration.utils import encode_and_expect_error, encode_and_verify

_REGISTERED_RECIPES = tuple({case.recipe.recipe_id: case.recipe for case in ALL_PUBLIC_CASES}.values())


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


@pytest.mark.parametrize(argnames="case", argvalues=V7_DIRECT_EQUALITY_CASES, ids=lambda case: f"{case.case_id}-direct")
def test_registered_v7_direct_recipes_build_deeply_fresh_requests(case: V7DirectEqualityCase) -> None:
    _assert_fresh_request_graph(recipe_id=case.case_id, build=case.build_direct_request)


def _assert_fresh_request_graph(*, recipe_id: str, build: Callable[[], BaseModel]) -> None:
    first = build()
    second = build()
    expected_structure = _structure(second)
    assert _structure(first) == expected_structure

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


@pytest.mark.parametrize(argnames="case", argvalues=RELEASED_SUCCESS_CASES, ids=lambda case: case.case_id)
def test_released_profile_public_chat_success(
    case: PublicChatSuccessCase,
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    if case.case_id == "chat-v7-released-user-audio-url":
        assert get_dummy_audio_url_chunk().get_url_type() == AudioURLType.base64
        with patch(
            target="mistral_common.tokens.tokenizers.audio._requests_lib.get",
            side_effect=AssertionError("network used"),
        ):
            encode_and_verify(case=case, public_tokenizer=public_tokenizer)
    else:
        encode_and_verify(case=case, public_tokenizer=public_tokenizer)


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

    with pytest.raises(FileNotFoundError, match=r"scripts/provision_test_tokenizers\.py"):
        configuration.load()


def test_public_chat_case_ids_are_globally_unique() -> None:
    case_ids = [case.case_id for case in ALL_PUBLIC_CASES]

    assert len(case_ids) == len(set(case_ids))
