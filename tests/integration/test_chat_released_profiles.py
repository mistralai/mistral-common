r"""Released-profile public chat behavior and full-result expectations."""

from collections.abc import Callable
from enum import Enum
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
    SAMPLE_ERROR_CASES,
    SAMPLE_SUCCESS_CASES,
    PublicChatErrorCase,
    PublicChatSuccessCase,
)
from tests.integration.chat_recipes import ChatRecipe
from tests.integration.chat_released_cases import (
    RELEASED_ERROR_CASES,
    RELEASED_SUCCESS_CASES,
)
from tests.integration.chat_v3_cases import V3_SUCCESS_CASES
from tests.integration.chat_v7_cases import V7_DIRECT_EQUALITY_CASES, V7_SUCCESS_CASES, V7DirectEqualityCase
from tests.integration.chat_v13_cases import V13_SUCCESS_CASES
from tests.integration.chat_v15_cases import V15_ERROR_CASES, V15_SUCCESS_CASES
from tests.integration.expected_results import assert_public_success, load_expected_success
from tests.integration.tokenizer_configurations import TokenizerConfiguration
from tests.utils import decode_keep

_ALL_PUBLIC_CASES: tuple[PublicChatSuccessCase | PublicChatErrorCase, ...] = (
    *SAMPLE_SUCCESS_CASES,
    *SAMPLE_ERROR_CASES,
    *V3_SUCCESS_CASES,
    *V7_SUCCESS_CASES,
    *V13_SUCCESS_CASES,
    *V15_SUCCESS_CASES,
    *V15_ERROR_CASES,
    *RELEASED_SUCCESS_CASES,
    *RELEASED_ERROR_CASES,
)
_REGISTERED_RECIPES = tuple({case.recipe.recipe_id: case.recipe for case in _ALL_PUBLIC_CASES}.values())


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


def _encode_and_verify(
    case: PublicChatSuccessCase,
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    request = case.recipe.build()
    tokenizer = public_tokenizer(case.configuration)
    tokenized = tokenizer.encode_chat_completion(request)
    decoded_text = decode_keep(tokenizer=tokenizer, tokenized=tokenized)
    expected = load_expected_success(
        case_id=case.case_id,
        tokenizer_configuration_id=case.configuration.configuration_id,
    )
    assert_public_success(expected=expected, tokenized=tokenized, decoded_text=decoded_text)


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
            _encode_and_verify(case=case, public_tokenizer=public_tokenizer)
    else:
        _encode_and_verify(case=case, public_tokenizer=public_tokenizer)


@pytest.mark.parametrize(argnames="case", argvalues=RELEASED_ERROR_CASES, ids=lambda case: case.case_id)
def test_released_profile_public_chat_error(
    case: PublicChatErrorCase,
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    request = case.recipe.build()
    tokenizer = public_tokenizer(case.configuration)

    with pytest.raises(expected_exception=case.expected_exception, match=case.message_pattern):
        tokenizer.encode_chat_completion(request)


def test_public_chat_case_ids_are_globally_unique() -> None:
    success_cases = (
        *SAMPLE_SUCCESS_CASES,
        *V3_SUCCESS_CASES,
        *V7_SUCCESS_CASES,
        *V13_SUCCESS_CASES,
        *V15_SUCCESS_CASES,
        *RELEASED_SUCCESS_CASES,
    )
    error_cases = (*SAMPLE_ERROR_CASES, *V15_ERROR_CASES, *RELEASED_ERROR_CASES)
    case_ids = [case.case_id for case in success_cases] + [case.case_id for case in error_cases]

    assert len(case_ids) == len(set(case_ids))
