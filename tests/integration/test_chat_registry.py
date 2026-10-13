r"""Registry, recipe and tokenizer-configuration invariants for public chat cases."""

from collections.abc import Callable
from dataclasses import replace
from enum import Enum
from importlib import import_module
from pathlib import Path
from pkgutil import iter_modules

import pytest
from PIL import Image
from pydantic import BaseModel

import tests.integration
from mistral_common.protocol.instruct.chunk import ImageChunk
from mistral_common.protocol.instruct.messages import ChatMessage, UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import Function, Tool
from mistral_common.protocol.instruct.validator import ValidationMode
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.integration.chat_cases import PublicChatErrorCase, PublicChatSuccessCase
from tests.integration.chat_recipes import ChatRecipe
from tests.integration.chat_registry import ALL_PUBLIC_CASES, ALL_SUCCESS_CASES
from tests.integration.chat_v3_cases import PairedChatCase
from tests.integration.chat_v7_cases import V7_DIRECT_EQUALITY_CASES, V7DirectEqualityCase
from tests.integration.expected_results import _EXPECTED_ROOT
from tests.integration.tokenizer_configurations import (
    BUNDLED_SPM_V1_TEST,
    BUNDLED_TEKKEN_V3_MM_PATCH2_TEST,
    BUNDLED_TEKKEN_V3_MM_TEST,
    PINNED_V7_IMAGE_TEST,
    SYNTHETIC_V13_AUDIO_TEST,
    SYNTHETIC_V15_AUDIO_TEST,
    TokenizerConfiguration,
)

_REGISTERED_RECIPES = tuple({case.recipe.recipe_id: case.recipe for case in ALL_PUBLIC_CASES}.values())
_REGISTERED_CONFIGURATIONS = tuple(
    {case.configuration.configuration_id: case.configuration for case in ALL_PUBLIC_CASES}.values()
)
_V7_DIRECT_BUILDERS = tuple(dict.fromkeys(case.build_direct_request for case in V7_DIRECT_EQUALITY_CASES))


def _mutable_object_ids(value: object) -> set[int]:
    object_ids: set[int] = set()
    if isinstance(value, (BaseModel, dict, list, Image.Image)):
        object_ids.add(id(value))
    if isinstance(value, BaseModel):
        for field_name in type(value).model_fields:
            object_ids.update(_mutable_object_ids(getattr(value, field_name)))
    elif isinstance(value, dict):
        for key, item in value.items():
            object_ids.update(_mutable_object_ids(key))
            object_ids.update(_mutable_object_ids(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            object_ids.update(_mutable_object_ids(item))
    return object_ids


def _structure(value: object) -> object:
    if isinstance(value, BaseModel):
        return (
            type(value),
            tuple((field_name, _structure(getattr(value, field_name))) for field_name in type(value).model_fields),
        )
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


@pytest.mark.parametrize(
    argnames="shared_kind",
    argvalues=["image", "user-message", "content-list", "dictionary", "nondeterministic-structure"],
    ids=[
        "shared-image",
        "shared-user-message",
        "shared-content-list",
        "shared-dictionary",
        "nondeterministic-structure",
    ],
)
def test_freshness_check_rejects_unfresh_builds(shared_kind: str) -> None:
    shared_image = Image.new(mode="RGB", size=(1, 1))
    shared_message = UserMessage(content="shared")
    shared_content: list[object] = []
    shared_parameters = {"type": "object", "properties": {}}
    contents = iter(("first request", "second request"))

    def build_shared_image_request() -> BaseModel:
        return ChatCompletionRequest[ChatMessage](messages=[UserMessage(content=[ImageChunk(image=shared_image)])])

    def build_shared_user_message_request() -> BaseModel:
        return ChatCompletionRequest[ChatMessage](messages=[shared_message])

    def build_shared_content_list_request() -> BaseModel:
        message = UserMessage.model_construct(role="user", content=shared_content)
        return ChatCompletionRequest[ChatMessage].model_construct(messages=[message])

    def build_shared_dictionary_request() -> BaseModel:
        function = Function.model_construct(name="lookup", parameters=shared_parameters)
        tool = Tool.model_construct(function=function)
        return ChatCompletionRequest[ChatMessage](messages=[UserMessage(content="query")], tools=[tool])

    def build_nondeterministic_request() -> BaseModel:
        return ChatCompletionRequest[ChatMessage](messages=[UserMessage(content=next(contents))])

    builders = {
        "image": build_shared_image_request,
        "user-message": build_shared_user_message_request,
        "content-list": build_shared_content_list_request,
        "dictionary": build_shared_dictionary_request,
        "nondeterministic-structure": build_nondeterministic_request,
    }
    with pytest.raises(expected_exception=AssertionError):
        _assert_fresh_request_graph(recipe_id=shared_kind, build=builders[shared_kind])


def _assert_fresh_request_graph(*, recipe_id: str, build: Callable[[], BaseModel]) -> None:
    first = build()
    second = build()
    expected_structure = _structure(second)
    assert _structure(first) == expected_structure
    shared_object_ids = _mutable_object_ids(first) & _mutable_object_ids(second)
    assert not shared_object_ids, f"Recipe {recipe_id} reuses mutable objects across builds"


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
    argnames="invalid_field",
    argvalues=[
        pytest.param("synthetic-path", id="synthetic-with-path"),
        pytest.param("synthetic-digest", id="synthetic-with-digest"),
        pytest.param("synthetic-loader", id="synthetic-without-loader"),
        pytest.param("file-path", id="file-backed-without-path"),
        pytest.param("file-digest", id="file-backed-without-digest"),
        pytest.param("file-loader", id="file-backed-with-synthetic-loader"),
    ],
)
def test_incomplete_tokenizer_configuration_is_rejected(invalid_field: str) -> None:
    with pytest.raises(expected_exception=ValueError, match="incomplete"):
        match invalid_field:
            case "synthetic-path":
                replace(SYNTHETIC_V13_AUDIO_TEST, tokenizer_path=Path("unused"))
            case "synthetic-digest":
                replace(SYNTHETIC_V13_AUDIO_TEST, sha256="0" * 64)
            case "synthetic-loader":
                replace(SYNTHETIC_V13_AUDIO_TEST, synthetic_loader=None)
            case "file-path":
                replace(BUNDLED_SPM_V1_TEST, tokenizer_path=None)
            case "file-digest":
                replace(BUNDLED_SPM_V1_TEST, sha256=None)
            case "file-loader":
                replace(BUNDLED_SPM_V1_TEST, synthetic_loader=SYNTHETIC_V13_AUDIO_TEST.synthetic_loader)


@pytest.mark.parametrize(
    argnames=("configuration", "mode"),
    argvalues=[
        pytest.param(SYNTHETIC_V13_AUDIO_TEST, ValidationMode.finetuning, id="v13-finetuning"),
        pytest.param(SYNTHETIC_V15_AUDIO_TEST, ValidationMode.serving, id="v15-serving"),
    ],
)
def test_synthetic_loaders_apply_the_declared_mode(configuration: TokenizerConfiguration, mode: ValidationMode) -> None:
    assert replace(configuration, mode=mode).load().mode == mode


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

    recipe_structures: list[tuple[str, object]] = []
    for recipe_id, recipe in recipes.items():
        request_structure = _structure(recipe.build())
        for other_recipe_id, other_structure in recipe_structures:
            assert request_structure != other_structure, (
                f"Recipes {recipe_id!r} and {other_recipe_id!r} build structurally identical requests"
            )
        recipe_structures.append((recipe_id, request_structure))


def test_expected_manifest_directories_match_success_cases() -> None:
    manifest_directories = {path.name for path in _EXPECTED_ROOT.iterdir() if path.is_dir()}
    success_case_ids = {case.case_id for case in ALL_SUCCESS_CASES}

    assert manifest_directories == success_case_ids


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


def test_registry_contains_every_defined_case() -> None:
    defined_cases: list[PublicChatSuccessCase | PublicChatErrorCase] = []

    def collect(value: object) -> None:
        if isinstance(value, (PublicChatSuccessCase, PublicChatErrorCase)):
            defined_cases.append(value)
        elif isinstance(value, PairedChatCase):
            collect(value.first)
            collect(value.second)
        elif isinstance(value, V7DirectEqualityCase):
            collect(value.case)
        elif isinstance(value, tuple):
            for item in value:
                collect(item)

    case_module_names = (
        module_info.name
        for module_info in iter_modules(path=tests.integration.__path__)
        if module_info.name == "chat_cases"
        or (module_info.name.startswith("chat_") and module_info.name.endswith("_cases"))
    )
    for module_name in case_module_names:
        module = import_module(name=f"tests.integration.{module_name}")
        for value in vars(module).values():
            collect(value)

    registered_identities = {id(case) for case in ALL_PUBLIC_CASES}
    missing_case_ids = {case.case_id for case in defined_cases if id(case) not in registered_identities}

    assert not missing_case_ids, f"Defined public chat cases missing from ALL_PUBLIC_CASES: {sorted(missing_case_ids)}"
