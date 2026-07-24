r"""Shared tokenizer fixtures for the mirror-tree tokenizer tests.

Every tokenizer a test needs is built here, never in a test file, and every fixture is
session-scoped so a given tokenizer is built at most once per run.

Two axes exist:

- **Synthetic** in-memory tokenizers, via `instruct_tokenizer` / `mistral_tokenizer`.
  Both are parametrized over the shared `TEKKEN_CONFIGS` matrix and derive from the same
  `TestConfig`, so a test narrows to the configs it needs::

      @pytest.mark.parametrize("mistral_tokenizer", [TestConfig(version=TokenizerVersion.v13)],
                               indirect=True, ids=config_id)

  Only the configs a test actually requests are ever built.

- **Shipped** tokenizers loaded from the packaged data files, via
  `shipped_mistral_tokenizer` / `shipped_instruct_tokenizer`, narrowed the same way with
  a key from `SHIPPED_TOKENIZER_KEYS`.

- **Keyed** tokenizers, via `keyed_mistral_tokenizer`, narrowed with any key from
  `tests.utils.tokenizers.TOKENIZER_FACTORIES` (shipped or synthetic). This is the one
  fixture that also covers the FIM-only synthetic versions (v11/v13/v15) and the golden
  registry keys used by `tests/tokenizers/test_registry_samples.py`.

`instruct_tokenizer_factory` / `mistral_tokenizer_factory` cover the one case a plain
fixture cannot: model settings vary per test case. Both cache their results.
"""

from collections.abc import Callable
from functools import lru_cache

import pytest

from mistral_common.protocol.instruct.request import ReasoningEffort
from mistral_common.tokens.tokenizers.base import InstructTokenizer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.utils.tokenizers import (
    SHIPPED_TOKENIZER_KEYS,
    build_instruct_tokenizer_from_config,
    build_mistral_tokenizer_from_config,
    build_model_settings_builder,
    load_mistral_tokenizer,
)
from tests.utils.versions import TEKKEN_CONFIGS, TestConfig, config_id


@pytest.fixture(params=TEKKEN_CONFIGS, ids=config_id, scope="session")
def instruct_tokenizer(request: pytest.FixtureRequest) -> InstructTokenizer:
    r"""Synthetic in-memory instruct tokenizer for one config of the Tekken matrix.

    Returns:
        An `InstructTokenizer` of the subclass matching the parametrized `TestConfig`.
    """
    config: TestConfig = request.param
    return build_instruct_tokenizer_from_config(config)


@pytest.fixture(params=TEKKEN_CONFIGS, ids=config_id, scope="session")
def mistral_tokenizer(request: pytest.FixtureRequest) -> MistralTokenizer:
    r"""Synthetic in-memory `MistralTokenizer` for one config of the Tekken matrix.

    Returns:
        A `MistralTokenizer` wrapping the instruct tokenizer of the parametrized `TestConfig`.
    """
    config: TestConfig = request.param
    return build_mistral_tokenizer_from_config(config)


@pytest.fixture(params=SHIPPED_TOKENIZER_KEYS, scope="session")
def shipped_mistral_tokenizer(request: pytest.FixtureRequest) -> MistralTokenizer:
    r"""Shipped `MistralTokenizer` loaded from the packaged data files.

    Returns:
        The shipped `MistralTokenizer` for the parametrized key.
    """
    key: str = request.param
    return load_mistral_tokenizer(key)


@pytest.fixture(params=SHIPPED_TOKENIZER_KEYS, scope="session")
def shipped_instruct_tokenizer(request: pytest.FixtureRequest) -> InstructTokenizer:
    r"""Instruct tokenizer of a shipped `MistralTokenizer`.

    Returns:
        The shipped instruct tokenizer for the parametrized key.
    """
    key: str = request.param
    return load_mistral_tokenizer(key).instruct_tokenizer


@pytest.fixture(scope="session")
def keyed_mistral_tokenizer(request: pytest.FixtureRequest) -> MistralTokenizer:
    r"""`MistralTokenizer` for an explicit key, shipped or synthetic.

    Covers every key in `tests.utils.tokenizers.TOKENIZER_FACTORIES`, including the
    FIM-only synthetic versions and the keys used by the golden registry, so tests
    needing those never construct a tokenizer themselves.

    Returns:
        The `MistralTokenizer` for the parametrized key.
    """
    key: str = request.param
    return load_mistral_tokenizer(key)


@pytest.fixture(scope="session")
def instruct_tokenizer_factory() -> Callable[..., InstructTokenizer]:
    r"""Factory building synthetic instruct tokenizers with explicit model settings.

    Model settings vary per test case, so this is exposed as a factory rather than a
    fixture per variant. Results are cached, so a given combination is built once.

    Returns:
        A callable taking a `TestConfig` and the accepted reasoning efforts (or `None`
        for a tokenizer that ignores model settings), returning an `InstructTokenizer`.
    """

    @lru_cache(maxsize=None)
    def factory(
        config: TestConfig,
        reasoning_efforts: tuple[ReasoningEffort, ...] | None = None,
        use_default: bool = True,
    ) -> InstructTokenizer:
        return build_instruct_tokenizer_from_config(
            config,
            model_settings_builder=build_model_settings_builder(reasoning_efforts, use_default=use_default),
        )

    return factory


@pytest.fixture(scope="session")
def mistral_tokenizer_factory() -> Callable[..., MistralTokenizer]:
    r"""Factory building synthetic `MistralTokenizer`s with explicit model settings.

    Returns:
        A callable taking a `TestConfig` and the accepted reasoning efforts (or `None`
        for a tokenizer that ignores model settings), returning a `MistralTokenizer`.
    """

    @lru_cache(maxsize=None)
    def factory(
        config: TestConfig,
        reasoning_efforts: tuple[ReasoningEffort, ...] | None = None,
        use_default: bool = True,
    ) -> MistralTokenizer:
        return build_mistral_tokenizer_from_config(
            config,
            model_settings_builder=build_model_settings_builder(reasoning_efforts, use_default=use_default),
        )

    return factory
