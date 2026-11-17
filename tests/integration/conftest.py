r"""Shared fixtures for public workflow integration tests.

Tokenizer artifacts are expensive to load, immutable during test execution
and verified at load time, so one session-scoped cache serves every case.
Under xdist the cache is per worker, which keeps the suite correct with
``--dist loadfile``. Requests are never cached: every case builds a fresh
one from its recipe.
"""

from collections.abc import Callable

import pytest

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.integration.tokenizer_configurations import TokenizerConfiguration


@pytest.fixture(scope="session")
def public_tokenizer() -> Callable[[TokenizerConfiguration], MistralTokenizer]:
    """Return a loader caching one verified tokenizer per configuration id."""

    cache: dict[str, MistralTokenizer] = {}

    def _load(configuration: TokenizerConfiguration) -> MistralTokenizer:
        tokenizer = cache.get(configuration.configuration_id)
        if tokenizer is None:
            tokenizer = configuration.load()
            cache[configuration.configuration_id] = tokenizer
        return tokenizer

    return _load
