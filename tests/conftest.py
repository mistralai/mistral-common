r"""Root conftest for the mistral-common test suite.

Provides parametrized tokenizer fixtures shared across test packages. These fixtures are
additive and have no autouse or global side effects.

`tekkenizer` and `spm` are parametrized over the full config matrix (all versions and
modalities). A plain `def test_x(tekkenizer)` therefore spans the whole matrix; narrow a
test to a subset with, e.g.:

    @pytest.mark.parametrize("tekkenizer", BASE_TEKKEN_CONFIGS, indirect=True, ids=config_id)
    def test_x(tekkenizer): ...

Structural fixtures that need a specific vocabulary layout (not a version/modality combo)
live next to the tests that use them.
"""

import pytest

from mistral_common.tokens.tokenizers.sentencepiece import SentencePieceTokenizer
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.utils.tokenizers import build_tekkenizer_from_config, load_sentencepiece
from tests.utils.versions import SPM_CONFIGS, TEKKEN_CONFIGS, TestConfig, config_id


@pytest.fixture(params=TEKKEN_CONFIGS, ids=config_id, scope="session")
def tekkenizer(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Tekkenizer:
    r"""Session-scoped `Tekkenizer` for one config of the full Tekken matrix.

    Returns:
        A `Tekkenizer` matching the parametrized `TestConfig`.
    """
    config: TestConfig = request.param
    output_dir = tmp_path_factory.mktemp("tekken") if (config.image or config.audio) else None
    return build_tekkenizer_from_config(config, output_dir=output_dir)


@pytest.fixture(params=SPM_CONFIGS, ids=config_id, scope="session")
def spm(request: pytest.FixtureRequest) -> SentencePieceTokenizer:
    r"""Session-scoped `SentencePieceTokenizer` for one config of the SPM matrix.

    Returns:
        A `SentencePieceTokenizer` matching the parametrized `TestConfig`.
    """
    config: TestConfig = request.param
    return load_sentencepiece(version=config.version.value)
