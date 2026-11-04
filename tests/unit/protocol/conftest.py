import numpy as np
import pytest


@pytest.fixture
def audio_samples() -> np.ndarray:
    r"""Supply distinct mono samples for in-memory request conversions."""
    return np.tile(np.array([0.0, 0.25, -0.5, 0.75]), 100)
