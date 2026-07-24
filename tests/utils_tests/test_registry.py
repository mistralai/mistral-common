from pathlib import Path

import numpy as np
import pytest

from tests.utils import registry


class TestLoadImageArrays:
    def test_load_image_arrays_preserves_order_beyond_ten_images(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(registry, "EXPECTED_DIR", tmp_path)
        key_dir = tmp_path / "instruct" / "test_key"
        key_dir.mkdir(parents=True)
        # 12 arrays: lexicographic key order ("arr_0", "arr_1", "arr_10", "arr_11",
        # "arr_2", ...) would misorder the array past index 9 if not sorted numerically.
        arrays = {f"arr_{i}": np.full((1, 1), i) for i in range(12)}
        np.savez_compressed(key_dir / "req.npz", **arrays)  # type: ignore[arg-type]

        loaded = registry.load_image_arrays(protocol="instruct", key="test_key", request_name="req")

        assert [int(arr.item()) for arr in loaded] == list(range(12))
