import builtins
from types import ModuleType
from typing import Any
from unittest.mock import Mock, patch

import pytest

import mistral_common.imports as imports


def test_package_predicate_and_assertion() -> None:
    with patch("importlib.util.find_spec", return_value=object()):
        assert imports.is_package_installed("package_name") is True
    with patch("importlib.util.find_spec", return_value=None):
        assert imports.is_package_installed("package_name") is False
    with patch("mistral_common.imports.is_package_installed", return_value=True):
        imports.assert_package_installed("package_name")
    with patch("mistral_common.imports.is_package_installed", return_value=False):
        with pytest.raises(ImportError, match="Package 'package_name' is required but not installed"):
            imports.assert_package_installed("package_name")
    with patch("mistral_common.imports.is_package_installed", return_value=False):
        with pytest.raises(ImportError, match="custom error"):
            imports.assert_package_installed("package_name", "custom error")


@pytest.mark.parametrize(
    ("predicate", "package"),
    [
        (imports.is_hf_hub_installed, "huggingface_hub"),
        (imports.is_jinja2_installed, "jinja2"),
        (imports.is_llguidance_installed, "llguidance"),
        (imports.is_sentencepiece_installed, "sentencepiece"),
        (imports.is_soundfile_installed, "soundfile"),
        (imports.is_soxr_installed, "soxr"),
    ],
)
def test_dependency_predicates_preserve_find_spec_result(predicate: Any, package: str) -> None:
    predicate.cache_clear()
    with patch("mistral_common.imports.is_package_installed", return_value=True) as check:
        assert predicate() is True
        check.assert_called_once_with(package)
    predicate.cache_clear()
    with patch("mistral_common.imports.is_package_installed", return_value=False):
        assert predicate() is False
    predicate.cache_clear()


def test_opencv_predicate_handles_success_missing_and_broken_imports() -> None:
    imports.is_opencv_installed.cache_clear()
    with patch.dict("sys.modules", {"cv2": Mock()}):
        assert imports.is_opencv_installed() is True
    imports.is_opencv_installed.cache_clear()
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> ModuleType:
        if name == "cv2":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        assert imports.is_opencv_installed() is False
    imports.is_opencv_installed.cache_clear()

    def broken_import(name: str, *args: Any, **kwargs: Any) -> ModuleType:
        if name == "cv2":
            raise RuntimeError("broken")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=broken_import), patch.object(imports.logger, "warning") as warning:
        assert imports.is_opencv_installed() is False
        warning.assert_called_once()
    imports.is_opencv_installed.cache_clear()


@pytest.mark.parametrize(
    ("predicate", "assertion", "package", "group"),
    [
        (imports.is_hf_hub_installed, imports.assert_hf_hub_installed, "huggingface_hub", "hf-hub"),
        (imports.is_jinja2_installed, imports.assert_jinja2_installed, "jinja2", "guidance"),
        (imports.is_llguidance_installed, imports.assert_llguidance_installed, "llguidance", "guidance"),
        (imports.is_opencv_installed, imports.assert_opencv_installed, "opencv", "opencv"),
        (imports.is_sentencepiece_installed, imports.assert_sentencepiece_installed, "sentencepiece", "sentencepiece"),
        (imports.is_soundfile_installed, imports.assert_soundfile_installed, "soundfile", "soundfile"),
        (imports.is_soxr_installed, imports.assert_soxr_installed, "soxr", "soxr"),
    ],
)
def test_dependency_assertions_include_install_hint(predicate: Any, assertion: Any, package: str, group: str) -> None:
    predicate.cache_clear()
    assertion.cache_clear()
    with patch("mistral_common.imports.is_package_installed", return_value=False):
        with pytest.raises(ImportError, match=rf"{package}.*mistral-common\[{group}\]"):
            assertion()
    predicate.cache_clear()
    assertion.cache_clear()
