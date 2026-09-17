import importlib.util
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


def _get_dependency_error_message(package_name: str, dependency_group: str) -> str:
    r"""Build the install hint shown when an optional dependency is missing."""
    return f"`{package_name}` is not installed. Please install it with `pip install mistral-common[{dependency_group}]`"


def is_package_installed(package_name: str) -> bool:
    r"""Check whether a package is importable in the current environment."""
    return importlib.util.find_spec(package_name) is not None


def assert_package_installed(package_name: str, error_message: str | None = None) -> None:
    r"""Raise ImportError if the package is not importable.

    Args:
        package_name: The package to check.
        error_message: Custom error message. If `None`, a generic message is used.

    Raises:
        ImportError: If the package is not installed.
    """
    if not is_package_installed(package_name):
        error_message = error_message or f"Package '{package_name}' is required but not installed."
        raise ImportError(error_message)


@lru_cache()
def is_hf_hub_installed() -> bool:
    r"""Check whether the `huggingface_hub` package is installed."""

    return is_package_installed("huggingface_hub")


@lru_cache()
def is_jinja2_installed() -> bool:
    r"""Check whether the `jinja2` package is installed."""

    return is_package_installed("jinja2")


@lru_cache()
def is_llguidance_installed() -> bool:
    r"""Check whether the `llguidance` package is installed."""

    return is_package_installed("llguidance")


@lru_cache()
def is_opencv_installed() -> bool:
    r"""Check whether the `cv2` package is importable.

    Broken cv2 installs that raise on import are treated as not installed;
    a warning is logged instead.

    Returns:
        True if `cv2` imports cleanly, False otherwise."""

    try:
        import cv2  # noqa: F401
    except ImportError:
        _cv2_available = False
    except Exception as e:
        # cv2 has lots of import problems: https://github.com/opencv/opencv-python/issues/884
        # for better UX, let's simply skip all errors that might arise from import for now
        _cv2_available = False
        logger.warning(
            f"Warning: Your installation of OpenCV appears to be broken: {e}."
            "Please follow the instructions at https://github.com/opencv/opencv-python/issues/884 "
            "to correct your environment. The import of cv2 has been skipped."
        )
    else:
        _cv2_available = True
    return _cv2_available


@lru_cache()
def is_sentencepiece_installed() -> bool:
    r"""Check whether the `sentencepiece` package is installed."""

    return is_package_installed("sentencepiece")


@lru_cache()
def is_soundfile_installed() -> bool:
    r"""Check whether the `soundfile` package is installed."""

    return is_package_installed("soundfile")


@lru_cache()
def is_soxr_installed() -> bool:
    r"""Check whether the `soxr` package is installed."""

    return is_package_installed("soxr")


@lru_cache()
def assert_hf_hub_installed() -> None:
    r"""Raise ImportError if the dependency is missing, with an install hint."""

    assert_package_installed("huggingface_hub", _get_dependency_error_message("huggingface_hub", "hf-hub"))


@lru_cache()
def assert_jinja2_installed() -> None:
    r"""Raise ImportError if the dependency is missing, with an install hint."""

    assert_package_installed("jinja2", _get_dependency_error_message("jinja2", "guidance"))


@lru_cache()
def assert_llguidance_installed() -> None:
    r"""Raise ImportError if the dependency is missing, with an install hint."""

    assert_package_installed("llguidance", _get_dependency_error_message("llguidance", "guidance"))


@lru_cache()
def assert_opencv_installed() -> None:
    r"""Raise ImportError if the dependency is missing, with an install hint."""

    assert_package_installed("cv2", _get_dependency_error_message("opencv", "opencv"))


@lru_cache()
def assert_sentencepiece_installed() -> None:
    r"""Raise ImportError if the dependency is missing, with an install hint."""

    assert_package_installed("sentencepiece", _get_dependency_error_message("sentencepiece", "sentencepiece"))


@lru_cache()
def assert_soundfile_installed() -> None:
    r"""Raise ImportError if the dependency is missing, with an install hint."""

    assert_package_installed("soundfile", _get_dependency_error_message("soundfile", "soundfile"))


@lru_cache()
def assert_soxr_installed() -> None:
    r"""Raise ImportError if the dependency is missing, with an install hint."""

    assert_package_installed("soxr", _get_dependency_error_message("soxr", "soxr"))
