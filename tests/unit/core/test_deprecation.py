import os.path
import warnings

import pytest

import mistral_common.deprecation
from mistral_common.deprecation import deprecated_import, warn_once


@pytest.fixture(autouse=True)
def _clear_warned_keys() -> None:
    mistral_common.deprecation._warned_keys.clear()


def test_deprecated_import_different_pairs_each_warn() -> None:
    with pytest.warns(DeprecationWarning, match="join"):
        deprecated_import("old.one", "os.path", "join", "1.0")
    with pytest.warns(DeprecationWarning, match="exists"):
        deprecated_import("old.two", "os.path", "exists", "1.0")


def test_deprecated_import_emits_deprecation_warning() -> None:
    with pytest.warns(DeprecationWarning, match=r"Importing join from old\.module") as record:
        deprecated_import("old.module", "os.path", "join", "99.0")
    assert len(record) == 1
    assert "Use os.path.join instead" in str(record[0].message)
    assert "Will be removed in 99.0" in str(record[0].message)


def test_deprecated_import_raises_attribute_error() -> None:
    with pytest.raises(AttributeError):
        deprecated_import("old", "os.path", "no_such_attr_xyz", "1.0")


def test_deprecated_import_raises_module_not_found_error() -> None:
    with pytest.raises(ModuleNotFoundError):
        deprecated_import("old", "no_such_module_xyz_abc", "Foo", "1.0")


def test_deprecated_import_returns_correct_object() -> None:
    assert deprecated_import("old.module", "os.path", "join", "99.0") is os.path.join


def test_deprecated_import_warns_only_once() -> None:
    with pytest.warns(DeprecationWarning):
        deprecated_import("old.module", "os.path", "join", "99.0")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        deprecated_import("old.module", "os.path", "join", "99.0")
    assert [warning for warning in caught if issubclass(warning.category, DeprecationWarning)] == []


def test_warn_once_different_keys_each_warn() -> None:
    with pytest.warns(DeprecationWarning, match="first"):
        warn_once("first-key", "first", DeprecationWarning, stacklevel=2)
    with pytest.warns(DeprecationWarning, match="second"):
        warn_once("second-key", "second", DeprecationWarning, stacklevel=2)


def test_warn_once_does_not_repeat() -> None:
    with pytest.warns(DeprecationWarning):
        warn_once("key", "message", DeprecationWarning, stacklevel=2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_once("key", "message", DeprecationWarning, stacklevel=2)
    assert [warning for warning in caught if issubclass(warning.category, DeprecationWarning)] == []


def test_warn_once_emits_warning() -> None:
    with pytest.warns(UserWarning, match="message"):
        warn_once("key", "message", UserWarning, stacklevel=2)


def test_deprecated_import_warn_key_is_old_path_and_name() -> None:
    with pytest.warns(DeprecationWarning):
        deprecated_import("old.module", "os.path", "join", "99.0")
    assert mistral_common.deprecation._warned_keys == {"import:old.module.join"}


def _call_deprecated_import() -> object:
    return deprecated_import("old.module", "os.path", "join", "99.0")


def test_deprecated_import_warning_stacklevel() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _call_deprecated_import()
    assert len(caught) == 1
    assert caught[0].filename == __file__
