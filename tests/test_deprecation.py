import os.path
import warnings

import pytest

import mistral_common.deprecation
from mistral_common.deprecation import deprecated_import, warn_once


@pytest.fixture(autouse=True)
def _clear_warned_keys() -> None:
    mistral_common.deprecation._warned_keys.clear()


def test_deprecated_import_returns_correct_object() -> None:
    result = deprecated_import("old.module", "os.path", "join", "99.0")
    assert result is os.path.join


def test_deprecated_import_emits_deprecation_warning() -> None:
    with pytest.warns(DeprecationWarning, match=r"Importing join from old\.mod") as record:
        deprecated_import("old.mod", "os.path", "join", "99.0")

    assert len(record) == 1
    msg = str(record[0].message)
    assert "Use os.path.join instead" in msg
    assert "Will be removed in 99.0" in msg


def test_deprecated_import_warns_only_once() -> None:
    with pytest.warns(DeprecationWarning):
        deprecated_import("old.mod", "os.path", "join", "99.0")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        deprecated_import("old.mod", "os.path", "join", "99.0")

    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert dep_warnings == []


def test_deprecated_import_different_pairs_each_warn() -> None:
    with pytest.warns(DeprecationWarning, match="join"):
        deprecated_import("pkg.a", "os.path", "join", "1.0")

    with pytest.warns(DeprecationWarning, match="exists"):
        deprecated_import("pkg.b", "os.path", "exists", "1.0")


def test_deprecated_import_raises_attribute_error() -> None:
    with pytest.raises(AttributeError):
        deprecated_import("old", "os.path", "no_such_attr_xyz", "1.0")


def test_deprecated_import_raises_module_not_found_error() -> None:
    with pytest.raises(ModuleNotFoundError):
        deprecated_import("old", "no_such_module_xyz_abc", "Foo", "1.0")


def test_warn_once_emits_warning() -> None:
    with pytest.warns(DeprecationWarning, match="something broke"):
        warn_once("k1", "something broke", DeprecationWarning, stacklevel=2)


def test_warn_once_does_not_repeat() -> None:
    with pytest.warns(DeprecationWarning):
        warn_once("k1", "msg", DeprecationWarning, stacklevel=2)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_once("k1", "msg", DeprecationWarning, stacklevel=2)

    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert dep_warnings == []


def test_warn_once_different_keys_each_warn() -> None:
    with pytest.warns(DeprecationWarning, match="first"):
        warn_once("a", "first", DeprecationWarning, stacklevel=2)

    with pytest.warns(DeprecationWarning, match="second"):
        warn_once("b", "second", DeprecationWarning, stacklevel=2)


def test_warn_once_respects_custom_category() -> None:
    with pytest.warns(FutureWarning, match="future thing"):
        warn_once("fw", "future thing", FutureWarning, stacklevel=2)
