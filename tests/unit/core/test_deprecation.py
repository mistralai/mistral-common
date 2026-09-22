import os.path
import warnings

import pytest

import mistral_common.deprecation
from mistral_common.deprecation import deprecated_import, warn_once


@pytest.fixture(autouse=True)
def clear_warned_keys() -> None:
    mistral_common.deprecation._warned_keys.clear()


def test_deprecated_import_returns_imported_symbol() -> None:
    assert deprecated_import("old.module", "os.path", "join", "99.0") is os.path.join


def test_deprecated_import_warns_once_per_old_symbol() -> None:
    with pytest.warns(DeprecationWarning, match=r"Importing join from old\.mod") as record:
        deprecated_import("old.mod", "os.path", "join", "99.0")
    assert len(record) == 1
    assert "Use os.path.join instead" in str(record[0].message)
    assert "Will be removed in 99.0" in str(record[0].message)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        deprecated_import("old.mod", "os.path", "join", "99.0")
    assert [warning for warning in caught if issubclass(warning.category, DeprecationWarning)] == []


def test_deprecated_import_distinguishes_symbols_and_reports_import_errors() -> None:
    with pytest.warns(DeprecationWarning, match="join"):
        deprecated_import("pkg.a", "os.path", "join", "1.0")
    with pytest.warns(DeprecationWarning, match="exists"):
        deprecated_import("pkg.b", "os.path", "exists", "1.0")
    with pytest.raises(AttributeError):
        deprecated_import("old", "os.path", "no_such_attr_xyz", "1.0")
    with pytest.raises(ModuleNotFoundError):
        deprecated_import("old", "no_such_module_xyz_abc", "Foo", "1.0")


def test_warn_once_warns_once_per_key() -> None:
    with pytest.warns(DeprecationWarning, match="something broke"):
        warn_once("k1", "something broke", DeprecationWarning, stacklevel=2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_once("k1", "something broke", DeprecationWarning, stacklevel=2)
    assert [warning for warning in caught if issubclass(warning.category, DeprecationWarning)] == []
    with pytest.warns(DeprecationWarning, match="different"):
        warn_once("k2", "different", DeprecationWarning, stacklevel=2)
