import importlib
import warnings

_warned_keys: set[str] = set()


def deprecated_import(old_path: str, new_module: str, name: str, version: str) -> object:
    r"""Warn once and lazily import a symbol that moved to a new module.

    Args:
        old_path: The old module path (e.g. `"mistral_common.audio"`).
        new_module: The new module path (e.g. `"mistral_common.tokens.tokenizers.audio"`).
        name: The symbol name (e.g. `"Audio"`).
        version: The version in which the symbol will be removed.

    Returns:
        The imported symbol from the new module.
    """
    key = f"import:{old_path}.{name}"
    if key not in _warned_keys:
        _warned_keys.add(key)
        warnings.warn(
            f"Importing {name} from {old_path} is deprecated. "
            f"Use {new_module}.{name} instead. "
            f"Will be removed in {version}.",
            DeprecationWarning,
            stacklevel=3,
        )
    mod = importlib.import_module(new_module)
    return getattr(mod, name)


def warn_once(key: str, message: str, category: type[Warning], stacklevel: int) -> None:
    r"""Emit a warning only on the first call for a given key.

    Args:
        key: Unique identifier for this warning.
        message: The warning message.
        category: The warning category class.
        stacklevel: Stack level for the warning.
    """
    if key not in _warned_keys:
        _warned_keys.add(key)
        warnings.warn(message, category, stacklevel=stacklevel)
