r"""Regenerate the golden expected-data registry.

Run manually and review the resulting diff in the PR; the test suite never invokes
this script, so goldens are never auto-overwritten during a normal run::

    uv run --no-sync python -m tests.utils.regenerate_registry

Every tokenizer, request, and selection rule this script writes comes from
`tests.utils.registry.SCENARIOS`, so the goldens it produces always match exactly what the
registry tests read back.
"""

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from tests.utils.registry import EXPECTED_DIR, PROTOCOL_ENCODERS, SCENARIOS, build_tokenizer, serialize_request


def _write_jsonl(path: Path, entries: Mapping[str, dict[str, object]]) -> None:
    r"""Write a name -> entry mapping as one JSON object per line, sorted by name.

    One line per scenario keeps a corpus of mostly-integer `token_ids` arrays navigable
    and diffable, unlike `json.dump(..., indent=2)`, which puts one integer per line.

    Args:
        path: Destination file path.
        entries: Mapping from request name to its `request`/`text`/`token_ids` entry.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for name in sorted(entries):
            record = {"name": name, **entries[name]}
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def main() -> None:
    r"""Regenerate every golden request, token ids, decoded text, and image array.

    Refusal scenarios (`scenario.raises is not None`) produce no output, so they get no
    line in `<key>.jsonl`; `TestRefusalScenarios` covers them by asserting the raise
    directly, never by reading a golden.
    """
    goldens: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
    for scenario in SCENARIOS:
        if scenario.raises is not None:
            continue
        tokenizer = build_tokenizer(scenario.key)
        request = scenario.build_request()
        encoded = PROTOCOL_ENCODERS[scenario.protocol](tokenizer, request)
        entries = goldens.setdefault((scenario.protocol, scenario.key), {})
        entries[scenario.name] = {
            "request": serialize_request(request),
            "text": encoded.text,
            "token_ids": encoded.tokens,
        }
        if scenario.has_images:
            key_dir = EXPECTED_DIR / scenario.protocol / scenario.key
            key_dir.mkdir(parents=True, exist_ok=True)
            arrays = {f"arr_{i}": np.asarray(img) for i, img in enumerate(encoded.images)}
            np.savez_compressed(key_dir / f"{scenario.name}.npz", **arrays)  # type: ignore[arg-type]

    written: set[Path] = set()
    for (protocol, key), entries in goldens.items():
        path = EXPECTED_DIR / protocol / f"{key}.jsonl"
        _write_jsonl(path, entries)
        written.add(path)
        written.update(
            EXPECTED_DIR / protocol / key / f"{name}.npz"
            for name in entries
            if (EXPECTED_DIR / protocol / key).is_dir()
        )

    pruned = _prune_orphans(written=written)
    print(f"Regenerated golden registry under {EXPECTED_DIR} ({len(pruned)} orphan(s) removed)")


def _prune_orphans(written: set[Path]) -> list[Path]:
    r"""Delete golden files the current matrix no longer produces.

    Without this a key or scenario dropped from the matrix leaves its golden behind forever,
    where nothing reads it and nothing notices it has gone stale.

    Args:
        written: Every path this run produced.

    Returns:
        The paths that were removed, sorted.
    """
    orphans = sorted(
        path
        for path in EXPECTED_DIR.rglob("*")
        if path.is_file() and path.suffix in {".jsonl", ".npz"} and path not in written
    )
    for path in orphans:
        path.unlink()
    for directory in sorted(EXPECTED_DIR.rglob("*"), reverse=True):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    return orphans


if __name__ == "__main__":
    main()
