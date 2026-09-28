r"""Propose expected manifests for public chat success cases.

Runs each registered success case's fresh request through the public encode
entry point and writes ``tests/data/expected/<case-id>/expected.json`` with
optional ``.npy`` sidecars. A generated manifest is only a **proposal**: it
becomes a reviewed golden after a human compares it against the legacy
expectations and independent format or round-trip evidence.

Run from the repository root:

    ./.venv/bin/python -m tests.integration.manifest_generator --case-id <id> [--force]

The generator refuses to overwrite an existing manifest unless ``--force``
is passed, so reviewed goldens are never silently regenerated.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from tests.integration.chat_cases import SAMPLE_SUCCESS_CASES, PublicChatSuccessCase
from tests.integration.chat_v3_cases import V3_SUCCESS_CASES
from tests.integration.chat_v7_cases import V7_SUCCESS_CASES
from tests.integration.chat_v13_cases import V13_SUCCESS_CASES
from tests.integration.expected_results import _EXPECTED_ROOT
from tests.utils import decode_keep

ALL_SUCCESS_CASES: tuple[PublicChatSuccessCase, ...] = (
    *SAMPLE_SUCCESS_CASES,
    *V3_SUCCESS_CASES,
    *V7_SUCCESS_CASES,
    *V13_SUCCESS_CASES,
)


def _write_sidecars(case_dir: Path, prefix: str, arrays: list[np.ndarray]) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for index, array in enumerate(arrays):
        relative = Path(prefix) / f"{index}.npy"
        (case_dir / prefix).mkdir(parents=True, exist_ok=True)
        np.save(file=case_dir / relative, arr=array)
        entries.append(
            {
                "path": str(relative),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
        )
    return entries


def propose_manifest(case: PublicChatSuccessCase, *, force: bool) -> Path:
    r"""Generate one case's manifest and sidecars from the current tokenizer.

    Args:
        case: The success case to propose.
        force: Overwrite an existing manifest.

    Returns:
        The written manifest path.
    """
    case_dir = _EXPECTED_ROOT / case.case_id
    manifest_path = case_dir / "expected.json"
    if manifest_path.exists():
        if not force:
            print(f"exists {case.case_id}: {manifest_path} (pass --force to regenerate)")
            return manifest_path

    tokenizer = case.configuration.load()
    request = case.recipe.build()
    tokenized = tokenizer.encode_chat_completion(request)
    decoded_text = decode_keep(tokenizer=tokenizer, tokenized=tokenized)

    images = [np.asarray(image) for image in tokenized.images]
    audios = [np.asarray(audio.audio_array) for audio in tokenized.audios]
    audio_entries: list[dict[str, object]] = []
    for entry, audio in zip(_write_sidecars(case_dir, "audios", audios), tokenized.audios, strict=True):
        entry["sampling_rate"] = audio.sampling_rate
        entry["format"] = audio.format
        audio_entries.append(entry)

    manifest = {
        "case_id": case.case_id,
        "tokenizer_configuration_id": case.configuration.configuration_id,
        "token_ids": tokenized.tokens,
        "decoded_text": decoded_text,
        "images": _write_sidecars(case_dir, "images", images),
        "audios": audio_entries,
    }
    case_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"proposed {case.case_id}: {manifest_path}")
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    """Propose manifests for the selected public chat success cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", action="append", required=False, help="Case id to propose; repeatable.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing manifests.")
    args = parser.parse_args(argv)

    selected = ALL_SUCCESS_CASES
    if args.case_id:
        wanted = set(args.case_id)
        known = {case.case_id for case in ALL_SUCCESS_CASES}
        unknown = wanted - known
        if unknown:
            raise SystemExit(f"Unknown case ids: {sorted(unknown)}")
        selected = tuple(case for case in ALL_SUCCESS_CASES if case.case_id in wanted)

    for case in selected:
        propose_manifest(case, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
