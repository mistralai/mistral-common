r"""Propose expected manifests for public chat, transcription and speech success cases.

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
from typing import Any, Protocol

import numpy as np

from mistral_common.protocol.speech.request import SpeechRequest
from mistral_common.protocol.transcription.request import TranscriptionRequest
from mistral_common.tokens.tokenizers.base import Tokenized
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.integration.chat_cases import PublicChatSuccessCase
from tests.integration.chat_registry import ALL_SUCCESS_CASES as CHAT_SUCCESS_CASES
from tests.integration.expected_results import _EXPECTED_ROOT
from tests.integration.speech_cases import SPEECH_SUCCESS_CASES, PublicSpeechSuccessCase
from tests.integration.transcription_cases import TRANSCRIPTION_SUCCESS_CASES, PublicTranscriptionSuccessCase
from tests.utils import decode_keep


class _RecipeLike(Protocol):
    """Structural type for the recipe bound to a success case."""

    def build(self) -> Any: ...


_ALL_SUCCESS_CASES: tuple[PublicChatSuccessCase | PublicTranscriptionSuccessCase | PublicSpeechSuccessCase, ...] = (
    *CHAT_SUCCESS_CASES,
    *TRANSCRIPTION_SUCCESS_CASES,
    *SPEECH_SUCCESS_CASES,
)


def _encode(
    case: PublicChatSuccessCase | PublicTranscriptionSuccessCase | PublicSpeechSuccessCase,
    tokenizer: MistralTokenizer,
) -> tuple[Tokenized, str]:
    request = case.recipe.build()
    if isinstance(request, TranscriptionRequest):
        tokenized = tokenizer.encode_transcription(request)
    elif isinstance(request, SpeechRequest):
        tokenized = tokenizer.encode_speech_request(request)
    else:
        tokenized = tokenizer.encode_chat_completion(request)
    return tokenized, decode_keep(tokenizer=tokenizer, tokenized=tokenized)


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


def propose_manifest(
    case: PublicChatSuccessCase | PublicTranscriptionSuccessCase | PublicSpeechSuccessCase, *, force: bool
) -> None:
    r"""Generate one case's manifest and sidecars from the current tokenizer.

    Args:
        case: The success case to propose.
        force: Overwrite an existing manifest.
    """
    case_dir = _EXPECTED_ROOT / case.case_id
    manifest_path = case_dir / "expected.json"
    if manifest_path.exists():
        if not force:
            print(f"exists {case.case_id}: {manifest_path} (pass --force to regenerate)")
            return

    tokenizer = case.configuration.load()
    tokenized, decoded_text = _encode(case=case, tokenizer=tokenizer)

    images = [np.asarray(image) for image in tokenized.images]
    audios = [np.asarray(audio.audio_array) for audio in tokenized.audios]
    audio_entries: list[dict[str, object]] = []
    for entry, audio in zip(
        _write_sidecars(case_dir=case_dir, prefix="audios", arrays=audios), tokenized.audios, strict=True
    ):
        entry["sampling_rate"] = audio.sampling_rate
        entry["format"] = audio.format
        audio_entries.append(entry)

    manifest = {
        "case_id": case.case_id,
        "tokenizer_configuration_id": case.configuration.configuration_id,
        "token_ids": tokenized.tokens,
        "decoded_text": decoded_text,
        "images": _write_sidecars(case_dir=case_dir, prefix="images", arrays=images),
        "audios": audio_entries,
    }
    case_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(obj=manifest, indent=2) + "\n")
    print(f"proposed {case.case_id}: {manifest_path}")


def main() -> int:
    """Propose manifests for the selected public success cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", action="append", required=False, help="Case id to propose; repeatable.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing manifests.")
    args = parser.parse_args()

    selected = _ALL_SUCCESS_CASES
    if args.case_id:
        wanted = set(args.case_id)
        success_case_ids = {success.case_id for success in _ALL_SUCCESS_CASES}
        unknown = wanted - success_case_ids
        if unknown:
            raise SystemExit(f"Unknown case ids: {sorted(unknown)}")
        selected = tuple(success for success in _ALL_SUCCESS_CASES if success.case_id in wanted)

    for case in selected:
        propose_manifest(case=case, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
