r"""Provision pinned released tokenizer files for offline public-workflow tests.

Downloads the six released tokenizer artifacts selected by the approved public
chat workflow test specification at immutable Hugging Face revisions, verifies
their exact byte size and SHA-256, and stores them under a gitignored cache
directory. The step fails closed: a missing or mismatched file is an error,
never a skipped test case.

Usage:

    uv run python scripts/provision_test_tokenizers.py [--directory PATH]

Tests load the provisioned files from local paths without network access.
"""

import argparse
import hashlib
import sys
from pathlib import Path

import requests

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DIRECTORY = _REPO_ROOT / ".cache" / "test-tokenizers"
_HUB_RESOLVE_URL = "https://huggingface.co/{repo}/resolve/{revision}/{filename}"


class PinnedTokenizer:
    """One released tokenizer file pinned to an immutable revision."""

    def __init__(self, *, profile_id: str, repo: str, revision: str, filename: str, size: int, sha256: str) -> None:
        self.profile_id = profile_id
        self.repo = repo
        self.revision = revision
        self.filename = filename
        self.size = size
        self.sha256 = sha256

    @property
    def local_name(self) -> str:
        """File name used in the provisioned directory."""
        return f"{self.profile_id}.tekken.json"

    def local_path(self, directory: Path) -> Path:
        """Absolute path of the provisioned file inside ``directory``."""
        return directory / self.local_name

    def matches(self, data: bytes) -> bool:
        """Whether ``data`` has the pinned size and SHA-256 digest."""
        return len(data) == self.size and hashlib.sha256(data).hexdigest() == self.sha256


# Selection recorded in the approved public-chat child plan (2026-09-28):
# one representative per distinct released version/capability profile.
PINNED_TOKENIZERS: list[PinnedTokenizer] = [
    PinnedTokenizer(
        profile_id="v7-image",
        repo="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        revision="68faf511d618ef198fef186659617cfd2eb8e33a",
        filename="tekken.json",
        size=14801330,
        sha256="c604f35d1035f534519622c0ec83fed6184978d4fdee92a5bd2a50bc05438094",
    ),
    PinnedTokenizer(
        profile_id="v7-audio",
        repo="mistralai/Voxtral-Mini-3B-2507",
        revision="3060fe34b35ba5d44202ce9ff3c097642914f8f3",
        filename="tekken.json",
        size=14894206,
        sha256="4aaf3836c2a5332f029ce85a7a62255c966f47b6797ef81dedd0ade9c862e4a8",
    ),
    PinnedTokenizer(
        profile_id="v11-image",
        repo="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        revision="95a6d26c4bfb886c58daf9d3f7332c857cb27b43",
        filename="tekken.json",
        size=19399895,
        sha256="6e2501687ccd0e1f30f36319eaf2b46958b897811e246cd8eb5d385b9e3de7d1",
    ),
    PinnedTokenizer(
        profile_id="v13-text",
        repo="mistralai/Magistral-Small-2507",
        revision="b3583a426ca23186681c4297926dfe9219be6161",
        filename="tekken.json",
        size=19399647,
        sha256="93a2d5af491c61f0b8f5233a1c0b91e5edb7332bf6000038f06a9b3ab92bfe8d",
    ),
    PinnedTokenizer(
        profile_id="v13-image",
        repo="mistralai/Ministral-3-3B-Instruct-2512",
        revision="b35d4dfe56c142746f54dbd64f579faab2744308",
        filename="tekken.json",
        size=16753784,
        sha256="600bb27946565481ecf51ba8aee252e49b9a68507866080ac9c30185bb312843",
    ),
    PinnedTokenizer(
        profile_id="v15-image-settings",
        repo="mistralai/Mistral-Small-4-119B-2603",
        revision="a11f36bebf709121056b1dbcc943d1c6afbe494d",
        filename="tekken.json",
        size=16275354,
        sha256="b1272b956bd6edd2d2c674c76896c7661308c9e723997b0afb55ecb429cb5dc7",
    ),
]


def _verify_pinned(pinned: PinnedTokenizer, data: bytes, *, source: str) -> None:
    r"""Fail when ``data`` does not match the pinned identity.

    Args:
        pinned: The expected artifact identity.
        data: The bytes read from ``source``.
        source: Human-readable location used in error messages.
    """
    actual_size = len(data)
    actual_sha256 = hashlib.sha256(data).hexdigest()
    if actual_size != pinned.size or actual_sha256 != pinned.sha256:
        raise ValueError(
            f"{pinned.profile_id} tokenizer mismatch at {source}: "
            f"expected size={pinned.size} sha256={pinned.sha256}, "
            f"got size={actual_size} sha256={actual_sha256}"
        )


def provision(pinned: PinnedTokenizer, *, directory: Path) -> None:
    r"""Ensure the pinned tokenizer file exists locally with verified bytes.

    Reuses an already provisioned file when its size and digest match;
    otherwise downloads it at the immutable revision and fails on any
    mismatch.

    Args:
        pinned: The artifact to provision.
        directory: Writable cache directory receiving the file.
    """
    target = pinned.local_path(directory)
    if target.exists():
        data = target.read_bytes()
        if pinned.matches(data):
            print(f"verified existing {pinned.profile_id}: {target}")
            return
        raise ValueError(
            f"{pinned.profile_id} exists at {target} but does not match the pinned bytes; "
            "delete it or pass a clean --directory"
        )

    url = _HUB_RESOLVE_URL.format(repo=pinned.repo, revision=pinned.revision, filename=pinned.filename)
    response = requests.get(url=url, timeout=120)
    response.raise_for_status()
    data = response.content
    _verify_pinned(pinned=pinned, data=data, source=url)

    directory.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    print(f"provisioned {pinned.profile_id}: {pinned.repo}@{pinned.revision} -> {target}")


def main(argv: list[str] | None = None) -> int:
    r"""Provision every pinned tokenizer or fail with a non-zero exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        default=_DEFAULT_DIRECTORY,
        help="Cache directory receiving the pinned tokenizer files (default: %(default)s).",
    )
    args = parser.parse_args(argv)

    for pinned in PINNED_TOKENIZERS:
        provision(pinned, directory=args.directory)
    return 0


if __name__ == "__main__":
    sys.exit(main())
