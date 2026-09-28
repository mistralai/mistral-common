r"""Tokenizer configurations for public chat workflow cases.

Each configuration names one artifact with a verified identity: bundled
files are checked against their recorded SHA-256, and pinned released files
are provisioned by ``scripts/provision_test_tokenizers.py`` and re-verified
on load. Loading always goes through the public ``MistralTokenizer.from_file``
entry point so cases exercise the same path production users take.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from mistral_common.protocol.instruct.validator import ValidationMode
from mistral_common.tokens.tokenizers.image import ImageEncoder
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUNDLED_DATA = _REPO_ROOT / "src" / "mistral_common" / "data"
_PINNED_CACHE = _REPO_ROOT / ".cache" / "test-tokenizers"


@dataclass(frozen=True)
class TokenizerConfiguration:
    """One identified tokenizer artifact bound to a validation mode."""

    configuration_id: str
    tokenizer_path: Path
    mode: ValidationMode
    sha256: str
    provenance: str
    post_load: Callable[[MistralTokenizer], None] | None = None

    def load(self) -> MistralTokenizer:
        r"""Load the public tokenizer after verifying the artifact bytes.

        Returns:
            The tokenizer loaded from the verified file in this configuration's mode.
        """
        data = self.tokenizer_path.read_bytes()
        actual_sha256 = hashlib.sha256(data).hexdigest()
        if actual_sha256 != self.sha256:
            raise ValueError(
                f"Tokenizer artifact mismatch for {self.configuration_id} at {self.tokenizer_path}: "
                f"expected sha256={self.sha256}, got sha256={actual_sha256}"
            )
        tokenizer = MistralTokenizer.from_file(tokenizer_filename=self.tokenizer_path, mode=self.mode)
        if self.post_load is not None:
            self.post_load(tokenizer)
        return tokenizer


def _bundled_spm_v1(mode: ValidationMode) -> TokenizerConfiguration:
    return TokenizerConfiguration(
        configuration_id=f"bundled-spm-v1-{mode.value}",
        tokenizer_path=_BUNDLED_DATA / "tokenizer.model.v1",
        mode=mode,
        sha256="dadfd56d766715c61d2ef780a525ab43b8e6da4de6865bda3d95fdef5e134055",
        provenance="bundled",
    )


def _bundled_spm_v2(mode: ValidationMode) -> TokenizerConfiguration:
    return TokenizerConfiguration(
        configuration_id=f"bundled-spm-v2-{mode.value}",
        tokenizer_path=_BUNDLED_DATA / "mistral_instruct_tokenizer_240216.model.v2",
        mode=mode,
        sha256="37f00374dea48658ee8f5d0f21895b9bc55cb0103939607c8185bfd1c6ca1f89",
        provenance="bundled",
    )


def _bundled_spm_v3(mode: ValidationMode) -> TokenizerConfiguration:
    return TokenizerConfiguration(
        configuration_id=f"bundled-spm-v3-{mode.value}",
        tokenizer_path=_BUNDLED_DATA / "mistral_instruct_tokenizer_240323.model.v3",
        mode=mode,
        sha256="9addc8bdce5988448ae81b729336f43a81262160ae8da760674badab9d4c7d33",
        provenance="bundled",
    )


def _pinned(profile_id: str, filename: str, sha256: str, mode: ValidationMode) -> TokenizerConfiguration:
    return TokenizerConfiguration(
        configuration_id=f"pinned-{profile_id}-{mode.value}",
        tokenizer_path=_PINNED_CACHE / filename,
        mode=mode,
        sha256=sha256,
        provenance="pinned-released",
    )


BUNDLED_SPM_V1_TEST = _bundled_spm_v1(ValidationMode.test)
BUNDLED_SPM_V2_TEST = _bundled_spm_v2(ValidationMode.test)
BUNDLED_SPM_V2_SERVING = _bundled_spm_v2(ValidationMode.serving)
BUNDLED_SPM_V2_FINETUNING = _bundled_spm_v2(ValidationMode.finetuning)
BUNDLED_SPM_V2_AGNOSTIC = _bundled_spm_v2(ValidationMode.agnostic)
BUNDLED_SPM_V3_TEST = _bundled_spm_v3(ValidationMode.test)


def _set_test_image_patch_size_2(tokenizer: MistralTokenizer) -> None:
    """Apply the test-only patch-size override to this configuration's own instance."""
    image_encoder = tokenizer.instruct_tokenizer.image_encoder
    assert isinstance(image_encoder, ImageEncoder)
    image_encoder.image_config.image_patch_size = 2


BUNDLED_TEKKEN_V3_TEXT_TEST = TokenizerConfiguration(
    configuration_id="bundled-tekken-v3-text-test",
    tokenizer_path=_BUNDLED_DATA / "tekken_240718.json",
    mode=ValidationMode.test,
    sha256="eccd1665d2e477697c33cb7f0daa6f6dfefc57a0a6bceb66d4be52952f827516",
    provenance="bundled",
)
BUNDLED_TEKKEN_V3_MM_TEST = TokenizerConfiguration(
    configuration_id="bundled-tekken-v3-mm-test",
    tokenizer_path=_BUNDLED_DATA / "tekken_240911.json",
    mode=ValidationMode.test,
    sha256="1948e2d48b0e7377f1bb5f1210f1ae5f984934e75713fc07e2452729b8365316",
    provenance="bundled",
)
# Bundled v3 multimodal bytes with a test-only image patch-size override; the
# mutation belongs to this configuration's own loaded instance, never to a
# shared tokenizer, and is not evidence of an unmodified released profile.
BUNDLED_TEKKEN_V3_MM_PATCH2_TEST = TokenizerConfiguration(
    configuration_id="bundled-tekken-v3-mm-patch2-test",
    tokenizer_path=_BUNDLED_DATA / "tekken_240911.json",
    mode=ValidationMode.test,
    sha256="1948e2d48b0e7377f1bb5f1210f1ae5f984934e75713fc07e2452729b8365316",
    provenance="modified-bundled",
    post_load=_set_test_image_patch_size_2,
)

# Pinned released profiles provisioned by scripts/provision_test_tokenizers.py.
PINNED_V7_IMAGE_TEST = _pinned(
    profile_id="v7-image",
    filename="v7-image.tekken.json",
    sha256="c604f35d1035f534519622c0ec83fed6184978d4fdee92a5bd2a50bc05438094",
    mode=ValidationMode.test,
)
PINNED_V7_AUDIO_TEST = _pinned(
    profile_id="v7-audio",
    filename="v7-audio.tekken.json",
    sha256="4aaf3836c2a5332f029ce85a7a62255c966f47b6797ef81dedd0ade9c862e4a8",
    mode=ValidationMode.test,
)
PINNED_V11_IMAGE_TEST = _pinned(
    profile_id="v11-image",
    filename="v11-image.tekken.json",
    sha256="6e2501687ccd0e1f30f36319eaf2b46958b897811e246cd8eb5d385b9e3de7d1",
    mode=ValidationMode.test,
)
PINNED_V13_TEXT_TEST = _pinned(
    profile_id="v13-text",
    filename="v13-text.tekken.json",
    sha256="93a2d5af491c61f0b8f5233a1c0b91e5edb7332bf6000038f06a9b3ab92bfe8d",
    mode=ValidationMode.test,
)
PINNED_V13_IMAGE_TEST = _pinned(
    profile_id="v13-image",
    filename="v13-image.tekken.json",
    sha256="600bb27946565481ecf51ba8aee252e49b9a68507866080ac9c30185bb312843",
    mode=ValidationMode.test,
)
PINNED_V15_IMAGE_SETTINGS_TEST = _pinned(
    profile_id="v15-image-settings",
    filename="v15-image-settings.tekken.json",
    sha256="b1272b956bd6edd2d2c674c76896c7661308c9e723997b0afb55ecb429cb5dc7",
    mode=ValidationMode.test,
)
