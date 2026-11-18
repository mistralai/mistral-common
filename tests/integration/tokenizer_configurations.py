r"""Tokenizer configurations for public chat workflow cases.

Each configuration names one artifact with a verified identity: bundled
files are checked against their recorded SHA-256, and pinned released files
are provisioned by ``tests/integration/provision_tokenizers.py`` and re-verified
on load. Loading always goes through the public ``MistralTokenizer.from_file``
entry point so cases exercise the same path production users take.
"""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from mistral_common.protocol.instruct.normalize import InstructRequestNormalizerV13, get_normalizer
from mistral_common.protocol.instruct.request import ReasoningEffort
from mistral_common.protocol.instruct.validator import MistralRequestValidatorV13, ValidationMode, get_validator
from mistral_common.tokens.tokenizers.audio import (
    AudioConfig,
    AudioEncoder,
    AudioSpectrogramConfig,
    SpecialAudioIDs,
    TranscriptionFormat,
)
from mistral_common.tokens.tokenizers.base import SpecialTokens, TokenizerVersion
from mistral_common.tokens.tokenizers.image import ImageEncoder
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV7, InstructTokenizerV13, InstructTokenizerV15
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer, load_audio_encoder
from mistral_common.tokens.tokenizers.model_settings_builder import ModelSettingsBuilder, ReasoningEffortEnumBuilder
from mistral_common.tokens.tokenizers.tekken import SpecialTokenInfo, Tekkenizer
from tests.integration.provision_tokenizers import PINNED_TOKENIZERS
from tests.utils import get_special_tokens, quick_vocab

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUNDLED_DATA = _REPO_ROOT / "src" / "mistral_common" / "data"
_PINNED_CACHE = _REPO_ROOT / ".cache" / "test-tokenizers"


class Provenance(Enum):
    """Artifact source identity for one tokenizer configuration."""

    BUNDLED = "bundled"
    MODIFIED_BUNDLED = "modified-bundled"
    PINNED_RELEASED = "pinned-released"
    SYNTHETIC = "synthetic"


@dataclass(frozen=True)
class TokenizerConfiguration:
    """One identified tokenizer artifact bound to a validation mode."""

    configuration_id: str
    tokenizer_path: Path | None
    mode: ValidationMode
    sha256: str | None
    provenance: Provenance
    post_load: Callable[[MistralTokenizer], None] | None = None
    synthetic_loader: Callable[[ValidationMode], MistralTokenizer] | None = None

    def __post_init__(self) -> None:
        if self.provenance is Provenance.SYNTHETIC:
            if self.tokenizer_path is not None:
                raise ValueError(f"Synthetic tokenizer configuration is incomplete: {self.configuration_id}")
            if self.sha256 is not None:
                raise ValueError(f"Synthetic tokenizer configuration is incomplete: {self.configuration_id}")
            if self.synthetic_loader is None:
                raise ValueError(f"Synthetic tokenizer configuration is incomplete: {self.configuration_id}")
        else:
            if self.tokenizer_path is None:
                raise ValueError(f"File-backed tokenizer configuration is incomplete: {self.configuration_id}")
            if self.sha256 is None:
                raise ValueError(f"File-backed tokenizer configuration is incomplete: {self.configuration_id}")
            if self.synthetic_loader is not None:
                raise ValueError(f"File-backed tokenizer configuration is incomplete: {self.configuration_id}")

    def load(self) -> MistralTokenizer:
        r"""Load the public tokenizer after verifying the artifact bytes.

        Returns:
            The tokenizer loaded from the verified file in this configuration's mode.
        """
        if self.provenance is Provenance.SYNTHETIC:
            assert self.synthetic_loader is not None
            return self.synthetic_loader(self.mode)

        assert self.tokenizer_path is not None
        assert self.sha256 is not None
        if self.provenance is Provenance.PINNED_RELEASED and not self.tokenizer_path.is_file():
            raise FileNotFoundError(
                f"Pinned tokenizer configuration {self.configuration_id} is missing at {self.tokenizer_path}; "
                "run `uv run python tests/integration/provision_tokenizers.py` first"
            )
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


_BUNDLED_SPM_V1_FILENAME = "tokenizer.model.v1"
_BUNDLED_SPM_V1_SHA256 = "dadfd56d766715c61d2ef780a525ab43b8e6da4de6865bda3d95fdef5e134055"
_BUNDLED_SPM_V2_FILENAME = "mistral_instruct_tokenizer_240216.model.v2"
_BUNDLED_SPM_V2_SHA256 = "37f00374dea48658ee8f5d0f21895b9bc55cb0103939607c8185bfd1c6ca1f89"
_BUNDLED_SPM_V3_FILENAME = "mistral_instruct_tokenizer_240323.model.v3"
_BUNDLED_SPM_V3_SHA256 = "9addc8bdce5988448ae81b729336f43a81262160ae8da760674badab9d4c7d33"
_BUNDLED_SPM_V7_MM_FILENAME = "mistral_instruct_tokenizer_241114.model.v7m1"
_BUNDLED_SPM_V7_MM_SHA256 = "1b968b8dc352f42192367337c78ccc61e1eaddc6d641a579372d4f20694beb7a"


def _bundled_spm(*, profile_id: str, filename: str, sha256: str, mode: ValidationMode) -> TokenizerConfiguration:
    r"""Build a bundled SentencePiece configuration for one profile and mode.

    Args:
        profile_id: Version-specific portion of the configuration identity.
        filename: Bundled tokenizer filename.
        sha256: Expected digest of the tokenizer bytes.
        mode: Validation mode applied when loading the tokenizer.

    Returns:
        A configuration pinned to the selected bundled file.
    """
    return TokenizerConfiguration(
        configuration_id=f"bundled-spm-{profile_id}-{mode.value}",
        tokenizer_path=_BUNDLED_DATA / filename,
        mode=mode,
        sha256=sha256,
        provenance=Provenance.BUNDLED,
    )


def _pinned(*, profile_id: str, sha256: str, mode: ValidationMode) -> TokenizerConfiguration:
    return TokenizerConfiguration(
        configuration_id=f"pinned-{profile_id}-{mode.value}",
        tokenizer_path=_PINNED_CACHE / f"{profile_id}.tekken.json",
        mode=mode,
        sha256=sha256,
        provenance=Provenance.PINNED_RELEASED,
    )


# Single source of truth for pinned artifact digests: provision_tokenizers.PINNED_TOKENIZERS.
_PINNED_SHA256_BY_PROFILE: dict[str, str] = {pinned.profile_id: pinned.sha256 for pinned in PINNED_TOKENIZERS}


BUNDLED_SPM_V1_TEST = _bundled_spm(
    profile_id="v1", filename=_BUNDLED_SPM_V1_FILENAME, sha256=_BUNDLED_SPM_V1_SHA256, mode=ValidationMode.test
)
BUNDLED_SPM_V1_FINETUNING = _bundled_spm(
    profile_id="v1",
    filename=_BUNDLED_SPM_V1_FILENAME,
    sha256=_BUNDLED_SPM_V1_SHA256,
    mode=ValidationMode.finetuning,
)
BUNDLED_SPM_V2_TEST = _bundled_spm(
    profile_id="v2", filename=_BUNDLED_SPM_V2_FILENAME, sha256=_BUNDLED_SPM_V2_SHA256, mode=ValidationMode.test
)
BUNDLED_SPM_V2_SERVING = _bundled_spm(
    profile_id="v2",
    filename=_BUNDLED_SPM_V2_FILENAME,
    sha256=_BUNDLED_SPM_V2_SHA256,
    mode=ValidationMode.serving,
)
BUNDLED_SPM_V2_FINETUNING = _bundled_spm(
    profile_id="v2",
    filename=_BUNDLED_SPM_V2_FILENAME,
    sha256=_BUNDLED_SPM_V2_SHA256,
    mode=ValidationMode.finetuning,
)
BUNDLED_SPM_V3_TEST = _bundled_spm(
    profile_id="v3", filename=_BUNDLED_SPM_V3_FILENAME, sha256=_BUNDLED_SPM_V3_SHA256, mode=ValidationMode.test
)
BUNDLED_SPM_V3_FINETUNING = _bundled_spm(
    profile_id="v3",
    filename=_BUNDLED_SPM_V3_FILENAME,
    sha256=_BUNDLED_SPM_V3_SHA256,
    mode=ValidationMode.finetuning,
)
BUNDLED_SPM_V7_MM_TEST = _bundled_spm(
    profile_id="v7-mm",
    filename=_BUNDLED_SPM_V7_MM_FILENAME,
    sha256=_BUNDLED_SPM_V7_MM_SHA256,
    mode=ValidationMode.test,
)


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
    provenance=Provenance.BUNDLED,
)
BUNDLED_TEKKEN_V3_MM_TEST = TokenizerConfiguration(
    configuration_id="bundled-tekken-v3-mm-test",
    tokenizer_path=_BUNDLED_DATA / "tekken_240911.json",
    mode=ValidationMode.test,
    sha256="1948e2d48b0e7377f1bb5f1210f1ae5f984934e75713fc07e2452729b8365316",
    provenance=Provenance.BUNDLED,
)
# Bundled v3 multimodal bytes with a test-only image patch-size override; the
# mutation belongs to this configuration's own loaded instance, never to a
# shared tokenizer, and is not evidence of an unmodified released profile.
BUNDLED_TEKKEN_V3_MM_PATCH2_TEST = TokenizerConfiguration(
    configuration_id="bundled-tekken-v3-mm-patch2-test",
    tokenizer_path=_BUNDLED_DATA / "tekken_240911.json",
    mode=ValidationMode.test,
    sha256="1948e2d48b0e7377f1bb5f1210f1ae5f984934e75713fc07e2452729b8365316",
    provenance=Provenance.MODIFIED_BUNDLED,
    post_load=_set_test_image_patch_size_2,
)

# Pinned released profiles provisioned by tests/integration/provision_tokenizers.py.
PINNED_V7_IMAGE_TEST = _pinned(
    profile_id="v7-image",
    sha256=_PINNED_SHA256_BY_PROFILE["v7-image"],
    mode=ValidationMode.test,
)
PINNED_V7_AUDIO_TEST = _pinned(
    profile_id="v7-audio",
    sha256=_PINNED_SHA256_BY_PROFILE["v7-audio"],
    mode=ValidationMode.test,
)
PINNED_V7_IMAGE_FINETUNING = _pinned(
    profile_id="v7-image",
    sha256=_PINNED_SHA256_BY_PROFILE["v7-image"],
    mode=ValidationMode.finetuning,
)
PINNED_V7_IMAGE_SERVING = _pinned(
    profile_id="v7-image",
    sha256=_PINNED_SHA256_BY_PROFILE["v7-image"],
    mode=ValidationMode.serving,
)
PINNED_V7_AUDIO_FINETUNING = _pinned(
    profile_id="v7-audio",
    sha256=_PINNED_SHA256_BY_PROFILE["v7-audio"],
    mode=ValidationMode.finetuning,
)
PINNED_V7_AUDIO_SERVING = _pinned(
    profile_id="v7-audio",
    sha256=_PINNED_SHA256_BY_PROFILE["v7-audio"],
    mode=ValidationMode.serving,
)
PINNED_V7_REALTIME_TEST = _pinned(
    profile_id="v7-realtime",
    sha256=_PINNED_SHA256_BY_PROFILE["v7-realtime"],
    mode=ValidationMode.test,
)
PINNED_V11_IMAGE_TEST = _pinned(
    profile_id="v11-image",
    sha256=_PINNED_SHA256_BY_PROFILE["v11-image"],
    mode=ValidationMode.test,
)
PINNED_V11_IMAGE_FINETUNING = _pinned(
    profile_id="v11-image",
    sha256=_PINNED_SHA256_BY_PROFILE["v11-image"],
    mode=ValidationMode.finetuning,
)
PINNED_V13_TEXT_TEST = _pinned(
    profile_id="v13-text",
    sha256=_PINNED_SHA256_BY_PROFILE["v13-text"],
    mode=ValidationMode.test,
)
PINNED_V13_TEXT_FINETUNING = _pinned(
    profile_id="v13-text",
    sha256=_PINNED_SHA256_BY_PROFILE["v13-text"],
    mode=ValidationMode.finetuning,
)
PINNED_V13_IMAGE_TEST = _pinned(
    profile_id="v13-image",
    sha256=_PINNED_SHA256_BY_PROFILE["v13-image"],
    mode=ValidationMode.test,
)
PINNED_V15_IMAGE_SETTINGS_TEST = _pinned(
    profile_id="v15-image-settings",
    sha256=_PINNED_SHA256_BY_PROFILE["v15-image-settings"],
    mode=ValidationMode.test,
)
PINNED_V15_IMAGE_SETTINGS_FINETUNING = _pinned(
    profile_id="v15-image-settings",
    sha256=_PINNED_SHA256_BY_PROFILE["v15-image-settings"],
    mode=ValidationMode.finetuning,
)
PINNED_V15_IMAGE_JSON_SCHEMA_TEST = _pinned(
    profile_id="v15-image-json-schema",
    sha256=_PINNED_SHA256_BY_PROFILE["v15-image-json-schema"],
    mode=ValidationMode.test,
)


def _synthetic_audio_encoder(*, tekkenizer: Tekkenizer) -> AudioEncoder:
    r"""Attach the shared synthetic audio profile to a tokenizer.

    Args:
        tekkenizer: Synthetic tokenizer supplying the audio token IDs.

    Returns:
        An audio encoder with the chat-test synthetic configuration.
    """
    audio_config = AudioConfig(
        sampling_rate=24_000,
        frame_rate=12.5,
        encoding_config=AudioSpectrogramConfig(num_mel_bins=128, window_size=400, hop_length=160),
    )
    special_audio_ids = SpecialAudioIDs(
        audio=tekkenizer.get_special_token(SpecialTokens.audio.value),
        begin_audio=tekkenizer.get_special_token(SpecialTokens.begin_audio.value),
        streaming_pad=None,
        text_to_audio=None,
        audio_to_text=None,
    )
    return AudioEncoder(audio_config=audio_config, special_ids=special_audio_ids)


def _load_synthetic_v13_audio(mode: ValidationMode) -> MistralTokenizer:
    special_tokens = get_special_tokens(tokenizer_version=TokenizerVersion.v13, add_think=False, add_audio=True)
    tekkenizer = Tekkenizer(
        vocab=quick_vocab(extra_toks=[b"a", b"b", b"c", b"f", b"de"]),
        special_tokens=special_tokens,
        pattern=r".+",
        vocab_size=256 + 100,
        num_special_tokens=100,
        version=TokenizerVersion.v13,
    )
    audio_encoder = _synthetic_audio_encoder(tekkenizer=tekkenizer)
    instruct_tokenizer = InstructTokenizerV13(tokenizer=tekkenizer, audio_encoder=audio_encoder)
    return MistralTokenizer(
        instruct_tokenizer=instruct_tokenizer,
        validator=MistralRequestValidatorV13(mode=mode),
        request_normalizer=InstructRequestNormalizerV13.normalizer(),
    )


def _synthetic_v15_loader(
    *,
    reasoning_effort: tuple[tuple[ReasoningEffort, ...], ReasoningEffort] | None,
    add_audio: bool,
) -> Callable[[ValidationMode], MistralTokenizer]:
    r"""Capture immutable v15 settings for a lazy synthetic configuration.

    Args:
        reasoning_effort: Allowed effort values and their required default, or `None` for no settings builder.
        add_audio: Whether to attach the legacy-compatible audio encoder.

    Returns:
        A loader creating the synthetic tokenizer in the requested mode.
    """

    def load(mode: ValidationMode) -> MistralTokenizer:
        if reasoning_effort is None:
            settings_builder = ModelSettingsBuilder.none()
        else:
            allowed_reasoning_effort, default_reasoning_effort = reasoning_effort
            settings_builder = ModelSettingsBuilder(
                reasoning_effort=ReasoningEffortEnumBuilder(
                    values=list(allowed_reasoning_effort), default=default_reasoning_effort
                )
            )

        tekkenizer = Tekkenizer(
            vocab=quick_vocab(extra_toks=[b"a", b"b", b"c", b"f", b"de"]),
            special_tokens=get_special_tokens(
                tokenizer_version=TokenizerVersion.v15,
                add_audio=add_audio,
                add_think=not add_audio,
            ),
            pattern=r".+",
            vocab_size=256 + 100,
            num_special_tokens=100,
            version=TokenizerVersion.v15,
            model_settings_builder=settings_builder,
        )

        audio_encoder: AudioEncoder | None = None
        if add_audio:
            audio_encoder = _synthetic_audio_encoder(tekkenizer=tekkenizer)

        instruct_tokenizer = InstructTokenizerV15(tokenizer=tekkenizer, audio_encoder=audio_encoder)
        return MistralTokenizer(
            instruct_tokenizer=instruct_tokenizer,
            validator=get_validator(version=TokenizerVersion.v15, mode=mode),
            request_normalizer=get_normalizer(version=TokenizerVersion.v15, model_settings_builder=settings_builder),
        )

    return load


SYNTHETIC_V13_AUDIO_TEST = TokenizerConfiguration(
    configuration_id="synthetic-v13-audio-test",
    tokenizer_path=None,
    mode=ValidationMode.test,
    sha256=None,
    provenance=Provenance.SYNTHETIC,
    synthetic_loader=_load_synthetic_v13_audio,
)


SYNTHETIC_V15_REASONING_NONE_ONLY_TEST = TokenizerConfiguration(
    configuration_id="synthetic-v15-reasoning-none-only-test",
    tokenizer_path=None,
    mode=ValidationMode.test,
    sha256=None,
    provenance=Provenance.SYNTHETIC,
    synthetic_loader=_synthetic_v15_loader(
        reasoning_effort=((ReasoningEffort.none,), ReasoningEffort.none),
        add_audio=False,
    ),
)
SYNTHETIC_V15_NO_SETTINGS_TEST = TokenizerConfiguration(
    configuration_id="synthetic-v15-no-settings-test",
    tokenizer_path=None,
    mode=ValidationMode.test,
    sha256=None,
    provenance=Provenance.SYNTHETIC,
    synthetic_loader=_synthetic_v15_loader(reasoning_effort=None, add_audio=False),
)
SYNTHETIC_V15_ALL_EFFORTS_TEST = TokenizerConfiguration(
    configuration_id="synthetic-v15-all-efforts-test",
    tokenizer_path=None,
    mode=ValidationMode.test,
    sha256=None,
    provenance=Provenance.SYNTHETIC,
    synthetic_loader=_synthetic_v15_loader(
        reasoning_effort=(tuple(ReasoningEffort), ReasoningEffort.none),
        add_audio=False,
    ),
)
SYNTHETIC_V15_ALL_EFFORTS_DEFAULT_MAX_TEST = TokenizerConfiguration(
    configuration_id="synthetic-v15-all-efforts-default-max-test",
    tokenizer_path=None,
    mode=ValidationMode.test,
    sha256=None,
    provenance=Provenance.SYNTHETIC,
    synthetic_loader=_synthetic_v15_loader(
        reasoning_effort=(tuple(ReasoningEffort), ReasoningEffort.max),
        add_audio=False,
    ),
)
SYNTHETIC_V15_AUDIO_TEST = TokenizerConfiguration(
    configuration_id="synthetic-v15-audio-test",
    tokenizer_path=None,
    mode=ValidationMode.test,
    sha256=None,
    provenance=Provenance.SYNTHETIC,
    synthetic_loader=_synthetic_v15_loader(
        reasoning_effort=(tuple(ReasoningEffort), ReasoningEffort.none),
        add_audio=True,
    ),
)


def _synthetic_v7_audio_loader(
    *,
    streaming: bool,
    omit_streaming_pad: bool = False,
    omit_text_to_audio: bool = False,
) -> Callable[[ValidationMode], MistralTokenizer]:
    r"""Capture immutable v7 audio profile settings for a lazy synthetic configuration.

    Args:
        streaming: Whether the audio config uses the streaming transcription format.
        omit_streaming_pad: Whether to omit the streaming_pad marker.
        omit_text_to_audio: Whether to omit the text_to_audio marker.

    Returns:
        A loader creating the synthetic v7 audio tokenizer in the requested mode.
    """

    def load(mode: ValidationMode) -> MistralTokenizer:
        special_tokens = get_special_tokens(tokenizer_version=TokenizerVersion.v7, add_audio=True)
        if streaming:
            special_tokens += [
                SpecialTokenInfo(rank=37, token_str="<SPCECIAL_37>", is_control=True),
                SpecialTokenInfo(rank=38, token_str="<SPCECIAL_38>", is_control=True),
                SpecialTokenInfo(rank=39, token_str=SpecialTokens.streaming_pad.value, is_control=True),
                SpecialTokenInfo(rank=40, token_str=SpecialTokens.streaming_word.value, is_control=True),
            ]
        if omit_streaming_pad:
            special_tokens = [
                SpecialTokenInfo(rank=token["rank"], token_str=f"<OMITTED_{token['rank']}>", is_control=True)
                if token["token_str"] == SpecialTokens.streaming_pad.value
                else token
                for token in special_tokens
            ]
        if omit_text_to_audio:
            special_tokens = [
                SpecialTokenInfo(rank=token["rank"], token_str=f"<OMITTED_{token['rank']}>", is_control=True)
                if token["token_str"] == SpecialTokens.text_to_audio.value
                else token
                for token in special_tokens
            ]
        audio_config = AudioConfig(
            sampling_rate=24_000,
            frame_rate=12.5,
            encoding_config=AudioSpectrogramConfig(num_mel_bins=128, window_size=400, hop_length=160),
            transcription_format=TranscriptionFormat.STREAMING if streaming else TranscriptionFormat.INSTRUCT,
            transcription_delay_ms=480.0 if streaming else None,
            streaming_look_ahead_ms=2.5 if streaming else None,
            streaming_look_back_ms=52.5 if streaming else None,
            streaming_n_left_pad_tokens=16 if streaming else None,
            voice_num_audio_tokens={"preset": 5},
        )
        tekkenizer = Tekkenizer(
            vocab=quick_vocab(extra_toks=[b"synthetic"]),
            special_tokens=special_tokens,
            pattern=r".+",
            vocab_size=356,
            num_special_tokens=100,
            version=TokenizerVersion.v7,
            audio_config=audio_config,
        )
        audio_encoder = load_audio_encoder(audio_config=audio_config, tokenizer=tekkenizer)
        instruct_tokenizer = InstructTokenizerV7(tokenizer=tekkenizer, audio_encoder=audio_encoder)
        return MistralTokenizer(
            instruct_tokenizer=instruct_tokenizer,
            validator=get_validator(version=TokenizerVersion.v7, mode=mode),
            request_normalizer=get_normalizer(version=TokenizerVersion.v7),
        )

    return load


SYNTHETIC_V7_INSTRUCT_AUDIO_TEST = TokenizerConfiguration(
    configuration_id="synthetic-v7-instruct-audio-test",
    tokenizer_path=None,
    mode=ValidationMode.test,
    sha256=None,
    provenance=Provenance.SYNTHETIC,
    synthetic_loader=_synthetic_v7_audio_loader(streaming=False),
)
SYNTHETIC_V7_STREAMING_AUDIO_TEST = TokenizerConfiguration(
    configuration_id="synthetic-v7-streaming-audio-test",
    tokenizer_path=None,
    mode=ValidationMode.test,
    sha256=None,
    provenance=Provenance.SYNTHETIC,
    synthetic_loader=_synthetic_v7_audio_loader(streaming=True),
)
SYNTHETIC_V7_STREAMING_NO_PAD_AUDIO_TEST = TokenizerConfiguration(
    configuration_id="synthetic-v7-streaming-no-pad-audio-test",
    tokenizer_path=None,
    mode=ValidationMode.test,
    sha256=None,
    provenance=Provenance.SYNTHETIC,
    synthetic_loader=_synthetic_v7_audio_loader(streaming=True, omit_streaming_pad=True),
)
SYNTHETIC_V7_SPEECH_AUDIO_TEST = TokenizerConfiguration(
    configuration_id="synthetic-v7-speech-audio-test",
    tokenizer_path=None,
    mode=ValidationMode.test,
    sha256=None,
    provenance=Provenance.SYNTHETIC,
    synthetic_loader=_synthetic_v7_audio_loader(streaming=False),
)
SYNTHETIC_V7_SPEECH_NO_MARKER_AUDIO_TEST = TokenizerConfiguration(
    configuration_id="synthetic-v7-speech-no-marker-audio-test",
    tokenizer_path=None,
    mode=ValidationMode.test,
    sha256=None,
    provenance=Provenance.SYNTHETIC,
    synthetic_loader=_synthetic_v7_audio_loader(streaming=False, omit_text_to_audio=True),
)
