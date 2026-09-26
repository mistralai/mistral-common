import hashlib
from dataclasses import dataclass

import numpy as np

from mistral_common.protocol.instruct.normalize import get_normalizer
from mistral_common.protocol.instruct.validator import ValidationMode, get_validator
from mistral_common.tokens.tokenizers.audio import Audio, AudioConfig, AudioSpectrogramConfig
from mistral_common.tokens.tokenizers.base import SpecialTokens, TokenizerVersion
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV7
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer, load_audio_encoder
from mistral_common.tokens.tokenizers.tekken import SpecialTokenInfo, Tekkenizer
from tests.test_tekken import get_special_tokens, quick_vocab

BUNDLED_V7_NO_AUDIO_CONFIGURATION_ID = "bundled-spm-v7-no-audio-test"
BUNDLED_V7_NO_AUDIO_SHA256 = "1b968b8dc352f42192367337c78ccc61e1eaddc6d641a579372d4f20694beb7a"


@dataclass(frozen=True)
class SyntheticV7AudioProfile:
    configuration_id: str
    unavailable_instruct_markers: tuple[str, ...]
    omitted_audio_markers: tuple[str, ...]
    voice_num_audio_tokens: tuple[tuple[str, int], ...] | None


SYNTHETIC_V7_INSTRUCT_NO_TRANSCRIBE = SyntheticV7AudioProfile(
    configuration_id="synthetic-v7-instruct-no-transcribe-test",
    unavailable_instruct_markers=(SpecialTokens.transcribe.value,),
    omitted_audio_markers=(),
    voice_num_audio_tokens=(("preset", 5),),
)
SYNTHETIC_V7_SPEECH_NO_MARKER = SyntheticV7AudioProfile(
    configuration_id="synthetic-v7-speech-no-marker-test",
    unavailable_instruct_markers=(),
    omitted_audio_markers=(SpecialTokens.text_to_audio.value,),
    voice_num_audio_tokens=(("preset", 5),),
)
SYNTHETIC_V7_SPEECH_NO_AUDIO_TO_TEXT = SyntheticV7AudioProfile(
    configuration_id="synthetic-v7-speech-no-audio-to-text-test",
    unavailable_instruct_markers=(),
    omitted_audio_markers=(SpecialTokens.audio_to_text.value,),
    voice_num_audio_tokens=(("preset", 5),),
)
SYNTHETIC_V7_SPEECH_NO_VOICE_MAP = SyntheticV7AudioProfile(
    configuration_id="synthetic-v7-speech-no-map-test",
    unavailable_instruct_markers=(),
    omitted_audio_markers=(),
    voice_num_audio_tokens=None,
)


def load_bundled_v7_no_audio_tokenizer(mode: ValidationMode) -> MistralTokenizer:
    path = MistralTokenizer._data_path() / "mistral_instruct_tokenizer_241114.model.v7"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == BUNDLED_V7_NO_AUDIO_SHA256

    tokenizer = MistralTokenizer.from_file(path, mode=mode)
    assert tokenizer.version == TokenizerVersion.v7
    assert tokenizer.mode == mode
    assert tokenizer.instruct_tokenizer.audio_encoder is None
    return tokenizer


def build_synthetic_v7_audio_tokenizer(profile: SyntheticV7AudioProfile, mode: ValidationMode) -> MistralTokenizer:
    special_tokens = get_special_tokens(tokenizer_version=TokenizerVersion.v7, add_audio=True)
    omitted_tokens = set(profile.unavailable_instruct_markers) | set(profile.omitted_audio_markers)
    special_tokens = [
        SpecialTokenInfo(
            rank=token["rank"],
            token_str=f"<OMITTED_{token['rank']}>",
            is_control=True,
        )
        if token["token_str"] in omitted_tokens
        else token
        for token in special_tokens
    ]
    audio_config = AudioConfig(
        sampling_rate=24_000,
        frame_rate=12.5,
        encoding_config=AudioSpectrogramConfig(num_mel_bins=128, window_size=400, hop_length=160),
        voice_num_audio_tokens=(
            dict(profile.voice_num_audio_tokens) if profile.voice_num_audio_tokens is not None else None
        ),
    )
    tekkenizer = Tekkenizer(
        vocab=quick_vocab([b"synthetic"]),
        special_tokens=special_tokens,
        pattern=r".+",
        vocab_size=356,
        num_special_tokens=100,
        version=TokenizerVersion.v7,
        audio_config=audio_config,
    )
    audio_encoder = load_audio_encoder(audio_config=audio_config, tokenizer=tekkenizer)
    if SpecialTokens.transcribe.value in profile.unavailable_instruct_markers:
        # The V7 constructor only binds TRANSCRIBE when an audio encoder is passed.
        instruct_tokenizer = InstructTokenizerV7(tokenizer=tekkenizer, audio_encoder=None)
        instruct_tokenizer.audio_encoder = audio_encoder
    else:
        instruct_tokenizer = InstructTokenizerV7(tokenizer=tekkenizer, audio_encoder=audio_encoder)
    return MistralTokenizer(
        instruct_tokenizer=instruct_tokenizer,
        validator=get_validator(version=TokenizerVersion.v7, mode=mode),
        request_normalizer=get_normalizer(version=TokenizerVersion.v7),
    )


def valid_reference_audio() -> str:
    audio = Audio(
        audio_array=np.zeros(shape=(2_400,), dtype=np.float32),
        sampling_rate=24_000,
        format="wav",
    )
    return audio.to_base64("wav")
