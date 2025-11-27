import argparse
from pathlib import Path

import regex as re  # type: ignore[import-untyped]

from mistral_common.tokens.tokenizers.base import TokenizerVersion

_TEMPLATE_PATH = Path(__file__).parent / "templates"

V1 = _TEMPLATE_PATH / "v1.jinja"
V1_SPM = _TEMPLATE_PATH / "v1_spm.jinja"
V2 = _TEMPLATE_PATH / "v2.jinja"
V2_SPM = _TEMPLATE_PATH / "v2_spm.jinja"
V3 = _TEMPLATE_PATH / "v3.jinja"
V3_SPM = _TEMPLATE_PATH / "v3_spm.jinja"
V3_IMAGE = _TEMPLATE_PATH / "v3_image.jinja"
V3_IMAGE_SPM = _TEMPLATE_PATH / "v3_image_spm.jinja"
V7 = _TEMPLATE_PATH / "v7.jinja"
V7_SPM = _TEMPLATE_PATH / "v7_spm.jinja"
V7_IMAGE = _TEMPLATE_PATH / "v7_image.jinja"
V7_IMAGE_SPM = _TEMPLATE_PATH / "v7_image_spm.jinja"
V7_AUDIO = _TEMPLATE_PATH / "v7_audio.jinja"
V11 = _TEMPLATE_PATH / "v11.jinja"
V11_IMAGE = _TEMPLATE_PATH / "v11_image.jinja"
V11_AUDIO = _TEMPLATE_PATH / "v11_audio.jinja"
V13 = _TEMPLATE_PATH / "v13.jinja"
V13_IMAGE = _TEMPLATE_PATH / "v13_image.jinja"
V13_IMAGE_THINK = _TEMPLATE_PATH / "v13_image_think.jinja"
V13_AUDIO = _TEMPLATE_PATH / "v13_audio.jinja"
V13_THINK = _TEMPLATE_PATH / "v13_think.jinja"


def _load_chat_template(path: Path, default_system_prompt: str | None) -> str:
    if default_system_prompt is not None:
        chat_template = path.read_text()
        chat_template_with_default_sp = re.sub(
            r"{%- set default_system_message = '()' %}",
            r"{%- set default_system_message = '" + default_system_prompt + r"' %}",
            chat_template,
        )
        assert isinstance(chat_template_with_default_sp, str), type(chat_template_with_default_sp)
        return chat_template_with_default_sp
    else:
        return path.read_text()


def get_chat_template(
    spm: bool,
    tokenizer_version: TokenizerVersion,
    image_support: bool,
    audio_support: bool,
    thinking_support: bool,
    default_system_prompt: str | None = None,
) -> str:
    if spm and (tokenizer_version >= TokenizerVersion.v11 or audio_support):
        raise ValueError("SPM tokenizer is not supported for tokenizer versions v11 and above or audio")
    if image_support and audio_support:
        raise ValueError("Image and audio support are mutually exclusive")
    if image_support and tokenizer_version < TokenizerVersion.v3:
        raise ValueError("Image support is only available for tokenizer versions v3 and above")
    if audio_support and tokenizer_version < TokenizerVersion.v7:
        raise ValueError("Audio support is only available for tokenizer versions v7 and above")
    if thinking_support and tokenizer_version < TokenizerVersion.v13:
        raise ValueError("Thinking support is only available for tokenizer versions v13 and above")

    if tokenizer_version == TokenizerVersion.v1:
        chat_path = V1_SPM if spm else V1
        return _load_chat_template(chat_path, default_system_prompt)
    elif tokenizer_version == TokenizerVersion.v2:
        chat_path = V2_SPM if spm else V2
        return _load_chat_template(chat_path, default_system_prompt)
    elif tokenizer_version == TokenizerVersion.v3:
        if image_support:
            chat_path = V3_IMAGE_SPM if spm else V3_IMAGE
        else:
            chat_path = V3_SPM if spm else V3
        return _load_chat_template(chat_path, default_system_prompt)
    elif tokenizer_version == TokenizerVersion.v7:
        if image_support:
            chat_path = V7_IMAGE_SPM if spm else V7_IMAGE
            return _load_chat_template(chat_path, default_system_prompt)
        elif audio_support:
            return _load_chat_template(V7_AUDIO, default_system_prompt)
        else:
            chat_path = V7_SPM if spm else V7
            return _load_chat_template(chat_path, default_system_prompt)
    elif tokenizer_version == TokenizerVersion.v11:
        if image_support:
            return _load_chat_template(V11_IMAGE, default_system_prompt)
        elif audio_support:
            return _load_chat_template(V11_AUDIO, default_system_prompt)
        else:
            return _load_chat_template(V11, default_system_prompt)
    elif tokenizer_version == TokenizerVersion.v13:
        if image_support and not thinking_support:
            return _load_chat_template(V13_IMAGE, default_system_prompt)
        elif image_support and thinking_support:
            return _load_chat_template(V13_IMAGE_THINK, default_system_prompt)
        elif audio_support:
            return _load_chat_template(V13_AUDIO, default_system_prompt)
        elif thinking_support:
            return _load_chat_template(V13_THINK, default_system_prompt)
        else:
            return _load_chat_template(V13, default_system_prompt)
    raise ValueError(
        f"Unknown configuration: tokenizer_version={tokenizer_version}, image_support={image_support}, "
        f"audio_support={audio_support}, thinking_support={thinking_support}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get chat template")
    parser.add_argument("--version", type=str, help="Tokenizer version")
    parser.add_argument("--spm", action="store_true", help="Use SPM tokenizer")
    parser.add_argument("--image", action="store_true", help="Image support")
    parser.add_argument("--audio", action="store_true", help="Audio support")
    parser.add_argument("--thinking", action="store_true", help="Thinking support")
    parser.add_argument("--default_system_prompt", type=str, required=False, default=None, help="Default system prompt")
    parser.add_argument(
        "--saving_path", type=str, required=False, default="./chat_template.jinja", help="Saving path for the template"
    )
    args = parser.parse_args()
    spm = args.spm
    tokenizer_version = TokenizerVersion(args.version)
    image_support = args.image
    audio_support = args.audio
    thinking_support = args.thinking
    default_system_prompt = args.default_system_prompt
    saving_path = args.saving_path

    with open(saving_path, "w") as f:
        f.write(
            get_chat_template(
                spm=spm,
                tokenizer_version=tokenizer_version,
                image_support=image_support,
                audio_support=audio_support,
                thinking_support=thinking_support,
                default_system_prompt=default_system_prompt,
            )
        )
