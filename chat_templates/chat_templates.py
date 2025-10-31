import argparse
from pathlib import Path

import regex as re  # type: ignore[import-untyped]

from mistral_common.tokens.tokenizers.base import TokenizerVersion

_TEMPLATE_PATH = Path(__file__).parent / "templates"

V1 = _TEMPLATE_PATH / "v1.jinja"
V2 = _TEMPLATE_PATH / "v2.jinja"
V3 = _TEMPLATE_PATH / "v3.jinja"
V3_IMAGE = _TEMPLATE_PATH / "v3_image.jinja"
V7 = _TEMPLATE_PATH / "v7.jinja"
V7_IMAGE = _TEMPLATE_PATH / "v7_image.jinja"
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
            r"{%- set default_system_message = '()' %}", default_system_prompt, chat_template
        )
        assert isinstance(chat_template_with_default_sp, str), type(chat_template_with_default_sp)
        return chat_template_with_default_sp
    else:
        return path.read_text()


def get_chat_template(
    tokenizer_version: TokenizerVersion,
    image_support: bool,
    audio_support: bool,
    thinking_support: bool,
    default_system_prompt: str | None = None,
) -> str:
    if tokenizer_version == TokenizerVersion.v1:
        return _load_chat_template(V1, default_system_prompt)
    elif tokenizer_version == TokenizerVersion.v2:
        return _load_chat_template(V2, default_system_prompt)
    elif tokenizer_version == TokenizerVersion.v3:
        if image_support:
            return _load_chat_template(V3_IMAGE, default_system_prompt)
        else:
            return _load_chat_template(V3, default_system_prompt)
    elif tokenizer_version == TokenizerVersion.v7:
        if image_support:
            return _load_chat_template(V7_IMAGE, default_system_prompt)
        elif audio_support:
            return _load_chat_template(V7_AUDIO, default_system_prompt)
        else:
            return _load_chat_template(V7, default_system_prompt)
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
    parser.add_argument("--image", type=bool, default=False, help="Image support")
    parser.add_argument("--audio", type=bool, default=False, help="Audio support")
    parser.add_argument("--thinking", type=bool, default=False, help="Thinking support")
    parser.add_argument("--default_system_prompt", type=str, required=False, default=None, help="Default system prompt")
    args = parser.parse_args()
    tokenizer_version = TokenizerVersion(args.version)
    image_support = args.image
    audio_support = args.audio
    thinking_support = args.thinking
    default_system_prompt = args.default_system_prompt

    print(
        get_chat_template(
            tokenizer_version=tokenizer_version,
            image_support=image_support,
            audio_support=audio_support,
            thinking_support=thinking_support,
            default_system_prompt=default_system_prompt,
        )
    )
