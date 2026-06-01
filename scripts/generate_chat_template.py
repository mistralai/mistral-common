import argparse

from mistral_common.integrations.chat_templates.chat_templates import generate_chat_template
from mistral_common.tokens.tokenizers.base import TokenizerVersion


def main() -> None:
    r"""Generate a chat template and save it to a file."""
    parser = argparse.ArgumentParser(description="Generate a Mistral chat template")
    parser.add_argument("--version", type=str, required=True, help="Tokenizer version (e.g., v1, v3, v7, v15)")
    parser.add_argument("--spm", action="store_true", help="Use SentencePiece tokenizer")
    parser.add_argument("--image", action="store_true", help="Enable image support")
    parser.add_argument("--audio", action="store_true", help="Enable audio support")
    parser.add_argument("--thinking", action="store_true", help="Enable thinking support (special tokens)")
    parser.add_argument("--plain_thinking", action="store_true", help="Enable plain text thinking (<think> tags)")
    parser.add_argument("--default_system_prompt", type=str, default=None, help="Default system prompt to embed")
    parser.add_argument("--saving_path", type=str, default="./chat_template.jinja", help="Output path for the template")
    parser.add_argument(
        "--no_special_token_variables",
        action="store_true",
        help="Embed literal BOS/EOS values instead of using bos_token/eos_token variables",
    )
    args = parser.parse_args()

    template = generate_chat_template(
        spm=args.spm,
        tokenizer_version=TokenizerVersion(args.version),
        image_support=args.image,
        audio_support=args.audio,
        thinking_support=args.thinking,
        default_system_prompt=args.default_system_prompt,
        plain_thinking_support=args.plain_thinking,
        use_special_token_variables=not args.no_special_token_variables,
    )

    with open(args.saving_path, "w", encoding="utf-8") as f:
        f.write(template)

    print(f"Template saved to {args.saving_path}")


if __name__ == "__main__":
    main()
