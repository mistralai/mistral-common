from pathlib import Path

from mistral_common.integrations.chat_templates.template_generator import (
    TemplateConfig,
)
from mistral_common.integrations.chat_templates.template_generator import (
    build_chat_template as _build_chat_template,
)
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.sentencepiece import SentencePieceTokenizer


def generate_chat_template(
    spm: bool,
    tokenizer_version: TokenizerVersion,
    image_support: bool,
    audio_support: bool,
    thinking_support: bool,
    default_system_prompt: str | None,
    plain_thinking_support: bool,
    use_special_token_variables: bool,
) -> str:
    r"""Generate a chat template based on configuration.

    Programmatically generates a Jinja2 chat template string that formats
    conversation messages for Mistral models. The generated template handles
    message roles, special tokens, tool calls, and multimodal content.

    Args:
        spm: Whether to use SentencePiece tokenizer.
        tokenizer_version: The tokenizer version.
        image_support: Whether to support image chunks.
        audio_support: Whether to support audio chunks.
        thinking_support: Whether to support thinking chunks with special tokens.
        default_system_prompt: Optional default system prompt to embed.
        plain_thinking_support: Whether to support thinking chunks with plain
            `<think>`/`</think>` text tags. Only available for v11.
            Mutually exclusive with `thinking_support`.
        use_special_token_variables: Whether to emit BOS/EOS as Jinja variable
            references (`bos_token`/`eos_token`) or as literal values.

    Returns:
        The generated Jinja2 template as a string.
    """
    config = TemplateConfig(
        version=tokenizer_version,
        spm=spm,
        image_support=image_support,
        audio_support=audio_support,
        thinking_support=thinking_support,
        plain_thinking_support=plain_thinking_support,
        use_special_token_variables=use_special_token_variables,
    )
    template = _build_chat_template(config)

    if default_system_prompt is not None:
        # Thinking chunks are not a concern here because default system prompts
        # are plain text strings, not structured content with chunk types.
        escaped_prompt = default_system_prompt.replace("\\", "\\\\").replace("'", "\\'")
        template = template.replace(
            "{%- set default_system_message = '' %}",
            "{%- set default_system_message = '" + escaped_prompt + "' %}",
        )

    return template


def convert_tokenizer_to_chat_template(
    tokenizer_file: str | Path,
    system_prompt: str | None = None,
    use_special_token_variables: bool = True,
) -> str:
    r"""Load a tokenizer file and auto-detect its capabilities to generate a matching chat template.

    Loads the tokenizer via `MistralTokenizer.from_file`, inspects the resulting
    instruct tokenizer to determine version, backend (SentencePiece vs Tekken),
    and supported modalities, then delegates to `generate_chat_template` with the
    detected flags.

    The `plain_thinking_support` flag is set heuristically: any v11 tokenizer uses
    plain `<think>`/`</think>` text tags instead of special `[THINK]`/`[/THINK]`
    tokens. `thinking_support` (special-token thinking) is detected by checking
    whether the instruct tokenizer exposes a non-`None` `BEGIN_THINK` attribute,
    which is only set on v13+ tokenizers that include think special tokens.

    Args:
        tokenizer_file: Path to the tokenizer file (tekken JSON or SentencePiece `.model.vX`).
        system_prompt: Optional default system prompt to embed in the template.
            When not `None`, maps to `generate_chat_template`'s `default_system_prompt`.
        use_special_token_variables: Whether to emit BOS/EOS as Jinja variable
            references (`bos_token`/`eos_token`) or as literal string values.

    Returns:
        The generated Jinja2 chat template string matching the tokenizer's capabilities.

    Raises:
        TokenizerException: If the tokenizer file is not recognized or invalid.
    """
    mt = MistralTokenizer.from_file(tokenizer_file)
    it = mt.instruct_tokenizer
    tok = it.tokenizer

    version = tok.version
    spm = isinstance(tok, SentencePieceTokenizer)
    image_support = it.image_encoder is not None
    audio_support = it.audio_encoder is not None
    thinking_support = getattr(it, "BEGIN_THINK", None) is not None
    plain_thinking_support = version == TokenizerVersion.v11

    return generate_chat_template(
        spm=spm,
        tokenizer_version=version,
        image_support=image_support,
        audio_support=audio_support,
        thinking_support=thinking_support,
        default_system_prompt=system_prompt,
        plain_thinking_support=plain_thinking_support,
        use_special_token_variables=use_special_token_variables,
    )
