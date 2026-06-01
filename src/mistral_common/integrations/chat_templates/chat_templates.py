from mistral_common.integrations.chat_templates.template_generator import (
    TemplateConfig,
)
from mistral_common.integrations.chat_templates.template_generator import (
    build_chat_template as _build_chat_template,
)
from mistral_common.tokens.tokenizers.base import TokenizerVersion


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
