from typing import Any

import pytest

from integrations.chat_templates.chat_templates import (
    generate_chat_template_dynamic,
    get_chat_template,
)
from integrations.chat_templates.template_generator import (
    TemplateConfig,
    generate_chat_template,
)
from mistral_common.tokens.tokenizers.base import TokenizerVersion


def render_template(
    template: str, messages: list[Any], tools: list[Any] | None = None, reasoning_effort: str | None = None
) -> str:
    """Render a jinja2 template with the given messages."""
    from jinja2 import BaseLoader
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(msg: str) -> None:
        raise ValueError(msg)

    env = ImmutableSandboxedEnvironment(loader=BaseLoader())
    env.globals["raise_exception"] = raise_exception
    jinja_template = env.from_string(template)

    render_kwargs = {
        "messages": messages,
        "tools": tools,
        "bos_token": "<s>",
        "eos_token": "</s>",
    }

    # Only add reasoning_effort for v15+ templates that support it
    if reasoning_effort is not None or "reasoning_effort" in template:
        render_kwargs["reasoning_effort"] = reasoning_effort

    return jinja_template.render(**render_kwargs)


# All configurations for output comparison tests (including SPM)
ALL_CONFIGS = [
    # Non-SPM
    (TokenizerVersion.v1, False, False, False, False),
    (TokenizerVersion.v2, False, False, False, False),
    (TokenizerVersion.v3, False, False, False, False),
    (TokenizerVersion.v3, False, True, False, False),
    (TokenizerVersion.v7, False, False, False, False),
    (TokenizerVersion.v7, False, True, False, False),
    (TokenizerVersion.v7, False, False, True, False),
    (TokenizerVersion.v11, False, False, False, False),
    (TokenizerVersion.v11, False, True, False, False),
    (TokenizerVersion.v11, False, False, True, False),
    (TokenizerVersion.v13, False, False, False, False),
    (TokenizerVersion.v13, False, True, False, False),
    (TokenizerVersion.v13, False, False, True, False),
    (TokenizerVersion.v13, False, False, False, True),
    (TokenizerVersion.v13, False, True, False, True),
    (TokenizerVersion.v15, False, False, False, False),
    (TokenizerVersion.v15, False, True, False, False),
    (TokenizerVersion.v15, False, False, True, False),
    (TokenizerVersion.v15, False, True, False, True),
    # SPM
    (TokenizerVersion.v1, True, False, False, False),
    (TokenizerVersion.v2, True, False, False, False),
    (TokenizerVersion.v3, True, False, False, False),
    (TokenizerVersion.v3, True, True, False, False),
    (TokenizerVersion.v7, True, False, False, False),
    (TokenizerVersion.v7, True, True, False, False),
]


@pytest.mark.parametrize(
    ("version", "spm", "image", "audio", "think"),
    ALL_CONFIGS,
)
def test_dynamic_template_produces_same_output(
    version: TokenizerVersion,
    spm: bool,
    image: bool,
    audio: bool,
    think: bool,
) -> None:
    """Verify dynamically generated templates produce same output as static templates."""
    # Get templates
    tokenizer_version = TokenizerVersion(version.value)
    static_template = get_chat_template(
        spm=spm,
        tokenizer_version=tokenizer_version,
        image_support=image,
        audio_support=audio,
        thinking_support=think,
    )

    config = TemplateConfig(
        version=version,
        spm=spm,
        image_support=image,
        audio_support=audio,
        thinking_support=think,
    )
    dynamic_template = generate_chat_template(config)

    # Test with a simple conversation
    messages: list[Any] = [
        {"role": "user", "content": "Hello"},
    ]

    # Only add assistant message for training validation
    if version != TokenizerVersion.v1:
        messages.append({"role": "assistant", "content": "Hi there!"})

    static_output = render_template(static_template, messages)
    dynamic_output = render_template(dynamic_template, messages)

    assert static_output == dynamic_output, (
        f"Output mismatch for version={version}, spm={spm}, image={image}, audio={audio}, think={think}\n\n"
        f"Static output: {static_output}\n"
        f"Dynamic output: {dynamic_output}"
    )


def test_invalid_config_spm_v11() -> None:
    """Test that SPM with v11+ raises error."""
    with pytest.raises(ValueError, match="SPM tokenizer is not supported"):
        TemplateConfig(version=TokenizerVersion.v11, spm=True)


def test_invalid_config_image_audio() -> None:
    """Test that image and audio together raises error."""
    with pytest.raises(ValueError, match="Image and audio support are mutually exclusive"):
        TemplateConfig(version=TokenizerVersion.v7, image_support=True, audio_support=True)


def test_invalid_config_image_v1() -> None:
    """Test that image support with v1/v2 raises error."""
    with pytest.raises(ValueError, match="Image support is only available"):
        TemplateConfig(version=TokenizerVersion.v1, image_support=True)


def test_invalid_config_audio_v3() -> None:
    """Test that audio support with v3 raises error."""
    with pytest.raises(ValueError, match="Audio support is only available"):
        TemplateConfig(version=TokenizerVersion.v3, audio_support=True)


def test_invalid_config_thinking_v7() -> None:
    """Test that thinking support with non-v13 raises error."""
    with pytest.raises(ValueError, match="Thinking support is only available"):
        TemplateConfig(version=TokenizerVersion.v7, thinking_support=True)


def test_invalid_config_audio_thinking() -> None:
    """Test that audio and thinking together raises error."""
    with pytest.raises(ValueError, match="Audio and thinking support are mutually exclusive"):
        TemplateConfig(version=TokenizerVersion.v13, audio_support=True, thinking_support=True)


def test_generate_chat_template_dynamic_function() -> None:
    """Test that generate_chat_template_dynamic function works correctly."""
    # Test basic generation
    template = generate_chat_template_dynamic(
        spm=False,
        tokenizer_version=TokenizerVersion.v13,
        image_support=False,
        audio_support=False,
        thinking_support=False,
    )
    assert "{%- set default_system_message = '' %}" in template
    assert "{{- '<s>' }}" in template

    # Test with default system prompt
    template_with_prompt = generate_chat_template_dynamic(
        spm=False,
        tokenizer_version=TokenizerVersion.v13,
        image_support=False,
        audio_support=False,
        thinking_support=False,
        default_system_prompt="You are a helpful assistant.",
    )
    assert "You are a helpful assistant." in template_with_prompt

    # Test that both static and dynamic produce same output
    static_template = get_chat_template(
        spm=False,
        tokenizer_version=TokenizerVersion.v13,
        image_support=False,
        audio_support=False,
        thinking_support=False,
    )

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    static_output = render_template(static_template, messages)
    dynamic_output = render_template(template, messages)

    assert static_output == dynamic_output


def test_v15_reasoning_effort() -> None:
    """Test that v15 templates correctly handle reasoning effort."""
    # Test v15 template generation
    template = generate_chat_template_dynamic(
        spm=False,
        tokenizer_version=TokenizerVersion.v15,
        image_support=False,
        audio_support=False,
        thinking_support=False,
    )

    # Verify the template contains MODEL_SETTINGS logic
    assert "[MODEL_SETTINGS]" in template
    assert "reasoning_effort" in template

    # Test rendering with different reasoning effort values
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    # Test with no reasoning effort (should default to 'none')
    output_none = render_template(template, messages, reasoning_effort=None)
    assert '[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]' in output_none

    # Test with reasoning effort='high'
    output_high = render_template(template, messages, reasoning_effort="high")
    assert '[MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS]' in output_high

    # Test with reasoning effort='none'
    output_none_explicit = render_template(template, messages, reasoning_effort="none")
    assert '[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]' in output_none_explicit

    # Test that v15 static and dynamic templates produce same output
    static_template = get_chat_template(
        spm=False,
        tokenizer_version=TokenizerVersion.v15,
        image_support=False,
        audio_support=False,
        thinking_support=False,
    )

    static_output = render_template(static_template, messages, reasoning_effort="high")
    dynamic_output = render_template(template, messages, reasoning_effort="high")

    assert static_output == dynamic_output


def test_v15_with_features() -> None:
    """Test that v15 templates work correctly with images and thinking."""
    # Test v15 with image support
    template_image = generate_chat_template_dynamic(
        spm=False,
        tokenizer_version=TokenizerVersion.v15,
        image_support=True,
        audio_support=False,
        thinking_support=False,
    )

    messages_with_image = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": "http://example.com/image.png"},
            ],
        },
        {"role": "assistant", "content": "It's an image."},
    ]

    output = render_template(template_image, messages_with_image, reasoning_effort="high")
    assert '[MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS]' in output
    assert "[IMG]" in output

    # Test v15 with thinking support
    template_think = generate_chat_template_dynamic(
        spm=False,
        tokenizer_version=TokenizerVersion.v15,
        image_support=False,
        audio_support=False,
        thinking_support=True,
    )

    messages_with_thinking = [
        {"role": "user", "content": "Solve this problem"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Let me think..."},
                {"type": "text", "text": "The answer is 42."},
            ],
        },
    ]

    output = render_template(template_think, messages_with_thinking, reasoning_effort="none")
    assert '[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]' in output
    assert "[THINK]Let me think...[/THINK]" in output


def test_v15_with_tools() -> None:
    """Test that v15 templates work correctly with tools."""
    template = generate_chat_template_dynamic(
        spm=False,
        tokenizer_version=TokenizerVersion.v15,
        image_support=False,
        audio_support=False,
        thinking_support=False,
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string", "description": "The city and state"}},
                    "required": ["location"],
                },
            },
        }
    ]

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather in Paris?"},
    ]

    output = render_template(template, messages, tools=tools, reasoning_effort="high")
    assert '[MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS]' in output
    assert "[AVAILABLE_TOOLS]" in output
    assert "get_weather" in output


# Comprehensive functional tests using the same test cases as test_chat_templates.py
@pytest.mark.parametrize(
    ("version", "spm", "image", "audio", "think"),
    [
        # Non-SPM configurations that we can fully test
        (TokenizerVersion.v1, False, False, False, False),
        (TokenizerVersion.v2, False, False, False, False),
        (TokenizerVersion.v3, False, False, False, False),
        (TokenizerVersion.v3, False, True, False, False),
        (TokenizerVersion.v7, False, False, False, False),
        (TokenizerVersion.v7, False, True, False, False),
        (TokenizerVersion.v7, False, False, True, False),
        (TokenizerVersion.v11, False, False, False, False),
        (TokenizerVersion.v11, False, True, False, False),
        (TokenizerVersion.v11, False, False, True, False),
        (TokenizerVersion.v13, False, False, False, False),
        (TokenizerVersion.v13, False, True, False, False),
        (TokenizerVersion.v13, False, False, True, False),
        (TokenizerVersion.v13, False, False, False, True),
        (TokenizerVersion.v13, False, True, False, True),
        (TokenizerVersion.v15, False, False, False, False),
        (TokenizerVersion.v15, False, True, False, False),
        (TokenizerVersion.v15, False, False, True, False),
        (TokenizerVersion.v15, False, True, False, True),
    ],
)
def test_dynamic_template_comprehensive(
    version: TokenizerVersion,
    spm: bool,
    image: bool,
    audio: bool,
    think: bool,
) -> None:
    """Test dynamically generated templates with comprehensive conversation examples."""
    tokenizer_version = TokenizerVersion(version.value)
    static_template = get_chat_template(
        spm=spm,
        tokenizer_version=tokenizer_version,
        image_support=image,
        audio_support=audio,
        thinking_support=think,
    )

    config = TemplateConfig(
        version=version,
        spm=spm,
        image_support=image,
        audio_support=audio,
        thinking_support=think,
    )
    dynamic_template = generate_chat_template(config)

    # Test cases
    test_cases = [
        # Simple one-turn
        {
            "name": "one_turn",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
        },
        # With system message
        {
            "name": "with_system",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
        },
        # Multi-turn
        {
            "name": "multi_turn",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm doing well!"},
            ],
        },
        # Content as list of chunks
        {
            "name": "content_chunks",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]},
            ],
        },
    ]

    # Add image test case if image support
    if image:
        test_cases.append(
            {
                "name": "with_image",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is this?"},
                            {"type": "image_url", "image_url": "http://example.com/image.png"},
                        ],
                    },
                    {"role": "assistant", "content": "It's an image."},
                ],
            }
        )

    # Add audio test case if audio support
    if audio:
        test_cases.append(
            {
                "name": "with_audio",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is this?"},
                            {"type": "audio_url", "audio_url": "http://example.com/audio.mp3"},
                        ],
                    },
                    {"role": "assistant", "content": "It's an audio file."},
                ],
            }
        )

    # Add thinking test case if thinking support
    if think:
        test_cases.append(
            {
                "name": "with_thinking",
                "messages": [
                    {"role": "user", "content": "Solve this problem"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "Let me think..."},
                            {"type": "text", "text": "The answer is 42."},
                        ],
                    },
                ],
            }
        )

    for test_case in test_cases:
        test_name = test_case["name"]

        # Explicitly skip unsupported configurations rather than catching exceptions
        # Image tests require image support
        if test_name == "with_image" and not image:
            continue

        # Audio tests require audio support
        if test_name == "with_audio" and not audio:
            continue

        # Thinking tests require thinking support
        if test_name == "with_thinking" and not think:
            continue

        static_output = render_template(static_template, test_case["messages"])  # type: ignore
        dynamic_output = render_template(dynamic_template, test_case["messages"])  # type: ignore

        assert static_output == dynamic_output, (
            f"Output mismatch for version={version}, case={test_name}\n\n"
            f"Static output: {static_output}\n"
            f"Dynamic output: {dynamic_output}"
        )
