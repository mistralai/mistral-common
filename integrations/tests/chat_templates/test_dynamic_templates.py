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

    # Test with no reasoning effort (None/undefined — defaults to 'none')
    output_none = render_template(template, messages, reasoning_effort=None)
    assert '[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]' in output_none

    # Test with reasoning effort='high'
    output_high = render_template(template, messages, reasoning_effort="high")
    assert '[MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS]' in output_high

    # Test with reasoning effort='none' (explicit string)
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

    # Add message aggregation test cases
    test_cases.extend(
        [
            {
                "name": "consecutive_users",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "user", "content": "World"},
                    {"role": "assistant", "content": "Hi there"},
                ],
            },
            {
                "name": "consecutive_users_with_system",
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello"},
                    {"role": "user", "content": "World"},
                    {"role": "assistant", "content": "Hi there"},
                ],
            },
            {
                "name": "consecutive_assistants",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                    {"role": "assistant", "content": "How can I help?"},
                    {"role": "user", "content": "Thanks"},
                    {"role": "assistant", "content": "Welcome"},
                ],
            },
            {
                "name": "multiple_systems",
                "messages": [
                    {"role": "system", "content": "System 1."},
                    {"role": "system", "content": "System 2."},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ],
            },
            {
                "name": "mid_conv_system",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "system", "content": "New instruction."},
                    {"role": "assistant", "content": "Got it"},
                ],
            },
            {
                "name": "mid_conv_system_with_consecutive_users",
                "messages": [
                    {"role": "system", "content": "Be helpful."},
                    {"role": "user", "content": "Hello"},
                    {"role": "user", "content": "World"},
                    {"role": "system", "content": "Now be concise."},
                    {"role": "assistant", "content": "Got it"},
                ],
            },
        ]
    )

    # Multi-chunk aggregation test cases
    test_cases.extend(
        [
            {
                "name": "consecutive_users_text_chunks",
                "messages": [
                    {"role": "user", "content": "First as string"},
                    {"role": "user", "content": [{"type": "text", "text": "Second as chunk"}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Third part A"},
                            {"type": "text", "text": "Third part B"},
                        ],
                    },
                    {"role": "assistant", "content": "Response"},
                ],
            },
            {
                "name": "system_text_chunks",
                "messages": [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "You are helpful."},
                            {"type": "text", "text": "Be concise."},
                        ],
                    },
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ],
            },
        ]
    )

    if image:
        test_cases.extend(
            [
                {
                    "name": "consecutive_users_with_image",
                    "messages": [
                        {"role": "user", "content": "What is this?"},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": "http://example.com/image.png"},
                                {"type": "text", "text": "Describe it"},
                            ],
                        },
                        {"role": "assistant", "content": "It's an image."},
                    ],
                },
                {
                    "name": "consecutive_users_multi_image",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe this"},
                                {"type": "image_url", "image_url": "http://example.com/a.png"},
                                {"type": "text", "text": "What color?"},
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Also this"},
                                {"type": "image_url", "image_url": "http://example.com/b.png"},
                                {"type": "text", "text": "What shape?"},
                            ],
                        },
                        {"role": "assistant", "content": "Both are red squares."},
                    ],
                },
            ]
        )

    if audio:
        test_cases.append(
            {
                "name": "consecutive_users_multi_audio",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Listen"},
                            {"type": "audio_url", "audio_url": "http://example.com/a.wav"},
                            {"type": "text", "text": "What language?"},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "And this"},
                            {"type": "audio_url", "audio_url": "http://example.com/b.wav"},
                            {"type": "text", "text": "Transcribe it"},
                        ],
                    },
                    {"role": "assistant", "content": "Both are in English."},
                ],
            }
        )

    if think:
        test_cases.extend(
            [
                {
                    "name": "consecutive_assistants_think",
                    "messages": [
                        {"role": "user", "content": "Solve this"},
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "Hmm."},
                                {"type": "thinking", "thinking": "Let me think..."},
                                {"type": "text", "text": "I need more context."},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "OK."},
                                {"type": "thinking", "thinking": "Now I understand."},
                                {"type": "text", "text": "The answer is 42."},
                            ],
                        },
                        {"role": "user", "content": "Thanks"},
                        {"role": "assistant", "content": "You're welcome"},
                    ],
                },
                {
                    "name": "consecutive_systems_think",
                    "messages": [
                        {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": "Rule A"},
                                {"type": "text", "text": "Rule B"},
                                {"type": "thinking", "thinking": "Think 1"},
                            ],
                        },
                        {
                            "role": "system",
                            "content": [
                                {"type": "thinking", "thinking": "Think 2"},
                                {"type": "text", "text": "Rule C"},
                                {"type": "text", "text": "Rule D"},
                            ],
                        },
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi"},
                    ],
                },
            ]
        )

    skip_names_image = {"with_image", "consecutive_users_with_image", "consecutive_users_multi_image"}
    skip_names_audio = {"with_audio", "consecutive_users_multi_audio"}
    skip_names_think = {"with_thinking", "consecutive_assistants_think", "consecutive_systems_think"}
    # ThinkChunks in system messages are only supported in v13 (not v15+)
    skip_names_think_system = {"consecutive_systems_think"}

    for test_case in test_cases:
        test_name = test_case["name"]

        if test_name in skip_names_image and not image:
            continue
        if test_name in skip_names_audio and not audio:
            continue
        if test_name in skip_names_think and not think:
            continue
        if test_name in skip_names_think_system and version >= TokenizerVersion.v15:
            continue

        static_output = render_template(static_template, test_case["messages"])  # type: ignore
        dynamic_output = render_template(dynamic_template, test_case["messages"])  # type: ignore

        assert static_output == dynamic_output, (
            f"Output mismatch for version={version}, case={test_name}\n\n"
            f"Static output: {static_output}\n"
            f"Dynamic output: {dynamic_output}"
        )


@pytest.mark.parametrize(
    ("version", "spm"),
    [
        (TokenizerVersion.v7, False),
        (TokenizerVersion.v7, True),
        (TokenizerVersion.v13, False),
        (TokenizerVersion.v15, False),
    ],
)
def test_aggregation_consecutive_assistants_both_tool_calls(version: TokenizerVersion, spm: bool) -> None:
    r"""Test consecutive assistant messages where both have tool_calls.

    This pattern is rejected by the validator but the normalizer handles it.
    We test the template directly to ensure the Jinja aggregation logic works.
    """
    template = generate_chat_template_dynamic(spm, version, False, False, False)

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": "Checking Paris.",
            "tool_calls": [
                {
                    "id": "123456789",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                },
            ],
        },
        {
            "role": "assistant",
            "content": "And London.",
            "tool_calls": [
                {
                    "id": "023456789",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "London"}'},
                },
            ],
        },
        {"role": "tool", "name": "get_weather", "content": "22", "tool_call_id": "123456789"},
        {"role": "tool", "name": "get_weather", "content": "15", "tool_call_id": "023456789"},
        {"role": "assistant", "content": "Paris: 22, London: 15"},
        {"role": "user", "content": "Thanks"},
        {"role": "assistant", "content": "Welcome"},
    ]

    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]

    output = render_template(template, messages, tools=tools)

    # Both assistant messages should be merged: content joined with \n\n
    assert "Checking Paris.\n\nAnd London." in output
    # Both tool calls should be present in the output
    assert '"city": "Paris"' in output
    assert '"city": "London"' in output


@pytest.mark.parametrize(
    ("has_system", "has_tools", "reasoning_effort"),
    [
        (True, True, "high"),
        (True, False, "high"),
        (False, True, "high"),
        (False, False, "high"),
        (True, True, "none"),
        (False, False, "none"),
        (True, True, None),
        (False, False, None),
    ],
)
def test_v15_tools_and_settings_ordering(has_system: bool, has_tools: bool, reasoning_effort: str | None) -> None:
    r"""Test that v15 emits system, tools, and model_settings in the correct order.

    Expected order: ``[SYSTEM_PROMPT]...[/SYSTEM_PROMPT]`` (if system) then
    ``[AVAILABLE_TOOLS]...[/AVAILABLE_TOOLS]`` (if tools) then
    ``[MODEL_SETTINGS]...[/MODEL_SETTINGS]`` (always, None defaults to 'none') then
    ``[INST]...[/INST]``.
    """
    template = generate_chat_template_dynamic(False, TokenizerVersion.v15, False, False, False)

    messages: list[dict[str, Any]] = []
    if has_system:
        messages.append({"role": "system", "content": "You are helpful."})
    messages.extend(
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
    )

    tools: list[dict[str, Any]] | None = None
    if has_tools:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "tool1",
                    "description": "",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    output = render_template(template, messages, tools=tools, reasoning_effort=reasoning_effort)

    # Verify ordering of special blocks
    if has_system:
        assert "[SYSTEM_PROMPT]You are helpful.[/SYSTEM_PROMPT]" in output
        sp_pos = output.index("[SYSTEM_PROMPT]")
    else:
        assert "[SYSTEM_PROMPT]" not in output
        sp_pos = -1

    if has_tools:
        assert "[AVAILABLE_TOOLS]" in output
        tools_pos = output.index("[AVAILABLE_TOOLS]")
        assert tools_pos > sp_pos
    else:
        assert "[AVAILABLE_TOOLS]" not in output
        tools_pos = sp_pos

    # MODEL_SETTINGS is always emitted for v15 (reasoning_effort is always provided)
    assert "[MODEL_SETTINGS]" in output
    settings_pos = output.index("[MODEL_SETTINGS]")
    assert settings_pos > tools_pos

    inst_pos = output.index("[INST]")
    assert inst_pos > settings_pos


# ── reasoning / reasoning_content → thinking chunk conversion ──────────


THINK_CONFIGS = [
    (TokenizerVersion.v13, False, False),
    (TokenizerVersion.v13, True, False),
    (TokenizerVersion.v15, False, False),
    (TokenizerVersion.v15, True, False),
]


@pytest.mark.parametrize(("version", "image", "audio"), THINK_CONFIGS)
def test_reasoning_content_to_thinking_chunk(version: TokenizerVersion, image: bool, audio: bool) -> None:
    r"""``reasoning_content`` on an assistant message produces a leading ``[THINK]...[/THINK]``."""
    template = generate_chat_template_dynamic(False, version, image, audio, thinking_support=True)

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "reasoning_content": "Let me add 2 and 2.", "content": "The answer is 4."},
    ]

    output = render_template(template, messages)
    assert "[THINK]Let me add 2 and 2.[/THINK]The answer is 4." in output


@pytest.mark.parametrize(("version", "image", "audio"), THINK_CONFIGS)
def test_reasoning_field_to_thinking_chunk(version: TokenizerVersion, image: bool, audio: bool) -> None:
    r"""``reasoning`` field (alias) produces the same ``[THINK]...[/THINK]``."""
    template = generate_chat_template_dynamic(False, version, image, audio, thinking_support=True)

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "What is 3+3?"},
        {"role": "assistant", "reasoning": "Adding three and three.", "content": "The answer is 6."},
    ]

    output = render_template(template, messages)
    assert "[THINK]Adding three and three.[/THINK]The answer is 6." in output


@pytest.mark.parametrize(("version", "image", "audio"), THINK_CONFIGS)
def test_reasoning_content_takes_precedence_over_reasoning(version: TokenizerVersion, image: bool, audio: bool) -> None:
    r"""When both ``reasoning_content`` and ``reasoning`` are present, ``reasoning_content`` wins."""
    template = generate_chat_template_dynamic(False, version, image, audio, thinking_support=True)

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "reasoning_content": "RC wins",
            "reasoning": "R loses",
            "content": "Hello!",
        },
    ]

    output = render_template(template, messages)
    assert "[THINK]RC wins[/THINK]Hello!" in output
    assert "R loses" not in output


@pytest.mark.parametrize(("version", "image", "audio"), THINK_CONFIGS)
def test_reasoning_with_existing_think_chunks(version: TokenizerVersion, image: bool, audio: bool) -> None:
    r"""``reasoning_content`` is prepended before existing inline ``ThinkChunk``\s."""
    template = generate_chat_template_dynamic(False, version, image, audio, thinking_support=True)

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "reasoning_content": "Top-level reasoning",
            "content": [
                {"type": "thinking", "thinking": "Inline thinking"},
                {"type": "text", "text": "Hello!"},
            ],
        },
    ]

    output = render_template(template, messages)
    assert "[THINK]Top-level reasoning[/THINK][THINK]Inline thinking[/THINK]Hello!" in output


@pytest.mark.parametrize(("version", "image", "audio"), THINK_CONFIGS)
def test_reasoning_content_only_no_text_content(version: TokenizerVersion, image: bool, audio: bool) -> None:
    r"""``reasoning_content`` with empty/null text content produces just the thinking chunk."""
    template = generate_chat_template_dynamic(False, version, image, audio, thinking_support=True)

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "reasoning_content": "Just thinking...", "content": ""},
    ]

    output = render_template(template, messages)
    assert "[THINK]Just thinking...[/THINK]" in output


@pytest.mark.parametrize(("version", "image", "audio"), THINK_CONFIGS)
def test_reasoning_content_with_tool_calls(version: TokenizerVersion, image: bool, audio: bool) -> None:
    r"""``reasoning_content`` is preserved alongside ``tool_calls``."""
    template = generate_chat_template_dynamic(False, version, image, audio, thinking_support=True)

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "reasoning_content": "I should call a tool.",
            "content": "",
            "tool_calls": [
                {
                    "id": "abc123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                },
            ],
        },
        {"role": "tool", "name": "get_weather", "content": "22C", "tool_call_id": "abc123"},
        {"role": "assistant", "content": "It's 22C in Paris."},
    ]

    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]

    output = render_template(template, messages, tools=tools)
    assert "[THINK]I should call a tool.[/THINK]" in output
    assert "get_weather" in output


@pytest.mark.parametrize(("version", "image", "audio"), THINK_CONFIGS)
def test_reasoning_aggregation_consecutive_assistants(version: TokenizerVersion, image: bool, audio: bool) -> None:
    r"""Consecutive assistant messages with ``reasoning_content`` aggregate correctly."""
    template = generate_chat_template_dynamic(False, version, image, audio, thinking_support=True)

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Solve this."},
        {"role": "assistant", "reasoning_content": "Step 1 reasoning", "content": "Partial answer."},
        {"role": "assistant", "reasoning_content": "Step 2 reasoning", "content": "Final answer."},
    ]

    output = render_template(template, messages)
    # Both reasoning traces should appear as thinking chunks
    assert "[THINK]Step 1 reasoning[/THINK]" in output
    assert "[THINK]Step 2 reasoning[/THINK]" in output
    # Text from both messages should be aggregated
    assert "Partial answer." in output
    assert "Final answer." in output


@pytest.mark.parametrize(("version", "image", "audio"), THINK_CONFIGS)
def test_reasoning_static_dynamic_parity(version: TokenizerVersion, image: bool, audio: bool) -> None:
    r"""Static and dynamic templates produce identical output for ``reasoning_content`` input."""
    static_template = get_chat_template(
        spm=False,
        tokenizer_version=version,
        image_support=image,
        audio_support=audio,
        thinking_support=True,
    )
    dynamic_template = generate_chat_template_dynamic(False, version, image, audio, thinking_support=True)

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "reasoning_content": "Thinking...", "content": "Hi!"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "reasoning": "Considering the question.", "content": "I'm well!"},
    ]

    static_output = render_template(static_template, messages)
    dynamic_output = render_template(dynamic_template, messages)

    assert static_output == dynamic_output


def test_non_think_template_ignores_reasoning_field() -> None:
    r"""Non-think templates silently ignore ``reasoning_content`` (no conversion happens)."""
    template = generate_chat_template_dynamic(False, TokenizerVersion.v13, False, False, thinking_support=False)

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "reasoning_content": "This is ignored", "content": "Hello!"},
    ]

    output = render_template(template, messages)
    assert "[THINK]" not in output
    assert "Hello!" in output


def test_empty_reasoning_content_is_ignored() -> None:
    r"""Empty-string ``reasoning_content`` does not produce a thinking chunk."""
    template = generate_chat_template_dynamic(False, TokenizerVersion.v13, False, False, thinking_support=True)

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "reasoning_content": "", "content": "Hello!"},
    ]

    output = render_template(template, messages)
    assert "[THINK]" not in output
    assert "Hello!" in output


def test_none_reasoning_content_is_ignored() -> None:
    r"""``None`` value for ``reasoning_content`` does not produce a thinking chunk."""
    template = generate_chat_template_dynamic(False, TokenizerVersion.v13, False, False, thinking_support=True)

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "reasoning_content": None, "content": "Hello!"},
    ]

    output = render_template(template, messages)
    assert "[THINK]" not in output
    assert "Hello!" in output


def test_invalid_config_plain_thinking_with_thinking() -> None:
    r"""``plain_thinking_support`` and ``thinking_support`` are mutually exclusive."""
    with pytest.raises(ValueError, match="Plain thinking support and thinking support are mutually exclusive"):
        TemplateConfig(version=TokenizerVersion.v11, thinking_support=True, plain_thinking_support=True)


def test_invalid_config_plain_thinking_non_v11() -> None:
    r"""``plain_thinking_support`` only works with v11."""
    with pytest.raises(ValueError, match="Plain thinking support is only available for tokenizer version v11"):
        TemplateConfig(version=TokenizerVersion.v15, plain_thinking_support=True)


def test_invalid_config_plain_thinking_with_audio() -> None:
    r"""``plain_thinking_support`` and ``audio_support`` are mutually exclusive."""
    with pytest.raises(ValueError, match="Audio and plain thinking support are mutually exclusive"):
        TemplateConfig(version=TokenizerVersion.v11, audio_support=True, plain_thinking_support=True)


def test_plain_think_template_produces_correct_output() -> None:
    r"""Plain think template emits ``<think>``/``</think>`` tags, not ``[THINK]``/``[/THINK]``."""
    template = generate_chat_template_dynamic(
        False, TokenizerVersion.v11, False, False, thinking_support=False, plain_thinking_support=True
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Solve this"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Let me think..."},
                {"type": "text", "text": "The answer is 42."},
            ],
        },
    ]

    output = render_template(template, messages)
    assert "<think>Let me think...</think>The answer is 42." in output
    assert "[THINK]" not in output
    assert "[/THINK]" not in output


@pytest.mark.parametrize("image", [False, True])
def test_plain_think_static_dynamic_parity(image: bool) -> None:
    r"""Static and dynamic plain think templates produce identical output."""
    static_template = get_chat_template(
        spm=False,
        tokenizer_version=TokenizerVersion.v11,
        image_support=image,
        audio_support=False,
        thinking_support=False,
        plain_thinking_support=True,
    )
    dynamic_template = generate_chat_template_dynamic(
        False, TokenizerVersion.v11, image, False, thinking_support=False, plain_thinking_support=True
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Thinking..."},
                {"type": "text", "text": "Hi!"},
            ],
        },
    ]

    static_output = render_template(static_template, messages)
    dynamic_output = render_template(dynamic_template, messages)
    assert static_output == dynamic_output


def test_plain_think_reasoning_content_conversion() -> None:
    r"""``reasoning_content`` produces ``<think>`` tags in plain think mode."""
    template = generate_chat_template_dynamic(
        False, TokenizerVersion.v11, False, False, thinking_support=False, plain_thinking_support=True
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "reasoning_content": "Let me add.", "content": "4."},
    ]

    output = render_template(template, messages)
    assert "<think>Let me add.</think>4." in output
    assert "[THINK]" not in output


def test_plain_think_closed_false() -> None:
    r"""``closed: false`` omits the ``</think>`` closing tag in plain think mode."""
    template = generate_chat_template_dynamic(
        False, TokenizerVersion.v11, False, False, thinking_support=False, plain_thinking_support=True
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Still thinking...", "closed": False},
            ],
        },
    ]

    output = render_template(template, messages)
    assert "<think>Still thinking..." in output
    assert "</think>" not in output


def test_plain_think_image_template() -> None:
    r"""Image + plain think template handles both ``[IMG]`` and ``<think>`` tags."""
    template = generate_chat_template_dynamic(
        False, TokenizerVersion.v11, True, False, thinking_support=False, plain_thinking_support=True
    )

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": "http://example.com/img.png"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "It looks like..."},
                {"type": "text", "text": "A red square."},
            ],
        },
    ]

    output = render_template(template, messages)
    assert "[IMG]" in output
    assert "<think>It looks like...</think>A red square." in output
    assert "[THINK]" not in output


# ── closed attribute handling for special token think ──────────────────


@pytest.mark.parametrize(
    ("version", "image"),
    [
        (TokenizerVersion.v13, False),
        (TokenizerVersion.v13, True),
        (TokenizerVersion.v15, False),
        (TokenizerVersion.v15, True),
    ],
)
def test_closed_false_special_token_think(version: TokenizerVersion, image: bool) -> None:
    r"""``closed: false`` omits ``[/THINK]`` for special token think templates."""
    template = generate_chat_template_dynamic(False, version, image, False, thinking_support=True)

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Ongoing thought", "closed": False},
            ],
        },
    ]

    output = render_template(template, messages)
    assert "[THINK]Ongoing thought" in output
    assert "[/THINK]" not in output


@pytest.mark.parametrize(
    ("version", "image"),
    [
        (TokenizerVersion.v13, False),
        (TokenizerVersion.v13, True),
        (TokenizerVersion.v15, False),
        (TokenizerVersion.v15, True),
    ],
)
def test_closed_true_special_token_think(version: TokenizerVersion, image: bool) -> None:
    r"""``closed: true`` (or absent) emits ``[/THINK]`` for special token think templates."""
    template = generate_chat_template_dynamic(False, version, image, False, thinking_support=True)

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Done thinking"},
                {"type": "text", "text": "Hello!"},
            ],
        },
    ]

    output = render_template(template, messages)
    assert "[THINK]Done thinking[/THINK]Hello!" in output
