from typing import Any

import pytest

from mistral_common.integrations.chat_templates.template_generator import build_chat_template
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.integrations.chat_templates.conftest import ALL_CONFIGS, _config_id
from tests.integrations.chat_templates.helpers import (
    TestConfig,
    _load_golden_template,
    _make_config,
    render_template,
)


@pytest.mark.parametrize(
    "config",
    [c for c in ALL_CONFIGS if not c.plain_think],
    ids=[_config_id(c) for c in ALL_CONFIGS if not c.plain_think],
)
def test_dynamic_matches_static(config: TestConfig) -> None:
    template_config = _make_config(config)
    dynamic = build_chat_template(template_config)
    static = _load_golden_template(template_config)
    assert dynamic == static


@pytest.mark.parametrize(
    "config",
    [c for c in ALL_CONFIGS if not c.plain_think],
    ids=[_config_id(c) for c in ALL_CONFIGS if not c.plain_think],
)
def test_dynamic_template_produces_same_output(config: TestConfig) -> None:
    template_config = _make_config(config)
    static_template = _load_golden_template(template_config)
    dynamic_template = build_chat_template(template_config)

    # Test with a simple conversation
    messages: list[Any] = [
        {"role": "user", "content": "Hello"},
    ]

    # Only add assistant message for training validation
    if config.version != TokenizerVersion.v1:
        messages.append({"role": "assistant", "content": "Hi there!"})

    static_output = render_template(static_template, messages)
    dynamic_output = render_template(dynamic_template, messages)

    assert static_output == dynamic_output, (
        f"Output mismatch for {config}\n\nStatic output: {static_output}\nDynamic output: {dynamic_output}"
    )


# Comprehensive functional tests — covers all configs except plain_think
# (plain think template parity is tested separately in test_unit.py::test_plain_think_static_dynamic_parity)
@pytest.mark.parametrize(
    "config",
    [c for c in ALL_CONFIGS if not c.plain_think],
    ids=[_config_id(c) for c in ALL_CONFIGS if not c.plain_think],
)
def test_dynamic_template_comprehensive(config: TestConfig) -> None:
    template_config = _make_config(config)
    static_template = _load_golden_template(template_config)
    dynamic_template = build_chat_template(template_config)

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

    # Tool call scenarios (v2+ only)
    if config.version > TokenizerVersion.v1:
        test_cases.extend(
            [
                {
                    "name": "with_tools_definition",
                    "messages": [
                        {"role": "user", "content": "What's the weather?"},
                        {"role": "assistant", "content": "It's sunny."},
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "description": "Get weather",
                                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                            },
                        }
                    ],
                },
                {
                    "name": "with_tool_call_and_result",
                    "messages": [
                        {"role": "user", "content": "What's the weather in Paris?"},
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "abc123def",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                                }
                            ],
                        },
                        {"role": "tool", "content": '{"temp": 20}', "tool_call_id": "abc123def", "name": "get_weather"},
                        {"role": "assistant", "content": "It's 20 degrees in Paris."},
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "description": "Get weather",
                                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                            },
                        }
                    ],
                },
            ]
        )

    # Add image test case if image support
    if config.image:
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
    if config.audio:
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
    if config.think:
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

    if config.image:
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

    if config.audio:
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

    if config.think:
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

    # V15+ multimodal: tool messages with image/audio content
    if config.version >= TokenizerVersion.v15 and config.image:
        test_cases.append(
            {
                "name": "v15_tool_with_image",
                "messages": [
                    {"role": "user", "content": "Use tool"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"id": "abc123def", "function": {"name": "fn", "arguments": "{}"}},
                        ],
                    },
                    {
                        "role": "tool",
                        "content": [
                            {"type": "text", "text": "result"},
                            {"type": "image_url", "image_url": "http://example.com/img.png"},
                        ],
                        "tool_call_id": "abc123def",
                    },
                    {"role": "assistant", "content": "Done"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "fn", "description": "test", "parameters": {}},
                    }
                ],
            }
        )

    if config.version >= TokenizerVersion.v15 and config.audio:
        test_cases.append(
            {
                "name": "v15_tool_with_audio",
                "messages": [
                    {"role": "user", "content": "Use tool"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"id": "abc123def", "function": {"name": "fn", "arguments": "{}"}},
                        ],
                    },
                    {
                        "role": "tool",
                        "content": [
                            {"type": "text", "text": "result"},
                            {"type": "input_audio", "input_audio": {"data": "abc", "format": "wav"}},
                        ],
                        "tool_call_id": "abc123def",
                    },
                    {"role": "assistant", "content": "Done"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "fn", "description": "test", "parameters": {}},
                    }
                ],
            }
        )

    # V15+ multimodal: system messages with audio content
    if config.version >= TokenizerVersion.v15 and config.audio:
        test_cases.append(
            {
                "name": "v15_system_with_audio",
                "messages": [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "Listen to context"},
                            {"type": "input_audio", "input_audio": {"data": "abc", "format": "wav"}},
                        ],
                    },
                    {"role": "user", "content": "Summarize"},
                    {"role": "assistant", "content": "Done"},
                ],
            }
        )

    skip_names_image = {"with_image", "consecutive_users_with_image", "consecutive_users_multi_image"}
    skip_names_audio = {"with_audio", "consecutive_users_multi_audio"}
    skip_names_think = {"with_thinking", "consecutive_assistants_think", "consecutive_systems_think"}
    # ThinkChunks in system messages are only supported in v13 (not v15+)
    skip_names_think_system = {"consecutive_systems_think"}
    skip_names_tools = {"with_tools_definition", "with_tool_call_and_result"}
    skip_names_v15_image = {"v15_tool_with_image"}
    skip_names_v15_audio = {"v15_tool_with_audio", "v15_system_with_audio"}

    # Not using parametrize here because test_cases are built dynamically based on the
    # config (version/image/audio/think) from the outer parametrize. Each sub-case is
    # identifiable via test_name in the assertion failure message.
    for test_case in test_cases:
        test_name = test_case["name"]
        messages = test_case["messages"]
        tools = test_case.get("tools")

        if test_name in skip_names_image and not config.image:
            continue
        if test_name in skip_names_audio and not config.audio:
            continue
        if test_name in skip_names_think and not config.think:
            continue
        if test_name in skip_names_think_system and config.version >= TokenizerVersion.v15:
            continue
        if test_name in skip_names_tools and config.version <= TokenizerVersion.v1:
            continue
        if test_name in skip_names_v15_image and (config.version < TokenizerVersion.v15 or not config.image):
            continue
        if test_name in skip_names_v15_audio and (config.version < TokenizerVersion.v15 or not config.audio):
            continue

        static_output = render_template(static_template, messages, tools=tools)  # type: ignore
        dynamic_output = render_template(dynamic_template, messages, tools=tools)  # type: ignore

        assert static_output == dynamic_output, (
            f"Output mismatch for {config}, case={test_name}\n\n"
            f"Static output: {static_output}\n"
            f"Dynamic output: {dynamic_output}"
        )
