from typing import Any

import pytest

from mistral_common.integrations.chat_templates.chat_templates import generate_chat_template
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.integrations.chat_templates.helpers import TestConfig, _load_golden_template, _make_config, render_template


class TestToolRoleSupport:
    def test_v1_tool_role_rejected(self) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v1,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "tool", "content": "result"},
        ]

        with pytest.raises(ValueError, match="Unexpected role 'tool' after role 'assistant'"):
            render_template(template, messages)

    @pytest.mark.parametrize(
        ("version", "spm"),
        [
            (TokenizerVersion.v2, False),
            (TokenizerVersion.v2, True),
            (TokenizerVersion.v3, False),
            (TokenizerVersion.v3, True),
            (TokenizerVersion.v7, False),
            (TokenizerVersion.v7, True),
            (TokenizerVersion.v11, False),
            (TokenizerVersion.v13, False),
            (TokenizerVersion.v15, False),
        ],
    )
    def test_user_after_tool_accepted(self, version: TokenizerVersion, spm: bool) -> None:
        template = generate_chat_template(
            spm=spm,
            tokenizer_version=version,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Search for info"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "aaaaaaaaa",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"q": "info"}'},
                    },
                ],
            },
            {"role": "tool", "name": "search", "content": "result1", "tool_call_id": "aaaaaaaaa"},
            {"role": "user", "content": "Now refine"},
            {"role": "assistant", "content": "Here is the refined answer"},
        ]

        tools: list[dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                },
            }
        ]

        output = render_template(template, messages, tools=tools)
        assert "Now refine" in output
        assert "Here is the refined answer" in output

    @pytest.mark.parametrize(
        ("version", "spm"),
        [
            (TokenizerVersion.v7, False),
            (TokenizerVersion.v7, True),
            (TokenizerVersion.v13, False),
            (TokenizerVersion.v15, False),
        ],
    )
    def test_user_after_tool_static_matches_dynamic(self, version: TokenizerVersion, spm: bool) -> None:
        config = _make_config(TestConfig(version=version, spm=spm))
        static_template = _load_golden_template(config)
        dynamic_template = generate_chat_template(
            spm=spm,
            tokenizer_version=version,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Search for info"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "aaaaaaaaa",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"q": "info"}'},
                    },
                ],
            },
            {"role": "tool", "name": "search", "content": "result1", "tool_call_id": "aaaaaaaaa"},
            {"role": "user", "content": "Now refine"},
            {"role": "assistant", "content": "Here is the refined answer"},
        ]

        tools: list[dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                },
            }
        ]

        static_output = render_template(static_template, messages, tools=tools)
        dynamic_output = render_template(dynamic_template, messages, tools=tools)

        assert static_output == dynamic_output, (
            f"Output mismatch for version={version}, spm={spm}\n\n"
            f"Static output: {static_output}\n"
            f"Dynamic output: {dynamic_output}"
        )


class TestV2ToolResultParsing:
    @pytest.fixture()
    def v2_template(self) -> str:
        return generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v2,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

    @pytest.mark.parametrize(
        ("tool_name", "user_content", "arguments", "tools", "content", "assistant_content", "expected"),
        [
            pytest.param(
                "add",
                "What is 2+2?",
                '{"a": 2, "b": 2}',
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "add",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "a": {"type": "integer"},
                                    "b": {"type": "integer"},
                                },
                            },
                        },
                    }
                ],
                "4",
                "4",
                '"content": 4',
                id="integer",
            ),
            pytest.param(
                "pi",
                "What is pi?",
                "{}",
                [{"type": "function", "function": {"name": "pi", "parameters": {}}}],
                "3.14",
                "Pi is 3.14",
                '"content": 3.14',
                id="float",
            ),
        ],
    )
    def test_numeric_tool_result(
        self,
        v2_template: str,
        tool_name: str,
        user_content: str,
        arguments: str,
        tools: list[dict[str, Any]],
        content: str,
        assistant_content: str,
        expected: str,
    ) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "abc123def",
                        "type": "function",
                        "function": {"name": tool_name, "arguments": arguments},
                    }
                ],
            },
            {"role": "tool", "content": content, "name": tool_name},
            {"role": "assistant", "content": assistant_content},
        ]

        output = render_template(v2_template, messages, tools=tools)
        assert expected in output

    def test_v2_tool_message_missing_name_raises(self, v2_template: str) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Add 2+2"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "abc123def",
                        "type": "function",
                        "function": {"name": "add", "arguments": '{"a": 2, "b": 2}'},
                    }
                ],
            },
            {"role": "tool", "content": "4"},
            {"role": "assistant", "content": "4"},
        ]

        with pytest.raises(ValueError, match="Tool message must have a name"):
            render_template(
                v2_template,
                messages,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "add",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "a": {"type": "integer"},
                                    "b": {"type": "integer"},
                                },
                            },
                        },
                    }
                ],
            )


class TestV3ToolResultParsing:
    @pytest.fixture()
    def v3_template(self) -> str:
        return generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v3,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

    @pytest.mark.parametrize(
        ("tool_name", "user_content", "arguments", "tools", "content", "assistant_content", "expected"),
        [
            pytest.param(
                "add",
                "What is 2+2?",
                '{"a": 2, "b": 2}',
                [{"type": "function", "function": {"name": "add", "parameters": {}}}],
                "4",
                "4",
                '"content": 4',
                id="integer",
            ),
            pytest.param(
                "pi",
                "What is pi?",
                "{}",
                [{"type": "function", "function": {"name": "pi", "parameters": {}}}],
                "3.14",
                "3.14",
                '"content": 3.14',
                id="float",
            ),
        ],
    )
    def test_numeric_tool_result(
        self,
        v3_template: str,
        tool_name: str,
        user_content: str,
        arguments: str,
        tools: list[dict[str, Any]],
        content: str,
        assistant_content: str,
        expected: str,
    ) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "abc123def",
                        "type": "function",
                        "function": {"name": tool_name, "arguments": arguments},
                    }
                ],
            },
            {"role": "tool", "content": content, "tool_call_id": "abc123def", "name": tool_name},
            {"role": "assistant", "content": assistant_content},
        ]

        output = render_template(v3_template, messages, tools=tools)
        assert expected in output

    @pytest.mark.parametrize("tool_call_id", [None, "short"])
    def test_invalid_call_id(self, v3_template: str, tool_call_id: str | None) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "abc123def",
                        "type": "function",
                        "function": {"name": "greet", "arguments": "{}"},
                    }
                ],
            },
        ]

        tool_msg: dict[str, Any] = {"role": "tool", "content": "hi", "name": "greet"}
        if tool_call_id is not None:
            tool_msg["tool_call_id"] = tool_call_id

        messages.append(tool_msg)
        messages.append({"role": "assistant", "content": "hi"})

        with pytest.raises(ValueError, match="call_id or tool_call_id"):
            render_template(
                v3_template,
                messages,
                tools=[{"type": "function", "function": {"name": "greet", "parameters": {}}}],
            )


class TestV7ToolCalls:
    def test_v7_content_and_tool_calls_accepted(self) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v7,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": "Let me check the weather for you.",
                "tool_calls": [
                    {
                        "id": "abc123def",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
            },
            {"role": "tool", "content": "Sunny, 25C", "tool_call_id": "abc123def", "name": "get_weather"},
            {"role": "assistant", "content": "It's sunny and 25C in Paris."},
        ]

        output = render_template(
            template,
            messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                    },
                }
            ],
        )
        assert "Let me check the weather for you." in output
        assert "[TOOL_CALLS]" in output

    @pytest.mark.parametrize(
        ("version", "spm"),
        [
            (TokenizerVersion.v7, False),
            (TokenizerVersion.v7, True),
            (TokenizerVersion.v13, False),
            (TokenizerVersion.v15, False),
        ],
    )
    def test_aggregation_consecutive_assistants_both_tool_calls(self, version: TokenizerVersion, spm: bool) -> None:
        template = generate_chat_template(
            spm=spm,
            tokenizer_version=version,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

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
        ("version", "spm"),
        [
            (TokenizerVersion.v7, False),
            (TokenizerVersion.v13, False),
            (TokenizerVersion.v15, False),
        ],
    )
    def test_tool_call_empty_string_arguments_defaults_to_empty_object(
        self, version: TokenizerVersion, spm: bool
    ) -> None:
        template = generate_chat_template(
            spm=spm,
            tokenizer_version=version,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "abc123def",
                        "type": "function",
                        "function": {"name": "greet", "arguments": ""},
                    }
                ],
            },
            {"role": "tool", "content": "hi", "tool_call_id": "abc123def", "name": "greet"},
            {"role": "assistant", "content": "Done"},
        ]

        output = render_template(
            template,
            messages,
            tools=[{"type": "function", "function": {"name": "greet", "parameters": {}}}],
        )
        assert "{}" in output
        assert "greet" in output


class TestV11ToolCalls:
    def test_v11_tool_call_uses_call_id_token(self) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v11,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Weather?"},
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
            {"role": "tool", "content": "Sunny", "tool_call_id": "abc123def"},
            {"role": "assistant", "content": "Sunny in Paris."},
        ]

        output = render_template(
            template,
            messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
        )
        assert "[TOOL_CALLS]get_weather[CALL_ID]abc123def[ARGS]" in output
        assert '"city": "Paris"' in output

    @pytest.mark.parametrize(
        ("version", "spm"),
        [
            (TokenizerVersion.v3, False),
            (TokenizerVersion.v7, False),
            (TokenizerVersion.v11, False),
        ],
    )
    def test_tool_result_call_id_alias(self, version: TokenizerVersion, spm: bool) -> None:
        template = generate_chat_template(
            spm=spm,
            tokenizer_version=version,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Add 2+2"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "abc123def",
                        "type": "function",
                        "function": {"name": "add", "arguments": '{"a": 2, "b": 2}'},
                    }
                ],
            },
            {"role": "tool", "content": "4", "call_id": "abc123def", "name": "add"},
            {"role": "assistant", "content": "4"},
        ]

        output = render_template(
            template,
            messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "add",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "integer"},
                                "b": {"type": "integer"},
                            },
                        },
                    },
                }
            ],
        )
        assert "abc123def" in output
        assert "[TOOL_RESULTS]" in output


class TestV13PlusToolCalls:
    @pytest.mark.parametrize("version", [TokenizerVersion.v13, TokenizerVersion.v15])
    def test_v13_v15_accept_short_tool_call_id(self, version: TokenizerVersion) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=version,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "x",
                        "type": "function",
                        "function": {"name": "greet", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "content": "hi", "tool_call_id": "x", "name": "greet"},
            {"role": "assistant", "content": "Done"},
        ]

        output = render_template(
            template,
            messages,
            tools=[{"type": "function", "function": {"name": "greet", "parameters": {}}}],
        )
        assert "greet" in output
        assert "Done" in output
