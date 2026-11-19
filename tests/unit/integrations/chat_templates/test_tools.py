from typing import Any

import pytest

from mistral_common.protocol.instruct.messages import AssistantMessage, ToolMessage, UserMessage
from mistral_common.protocol.instruct.tool_calls import FunctionCall, ToolCall
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.integration.chat_recipes import (
    build_aggregated_tool_calls_request,
    build_conversation_request,
    build_function_tool,
    build_tool_call_request,
    build_tool_results_request,
)
from tests.utils.chat_templates.configs import TestConfig
from tests.utils.chat_templates.helpers import (
    build_render_args,
    build_template_config,
    build_test_template,
    load_golden_template,
    render_template,
)

_ADD_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
}
_SEARCH_PARAMETERS: dict[str, Any] = {"type": "object", "properties": {"q": {"type": "string"}}}
_WEATHER_PARAMETERS: dict[str, Any] = {"type": "object", "properties": {"city": {"type": "string"}}}


class TestToolRoleSupport:
    def test_v1_tool_role_rejected(self) -> None:
        template = build_test_template(TestConfig(version=TokenizerVersion.v1))
        request = build_conversation_request(
            messages=[
                UserMessage(content="Hello"),
                AssistantMessage(content="Hi"),
                ToolMessage(content="result", tool_call_id="x"),
            ]
        )

        with pytest.raises(ValueError, match="Unexpected role 'tool' after role 'assistant'"):
            render_template(template, **build_render_args(request))

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
        template = build_test_template(TestConfig(version=version, spm=spm))
        request = build_tool_call_request(
            tool_name="search",
            call_id="aaaaaaaaa",
            arguments='{"q": "info"}',
            result_content="result1",
            result_name="search",
            assistant_content="",
            user_content="Search for info",
            follow_up=("Now refine", "Here is the refined answer"),
            tools=[build_function_tool(name="search", description="", parameters=_SEARCH_PARAMETERS)],
        )

        output = render_template(template, **build_render_args(request))
        assert "Now refine" in output
        assert "Here is the refined answer" in output

    @pytest.mark.parametrize(
        ("version", "spm"),
        [
            (TokenizerVersion.v7, False),
            (TokenizerVersion.v7, True),
            (TokenizerVersion.v13, False),
        ],
    )
    def test_user_after_tool_static_matches_dynamic(self, version: TokenizerVersion, spm: bool) -> None:
        config = TestConfig(version=version, spm=spm)
        static_template = load_golden_template(build_template_config(config))
        dynamic_template = build_test_template(config)
        request = build_tool_call_request(
            tool_name="search",
            call_id="aaaaaaaaa",
            arguments='{"q": "info"}',
            result_content="result1",
            result_name="search",
            assistant_content="",
            user_content="Search for info",
            follow_up=("Now refine", "Here is the refined answer"),
            tools=[build_function_tool(name="search", description="", parameters=_SEARCH_PARAMETERS)],
        )

        static_output = render_template(static_template, **build_render_args(request))
        dynamic_output = render_template(dynamic_template, **build_render_args(request))

        assert static_output == dynamic_output, (
            f"Output mismatch for version={version}, spm={spm}\n\n"
            f"Static output: {static_output}\n"
            f"Dynamic output: {dynamic_output}"
        )


class TestV2ToolResultParsing:
    @pytest.fixture()
    def v2_template(self) -> str:
        return build_test_template(TestConfig(version=TokenizerVersion.v2))

    @pytest.mark.parametrize(
        ("tool_name", "user_content", "arguments", "parameters", "content", "assistant_content", "expected"),
        [
            pytest.param(
                "add",
                "What is 2+2?",
                '{"a": 2, "b": 2}',
                _ADD_PARAMETERS,
                "4",
                "4",
                '"content": 4',
                id="integer",
            ),
            pytest.param(
                "pi",
                "What is pi?",
                "{}",
                {},
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
        parameters: dict[str, Any],
        content: str,
        assistant_content: str,
        expected: str,
    ) -> None:
        request = build_tool_call_request(
            tool_name=tool_name,
            call_id="abc123def",
            arguments=arguments,
            result_content=content,
            result_name=tool_name,
            assistant_content="",
            user_content=user_content,
            terminal_content=assistant_content,
            tools=[build_function_tool(name=tool_name, description=None, parameters=parameters)],
        )

        output = render_template(v2_template, **build_render_args(request))
        assert expected in output

    def test_v2_tool_message_missing_name_raises(self, v2_template: str) -> None:
        request = build_tool_call_request(
            tool_name="add",
            call_id="abc123def",
            arguments='{"a": 2, "b": 2}',
            result_content="4",
            result_name=None,
            assistant_content="",
            user_content="Add 2+2",
            terminal_content="4",
            tools=[build_function_tool(name="add", description=None, parameters=_ADD_PARAMETERS)],
        )
        # Render the raw OpenAI conversion: build_render_args would re-add the
        # name from the tool call, defeating the missing-name claim.
        openai = request.to_openai()

        with pytest.raises(ValueError, match="Tool message must have a name"):
            render_template(v2_template, messages=openai["messages"], tools=openai["tools"])

    @pytest.mark.parametrize("spm", [False, True])
    def test_parallel_tool_results_rejected(self, spm: bool) -> None:
        template = build_test_template(TestConfig(version=TokenizerVersion.v2, spm=spm))
        request = build_tool_results_request(
            call_count=2,
            terminal_assistant=False,
            tools=[
                build_function_tool(name="tool_1", description=None, parameters={}),
                build_function_tool(name="tool_2", description=None, parameters={}),
            ],
        )

        with pytest.raises(ValueError, match=r"v2 does not support multiple tool results for one assistant turn"):
            render_template(template, **build_render_args(request))


class TestV3ToolResultParsing:
    @pytest.fixture()
    def v3_template(self) -> str:
        return build_test_template(TestConfig(version=TokenizerVersion.v3))

    @pytest.mark.parametrize(
        ("tool_name", "user_content", "arguments", "content", "assistant_content", "expected"),
        [
            pytest.param("add", "What is 2+2?", '{"a": 2, "b": 2}', "4", "4", '"content": 4', id="integer"),
            pytest.param("pi", "What is pi?", "{}", "3.14", "3.14", '"content": 3.14', id="float"),
        ],
    )
    def test_numeric_tool_result(
        self,
        v3_template: str,
        tool_name: str,
        user_content: str,
        arguments: str,
        content: str,
        assistant_content: str,
        expected: str,
    ) -> None:
        request = build_tool_call_request(
            tool_name=tool_name,
            call_id="abc123def",
            arguments=arguments,
            result_content=content,
            result_name=tool_name,
            assistant_content="",
            user_content=user_content,
            terminal_content=assistant_content,
            tools=[build_function_tool(name=tool_name, description=None, parameters={})],
        )

        output = render_template(v3_template, **build_render_args(request))
        assert expected in output

    @pytest.mark.parametrize("tool_call_id", [None, "short"])
    def test_invalid_call_id(self, v3_template: str, tool_call_id: str | None) -> None:
        if tool_call_id is None:
            # ToolMessage.to_openai requires a tool_call_id, so the missing-id
            # claim is exercised on the raw OpenAI dict the template receives.
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
                {"role": "tool", "content": "hi", "name": "greet"},
                {"role": "assistant", "content": "hi"},
            ]
        else:
            request = build_conversation_request(
                messages=[
                    UserMessage(content="Hello"),
                    AssistantMessage(
                        content="",
                        tool_calls=[ToolCall(id="abc123def", function=FunctionCall(name="greet", arguments="{}"))],
                    ),
                    ToolMessage(content="hi", name="greet", tool_call_id=tool_call_id),
                    AssistantMessage(content="hi"),
                ],
                tools=[build_function_tool(name="greet", description=None, parameters={})],
            )
            messages = build_render_args(request)["messages"]

        with pytest.raises(ValueError, match="call_id or tool_call_id"):
            render_template(
                v3_template,
                messages,
                tools=[{"type": "function", "function": {"name": "greet", "parameters": {}}}],
            )


class TestV7ToolCalls:
    def test_v7_content_and_tool_calls_accepted(self) -> None:
        template = build_test_template(TestConfig(version=TokenizerVersion.v7))
        request = build_tool_call_request(
            tool_name="get_weather",
            call_id="abc123def",
            arguments='{"city": "Paris"}',
            result_content="Sunny, 25C",
            result_name="get_weather",
            assistant_content="Let me check the weather for you.",
            user_content="What's the weather?",
            terminal_content="It's sunny and 25C in Paris.",
            tools=[build_function_tool(name="get_weather", description=None, parameters=_WEATHER_PARAMETERS)],
        )

        output = render_template(template, **build_render_args(request))
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
        template = build_test_template(TestConfig(version=version, spm=spm))
        request = build_aggregated_tool_calls_request(
            tools=[build_function_tool(name="get_weather", description="", parameters=_WEATHER_PARAMETERS)]
        )

        output = render_template(template, **build_render_args(request))

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
        template = build_test_template(TestConfig(version=version, spm=spm))
        request = build_tool_call_request(
            tool_name="greet",
            call_id="abc123def",
            arguments="",
            result_content="hi",
            result_name="greet",
            assistant_content="",
            terminal_content="Done",
            tools=[build_function_tool(name="greet", description=None, parameters={})],
        )

        output = render_template(template, **build_render_args(request))
        assert "{}" in output
        assert "greet" in output


class TestV11ToolCalls:
    def test_v11_tool_call_uses_call_id_token(self) -> None:
        template = build_test_template(TestConfig(version=TokenizerVersion.v11))
        request = build_tool_call_request(
            tool_name="get_weather",
            call_id="abc123def",
            arguments='{"city": "Paris"}',
            result_content="Sunny",
            result_name=None,
            user_content="Weather?",
            terminal_content="Sunny in Paris.",
            tools=[build_function_tool(name="get_weather", description=None, parameters=_WEATHER_PARAMETERS)],
        )

        output = render_template(template, **build_render_args(request))
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
        # The `call_id` alias is a raw-dict template feature; the request model
        # only carries `tool_call_id`, so this case renders the OpenAI dict directly.
        template = build_test_template(TestConfig(version=version, spm=spm))
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
                        "parameters": _ADD_PARAMETERS,
                    },
                }
            ],
        )
        assert "abc123def" in output
        assert "[TOOL_RESULTS]" in output


class TestV13PlusToolCalls:
    @pytest.mark.parametrize("version", [TokenizerVersion.v13, TokenizerVersion.v15])
    def test_v13_v15_accept_short_tool_call_id(self, version: TokenizerVersion) -> None:
        template = build_test_template(TestConfig(version=version))
        request = build_tool_call_request(
            tool_name="greet",
            call_id="x",
            arguments="{}",
            result_content="hi",
            result_name="greet",
            terminal_content="Done",
            tools=[build_function_tool(name="greet", description=None, parameters={})],
        )

        output = render_template(template, **build_render_args(request))
        assert "greet" in output
        assert "Done" in output
