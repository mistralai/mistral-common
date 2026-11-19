from typing import Any

import pytest

from mistral_common.protocol.instruct.chunk import TextChunk, ThinkChunk
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    ReasoningFieldFormat,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.tool_calls import FunctionCall, ToolCall
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.integration.chat_recipes import (
    build_conversation_request,
    build_function_tool,
    build_thinking_assistant_request,
)
from tests.utils.chat_templates.configs import TestConfig, has_released_template
from tests.utils.chat_templates.helpers import (
    build_render_args,
    build_template_config,
    build_test_template,
    load_golden_template,
    render_template,
)

THINK_CONFIGS = [
    (TokenizerVersion.v13, False, False),
    (TokenizerVersion.v13, True, False),
    (TokenizerVersion.v15, False, False),
    (TokenizerVersion.v15, True, False),
]
GOLDEN_THINK_CONFIGS = [
    (version, image, audio)
    for version, image, audio in THINK_CONFIGS
    if has_released_template(TestConfig(version=version, image=image, audio=audio, think=True))
]

_WEATHER_PARAMETERS: dict[str, Any] = {"type": "object", "properties": {"city": {"type": "string"}}}


class TestReasoningContentConversion:
    @pytest.mark.parametrize(("version", "image", "audio"), THINK_CONFIGS)
    def test_reasoning_content_to_thinking_chunk(self, version: TokenizerVersion, image: bool, audio: bool) -> None:
        template = build_test_template(TestConfig(version=version, image=image, audio=audio, think=True))
        request = build_thinking_assistant_request(
            user_content="What is 2+2?", thinking="Let me add 2 and 2.", text="The answer is 4."
        )

        output = render_template(
            template, **build_render_args(request, reasoning_field_format=ReasoningFieldFormat.reasoning_content)
        )
        assert "[THINK]Let me add 2 and 2.[/THINK]The answer is 4." in output

    @pytest.mark.parametrize(("version", "image", "audio"), THINK_CONFIGS)
    def test_reasoning_field_to_thinking_chunk(self, version: TokenizerVersion, image: bool, audio: bool) -> None:
        template = build_test_template(TestConfig(version=version, image=image, audio=audio, think=True))
        request = build_thinking_assistant_request(
            user_content="What is 3+3?", thinking="Adding three and three.", text="The answer is 6."
        )

        output = render_template(
            template, **build_render_args(request, reasoning_field_format=ReasoningFieldFormat.reasoning)
        )
        assert "[THINK]Adding three and three.[/THINK]The answer is 6." in output

    @pytest.mark.parametrize(("version", "image", "audio"), THINK_CONFIGS)
    def test_reasoning_content_takes_precedence_over_reasoning(
        self, version: TokenizerVersion, image: bool, audio: bool
    ) -> None:
        # Both flat fields on one assistant message are not representable as a
        # ChatCompletionRequest (one ThinkChunk serializes to one field), so this
        # precedence claim renders the raw OpenAI dict directly.
        template = build_test_template(TestConfig(version=version, image=image, audio=audio, think=True))
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
    def test_reasoning_content_only_no_text_content(self, version: TokenizerVersion, image: bool, audio: bool) -> None:
        template = build_test_template(TestConfig(version=version, image=image, audio=audio, think=True))
        request = build_thinking_assistant_request(user_content="Hi", thinking="Just thinking...", text="")

        output = render_template(
            template, **build_render_args(request, reasoning_field_format=ReasoningFieldFormat.reasoning_content)
        )
        assert "[THINK]Just thinking...[/THINK]" in output

    @pytest.mark.parametrize(("version", "image", "audio"), THINK_CONFIGS)
    def test_reasoning_content_with_tool_calls(self, version: TokenizerVersion, image: bool, audio: bool) -> None:
        template = build_test_template(TestConfig(version=version, image=image, audio=audio, think=True))
        request = build_conversation_request(
            messages=[
                UserMessage(content="What's the weather?"),
                AssistantMessage(
                    content=[ThinkChunk(thinking="I should call a tool."), TextChunk(text="")],
                    tool_calls=[
                        ToolCall(id="abc123", function=FunctionCall(name="get_weather", arguments='{"city": "Paris"}'))
                    ],
                ),
                ToolMessage(name="get_weather", content="22C", tool_call_id="abc123"),
                AssistantMessage(content="It's 22C in Paris."),
            ],
            tools=[build_function_tool(name="get_weather", description="", parameters=_WEATHER_PARAMETERS)],
        )

        output = render_template(
            template, **build_render_args(request, reasoning_field_format=ReasoningFieldFormat.reasoning_content)
        )
        assert "[THINK]I should call a tool.[/THINK]" in output
        assert "get_weather" in output

    @pytest.mark.parametrize(("version", "image", "audio"), THINK_CONFIGS)
    def test_reasoning_aggregation_consecutive_assistants(
        self, version: TokenizerVersion, image: bool, audio: bool
    ) -> None:
        template = build_test_template(TestConfig(version=version, image=image, audio=audio, think=True))
        request = build_conversation_request(
            messages=[
                UserMessage(content="Solve this."),
                AssistantMessage(content=[ThinkChunk(thinking="Step 1 reasoning"), TextChunk(text="Partial answer.")]),
                AssistantMessage(content=[ThinkChunk(thinking="Step 2 reasoning"), TextChunk(text="Final answer.")]),
            ]
        )

        output = render_template(
            template, **build_render_args(request, reasoning_field_format=ReasoningFieldFormat.reasoning_content)
        )
        # Both reasoning traces should appear as thinking chunks
        assert "[THINK]Step 1 reasoning[/THINK]" in output
        assert "[THINK]Step 2 reasoning[/THINK]" in output
        # Text from both messages should be aggregated
        assert "Partial answer." in output
        assert "Final answer." in output

    @pytest.mark.parametrize(("version", "image", "audio"), GOLDEN_THINK_CONFIGS)
    def test_reasoning_static_dynamic_parity(self, version: TokenizerVersion, image: bool, audio: bool) -> None:
        config = TestConfig(version=version, image=image, audio=audio, think=True)
        static_template = load_golden_template(build_template_config(config))
        dynamic_template = build_test_template(config)
        request = build_conversation_request(
            messages=[
                UserMessage(content="Hello"),
                AssistantMessage(content=[ThinkChunk(thinking="Thinking..."), TextChunk(text="Hi!")]),
                UserMessage(content="How are you?"),
                AssistantMessage(
                    content=[ThinkChunk(thinking="Considering the question."), TextChunk(text="I'm well!")]
                ),
            ]
        )
        render_kwargs = build_render_args(request, reasoning_field_format=ReasoningFieldFormat.reasoning_content)

        static_output = render_template(static_template, **render_kwargs)
        dynamic_output = render_template(dynamic_template, **render_kwargs)

        assert static_output == dynamic_output

    def test_reasoning_content_with_none_content(self) -> None:
        # A top-level reasoning field with `content: None` is not representable
        # as a ChatCompletionRequest, so this renders the raw OpenAI dict.
        from mistral_common.integrations.chat_templates.template_generator import TemplateConfig, build_chat_template

        config = TemplateConfig(
            version=TokenizerVersion.v15,
            thinking_support=True,
            reasoning_effort_default="none",
            reasoning_effort_values=("none", "high"),
        )
        template = build_chat_template(config)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "Let me think about this.",
            },
        ]
        result = render_template(template, messages=messages)
        assert "[THINK]" in result
        assert "Let me think about this." in result

    @pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
    @pytest.mark.parametrize(("version", "image", "audio"), THINK_CONFIGS)
    def test_reasoning_with_existing_think_chunks_raises(
        self, version: TokenizerVersion, image: bool, audio: bool, field: str
    ) -> None:
        # A message carrying both inline thinking chunks and a flat reasoning
        # field cannot be produced by to_openai (it converts one into the other),
        # so this rejection renders the raw OpenAI dict directly.
        template = build_test_template(TestConfig(version=version, image=image, audio=audio, think=True))
        reasoning_text = (
            "Top-level reasoning" if field == "reasoning_content" else "Top-level reasoning via reasoning field"
        )
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                field: reasoning_text,
                "content": [
                    {"type": "thinking", "thinking": "Inline thinking"},
                    {"type": "text", "text": "Hello!"},
                ],
            },
        ]

        with pytest.raises(
            ValueError,
            match="Message cannot have both thinking chunks in content and a top-level"
            " `reasoning` or `reasoning_content` field",
        ):
            render_template(template, messages)


class TestReasoningEdgeCases:
    def test_non_think_template_ignores_reasoning_field(self) -> None:
        template = build_test_template(TestConfig(version=TokenizerVersion.v13))
        request = build_thinking_assistant_request(user_content="Hi", thinking="This is ignored", text="Hello!")

        output = render_template(
            template, **build_render_args(request, reasoning_field_format=ReasoningFieldFormat.reasoning_content)
        )
        assert "[THINK]" not in output
        assert "Hello!" in output

    @pytest.mark.parametrize("value", ["", None])
    def test_empty_or_none_reasoning_ignored(self, value: str | None) -> None:
        template = build_test_template(TestConfig(version=TokenizerVersion.v13, think=True))
        request = build_thinking_assistant_request(user_content="Hi", thinking=value, text="Hello!")

        output = render_template(
            template, **build_render_args(request, reasoning_field_format=ReasoningFieldFormat.reasoning_content)
        )
        assert "[THINK]" not in output
        assert "Hello!" in output


class TestClosedAttribute:
    @pytest.mark.parametrize(
        ("version", "image"),
        [
            (TokenizerVersion.v13, False),
            (TokenizerVersion.v13, True),
            (TokenizerVersion.v15, False),
            (TokenizerVersion.v15, True),
        ],
    )
    def test_closed_true_emits_close_tag(self, version: TokenizerVersion, image: bool) -> None:
        template = build_test_template(TestConfig(version=version, image=image, think=True))
        request = build_thinking_assistant_request(
            user_content="Hi", thinking="Done thinking", text="Hello!", closed=True
        )

        output = render_template(
            template, **build_render_args(request, reasoning_field_format=ReasoningFieldFormat.thinking_chunks)
        )
        assert "[THINK]Done thinking[/THINK]Hello!" in output

    @pytest.mark.parametrize(
        ("version", "image"),
        [
            (TokenizerVersion.v13, False),
            (TokenizerVersion.v13, True),
            (TokenizerVersion.v15, False),
            (TokenizerVersion.v15, True),
        ],
    )
    def test_closed_false_omits_close_tag(self, version: TokenizerVersion, image: bool) -> None:
        template = build_test_template(TestConfig(version=version, image=image, think=True))
        request = build_thinking_assistant_request(user_content="Hi", thinking="Ongoing thought", closed=False)

        output = render_template(
            template, **build_render_args(request, reasoning_field_format=ReasoningFieldFormat.thinking_chunks)
        )
        assert "[THINK]Ongoing thought" in output
        assert "[/THINK]" not in output


class TestPlainThink:
    def test_plain_think_template_produces_correct_output(self) -> None:
        template = build_test_template(TestConfig(version=TokenizerVersion.v11, plain_think=True))
        request = build_thinking_assistant_request(
            user_content="Solve this", thinking="Let me think...", text="The answer is 42."
        )

        output = render_template(
            template, **build_render_args(request, reasoning_field_format=ReasoningFieldFormat.thinking_chunks)
        )
        assert "<think>Let me think...</think>The answer is 42." in output
        assert "[THINK]" not in output
        assert "[/THINK]" not in output

    @pytest.mark.parametrize("image", [False, True])
    def test_plain_think_static_dynamic_parity(self, image: bool) -> None:
        config = TestConfig(version=TokenizerVersion.v11, image=image, plain_think=True)
        static_template = load_golden_template(build_template_config(config))
        dynamic_template = build_test_template(config)
        request = build_thinking_assistant_request(user_content="Hello", thinking="Thinking...", text="Hi!")
        render_kwargs = build_render_args(request, reasoning_field_format=ReasoningFieldFormat.thinking_chunks)

        static_output = render_template(static_template, **render_kwargs)
        dynamic_output = render_template(dynamic_template, **render_kwargs)
        assert static_output == dynamic_output

    def test_plain_think_reasoning_content_conversion(self) -> None:
        template = build_test_template(TestConfig(version=TokenizerVersion.v11, plain_think=True))
        request = build_thinking_assistant_request(user_content="What is 2+2?", thinking="Let me add.", text="4.")

        output = render_template(
            template, **build_render_args(request, reasoning_field_format=ReasoningFieldFormat.reasoning_content)
        )
        assert "<think>Let me add.</think>4." in output
        assert "[THINK]" not in output

    def test_plain_think_closed_false(self) -> None:
        template = build_test_template(TestConfig(version=TokenizerVersion.v11, plain_think=True))
        request = build_thinking_assistant_request(user_content="Hi", thinking="Still thinking...", closed=False)

        output = render_template(
            template, **build_render_args(request, reasoning_field_format=ReasoningFieldFormat.thinking_chunks)
        )
        assert "<think>Still thinking..." in output
        assert "</think>" not in output

    def test_plain_think_image_template(self) -> None:
        from mistral_common.protocol.instruct.chunk import ImageURLChunk

        template = build_test_template(TestConfig(version=TokenizerVersion.v11, image=True, plain_think=True))
        request = build_conversation_request(
            messages=[
                UserMessage(
                    content=[
                        TextChunk(text="What is this?"),
                        ImageURLChunk(image_url="http://example.com/img.png"),
                    ]
                ),
                AssistantMessage(content=[ThinkChunk(thinking="It looks like..."), TextChunk(text="A red square.")]),
            ]
        )

        output = render_template(
            template, **build_render_args(request, reasoning_field_format=ReasoningFieldFormat.thinking_chunks)
        )
        assert "[IMG]" in output
        assert "<think>It looks like...</think>A red square." in output
        assert "[THINK]" not in output

    def test_plain_think_reasoning_with_existing_think_chunks_raises(self) -> None:
        # Both inline thinking chunks and a flat reasoning field on one message
        # are not representable as a ChatCompletionRequest; render the raw dict.
        template = build_test_template(TestConfig(version=TokenizerVersion.v11, plain_think=True))
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

        with pytest.raises(
            ValueError,
            match="Message cannot have both thinking chunks in content and a top-level"
            " `reasoning` or `reasoning_content` field",
        ):
            render_template(template, messages)
