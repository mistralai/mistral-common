from mistral_common.experimental.app.models import OpenAIChatCompletionRequest
from mistral_common.protocol.instruct.request import ChatCompletionRequest


def test_openai_request_accepts_null_assistant_content_and_converts() -> None:
    request = OpenAIChatCompletionRequest.model_validate(
        {
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location":"Paris"}',
                            },
                        }
                    ],
                },
            ]
        }
    )

    ChatCompletionRequest.from_openai(**request.model_dump())
