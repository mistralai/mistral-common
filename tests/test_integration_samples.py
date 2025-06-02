import json
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

from mistral_common.protocol.instruct.messages import ChatMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer


@pytest.fixture()
def samples_dir() -> Path:
    return Path(__file__).parent.joinpath("data").joinpath("samples")


def load_sample(samples_dir: Path, sample_name: str, version: int) -> Tuple[Any, str, Any]:
    with open(samples_dir / f"{sample_name}/sample.json", "r") as f:
        sample = json.load(f)

    with open(samples_dir / f"{sample_name}/text_v{version}.txt", "r") as f:
        text = f.read()

    with open(samples_dir / f"{sample_name}/tokens_v{version}.json", "r") as f:
        tokens = json.load(f)

    return sample, text, tokens["tokens"]


def get_tokenizer(version: int) -> MistralTokenizer:
    if version == 1:
        return MistralTokenizer.v1()
    elif version == 2:
        return MistralTokenizer.v2()
    elif version == 3:
        return MistralTokenizer.v3()
    else:
        raise ValueError(f"Invalid version: {version}")


@pytest.mark.parametrize(
    "sample",
    [
        {"sample_name": "get_weather_full", "versions": [2, 3]},
        {"sample_name": "get_weather_no_history", "versions": [2, 3]},
        {"sample_name": "several_calls", "versions": [2, 3]},
        {"sample_name": "no_tools", "versions": [1, 2, 3]},
        {"sample_name": "get_weather_no_system_prompt", "versions": [2, 3]},
        {"sample_name": "parallel_calls", "versions": [3]},
    ],
    ids=[
        "get_weather_full",
        "get_weather_no_history",
        "several_calls",
        "no_tools",
        "get_weather_no_system_prompt",
        "parallel_calls",
    ],
)
@pytest.mark.parametrize("version", [1, 2, 3], ids=["v1", "v2", "v3"])
def test_samples(sample: Dict[str, Any], version: int, samples_dir: Path) -> None:
    if version not in sample["versions"]:
        pytest.skip(f"Sample {sample['sample_name']} not available for version {version}")

    mistral_tokenizer = get_tokenizer(version)
    instruct_request, text, tokens = load_sample(samples_dir, sample["sample_name"], version)
    chat_completion_string = json.dumps(
        {"model": "debug", "messages": instruct_request["messages"], "tools": instruct_request["tools"]}
    )
    chat_completion_request = ChatCompletionRequest[ChatMessage].model_validate_json(chat_completion_string)
    tokenized = mistral_tokenizer.encode_chat_completion(chat_completion_request)
    assert tokenized.text == text
    assert tokenized.tokens == tokens
