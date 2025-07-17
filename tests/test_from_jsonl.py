import json
from pathlib import Path

from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.instruct.request import FIMRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer


def test_from_jsonl_single(tmp_path: Path) -> None:
    file = tmp_path / "data.jsonl"
    req = {"messages": [{"role": "user", "content": "a"}]}
    file.write_text(json.dumps(req) + "\n")

    tokenizer = MistralTokenizer.v2()
    tokenized_file = tokenizer.from_jsonl(str(file))
    direct = tokenizer.encode_chat_completion(
        ChatCompletionRequest(messages=[UserMessage(content="a")])
    )
    assert tokenized_file.tokens == direct.tokens
    assert tokenized_file.text == direct.text


def test_from_jsonl_multiple(tmp_path: Path) -> None:
    file = tmp_path / "multi.jsonl"
    reqs = [
        {"messages": [{"role": "user", "content": "a"}]},
        {"messages": [{"role": "user", "content": "b"}]},
    ]
    file.write_text("\n".join(json.dumps(r) for r in reqs))

    tokenizer = MistralTokenizer.v2()
    tokenized = tokenizer.from_jsonl(str(file))
    assert isinstance(tokenized, list)
    assert len(tokenized) == 2
    direct0 = tokenizer.encode_chat_completion(
        ChatCompletionRequest(messages=[UserMessage(content="a")])
    )
    direct1 = tokenizer.encode_chat_completion(
        ChatCompletionRequest(messages=[UserMessage(content="b")])
    )
    assert [t.tokens for t in tokenized] == [direct0.tokens, direct1.tokens]


def test_from_jsonl_fim(tmp_path: Path) -> None:
    file = tmp_path / "fim.jsonl"
    req = {"prompt": "a", "suffix": "b"}
    file.write_text(json.dumps(req))

    tokenizer = MistralTokenizer.v2()
    tokenized_file = tokenizer.from_jsonl(str(file))
    direct = tokenizer.encode_fim(FIMRequest(prompt="a", suffix="b"))
    assert tokenized_file.tokens == direct.tokens
