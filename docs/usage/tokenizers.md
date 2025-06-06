# Tokenizers

## How do we perform tokenization ?

Tokenization is the process of converting a sequence of characters into a sequence of tokens. To dive deeper into the tokenization process, you can read the [tokenization guide](https://docs.mistral.ai/guides/tokenization/) on our website.

Historically, we based our tokenizers on the following libraries:

- [**SentencePiece**](https://github.com/google/sentencepiece): we moved from this library to the one below.  
- [**tiktoken**](https://github.com/openai/tiktoken): we use this library for our **tekken** tokenizers. It is more efficient for multilingual tokenization.

## Tokenizer

For our releases, we provide a tokenizer file that you can use to load a tokenizer. If the tokenizer is based on the `tiktoken` library, the file is generally named `tekken.json`. Here is how you could load the [Mistral Small 3.1 Instruct](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503)'s tokenizer:

```python
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

tokenizer = MistralTokenizer.from_hf_hub("mistralai/Mistral-Small-3.1-24B-Instruct-2503", token="your_hf_token")

chat_completion = ChatCompletionRequest(
    messages=[
        UserMessage(role="user", content="Hello, how are you?"),
    ]
)
tokenizer.encode_chat_completion(chat_completion).tokens
# [1, 3, 22177, 1044, 2606, 1584, 1636, 1063, 4]
```

We provide three layer of abstraction for our tokenizers:

- [Tekkenizer][mistral_common.tokens.tokenizers.tekken.Tekkenizer] or [SentencePieceTokenizer][mistral_common.tokens.tokenizers.sentencepiece.SentencePieceTokenizer]: raw tokenizers to handle raw data to convert them into ids and vice versa.
- [InstructTokenizer][mistral_common.tokens.tokenizers.instruct.InstructTokenizer]: instruct tokenizers that wrap the raw tokenizers to add several helper methods for the different tasks (chat completion or FIM). They are versioned.
- [MistralTokenizer][mistral_common.tokens.tokenizers.mistral.MistralTokenizer]: mistral tokenizers that validate the requests, see [requests section](./requests.md), and call the instruct tokenizers.

For instance, you can directly load the [Tekkenizer][mistral_common.tokens.tokenizers.tekken.Tekkenizer]:

```python
from huggingface_hub import hf_hub_download

from mistral_common.tokens.tokenizers.tekken  import Tekkenizer

model_id = "mistralai/Devstral-Small-2505"
tekken_file = hf_hub_download(repo_id=model_id, filename="tekken.json", token="your_hf_token")

tokenizer = Tekkenizer.from_file(tekken_file)
tokenizer.encode("Plz tekkenize me", bos=True, eos=True)
# [1, 4444, 1122, 16058, 3569, 2033, 1639, 2]
```

See the [Examples section](../examples/index.md) for examples on how to use the tokenizers with our models.