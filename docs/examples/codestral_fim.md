# Fill-In-the-Middle (FIM) using Codestral

We will use [mistral-inference](https://github.com/mistralai/mistral-inference) to run [Codestral](https://huggingface.co/mistralai/Codestral-22B-v0.1).

## Setup

First, install `mistral-inference` and `mistral-common`:
```sh
pip install mistral-inference mistral-common
```

## Running the model

### Download the model

Download the model from [Hugging Face](https://huggingface.co/mistralai/Codestral-22B-v0.1).
```sh
pip install huggingface_hub[cli]

huggingface-cli login --token your_hf_token
huggingface-cli download mistralai/Codestral-22B-v0.1 --local-dir ~/codestral-22B-240529
```

### Perform the FIM task

The Codestral tokenizer is the v3 [MistralTokenizer][mistral_common.tokens.tokenizers.mistral.MistralTokenizer] from Mistral.

```python
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.instruct.request import FIMRequest

tokenizer = MistralTokenizer.v3()
model = Transformer.from_folder("~/codestral-22B-240529")

prefix = """def add("""
suffix = """    return sum"""

request = FIMRequest(prompt=prefix, suffix=suffix)

tokens = tokenizer.encode_fim(request).tokens

out_tokens, _ = generate([tokens], model, max_tokens=256, temperature=0.0, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
result = tokenizer.decode(out_tokens[0])

middle = result.split(suffix)[0].strip()
print(middle)
```

The output should be:
```python
"""num1, num2):

    sum = num1 + num2"""
```