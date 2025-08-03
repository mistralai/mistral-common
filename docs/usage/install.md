# Install

## Pip

You can install the library using pip:
```sh
pip install mistral-common
```

We propose different dependencies to install depending on your needs:
- `image`: to use the image tokenizers.
- `audio`: to use the audio tokenizers.
- `hf-hub`: to download the tokenizers from the Hugging Face Hub.
- `sentencepiece`: to allow the use of SentencePiece tokenizers. This is now optional as we only release `Tekken` tokenizers for recent models.
- \[Experimental\] `server`: to use our tokenizers in a server mode.

Each dependency is optional and can be installed separately or all together using the following commands:
```sh
pip install "mistral-common[image]"
pip install "mistral-common[audio]"
pip install "mistral-common[hf-hub]"
pip install "mistral-common[sentencepiece]"
pip install "mistral-common[server]"
pip install "mistral-common[image,audio,hf-hub,sentencepiece,server]"
```

## From source

To build it for source, you can clone the repository and install it using [uv](https://github.com/astral-sh/uv) or pip. We recommend using uv for faster and more reliable dependency resolution:
```sh
git clone https://github.com/mistralai/mistral-common.git
cd mistral-common
uv sync --frozen --extra image # or --all-extras to install all dependencies.
```

For development, you can install the `dev` group and/or the `docs` groups:
```sh
uv sync --frozen --all-extras --group dev # and/or --group docs.
```
