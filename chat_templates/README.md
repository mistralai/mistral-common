# Mistral Common - Chat Templates

This directory contains JINJA chat templates for Mistral AI models, designed to work similarly to our tokenization library [`mistral-common`](https://github.com/mistralai/mistral-common).

## Community Chat Templates

While we recommend using `mistral-common` we created this folder due to high demand during our releases for chat templates. The motivation to use the ones stored in this folder are the following:

- **Official Source**: Maintained by Mistral AI team.
- **Open Collaboration**: Community can submit PRs to improve templates.
- **Freshest Versions**: Always contains the latest officially supported templates. They can be often updated especially around new releases and this makes it easy to keep up to date.

You can find the templates of all our tokenizers inside the [`templates/`](./templates/) folder. They are named following this format:
```txt
<tokenizer_version>_<feature1>_<feature2>_..._<featureN>.jinja`
```

The version corresponds to the `TokenizerVersion` as defined by `mistral-common`. They handle differently system prompts, tool calls, multimodalities, ... Make sure to select the correct one for the model you're using. You can find the correct version usually by taking a look at the official tokenizer files we release:
- For Sentencepiece: `tokenizer.model.<version_number>`
- For Tekken: open the `tekken.json` file and look the entry `["config"]["version"]`

Supported features are:
- image
- audio
- think

The JINJA files do not contain a default system prompt as it should be defined per model. To customize the system prompt, you can either copy the template and modify it or use the `chat_templates.py` script:
```bash
python chat_templates.py -h
```

## Why we still recommend to use `mistral-common`?

While chat templates are widely available in the open-source ecosystem, we recommend using `mistral-common` for:
- **Performance**: Performance improvements over using chat templates can be observed for training and inference. This is due to the fact that `mistral-common` is used for training our models and what we use also for inference. This means that there are no conversion issues, no subtle errors and full support for all the capabilities of our models.
- **Validation**: Ensures templates are correctly formatted and compatible with Mistral models.
- **Fully Tested**: Rigorously tested for all our tokenizers and backward compatible.


## Why use chat templates over `mistral-common` ?

- **Cross-Platform**: Templates work across different implementations (Python, JS, etc.).
- **Integration**: Already integrated in lots of different libraries such as Transformers, llama.cpp... We try to also spread `mistral-common` but it is not always possible.


## Contributing

Found an issue or want to add a new template? Open an issue or a PR !
