# Contributing to mistral-common

Thank you for helping improve mistral-common. Contributions can include bug reports, feature proposals, documentation, tests, and code changes.

## Before you contribute

1. Search the [existing issues](https://github.com/mistralai/mistral-common/issues) for related discussions.
2. Open an issue before starting work. Explain the problem, use case, or feature you would like to discuss. If an issue already covers it, add your context there instead of opening a duplicate.
3. A pull request is optional. An issue or discussion that clarifies a problem is still valuable, even when no implementation follows.

### Guidance for agentic contributions

Agent contributions are welcome.

Agents must read and carefully follow the [`AGENTS.md`](AGENTS.md) file in the repository root before investigating or changing the project. It is the authoritative source for repository structure, style, testing, documentation, and workflow conventions.

Although such contributions are welcome, sloppy PRs or issues will be closed without notice.

## Pull requests

When you open a pull request:

- Make it self-contained. The description must clearly explain the problem and motivation, or link to an issue that does so.
- Describe what changed and why.
- Add or update meaningful tests for changed behavior.
- Update documentation and public API examples when needed.
- Follow the project style: use type hints, required Google-style docstrings, and concise comments that explain intent rather than restating code.
- Run Ruff, mypy, and pytest.

Use the pull request template to confirm these expectations. Maintainers may ask for changes before review or merge.

## Local development

The project uses Python 3.10–3.14, [uv](https://docs.astral.sh/uv/), pytest, Ruff, and mypy. A typical setup is:

```bash
uv venv
source .venv/bin/activate
uv sync --frozen --all-extras --group dev
uv run pre-commit install
```

To build the documentation locally (uses [Zensical](https://zensical.org), which reads `mkdocs.yml`):

```bash
uv sync --frozen --all-extras --group docs
uv run python docs/gen_api_pages.py  # generate the code reference pages
uv run zensical serve                # live preview at http://localhost:8000
```

Run the checks relevant to your change before opening a pull request:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
```

## Respectful collaboration

We are committed to a welcoming and respectful community. Be kind, professional, and constructive; assume good intent; and focus criticism on ideas, code, and documentation rather than people. Harassment, discrimination, personal attacks, and other disrespectful behavior are not acceptable. If a discussion becomes difficult, pause and ask a maintainer for help.
