r"""Shared Fill-In-the-Middle (FIM) request builders.

Single source of truth for `FIMRequest` inputs used by the FIM tokenizer tests
and the golden registry.
"""

from mistral_common.protocol.fim.request import FIMRequest


def fim_request(prompt: str = "def f(", suffix: str | None = "return a + b") -> FIMRequest:
    r"""Build a `FIMRequest` with a deterministic prompt and suffix.

    Args:
        prompt: The prompt to be completed.
        suffix: The suffix following the completion, or `None` for prompt-only.

    Returns:
        A `FIMRequest` with the given prompt and suffix.
    """
    return FIMRequest(prompt=prompt, suffix=suffix)


def prompt_suffix_request() -> FIMRequest:
    r"""Build the baseline FIM request with both a prompt and a suffix.

    Returns:
        A `FIMRequest` with a non-empty prompt and suffix.
    """
    return fim_request(prompt="def f(", suffix="return a + b")


def prompt_only_request() -> FIMRequest:
    r"""Build a FIM request with no suffix.

    Returns:
        A `FIMRequest` with `suffix=None`.
    """
    return fim_request(prompt="def f(", suffix=None)


def empty_suffix_request() -> FIMRequest:
    r"""Build a FIM request with an empty-string suffix.

    An empty string is falsy, so `InstructTokenizerV2.encode_fim` takes the same branch
    as `suffix=None`; this builder exists to make that equivalence an explicit, tested
    case rather than an accidental one.

    Returns:
        A `FIMRequest` with `suffix=""`.
    """
    return fim_request(prompt="def f(", suffix="")


def empty_prompt_request() -> FIMRequest:
    r"""Build a FIM request with an empty prompt and no suffix.

    Returns:
        A `FIMRequest` with `prompt=""` and `suffix=None`.
    """
    return fim_request(prompt="", suffix=None)


def newline_suffix_request() -> FIMRequest:
    r"""Build a FIM request whose suffix starts with a newline and leading whitespace.

    The FIM sentinel trick in `InstructTokenizerV2._encode_infilling` exists precisely to
    preserve leading whitespace like this in the suffix, so this is the highest-value
    suffix shape to exercise.

    Returns:
        A `FIMRequest` whose suffix starts with `"\n    "`.
    """
    return fim_request(prompt="def f(", suffix="\n    return a + b")


def leading_space_suffix_request() -> FIMRequest:
    r"""Build a FIM request whose suffix starts with a single leading space.

    Returns:
        A `FIMRequest` whose suffix starts with a space.
    """
    return fim_request(prompt="def f(", suffix=" return a + b")


def unicode_suffix_request() -> FIMRequest:
    r"""Build a FIM request whose suffix contains a non-ASCII character.

    Returns:
        A `FIMRequest` whose suffix contains `"café"`.
    """
    return fim_request(prompt="def f(", suffix="return café")


def registry_fim_requests() -> dict[str, FIMRequest]:
    r"""Return the curated FIM requests encoded by the golden registry.

    Returns:
        Ordered mapping from request name to `FIMRequest`.
    """
    return {
        "prompt_suffix": prompt_suffix_request(),
        "prompt_only": prompt_only_request(),
        "empty_suffix": empty_suffix_request(),
        "empty_prompt": empty_prompt_request(),
        "newline_suffix": newline_suffix_request(),
        "leading_space_suffix": leading_space_suffix_request(),
        "unicode_suffix": unicode_suffix_request(),
    }
