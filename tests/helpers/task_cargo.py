import re
from collections.abc import Mapping

APPROVED_PLACEHOLDERS = (
    "TASK_ID",
    "SPEC_ID",
    "CONTRACT_REVISION",
    "SPEC_BRANCH",
    "TASK_BASE_SHA",
    "TASK_BRANCH",
    "TASK_WORKTREE",
    "EXPECTED_TASK_HEAD",
    "REPO_ROOT",
)

_TOKEN = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")


class TaskCargoBindingError(ValueError):
    """Raised when task cargo contains an unbound or stale placeholder."""


def render_task_cargo(template: str, *, values: Mapping[str, object]) -> str:
    """Render approved task placeholders without consuming tokenizer markers."""
    unknown_values = set(values) - set(APPROVED_PLACEHOLDERS)
    if unknown_values:
        names = ", ".join(sorted(unknown_values))
        raise TaskCargoBindingError(f"unknown task values: {names}")

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in APPROVED_PLACEHOLDERS:
            if name not in values:
                raise TaskCargoBindingError(f"missing task value for {name}")
            return str(values[name])
        if name.endswith("_token"):
            return match.group(0)
        raise TaskCargoBindingError(f"unknown task placeholder: {name}")

    return _TOKEN.sub(replace, template)
