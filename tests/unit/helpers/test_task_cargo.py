import pytest

from tests.helpers.task_cargo import (
    APPROVED_PLACEHOLDERS,
    TaskCargoBindingError,
    render_task_cargo,
)


def test_all_approved_placeholders_render() -> None:
    values = {name: f"value-{name.lower()}" for name in APPROVED_PLACEHOLDERS}
    template = " ".join(f"{{{{{name}}}}}" for name in APPROVED_PLACEHOLDERS)
    assert render_task_cargo(template, values=values) == " ".join(values[name] for name in APPROVED_PLACEHOLDERS)


@pytest.mark.parametrize("placeholder", ["BASE_SHA", "SPEC_SHA", "OLD_TASK_BASE_SHA", "TASK_SHA"])
def test_legacy_placeholders_fail_binding(placeholder: str) -> None:
    with pytest.raises(TaskCargoBindingError, match=placeholder):
        render_task_cargo(f"{{{{{placeholder}}}}}", values={})


def test_literal_tokenizer_markers_remain_untouched() -> None:
    template = "{{ bos_token }}<s>[INST]{{ eos_token }}"
    assert render_task_cargo(template, values={}) == template
