r"""Subprocess workflow tests for `scripts/generate_chat_template.py`.

Each case invokes the real script as a process and observes the exit status,
stdout/stderr, and the output file. Successes must save the expected content
and print the save path; failures must exit nonzero with the intended
diagnostic and leave no output file.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.utils.chat_templates.helpers import TestConfig, build_tekken_json

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "generate_chat_template.py"


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    r"""Run the generator script as a subprocess and capture its result."""
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        capture_output=True,
        text=True,
    )


def assert_cli_success(result: subprocess.CompletedProcess[str], output_path: Path) -> str:
    r"""Assert a successful run: zero exit, saved file, and the printed save path."""
    assert result.returncode == 0, result.stderr
    assert output_path.exists()
    assert f"Template saved to {output_path}" in result.stdout
    return output_path.read_text(encoding="utf-8")


def assert_cli_failure(result: subprocess.CompletedProcess[str], output_path: Path, diagnostic: str) -> None:
    r"""Assert a failed run: nonzero exit, the intended diagnostic, and no output file."""
    assert result.returncode != 0
    assert diagnostic in result.stderr
    assert not output_path.exists()


@pytest.fixture(scope="session")
def tekken_think_v13_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Path to a v13 think tekken.json used for auto-detect CLI tests."""
    build_dir = tmp_path_factory.mktemp("tokenizer")
    config = TestConfig(version=TokenizerVersion.v13, think=True)
    return build_tekken_json(config=config, output_dir=build_dir)


@pytest.fixture(scope="session")
def tekken_v11_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Path to a plain v11 tekken.json used for auto-detect CLI plain thinking tests."""
    build_dir = tmp_path_factory.mktemp("tokenizer")
    config = TestConfig(version=TokenizerVersion.v11)
    return build_tekken_json(config=config, output_dir=build_dir)


@pytest.fixture(scope="session")
def tekken_v15_json_schema_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    r"""Path to a v15 Tekken tokenizer with both model settings builders."""
    build_dir = tmp_path_factory.mktemp("tokenizer")
    config = TestConfig(
        version=TokenizerVersion.v15,
        model_settings_fields=frozenset({"reasoning_effort", "json_schema"}),
    )
    return build_tekken_json(config=config, output_dir=build_dir)


def test_cli_generates_template(tmp_path: Path) -> None:
    """CLI generates a template file with expected content."""
    output_path = tmp_path / "template.jinja"
    result = run_cli(
        "--version",
        "v15",
        "--reasoning_effort_values",
        "none,high",
        "--reasoning_effort_default",
        "none",
        "--saving_path",
        str(output_path),
    )
    content = assert_cli_success(result, output_path)
    assert "bos_token" in content
    assert "[MODEL_SETTINGS]" in content


def test_cli_model_settings_fields_json_schema(tmp_path: Path) -> None:
    output_path = tmp_path / "template.jinja"
    result = run_cli(
        "--version",
        "v15",
        "--model_settings_fields",
        "json_schema",
        "--saving_path",
        str(output_path),
    )
    content = assert_cli_success(result, output_path)
    assert "response_format" in content


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        pytest.param(["--version", "v15"], "Reasoning effort values are required", id="v15_missing_values"),
        pytest.param(
            ["--version", "v15", "--reasoning_effort_values", "none,high"],
            "Reasoning effort default is required",
            id="v15_missing_default",
        ),
        pytest.param(
            ["--version", "v15", "--reasoning_effort_default", "none"],
            "Reasoning effort values are required",
            id="v15_default_without_values",
        ),
        pytest.param(
            ["--version", "v15", "--model_settings_fields", "reasoning_effort"],
            "Reasoning effort values are required",
            id="v15_reasoning_effort_field_missing_values",
        ),
    ],
)
def test_cli_requires_reasoning_effort_arguments(tmp_path: Path, arguments: list[str], message: str) -> None:
    output_path = tmp_path / "template.jinja"
    result = run_cli(*arguments, "--saving_path", str(output_path))
    assert_cli_failure(result, output_path, message)


def test_cli_model_settings_fields_default(tmp_path: Path) -> None:
    output_path = tmp_path / "template.jinja"
    result = run_cli(
        "--version",
        "v15",
        "--reasoning_effort_values",
        "none,high",
        "--reasoning_effort_default",
        "none",
        "--saving_path",
        str(output_path),
    )
    content = assert_cli_success(result, output_path)
    assert "[MODEL_SETTINGS]" in content
    assert "response_format" not in content


def test_cli_model_settings_fields_empty_value_rejected(tmp_path: Path) -> None:
    output_path = tmp_path / "template.jinja"
    result = run_cli(
        "--version",
        "v15",
        "--model_settings_fields",
        "",
        "--saving_path",
        str(output_path),
    )
    assert_cli_failure(result, output_path, "--model_settings_fields must list at least one field")


def test_cli_model_settings_fields_invalid_choice(tmp_path: Path) -> None:
    output_path = tmp_path / "template.jinja"
    result = run_cli(
        "--version",
        "v15",
        "--model_settings_fields",
        "bogus",
        "--saving_path",
        str(output_path),
    )
    assert_cli_failure(result, output_path, "Unknown model settings fields")


def test_cli_model_settings_fields_below_v15(tmp_path: Path) -> None:
    output_path = tmp_path / "template.jinja"
    result = run_cli(
        "--version",
        "v13",
        "--model_settings_fields",
        "reasoning_effort",
        "--saving_path",
        str(output_path),
    )
    assert_cli_failure(
        result,
        output_path,
        "Model settings fields are only available for tokenizer versions v15 and above",
    )


def test_cli_model_settings_fields_rejected_in_autodetect(tmp_path: Path, tekken_think_v13_path: Path) -> None:
    output_path = tmp_path / "template.jinja"
    result = run_cli(
        "--tokenizer_file",
        str(tekken_think_v13_path),
        "--model_settings_fields",
        "reasoning_effort",
        "--saving_path",
        str(output_path),
    )
    assert_cli_failure(
        result,
        output_path,
        "--tokenizer_file cannot be combined with manual capability flags: --model_settings_fields",
    )


def test_cli_reasoning_effort_flags(tmp_path: Path) -> None:
    output_path = tmp_path / "template.jinja"
    result = run_cli(
        "--version",
        "v15",
        "--reasoning_effort_values",
        "minimal,max",
        "--reasoning_effort_default",
        "max",
        "--saving_path",
        str(output_path),
    )
    content = assert_cli_success(result, output_path)
    assert "'minimal', 'max'" in content
    assert "reasoning_effort = 'max'" in content


def test_cli_reasoning_effort_default_rejected_in_autodetect(tmp_path: Path, tekken_think_v13_path: Path) -> None:
    output_path = tmp_path / "template.jinja"
    result = run_cli(
        "--tokenizer_file",
        str(tekken_think_v13_path),
        "--reasoning_effort_default",
        "none",
        "--saving_path",
        str(output_path),
    )
    assert_cli_failure(
        result,
        output_path,
        "--tokenizer_file cannot be combined with manual capability flags: --reasoning_effort_default",
    )


def test_cli_reasoning_effort_values_rejected_in_autodetect(tmp_path: Path, tekken_think_v13_path: Path) -> None:
    output_path = tmp_path / "template.jinja"
    result = run_cli(
        "--tokenizer_file",
        str(tekken_think_v13_path),
        "--reasoning_effort_values",
        "minimal,max",
        "--saving_path",
        str(output_path),
    )
    assert_cli_failure(
        result,
        output_path,
        "--tokenizer_file cannot be combined with manual capability flags: --reasoning_effort_values",
    )


def test_cli_autodetect_v15_json_schema_builder(tmp_path: Path, tekken_v15_json_schema_path: Path) -> None:
    output_path = tmp_path / "template.jinja"
    result = run_cli(
        "--tokenizer_file",
        str(tekken_v15_json_schema_path),
        "--saving_path",
        str(output_path),
    )
    content = assert_cli_success(result, output_path)
    assert "response_format" in content
    assert "[MODEL_SETTINGS]" in content


def test_cli_with_image_flag(tmp_path: Path) -> None:
    """CLI --image flag produces template with image support."""
    output_path = tmp_path / "template.jinja"
    result = run_cli(
        "--version",
        "v15",
        "--image",
        "--reasoning_effort_values",
        "none,high",
        "--reasoning_effort_default",
        "none",
        "--saving_path",
        str(output_path),
    )
    content = assert_cli_success(result, output_path)
    assert "image" in content.lower()


def test_cli_with_thinking_flag(tmp_path: Path) -> None:
    """CLI --thinking flag produces template with thinking support."""
    output_path = tmp_path / "template.jinja"
    result = run_cli(
        "--version",
        "v15",
        "--thinking",
        "--reasoning_effort_values",
        "none,high",
        "--reasoning_effort_default",
        "none",
        "--saving_path",
        str(output_path),
    )
    content = assert_cli_success(result, output_path)
    assert "[THINK]" in content


def test_cli_with_default_system_prompt(tmp_path: Path) -> None:
    """CLI --default_system_prompt embeds the prompt."""
    output_path = tmp_path / "template.jinja"
    result = run_cli(
        "--version",
        "v7",
        "--default_system_prompt",
        "You are helpful.",
        "--saving_path",
        str(output_path),
    )
    content = assert_cli_success(result, output_path)
    assert "You are helpful." in content


def test_cli_invalid_version(tmp_path: Path) -> None:
    """CLI rejects an unknown version with the enum diagnostic and writes no file."""
    output_path = tmp_path / "template.jinja"
    result = run_cli("--version", "v99", "--saving_path", str(output_path))
    assert_cli_failure(result, output_path, "'v99' is not a valid TokenizerVersion")


def test_cli_invalid_config(tmp_path: Path) -> None:
    """CLI rejects image + audio with the mutual-exclusion diagnostic and writes no file."""
    output_path = tmp_path / "template.jinja"
    result = run_cli("--version", "v15", "--image", "--audio", "--saving_path", str(output_path))
    assert_cli_failure(result, output_path, "Image and audio support are mutually exclusive")


def test_cli_missing_version(tmp_path: Path) -> None:
    """CLI requires --version in manual mode and emits the specific error message."""
    output_path = tmp_path / "template.jinja"
    result = run_cli("--saving_path", str(output_path))
    assert_cli_failure(result, output_path, "--version is required when --tokenizer_file is not provided")


def test_cli_with_audio_flag(tmp_path: Path) -> None:
    """CLI --audio flag produces template with audio support."""
    output_path = tmp_path / "template.jinja"
    result = run_cli("--version", "v7", "--audio", "--saving_path", str(output_path))
    content = assert_cli_success(result, output_path)
    assert "[AUDIO]" in content


def test_cli_with_plain_thinking_flag(tmp_path: Path) -> None:
    """CLI --plain_thinking flag produces template with plain text thinking."""
    output_path = tmp_path / "template.jinja"
    result = run_cli("--version", "v11", "--plain_thinking", "--saving_path", str(output_path))
    content = assert_cli_success(result, output_path)
    assert "<think>" in content
    assert "[THINK]" not in content


def test_cli_plain_thinking_invalid_version(tmp_path: Path) -> None:
    """CLI rejects --plain_thinking with a non-v11 version and writes no file."""
    output_path = tmp_path / "template.jinja"
    result = run_cli("--version", "v15", "--plain_thinking", "--saving_path", str(output_path))
    assert_cli_failure(
        result,
        output_path,
        "Plain thinking support is only available for tokenizer version v11",
    )


def test_cli_spm_flag(tmp_path: Path) -> None:
    """CLI --spm flag produces SPM template."""
    output_path = tmp_path / "template.jinja"
    result = run_cli("--version", "v7", "--spm", "--saving_path", str(output_path))
    content = assert_cli_success(result, output_path)
    assert "bos_token" in content


def test_cli_spm_with_audio_invalid(tmp_path: Path) -> None:
    """CLI --spm --audio together fails validation and writes no file."""
    output_path = tmp_path / "template.jinja"
    result = run_cli("--version", "v7", "--spm", "--audio", "--saving_path", str(output_path))
    assert_cli_failure(
        result,
        output_path,
        "SPM tokenizer is not supported for tokenizer versions v11 and above or audio",
    )


def test_cli_no_special_token_variables_flag(tmp_path: Path) -> None:
    """CLI --no_special_token_variables embeds literal BOS/EOS values."""
    output_path = tmp_path / "template.jinja"
    result = run_cli(
        "--version",
        "v7",
        "--no_special_token_variables",
        "--saving_path",
        str(output_path),
    )
    content = assert_cli_success(result, output_path)
    assert "'<s>'" in content
    assert "'</s>'" in content
    assert "bos_token" not in content
    assert "eos_token" not in content


@pytest.mark.parametrize("no_generation_blocks", [False, True])
def test_cli_no_generation_blocks_flag(tmp_path: Path, no_generation_blocks: bool) -> None:
    """CLI keeps the generation blocks by default; --no_generation_blocks leaves them out."""
    output_path = tmp_path / "template.jinja"
    flags = ["--no_generation_blocks"] if no_generation_blocks else []
    result = run_cli("--version", "v7", *flags, "--saving_path", str(output_path))
    content = assert_cli_success(result, output_path)
    assert ("{%- generation -%}" in content) is not no_generation_blocks
    assert ("{%- endgeneration -%}" in content) is not no_generation_blocks


def test_autodetect_no_generation_blocks_flag(tmp_path: Path, tekken_think_v13_path: Path) -> None:
    """Auto-detect mode with --no_generation_blocks leaves out the generation blocks."""
    output_path = tmp_path / "template.jinja"
    result = run_cli(
        "--tokenizer_file",
        str(tekken_think_v13_path),
        "--no_generation_blocks",
        "--saving_path",
        str(output_path),
    )
    content = assert_cli_success(result, output_path)
    assert "[THINK]" in content
    assert "generation" not in content


def test_autodetect_happy_path(tmp_path: Path, tekken_think_v13_path: Path) -> None:
    """Auto-detect mode with v13 think tokenizer generates template containing [THINK]."""
    output_path = tmp_path / "template.jinja"
    result = run_cli(
        "--tokenizer_file",
        str(tekken_think_v13_path),
        "--saving_path",
        str(output_path),
    )
    content = assert_cli_success(result, output_path)
    assert "[THINK]" in content


def test_autodetect_conflict_version(tmp_path: Path, tekken_think_v13_path: Path) -> None:
    """Auto-detect + --version exits non-zero with mutual-exclusion error."""
    output_path = tmp_path / "template.jinja"
    result = run_cli(
        "--tokenizer_file",
        str(tekken_think_v13_path),
        "--version",
        "v13",
        "--saving_path",
        str(output_path),
    )
    assert_cli_failure(
        result,
        output_path,
        "--tokenizer_file cannot be combined with manual capability flags: --version",
    )


def test_autodetect_conflict_capability_flag(tmp_path: Path, tekken_think_v13_path: Path) -> None:
    """Auto-detect + --image exits non-zero with mutual-exclusion error."""
    output_path = tmp_path / "template.jinja"
    result = run_cli(
        "--tokenizer_file",
        str(tekken_think_v13_path),
        "--image",
        "--saving_path",
        str(output_path),
    )
    assert_cli_failure(
        result,
        output_path,
        "--tokenizer_file cannot be combined with manual capability flags: --image",
    )


def test_autodetect_with_system_prompt(tmp_path: Path, tekken_think_v13_path: Path) -> None:
    """Auto-detect mode with --default_system_prompt embeds the prompt."""
    output_path = tmp_path / "template.jinja"
    result = run_cli(
        "--tokenizer_file",
        str(tekken_think_v13_path),
        "--default_system_prompt",
        "You are helpful.",
        "--saving_path",
        str(output_path),
    )
    content = assert_cli_success(result, output_path)
    assert "You are helpful." in content


def test_autodetect_no_special_token_variables(tmp_path: Path, tekken_think_v13_path: Path) -> None:
    """Auto-detect + --no_special_token_variables embeds literal BOS/EOS values."""
    output_path = tmp_path / "template.jinja"
    result = run_cli(
        "--tokenizer_file",
        str(tekken_think_v13_path),
        "--no_special_token_variables",
        "--saving_path",
        str(output_path),
    )
    content = assert_cli_success(result, output_path)
    assert "'<s>'" in content
    assert "bos_token" not in content


def test_autodetect_with_plain_thinking_flag(tmp_path: Path, tekken_v11_path: Path) -> None:
    """Auto-detect accepts --plain_thinking, which restates what the heuristic already detects."""
    output_path = tmp_path / "template.jinja"
    result = run_cli(
        "--tokenizer_file",
        str(tekken_v11_path),
        "--plain_thinking",
        "--saving_path",
        str(output_path),
    )
    content = assert_cli_success(result, output_path)
    assert "<think>" in content
    assert "[THINK]" not in content


def test_autodetect_with_no_plain_thinking_flag(tmp_path: Path, tekken_v11_path: Path) -> None:
    """Auto-detect + --no_plain_thinking forces plain thinking off, overriding the heuristic."""
    output_path = tmp_path / "template.jinja"
    result = run_cli(
        "--tokenizer_file",
        str(tekken_v11_path),
        "--no_plain_thinking",
        "--saving_path",
        str(output_path),
    )
    content = assert_cli_success(result, output_path)
    assert "<think>" not in content


def test_plain_thinking_and_no_plain_thinking_conflict(tmp_path: Path, tekken_v11_path: Path) -> None:
    """--plain_thinking and --no_plain_thinking together exit non-zero with a mutual-exclusion error."""
    output_path = tmp_path / "template.jinja"
    result = run_cli(
        "--tokenizer_file",
        str(tekken_v11_path),
        "--plain_thinking",
        "--no_plain_thinking",
        "--saving_path",
        str(output_path),
    )
    assert_cli_failure(
        result,
        output_path,
        "--plain_thinking and --no_plain_thinking are mutually exclusive",
    )


def test_manual_mode_no_plain_thinking_is_a_no_op(tmp_path: Path) -> None:
    """--no_plain_thinking is accepted but inert in manual mode, where plain thinking is off anyway."""
    with_flag = tmp_path / "with_flag.jinja"
    without_flag = tmp_path / "without_flag.jinja"
    for output_path, extra_args in ((with_flag, ["--no_plain_thinking"]), (without_flag, [])):
        result = run_cli("--version", "v11", *extra_args, "--saving_path", str(output_path))
        assert_cli_success(result, output_path)

    assert with_flag.read_text(encoding="utf-8") == without_flag.read_text(encoding="utf-8")
    assert "<think>" not in with_flag.read_text(encoding="utf-8")


def test_autodetect_plain_thinking_incompatible_tokenizer_errors_cleanly(
    tmp_path: Path, tekken_think_v13_path: Path
) -> None:
    """Forcing --plain_thinking on a special-token thinking tokenizer reports a CLI error, not a traceback."""
    output_path = tmp_path / "template.jinja"
    result = run_cli(
        "--tokenizer_file",
        str(tekken_think_v13_path),
        "--plain_thinking",
        "--saving_path",
        str(output_path),
    )
    assert result.returncode != 0
    assert "Plain thinking support and thinking support are mutually exclusive" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output_path.exists()
