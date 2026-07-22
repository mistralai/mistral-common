import subprocess
import sys
from pathlib import Path

import pytest

from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.integrations.chat_templates.helpers import TestConfig, _build_tekken_json

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "generate_chat_template.py"


@pytest.fixture(scope="session")
def tekken_think_v13_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Path to a v13 think tekken.json used for auto-detect CLI tests."""
    build_dir = tmp_path_factory.mktemp("tokenizer")
    config = TestConfig(version=TokenizerVersion.v13, think=True)
    return _build_tekken_json(config=config, output_dir=build_dir)


def test_cli_generates_template(tmp_path: Path) -> None:
    """CLI generates a template file with expected content."""
    output_path = tmp_path / "template.jinja"
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--version", "v15", "--saving_path", str(output_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert output_path.exists()
    content = output_path.read_text()
    assert "bos_token" in content
    assert "[MODEL_SETTINGS]" in content


def test_cli_with_image_flag(tmp_path: Path) -> None:
    """CLI --image flag produces template with image support."""
    output_path = tmp_path / "template.jinja"
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--version", "v15", "--image", "--saving_path", str(output_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    content = output_path.read_text()
    assert "image" in content.lower()


def test_cli_with_thinking_flag(tmp_path: Path) -> None:
    """CLI --thinking flag produces template with thinking support."""
    output_path = tmp_path / "template.jinja"
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--version", "v15", "--thinking", "--saving_path", str(output_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    content = output_path.read_text()
    assert "[THINK]" in content


def test_cli_with_default_system_prompt(tmp_path: Path) -> None:
    """CLI --default_system_prompt embeds the prompt."""
    output_path = tmp_path / "template.jinja"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--version",
            "v7",
            "--default_system_prompt",
            "You are helpful.",
            "--saving_path",
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    content = output_path.read_text()
    assert "You are helpful." in content


def test_cli_invalid_version(tmp_path: Path) -> None:
    """CLI rejects invalid version."""
    output_path = tmp_path / "template.jinja"
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--version", "v99", "--saving_path", str(output_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_cli_invalid_config(tmp_path: Path) -> None:
    """CLI rejects invalid configuration (e.g., image + audio)."""
    output_path = tmp_path / "template.jinja"
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--version", "v15", "--image", "--audio", "--saving_path", str(output_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_cli_missing_version(tmp_path: Path) -> None:
    """CLI requires --version in manual mode and emits the specific error message."""
    output_path = tmp_path / "template.jinja"
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--saving_path", str(output_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--version is required when --tokenizer_file is not provided" in result.stderr


def test_cli_with_audio_flag(tmp_path: Path) -> None:
    """CLI --audio flag produces template with audio support."""
    output_path = tmp_path / "template.jinja"
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--version", "v7", "--audio", "--saving_path", str(output_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    content = output_path.read_text()
    assert "[AUDIO]" in content


def test_cli_with_plain_thinking_flag(tmp_path: Path) -> None:
    """CLI --plain_thinking flag produces template with plain text thinking."""
    output_path = tmp_path / "template.jinja"
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--version", "v11", "--plain_thinking", "--saving_path", str(output_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    content = output_path.read_text()
    assert "<think>" in content
    assert "[THINK]" not in content


def test_cli_plain_thinking_invalid_version(tmp_path: Path) -> None:
    """CLI rejects --plain_thinking with non-v11 version."""
    output_path = tmp_path / "template.jinja"
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--version", "v15", "--plain_thinking", "--saving_path", str(output_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_cli_spm_flag(tmp_path: Path) -> None:
    """CLI --spm flag produces SPM template."""
    output_path = tmp_path / "template.jinja"
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--version", "v7", "--spm", "--saving_path", str(output_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    content = output_path.read_text()
    assert "bos_token" in content


def test_cli_spm_with_audio_invalid(tmp_path: Path) -> None:
    """CLI --spm --audio together fails validation."""
    output_path = tmp_path / "template.jinja"
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--version", "v7", "--spm", "--audio", "--saving_path", str(output_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_cli_no_special_token_variables_flag(tmp_path: Path) -> None:
    """CLI --no_special_token_variables embeds literal BOS/EOS values."""
    output_path = tmp_path / "template.jinja"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--version",
            "v7",
            "--no_special_token_variables",
            "--saving_path",
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    content = output_path.read_text()
    assert "'<s>'" in content
    assert "'</s>'" in content
    assert "bos_token" not in content
    assert "eos_token" not in content


def test_autodetect_happy_path(tmp_path: Path, tekken_think_v13_path: Path) -> None:
    """Auto-detect mode with v13 think tokenizer generates template containing [THINK]."""
    output_path = tmp_path / "template.jinja"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--tokenizer_file",
            str(tekken_think_v13_path),
            "--saving_path",
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert output_path.exists()
    content = output_path.read_text()
    assert "[THINK]" in content


def test_autodetect_conflict_version(tmp_path: Path, tekken_think_v13_path: Path) -> None:
    """Auto-detect + --version exits non-zero with mutual-exclusion error."""
    output_path = tmp_path / "template.jinja"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--tokenizer_file",
            str(tekken_think_v13_path),
            "--version",
            "v13",
            "--saving_path",
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--tokenizer_file cannot be combined with manual capability flags" in result.stderr


def test_autodetect_conflict_capability_flag(tmp_path: Path, tekken_think_v13_path: Path) -> None:
    """Auto-detect + --image exits non-zero with mutual-exclusion error."""
    output_path = tmp_path / "template.jinja"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--tokenizer_file",
            str(tekken_think_v13_path),
            "--image",
            "--saving_path",
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--tokenizer_file cannot be combined with manual capability flags" in result.stderr


def test_autodetect_with_system_prompt(tmp_path: Path, tekken_think_v13_path: Path) -> None:
    """Auto-detect mode with --default_system_prompt embeds the prompt."""
    output_path = tmp_path / "template.jinja"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--tokenizer_file",
            str(tekken_think_v13_path),
            "--default_system_prompt",
            "You are helpful.",
            "--saving_path",
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    content = output_path.read_text()
    assert "You are helpful." in content


def test_autodetect_no_special_token_variables(tmp_path: Path, tekken_think_v13_path: Path) -> None:
    """Auto-detect + --no_special_token_variables embeds literal BOS/EOS values."""
    output_path = tmp_path / "template.jinja"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--tokenizer_file",
            str(tekken_think_v13_path),
            "--no_special_token_variables",
            "--saving_path",
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    content = output_path.read_text()
    assert "'<s>'" in content
    assert "bos_token" not in content
