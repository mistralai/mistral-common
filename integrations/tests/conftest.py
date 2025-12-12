import sys
from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent.parent
INTEGRATION_PATH = ROOT_PATH / "integrations"
COMMON_TESTS_PATH = ROOT_PATH / "tests"

sys.path.append(INTEGRATION_PATH.as_posix())
sys.path.append(COMMON_TESTS_PATH.as_posix())
