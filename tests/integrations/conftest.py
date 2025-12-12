import sys
from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent.parent

sys.path.append((ROOT_PATH / "integrations").as_posix())
