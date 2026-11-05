"""Make test-only helpers (tests/fake_llm.py) importable."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
