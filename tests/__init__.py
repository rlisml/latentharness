"""Test package for the harness infrastructure. Puts the repo root on sys.path so
`harness` and `utils` are importable under unittest discovery from any cwd."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
