"""
conftest.py — pytest configuration for the ALPR test suite.

Adds the project root to sys.path so that `src`, `utils`, and `config`
can be imported without installing the package.
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
