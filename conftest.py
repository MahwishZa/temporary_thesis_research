"""Make ``src/`` importable so the tests run in a fresh checkout.

The package is a src-layout distribution, so ``import thesis`` normally requires
``pip install -e .``. A thesis has to stay runnable years from now on a machine
where that install may not have happened, so pytest is given the path directly.
This is the only reason this file exists; it declares no fixtures.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
