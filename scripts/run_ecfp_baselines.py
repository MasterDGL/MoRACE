#!/usr/bin/env python
"""Command-line entrypoint for Morgan/ECFP baselines."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from morace.ecfp_baselines import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
