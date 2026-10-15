#!/usr/bin/env python
"""Run deterministic raw-candidate closure without model generation or RF fitting."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from morace.raw_closure import main


if __name__ == "__main__":
    raise SystemExit(main())
