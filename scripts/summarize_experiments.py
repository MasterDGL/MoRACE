#!/usr/bin/env python
"""Source-checkout entry point for experiment report generation."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from morace.experiment_reports import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
