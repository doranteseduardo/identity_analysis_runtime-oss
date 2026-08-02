#!/usr/bin/env python3
"""Backward-compatible entry point for the packaged CLI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from identity_analysis.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
