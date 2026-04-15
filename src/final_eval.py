#!/usr/bin/env python3
"""placeholder — later: load every trained agent and write outputs/final_results/*."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utilities import paths  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="final combined eval (stub)")
    p.parse_args()
    out = paths.OUTPUTS_DIR / "final_results"
    out.mkdir(parents=True, exist_ok=True)
    print(f"nothing to do yet — use {out} when ready")


if __name__ == "__main__":
    main()
