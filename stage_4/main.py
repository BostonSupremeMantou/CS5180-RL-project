#!/usr/bin/env python3
"""
Stage 4 入口（仓库根目录）:
  python -m stage_4.main standard --quick --device mps
  python -m stage_4.main collect

说明见 stage_4/README.md 与 research_targets.md §6.2。
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> None:
    from stage_4.tools.sweep_stage4 import main as s4_main

    s4_main(sys.argv[1:])


if __name__ == "__main__":
    main()
