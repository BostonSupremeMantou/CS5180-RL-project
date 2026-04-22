#!/usr/bin/env python3
"""Stage 5 入口（仓库根目录）：python -m stage_5.main standard --device mps"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> None:
    from stage_5.tools.run_stage5 import main as s5_main

    s5_main(sys.argv[1:])


if __name__ == "__main__":
    main()
