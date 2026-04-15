#!/usr/bin/env python3
"""
Stage 3 入口（仓库根目录）:
  python -m stage_3.main sweep-stack-k --quick --dry-run
  python -m stage_3.main sweep-ablation --quick --stack-k 4
  python -m stage_3.main sweep-grid --quick --stack-ks 1,4 --ablations none,no_iou
  python -m stage_3.main collect
  python -m stage_3.main compare --lambda-from-ckpt --out-csv stage_3/outputs/eval_all_methods_teacher.csv --teacher-npz <npz>

说明见 stage_3/README.md 与 old_research_targets.md §6.1。
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        from stage_3.tools.compare_stage3 import main as compare_main

        compare_main(sys.argv[2:])
        return
    from stage_3.tools.sweep_experiments import main as sweep_main

    sweep_main(sys.argv[1:])


if __name__ == "__main__":
    main()
