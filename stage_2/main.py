#!/usr/bin/env python3
"""
Stage 2 入口（仓库根目录）:
  python -m stage_2.main compare [--lambda-from-ckpt]

Pareto:
  python -m stage_2.tools.run_pareto --train --eval

主训练（默认 stage2_intense + stack-k=4）:
  python -m stage_2.training.train_stage2 --device mps
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _cmd_compare(args: argparse.Namespace) -> None:
    from stage_2.evaluation.compare_stage2 import run_compare

    run_compare(args)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2: temporal DQN + compare")
    sub = ap.add_subparsers(dest="cmd", required=True)

    from stage_2.evaluation.compare_stage2 import add_compare_cli

    cp = sub.add_parser("compare", help="baselines + Stage1/Stage2 DQN，写 eval_all_methods.csv")
    add_compare_cli(cp)
    cp.set_defaults(func=_cmd_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
