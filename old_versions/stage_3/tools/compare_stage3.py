#!/usr/bin/env python3
"""Stage 3：与 Stage2 `compare` 相同协议（baseline + S1 + S2），并追加各 Stage3 checkpoint 行。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def add_stage3_compare_cli(p: argparse.ArgumentParser) -> None:
    from stage_2.evaluation.compare_stage2 import add_compare_cli
    from stage_3.utils.paths import STAGE3_OUTPUTS

    add_compare_cli(p)
    for action in p._actions:
        if getattr(action, "dest", None) == "out_csv":
            action.default = STAGE3_OUTPUTS / "eval_all_methods.csv"
            break
    p.add_argument(
        "--artifact-suffix",
        type=str,
        default="",
        help="匹配 checkpoints/dqn_stage3_sk*_ablnone{SUFFIX}.pt",
    )
    p.add_argument(
        "--skip-stage3",
        action="store_true",
        help="不写 Stage3 多 checkpoint 行（与 Stage2 compare 等价）",
    )


def run_stage3_compare(args: argparse.Namespace) -> None:
    from stage_1.evaluation.run_eval import write_eval_summary_csv
    from stage_1.utils.paths import VIDEO_PATH, WEIGHTS_PATH
    from stage_2.evaluation.compare_stage2 import collect_compare_rows
    from stage_3.tools.standard_eval import eval_stage2_row
    from stage_3.utils.paths import STAGE3_CHECKPOINTS, STAGE3_OUTPUTS

    out_csv = args.out_csv
    if out_csv is None:
        out_csv = STAGE3_OUTPUTS / "eval_all_methods.csv"

    rows = collect_compare_rows(args)

    if not args.skip_stage3:
        vp = Path(args.video_path) if args.video_path is not None else VIDEO_PATH
        if not vp.is_file():
            raise SystemExit(f"找不到视频: {vp}")
        tn: Path | None = None
        if args.teacher_npz is not None:
            tnp = Path(args.teacher_npz)
            if tnp.is_file():
                tn = tnp
            else:
                print(f"[warn] --teacher-npz 不存在 ({tnp})，按无 teacher 评估")
        base_kw = dict(
            video_path=vp,
            yolo_weights=WEIGHTS_PATH,
            teacher_npz=tn,
            lambda_cost=args.lambda_cost,
            max_episode_steps=args.max_episode_steps,
            imgsz=args.imgsz,
            device=args.device,
        )
        seeds = [0, 1, 2, 3, 4][: max(1, args.n_episodes)]
        dev = torch.device("cpu")
        suf = args.artifact_suffix or ""
        paths = sorted(STAGE3_CHECKPOINTS.glob(f"dqn_stage3_sk*_ablnone{suf}.pt"))
        if not paths:
            print(
                f"[warn] 未找到 Stage3 checkpoint: "
                f"{STAGE3_CHECKPOINTS}/dqn_stage3_sk*_ablnone{suf}.pt"
            )
        for ck in paths:
            row = eval_stage2_row(
                ckpt_path=ck,
                base_kw=base_kw,
                seeds=seeds,
                n_episodes=args.n_episodes,
                lambda_from_ckpt=args.lambda_from_ckpt,
                dev=dev,
            )
            row.pop("ckpt_path", None)
            pid = row["policy_id"]
            if pid.startswith("dqn_stage2_sk"):
                row["policy_id"] = "dqn_stage3" + pid.removeprefix("dqn_stage2")
            row["policy_name"] = row["policy_name"].replace(
                "DQN Stage2", "DQN Stage3", 1
            )
            rows.append(row)

    if not args.no_csv:
        write_eval_summary_csv(rows, out_csv)
        print(f"\n[ok] Stage3 对比表 -> {out_csv}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Stage3 eval：同 Stage2 compare + 各 stack_k ckpt")
    add_stage3_compare_cli(p)
    run_stage3_compare(p.parse_args(argv))


if __name__ == "__main__":
    main()
