#!/usr/bin/env python3
"""
Stage 5 编排：§6.4 四种 robust 训练（teacher / notchr）+ 评估（复用 Stage4 baseline62 ckpt，不重训）。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

S5_MODES = ("per", "softq", "c51", "gru")


def _run_one(cmd: list[str], *, dry_run: bool) -> int:
    if dry_run:
        print(" ".join(cmd))
        return 0
    print("\n>>>", " ".join(cmd), flush=True)
    return int(subprocess.run(cmd, cwd=str(_REPO_ROOT)).returncode)


def build_train_cmd(
    *,
    mode: str,
    artifact_suffix: str,
    no_teacher_reward: bool,
    device: str,
    stage2_preset: str,
) -> list[str]:
    from stage_5.utils.paths import STAGE5_CHECKPOINTS, STAGE5_OUTPUTS

    suf = artifact_suffix or ""
    save = STAGE5_CHECKPOINTS / f"dqn_stage5_{mode}{suf}.pt"
    metrics = STAGE5_OUTPUTS / f"train_metrics_{mode}{suf}.csv"
    cmd: list[str] = [
        sys.executable,
        "-m",
        "stage_5.training.train_stage5",
        "--preset",
        "none",
        "--stage2-preset",
        stage2_preset,
        "--stack-k",
        "4",
        "--ablation",
        "none",
        "--stage5-mode",
        mode,
        "--save",
        str(save),
        "--metrics-csv",
        str(metrics),
        "--device",
        device,
        "--lambda-min",
        "0.25",
        "--target-mean-cost",
        "0.45",
    ]
    if no_teacher_reward:
        cmd.append("--no-teacher-reward")
    return cmd


def cmd_standard(
    *,
    device: str,
    dry_run: bool,
    stage2_preset: str,
) -> None:
    from stage_5.utils.paths import STAGE5_CHECKPOINTS, STAGE5_OUTPUTS

    STAGE5_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    STAGE5_OUTPUTS.mkdir(parents=True, exist_ok=True)
    for suf, no_chr in (("_teacher", False), ("_notchr", True)):
        for mode in S5_MODES:
            cmd = build_train_cmd(
                mode=mode,
                artifact_suffix=suf,
                no_teacher_reward=no_chr,
                device=device,
                stage2_preset=stage2_preset,
            )
            rc = _run_one(cmd, dry_run=dry_run)
            if rc != 0:
                raise SystemExit(f"Stage5 训练失败 rc={rc} mode={mode} suffix={suf}")


def cmd_evaluate(
    *,
    device: str,
    n_episodes: int,
    baseline_lambda: float,
    video_path: Path | None,
    teacher_npz: Path | None,
    dry_run: bool,
) -> None:
    if dry_run:
        print("[dry-run] evaluate")
        return
    from stage_1.utils.paths import DEFAULT_TEACHER_NPZ, VIDEO_PATH, WEIGHTS_PATH
    from stage_3.tools.standard_eval import (
        baseline_rows,
        eval_stage2_row,
        write_standard_eval_csv,
    )
    from stage_4.utils.paths import STAGE4_CHECKPOINTS
    from stage_5.tools.eval_policy import eval_custom_arch_row
    from stage_5.utils.paths import STAGE5_CHECKPOINTS, STAGE5_OUTPUTS

    import torch

    STAGE5_OUTPUTS.mkdir(parents=True, exist_ok=True)

    vp = Path(video_path) if video_path is not None else VIDEO_PATH
    if not vp.is_file():
        raise SystemExit(f"找不到视频: {vp}")

    def run_one_suite(
        *,
        suffix: str,
        use_default_teacher: bool,
        out_name: str,
    ) -> None:
        tn: Path | None = None
        if teacher_npz is not None:
            tnp = Path(teacher_npz)
            if tnp.is_file():
                tn = tnp
        elif use_default_teacher and DEFAULT_TEACHER_NPZ.is_file():
            tn = DEFAULT_TEACHER_NPZ

        base_kw = dict(
            video_path=vp,
            yolo_weights=WEIGHTS_PATH,
            teacher_npz=tn,
            lambda_cost=baseline_lambda,
            max_episode_steps=200,
            imgsz=640,
            device=device,
        )
        seeds = [0, 1, 2, 3, 4][: max(1, n_episodes)]
        rows: list = list(baseline_rows(base_kw=base_kw, seeds=seeds, n_episodes=n_episodes))

        dev = torch.device("cpu")

        b4 = STAGE4_CHECKPOINTS / f"dqn_stage4_baseline62{suffix}.pt"
        if b4.is_file():
            rows.append(
                eval_stage2_row(
                    ckpt_path=b4,
                    base_kw=base_kw,
                    seeds=seeds,
                    n_episodes=n_episodes,
                    lambda_from_ckpt=True,
                    dev=dev,
                )
            )
            rows[-1]["policy_id"] = "stage4_baseline62"
            rows[-1]["policy_name"] = f"Stage4 §6.2 baseline62 reuse ({b4.name})"

        for ck in sorted(STAGE5_CHECKPOINTS.glob(f"dqn_stage5_*{suffix}.pt")):
            try:
                c = torch.load(ck, map_location="cpu", weights_only=False)
            except TypeError:
                c = torch.load(ck, map_location="cpu")
            mode = str(c.get("stage5_mode", ck.stem))
            rows.append(
                eval_custom_arch_row(
                    ckpt_path=ck,
                    base_kw=base_kw,
                    seeds=seeds,
                    n_episodes=n_episodes,
                    lambda_from_ckpt=True,
                    dev=dev,
                    policy_id=f"s5_{mode}",
                    policy_name=f"Stage5 §6.4 {mode} ({ck.name})",
                )
            )

        out_csv = STAGE5_OUTPUTS / out_name
        write_standard_eval_csv(out_csv=out_csv, rows=rows)
        print(f"[ok] 评估表 -> {out_csv}")

    run_one_suite(
        suffix="_teacher",
        use_default_teacher=True,
        out_name="eval_suite_teacher.csv",
    )
    run_one_suite(
        suffix="_notchr",
        use_default_teacher=False,
        out_name="eval_suite_notchr.csv",
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Stage5 §6.4 编排")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("standard", help="四种 §6.4 算法 × teacher / notchr（stage2_robust）")
    p1.add_argument("--device", default="mps")
    p1.add_argument("--dry-run", action="store_true")
    p1.add_argument(
        "--stage2-preset",
        choices=("none", "stage2_robust", "stage2_intense"),
        default="stage2_robust",
    )
    p1.set_defaults(func="standard")

    p2 = sub.add_parser(
        "evaluate",
        help="baselines + 复用 Stage4 baseline62 + 本 stage_5 ckpt，写 eval_suite_*.csv",
    )
    p2.add_argument("--device", default="mps")
    p2.add_argument("--n-episodes", type=int, default=10)
    p2.add_argument("--baseline-lambda", type=float, default=0.35)
    p2.add_argument("--video-path", type=Path, default=None)
    p2.add_argument("--teacher-npz", type=Path, default=None)
    p2.add_argument("--dry-run", action="store_true")
    p2.set_defaults(func="evaluate")

    args = ap.parse_args(argv)
    if args.func == "standard":
        cmd_standard(
            device=args.device,
            dry_run=bool(getattr(args, "dry_run", False)),
            stage2_preset=str(args.stage2_preset),
        )
        print("[ok] Stage5 standard 全部训练完成", flush=True)
        return

    cmd_evaluate(
        device=args.device,
        n_episodes=int(args.n_episodes),
        baseline_lambda=float(args.baseline_lambda),
        video_path=getattr(args, "video_path", None),
        teacher_npz=getattr(args, "teacher_npz", None),
        dry_run=bool(getattr(args, "dry_run", False)),
    )
    print("[ok] Stage5 evaluate 完成", flush=True)


if __name__ == "__main__":
    main()
