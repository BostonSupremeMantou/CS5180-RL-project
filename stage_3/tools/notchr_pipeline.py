#!/usr/bin/env python3
"""
无 teacher 参与奖励：依次训练 Stage1、Stage2（单栈 ckpt）、Stage3 standard，
并写出 Stage1 eval、Stage2 compare、Stage3 eval 的 CSV。

仓库根目录执行:
  python -m stage_3.tools.notchr_pipeline --quick
  python -m stage_3.tools.notchr_pipeline --artifact-suffix _notchr
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=str(_REPO_ROOT))
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def _teacher_arg() -> list[str]:
    from stage_1.utils.paths import DEFAULT_TEACHER_NPZ

    if DEFAULT_TEACHER_NPZ.is_file():
        return ["--teacher-npz", str(DEFAULT_TEACHER_NPZ)]
    return []


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="无 teacher 奖励：训练 S1/S2 + Stage3 standard + 评估")
    ap.add_argument("--device", default="mps")
    ap.add_argument(
        "--quick",
        action="store_true",
        help="短跑（与 sweep --quick 同量级）；正式实验勿加此项",
    )
    ap.add_argument(
        "--artifact-suffix",
        type=str,
        default="_notchr",
        help="Stage1/2 权重与 CSV 后缀；Stage3 透传给 standard（需与 teacher 跑区分）",
    )
    ap.add_argument("--skip-stage1-train", action="store_true")
    ap.add_argument("--skip-stage2-train", action="store_true")
    ap.add_argument("--skip-stage3", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args(argv)

    suf = args.artifact_suffix or ""
    n_eval_eps = "3" if args.quick else "10"
    py = sys.executable
    s1_ckpt = _REPO_ROOT / "stage_1" / "checkpoints" / f"dqn_stage1{suf}.pt"
    s1_metrics = _REPO_ROOT / "stage_1" / "outputs" / f"train_metrics{suf}.csv"
    s2_ckpt = _REPO_ROOT / "stage_2" / "checkpoints" / f"dqn_stage2{suf}.pt"
    s2_metrics = _REPO_ROOT / "stage_2" / "outputs" / f"train_metrics{suf}.csv"
    s1_eval_csv = _REPO_ROOT / "stage_1" / "outputs" / f"eval_summary{suf}.csv"
    s2_compare_csv = _REPO_ROOT / "stage_2" / "outputs" / f"eval_all_methods{suf}.csv"

    if not args.skip_stage1_train:
        s1_cmd = [
            py,
            "-m",
            "stage_1.main",
            "train",
            "--no-teacher-reward",
            "--device",
            args.device,
            "--save",
            str(s1_ckpt),
            "--metrics-csv",
            str(s1_metrics),
            "--lambda-min",
            "0.25",
            "--target-mean-cost",
            "0.45",
        ]
        if args.quick:
            s1_cmd += [
                "--preset",
                "none",
                "--total-steps",
                "8000",
                "--learning-starts",
                "600",
                "--log-every",
                "800",
                "--epsilon-decay-steps",
                "6500",
                "--replay-capacity",
                "60000",
                "--batch-size",
                "64",
                "--train-freq",
                "2",
                "--grad-updates",
                "1",
                "--target-update-every",
                "500",
            ]
        else:
            s1_cmd += ["--preset", "robust"]
        _run(s1_cmd)

    if not args.skip_stage2_train:
        s2_cmd = [
            py,
            "-m",
            "stage_2.training.train_stage2",
            "--no-teacher-reward",
            "--device",
            args.device,
            "--stack-k",
            "4",
            "--ablation",
            "none",
            "--save",
            str(s2_ckpt),
            "--metrics-csv",
            str(s2_metrics),
            "--lambda-min",
            "0.25",
            "--target-mean-cost",
            "0.45",
            "--preset",
            "none",
            "--stage2-preset",
            "none" if args.quick else "stage2_robust",
        ]
        if args.quick:
            s2_cmd += [
                "--total-steps",
                "8000",
                "--learning-starts",
                "600",
                "--log-every",
                "800",
                "--epsilon-decay-steps",
                "6500",
                "--replay-capacity",
                "60000",
                "--batch-size",
                "64",
                "--train-freq",
                "2",
                "--grad-updates",
                "1",
                "--target-update-every",
                "500",
            ]
        _run(s2_cmd)

    if not args.skip_stage3:
        s3_cmd = [
            py,
            "-m",
            "stage_3.tools.sweep_experiments",
            "standard",
            "--device",
            args.device,
            "--no-teacher-reward",
            "--artifact-suffix",
            suf,
            "--n-episodes",
            n_eval_eps,
        ]
        if args.quick:
            s3_cmd.append("--quick")
            s3_cmd += ["--stage2-preset", "none"]
        _run(s3_cmd)

    if not args.skip_eval:
        te = _teacher_arg()
        _run(
            [
                py,
                "-m",
                "stage_1.main",
                "eval",
                "--device",
                args.device,
                "--dqn-ckpt",
                str(s1_ckpt),
                "--lambda-from-ckpt",
                "--eval-csv",
                str(s1_eval_csv),
                "--n-episodes",
                n_eval_eps,
            ]
            + te
        )
        _run(
            [
                py,
                "-m",
                "stage_2.main",
                "compare",
                "--device",
                args.device,
                "--stage1-ckpt",
                str(s1_ckpt),
                "--stage2-ckpt",
                str(s2_ckpt),
                "--lambda-from-ckpt",
                "--out-csv",
                str(s2_compare_csv),
                "--n-episodes",
                n_eval_eps,
            ]
            + te
        )

    print(
        "\n[ok] 无 teacher 奖励流水线结束。\n"
        f"  Stage1 ckpt/metrics: {s1_ckpt} , {s1_metrics}\n"
        f"  Stage2 ckpt/metrics: {s2_ckpt} , {s2_metrics}\n"
        f"  Stage1 eval CSV: {s1_eval_csv}\n"
        f"  Stage2 compare CSV: {s2_compare_csv}\n"
        f"  Stage3（若已跑）: stage_3/outputs/sweep_train_summary{suf}.csv , "
        f"eval_standard_suite{suf}.csv"
    )


if __name__ == "__main__":
    main()
