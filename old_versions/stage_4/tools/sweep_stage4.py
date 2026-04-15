#!/usr/bin/env python3
"""
old_research_targets.md §6.2：奖励/λ 修正、n-step TD、ε 尾部退火。

通过子进程调用 `python -m stage_2.training.train_stage2`，权重写入
`stage_4/checkpoints/`，曲线写入 `stage_4/outputs/`。
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# (标签, 透传给 train_stage2 的额外 CLI 片段)
# baseline62：与 Stage2 默认一致（§6.2 对照组）
VARIANTS_62: tuple[tuple[str, list[str]], ...] = (
    ("baseline62", []),
    ("ssf05", ["--ssf-reward-penalty", "0.05"]),
    ("nstep3", ["--n-step", "3"]),
    ("epstail", ["--epsilon-tail", "0.03", "--epsilon-tail-steps", "4000"]),
    (
        "ioufloor",
        [
            "--lambda-teacher-iou-floor",
            "0.38",
            "--lambda-teacher-iou-below",
            "0.42",
        ],
    ),
    ("combo_ssf_n3", ["--ssf-reward-penalty", "0.05", "--n-step", "3"]),
)


def parse_last_metrics_row(metrics_csv: Path) -> dict[str, Any]:
    if not metrics_csv.is_file():
        return {}
    with metrics_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    last = rows[-1]
    out: dict[str, Any] = {}
    for k, v in last.items():
        try:
            out[k] = float(v) if "." in str(v) or "e" in str(v).lower() else int(v)
        except ValueError:
            out[k] = v
    return out


def append_manifest(manifest: Path, row: dict[str, Any]) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    exists = manifest.is_file()
    with manifest.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def run_one(cmd: list[str], *, dry_run: bool) -> int:
    if dry_run:
        print(" ".join(cmd))
        return 0
    print("\n>>>", " ".join(cmd), flush=True)
    return int(subprocess.run(cmd, cwd=str(_REPO_ROOT)).returncode)


def sweep_62_variants(
    *,
    device: str,
    quick: bool,
    dry_run: bool,
    stage2_preset: str,
    lambda_min: float,
    target_mean_cost: float,
    no_teacher_reward: bool,
    extra_args: list[str],
    stack_k: int,
    artifact_suffix: str = "",
) -> None:
    from stage_3.tools.sweep_experiments import build_train_cmd
    from stage_4.utils.paths import STAGE4_CHECKPOINTS, STAGE4_OUTPUTS

    STAGE4_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    STAGE4_OUTPUTS.mkdir(parents=True, exist_ok=True)
    suf = artifact_suffix or ""
    manifest = STAGE4_OUTPUTS / f"sweep_manifest_62{suf}.csv"
    if not dry_run and manifest.is_file():
        manifest.unlink()

    s2p = "none" if quick else stage2_preset
    for tag, v_extra in VARIANTS_62:
        save = STAGE4_CHECKPOINTS / f"dqn_stage4_{tag}{suf}.pt"
        metrics = STAGE4_OUTPUTS / f"train_metrics_{tag}{suf}.csv"
        cmd = build_train_cmd(
            stack_k=stack_k,
            ablation="none",
            save=save,
            metrics_csv=metrics,
            device=device,
            lambda_min=lambda_min,
            target_mean_cost=target_mean_cost,
            quick=quick,
            stage2_preset=s2p,
            extra_args=list(extra_args) + list(v_extra),
            no_teacher_reward=no_teacher_reward,
        )
        rc = run_one(cmd, dry_run=dry_run)
        if rc != 0:
            raise SystemExit(f"训练失败 rc={rc} variant={tag}")
        if not dry_run:
            last = parse_last_metrics_row(metrics)
            append_manifest(
                manifest,
                {
                    "kind": "stage4_62",
                    "variant": tag,
                    "artifact_suffix": suf,
                    "stack_k": stack_k,
                    "no_teacher_reward": int(no_teacher_reward),
                    "ckpt": str(save),
                    "metrics_csv": str(metrics),
                    "exit_code": rc,
                    **{f"final_{kk}": last.get(kk, "") for kk in last},
                },
            )


def collect_summary(*, out_csv: Path, glob_pattern: str) -> None:
    from stage_4.utils.paths import STAGE4_OUTPUTS

    files = sorted(STAGE4_OUTPUTS.glob(glob_pattern))
    if not files:
        raise SystemExit(f"未找到 {STAGE4_OUTPUTS}/{glob_pattern}")
    rows: list[dict[str, Any]] = []
    for p in files:
        rows.append({"metrics_csv": str(p), **parse_last_metrics_row(p)})
    keys: set[str] = set()
    for r in rows:
        keys.update(r.keys())
    fieldnames = sorted(keys)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"[ok] 汇总 {len(rows)} 行 -> {out_csv}")


def main(argv: list[str] | None = None) -> None:
    from stage_4.utils.paths import STAGE4_OUTPUTS

    ap = argparse.ArgumentParser(description="Stage4 §6.2 扫描")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("standard", help="跑 VARIANTS_62 全套（固定 stack_k=4、ablation=none）")
    p1.add_argument("--device", default="mps")
    p1.add_argument("--quick", action="store_true")
    p1.add_argument("--dry-run", action="store_true")
    p1.add_argument(
        "--stage2-preset",
        choices=("none", "stage2_robust", "stage2_intense"),
        default="stage2_robust",
    )
    p1.add_argument("--lambda-min", type=float, default=0.25)
    p1.add_argument("--target-mean-cost", type=float, default=0.45)
    p1.add_argument("--no-teacher-reward", action="store_true")
    p1.add_argument("--stack-k", type=int, default=4, help="与 Stage3 主实验对齐，默认 4")
    p1.add_argument(
        "--artifact-suffix",
        type=str,
        default="",
        help="写入 ckpt/metrics/manifest 文件名后缀，如 _teacher / _notchr",
    )
    p1.add_argument("extra", nargs="*", help="透传给 train_stage2")
    p1.set_defaults(func="standard")

    p3 = sub.add_parser(
        "evaluate",
        help="对 stage_4/checkpoints/dqn_stage4_*{suffix}.pt 做与 Stage3 相同的标准贪心评估",
    )
    p3.add_argument("--device", default="mps")
    p3.add_argument("--n-episodes", type=int, default=10)
    p3.add_argument("--baseline-lambda", type=float, default=0.35)
    p3.add_argument("--video-path", type=Path, default=None)
    p3.add_argument("--teacher-npz", type=Path, default=None)
    p3.add_argument(
        "--artifact-suffix",
        type=str,
        default="",
        help="与训练时一致，如 _teacher / _notchr",
    )
    p3.add_argument(
        "--no-teacher-eval",
        action="store_true",
        help="评估环境不传 teacher（mean_iou_teacher 为 nan）；与 notchr 训练对照",
    )
    p3.set_defaults(func="evaluate")

    p2 = sub.add_parser("collect", help="汇总 train_metrics_*.csv 末行")
    p2.add_argument(
        "--out",
        type=Path,
        default=None,
        help="默认 stage_4/outputs/sweep_train_summary_62{--artifact-suffix}.csv",
    )
    p2.add_argument("--glob", dest="glob_pattern", default=None, help="默认 train_metrics_*{suffix}.csv")
    p2.add_argument(
        "--artifact-suffix",
        type=str,
        default="",
        help="与 standard 训练时后缀一致，用于默认 out/glob",
    )
    p2.set_defaults(func="collect")

    args = ap.parse_args(argv)
    if args.func == "collect":
        suf = getattr(args, "artifact_suffix", "") or ""
        out = args.out
        if out is None:
            out = STAGE4_OUTPUTS / f"sweep_train_summary_62{suf}.csv"
        gpat = args.glob_pattern
        if gpat is None:
            gpat = f"train_metrics_*{suf}.csv" if suf else "train_metrics_*.csv"
        collect_summary(out_csv=out, glob_pattern=gpat)
        return

    if args.func == "evaluate":
        from stage_1.utils.paths import DEFAULT_TEACHER_NPZ, VIDEO_PATH
        from stage_3.tools.standard_eval import run_standard_eval
        from stage_4.utils.paths import STAGE4_CHECKPOINTS, STAGE4_OUTPUTS

        suf = getattr(args, "artifact_suffix", "") or ""
        out_csv = STAGE4_OUTPUTS / f"eval_suite{suf}.csv"
        tnp = args.teacher_npz
        use_def = not bool(getattr(args, "no_teacher_eval", False))
        if tnp is not None and not tnp.is_file():
            raise SystemExit(f"找不到 teacher npz: {tnp}")
        run_standard_eval(
            checkpoint_glob=str(STAGE4_CHECKPOINTS / f"dqn_stage4_*{suf}.pt"),
            out_csv=out_csv,
            video_path=getattr(args, "video_path", None),
            teacher_npz=Path(tnp) if tnp is not None else None,
            baseline_lambda=float(args.baseline_lambda),
            n_episodes=int(args.n_episodes),
            lambda_from_ckpt=True,
            device=str(args.device),
            use_default_teacher=use_def
            and (tnp is None),
        )
        print(f"[ok] 评估表 -> {out_csv}")
        return

    suf = getattr(args, "artifact_suffix", "") or ""
    sweep_62_variants(
        device=args.device,
        quick=args.quick,
        dry_run=args.dry_run,
        stage2_preset=args.stage2_preset,
        lambda_min=args.lambda_min,
        target_mean_cost=args.target_mean_cost,
        no_teacher_reward=args.no_teacher_reward,
        extra_args=list(getattr(args, "extra", []) or []),
        stack_k=int(args.stack_k),
        artifact_suffix=suf,
    )
    if not args.dry_run:
        collect_summary(
            out_csv=STAGE4_OUTPUTS / f"sweep_train_summary_62{suf}.csv",
            glob_pattern=f"train_metrics_*{suf}.csv" if suf else "train_metrics_*.csv",
        )


if __name__ == "__main__":
    main()
