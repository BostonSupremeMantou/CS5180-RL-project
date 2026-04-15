#!/usr/bin/env python3
"""
research_targets.md §6.1：对 Stage2 训练做网格/扫描编排。

通过子进程调用 `python -m stage_2.training.train_stage2`，将权重与 metrics
写入 stage_3/checkpoints 与 stage_3/outputs，并维护 sweep_manifest.csv。
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

# 6.1 默认网格（可按 CLI 覆盖）
DEFAULT_STACK_KS = (1, 2, 4, 8)
DEFAULT_ABLATIONS = (
    "none",
    "no_iou",
    "no_velocity",
    "no_frame_diff",
    "bbox_only",
)


def _safe_tag(s: str) -> str:
    return s.replace("/", "_").replace(" ", "")


def parse_last_metrics_row(metrics_csv: Path) -> dict[str, Any]:
    """读取 train_metrics CSV 最后一行数值列。"""
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


def append_manifest(
    manifest: Path,
    row: dict[str, Any],
) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    exists = manifest.is_file()
    with manifest.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def build_train_cmd(
    *,
    stack_k: int,
    ablation: str,
    save: Path,
    metrics_csv: Path,
    device: str,
    lambda_min: float,
    target_mean_cost: float,
    quick: bool,
    stage2_preset: str,
    extra_args: list[str],
    no_teacher_reward: bool = False,
) -> list[str]:
    s2p = "none" if quick else stage2_preset
    cmd: list[str] = [
        sys.executable,
        "-m",
        "stage_2.training.train_stage2",
        "--preset",
        "none",
        "--stage2-preset",
        s2p,
        "--stack-k",
        str(int(stack_k)),
        "--ablation",
        ablation,
        "--save",
        str(save),
        "--metrics-csv",
        str(metrics_csv),
        "--device",
        device,
        "--lambda-min",
        str(lambda_min),
        "--target-mean-cost",
        str(target_mean_cost),
    ]
    if no_teacher_reward:
        cmd.append("--no-teacher-reward")
    if quick:
        cmd += [
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
    cmd += extra_args
    return cmd


def run_one(
    cmd: list[str],
    *,
    dry_run: bool,
) -> int:
    if dry_run:
        print(" ".join(cmd))
        return 0
    print("\n>>>", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=str(_REPO_ROOT))
    return int(p.returncode)


def sweep_stack_k(
    *,
    stack_ks: tuple[int, ...],
    ablation: str,
    device: str,
    quick: bool,
    dry_run: bool,
    stage2_preset: str,
    lambda_min: float,
    target_mean_cost: float,
    extra_args: list[str],
    artifact_suffix: str = "",
    no_teacher_reward: bool = False,
) -> None:
    from stage_3.utils.paths import STAGE3_CHECKPOINTS, STAGE3_OUTPUTS

    STAGE3_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    STAGE3_OUTPUTS.mkdir(parents=True, exist_ok=True)
    manifest = STAGE3_OUTPUTS / "sweep_manifest.csv"

    for k in stack_ks:
        tag = f"sk{k}_abl{_safe_tag(ablation)}{artifact_suffix}"
        save = STAGE3_CHECKPOINTS / f"dqn_stage3_{tag}.pt"
        metrics = STAGE3_OUTPUTS / f"train_metrics_{tag}.csv"
        cmd = build_train_cmd(
            stack_k=k,
            ablation=ablation,
            save=save,
            metrics_csv=metrics,
            device=device,
            lambda_min=lambda_min,
            target_mean_cost=target_mean_cost,
            quick=quick,
            stage2_preset=stage2_preset,
            extra_args=extra_args,
            no_teacher_reward=no_teacher_reward,
        )
        rc = run_one(cmd, dry_run=dry_run)
        if rc != 0:
            raise SystemExit(f"训练失败 rc={rc} stack_k={k} ablation={ablation}")
        if not dry_run:
            last = parse_last_metrics_row(metrics)
            append_manifest(
                manifest,
                {
                    "kind": "stack_k",
                    "stack_k": k,
                    "ablation": ablation,
                    "artifact_suffix": artifact_suffix,
                    "no_teacher_reward": int(no_teacher_reward),
                    "ckpt": str(save),
                    "metrics_csv": str(metrics),
                    "exit_code": rc,
                    **{f"final_{kk}": last.get(kk, "") for kk in last},
                },
            )


def sweep_ablation(
    *,
    stack_k: int,
    ablations: tuple[str, ...],
    device: str,
    quick: bool,
    dry_run: bool,
    stage2_preset: str,
    lambda_min: float,
    target_mean_cost: float,
    extra_args: list[str],
    artifact_suffix: str = "",
    no_teacher_reward: bool = False,
) -> None:
    from stage_3.utils.paths import STAGE3_CHECKPOINTS, STAGE3_OUTPUTS

    STAGE3_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    STAGE3_OUTPUTS.mkdir(parents=True, exist_ok=True)
    manifest = STAGE3_OUTPUTS / "sweep_manifest.csv"

    for ab in ablations:
        tag = f"sk{stack_k}_abl{_safe_tag(ab)}{artifact_suffix}"
        save = STAGE3_CHECKPOINTS / f"dqn_stage3_{tag}.pt"
        metrics = STAGE3_OUTPUTS / f"train_metrics_{tag}.csv"
        cmd = build_train_cmd(
            stack_k=stack_k,
            ablation=ab,
            save=save,
            metrics_csv=metrics,
            device=device,
            lambda_min=lambda_min,
            target_mean_cost=target_mean_cost,
            quick=quick,
            stage2_preset=stage2_preset,
            extra_args=extra_args,
            no_teacher_reward=no_teacher_reward,
        )
        rc = run_one(cmd, dry_run=dry_run)
        if rc != 0:
            raise SystemExit(f"训练失败 rc={rc} stack_k={stack_k} ablation={ab}")
        if not dry_run:
            last = parse_last_metrics_row(metrics)
            append_manifest(
                manifest,
                {
                    "kind": "ablation",
                    "stack_k": stack_k,
                    "ablation": ab,
                    "artifact_suffix": artifact_suffix,
                    "no_teacher_reward": int(no_teacher_reward),
                    "ckpt": str(save),
                    "metrics_csv": str(metrics),
                    "exit_code": rc,
                    **{f"final_{kk}": last.get(kk, "") for kk in last},
                },
            )


def sweep_grid(
    *,
    stack_ks: tuple[int, ...],
    ablations: tuple[str, ...],
    device: str,
    quick: bool,
    dry_run: bool,
    stage2_preset: str,
    lambda_min: float,
    target_mean_cost: float,
    extra_args: list[str],
    artifact_suffix: str = "",
    no_teacher_reward: bool = False,
) -> None:
    from stage_3.utils.paths import STAGE3_CHECKPOINTS, STAGE3_OUTPUTS

    STAGE3_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    STAGE3_OUTPUTS.mkdir(parents=True, exist_ok=True)
    manifest = STAGE3_OUTPUTS / "sweep_manifest.csv"

    for k in stack_ks:
        for ab in ablations:
            tag = f"sk{k}_abl{_safe_tag(ab)}{artifact_suffix}"
            save = STAGE3_CHECKPOINTS / f"dqn_stage3_{tag}.pt"
            metrics = STAGE3_OUTPUTS / f"train_metrics_{tag}.csv"
            cmd = build_train_cmd(
                stack_k=k,
                ablation=ab,
                save=save,
                metrics_csv=metrics,
                device=device,
                lambda_min=lambda_min,
                target_mean_cost=target_mean_cost,
                quick=quick,
                stage2_preset=stage2_preset,
                extra_args=extra_args,
                no_teacher_reward=no_teacher_reward,
            )
            rc = run_one(cmd, dry_run=dry_run)
            if rc != 0:
                raise SystemExit(f"训练失败 rc={rc} stack_k={k} ablation={ab}")
            if not dry_run:
                last = parse_last_metrics_row(metrics)
                append_manifest(
                    manifest,
                    {
                        "kind": "grid",
                        "stack_k": k,
                        "ablation": ab,
                        "artifact_suffix": artifact_suffix,
                        "no_teacher_reward": int(no_teacher_reward),
                        "ckpt": str(save),
                        "metrics_csv": str(metrics),
                        "exit_code": rc,
                        **{f"final_{kk}": last.get(kk, "") for kk in last},
                    },
                )


def collect_summary(
    *,
    out_csv: Path | None,
    glob_pattern: str = "train_metrics_*.csv",
) -> None:
    """从 stage_3/outputs/ 下匹配 glob 的 metrics CSV 汇总末步指标。"""
    from stage_3.utils.paths import STAGE3_OUTPUTS

    out_csv = out_csv or (STAGE3_OUTPUTS / "sweep_summary.csv")
    files = sorted(STAGE3_OUTPUTS.glob(glob_pattern))
    if not files:
        raise SystemExit(f"未找到 {STAGE3_OUTPUTS}/{glob_pattern}")

    rows: list[dict[str, Any]] = []
    for p in files:
        last = parse_last_metrics_row(p)
        stem = p.stem  # train_metrics_sk4_ablnone
        rows.append(
            {
                "metrics_csv": str(p),
                **last,
            }
        )

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


# 与 research_targets / Stage1 v2 对齐的「标准」stack_k 扫描（ablation=none）
STANDARD_STACK_KS: tuple[int, ...] = (1, 2, 4, 8)
STANDARD_ABLATION = "none"


def run_standard_suite(
    *,
    device: str,
    quick: bool,
    dry_run: bool,
    skip_train: bool,
    skip_collect: bool,
    skip_eval: bool,
    stage2_preset: str,
    lambda_min: float,
    target_mean_cost: float,
    n_episodes: int,
    baseline_lambda: float,
    video_path: Path | None,
    teacher_npz: Path | None,
    extra_args: list[str],
    artifact_suffix: str = "",
    no_teacher_reward: bool = False,
) -> None:
    from stage_3.tools.standard_eval import run_standard_eval
    from stage_3.utils.paths import STAGE3_CHECKPOINTS, STAGE3_OUTPUTS

    STAGE3_OUTPUTS.mkdir(parents=True, exist_ok=True)
    train_summary = STAGE3_OUTPUTS / f"sweep_train_summary{artifact_suffix}.csv"
    eval_suite = STAGE3_OUTPUTS / f"eval_standard_suite{artifact_suffix}.csv"

    preset = "none" if quick else stage2_preset
    if not skip_train:
        sweep_stack_k(
            stack_ks=STANDARD_STACK_KS,
            ablation=STANDARD_ABLATION,
            device=device,
            quick=quick,
            dry_run=dry_run,
            stage2_preset=preset,
            lambda_min=lambda_min,
            target_mean_cost=target_mean_cost,
            extra_args=extra_args,
            artifact_suffix=artifact_suffix,
            no_teacher_reward=no_teacher_reward,
        )
    if dry_run:
        print(f"[dry-run] 将写入训练汇总: {train_summary}")
        print(f"[dry-run] 将写入评估表: {eval_suite}")
        return
    if not skip_collect:
        collect_summary(
            out_csv=train_summary,
            glob_pattern=f"train_metrics_sk*_ablnone{artifact_suffix}.csv",
        )
    if not skip_eval:
        run_standard_eval(
            checkpoint_glob=str(
                STAGE3_CHECKPOINTS / f"dqn_stage3_sk*_ablnone{artifact_suffix}.pt"
            ),
            out_csv=eval_suite,
            video_path=video_path,
            teacher_npz=teacher_npz,
            baseline_lambda=baseline_lambda,
            n_episodes=n_episodes,
            lambda_from_ckpt=True,
            device=device,
        )


def main(argv: list[str] | None = None) -> None:
    from stage_3.utils.paths import STAGE3_OUTPUTS

    ap = argparse.ArgumentParser(description="Stage3 §6.1 网格训练编排")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--device", default="mps")
        p.add_argument(
            "--quick",
            action="store_true",
            help="短跑（~8k 步 + none preset），扫网格用",
        )
        p.add_argument(
            "--dry-run",
            action="store_true",
            help="只打印子进程命令，不执行",
        )
        p.add_argument(
            "--stage2-preset",
            choices=("none", "stage2_robust", "stage2_intense"),
            default="none",
            help="非 --quick 时建议 stage2_robust；quick 下通常配合 none",
        )
        p.add_argument("--lambda-min", type=float, default=0.25)
        p.add_argument("--target-mean-cost", type=float, default=0.45)
        p.add_argument(
            "extra",
            nargs="*",
            help="透传给 train_stage2 的额外参数，例如 --seed 1",
        )

    p1 = sub.add_parser("sweep-stack-k", help="固定 ablation，扫 stack_k")
    add_common(p1)
    p1.add_argument(
        "--stack-ks",
        type=str,
        default=",".join(str(x) for x in DEFAULT_STACK_KS),
        help="逗号分隔，如 1,2,4,8",
    )
    p1.add_argument("--ablation", type=str, default="none")
    p1.set_defaults(func="stack_k")

    p2 = sub.add_parser("sweep-ablation", help="固定 stack_k，扫 ablation")
    add_common(p2)
    p2.add_argument("--stack-k", type=int, default=4)
    p2.add_argument(
        "--ablations",
        type=str,
        default=",".join(DEFAULT_ABLATIONS),
        help="逗号分隔消融名",
    )
    p2.set_defaults(func="ablation")

    p3 = sub.add_parser("sweep-grid", help="stack_k × ablation 全组合（耗时长）")
    add_common(p3)
    p3.add_argument("--stack-ks", type=str, default=",".join(str(x) for x in DEFAULT_STACK_KS))
    p3.add_argument("--ablations", type=str, default=",".join(DEFAULT_ABLATIONS))
    p3.set_defaults(func="grid")

    p4 = sub.add_parser("collect", help="汇总 stage_3/outputs/ 下匹配 glob 的 metrics CSV 末行")
    p4.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"默认 {STAGE3_OUTPUTS}/sweep_summary.csv",
    )
    p4.add_argument(
        "--glob",
        dest="glob_pattern",
        default="train_metrics_*.csv",
        help="相对 stage_3/outputs 的文件 glob，默认 train_metrics_*.csv",
    )
    p4.set_defaults(func="collect")

    p5 = sub.add_parser(
        "standard",
        help="标准实验：stack_k∈{1,2,4,8}、ablation=none；训练汇总与评估写入 stage_3/outputs/",
    )
    p5.add_argument("--device", default="mps")
    p5.add_argument(
        "--quick",
        action="store_true",
        help="每组 ~8k 步（preset none）；正式实验去掉本项并用 --stage2-preset stage2_robust",
    )
    p5.add_argument("--dry-run", action="store_true")
    p5.add_argument("--skip-train", action="store_true")
    p5.add_argument("--skip-collect", action="store_true")
    p5.add_argument("--skip-eval", action="store_true")
    p5.add_argument(
        "--stage2-preset",
        choices=("none", "stage2_robust", "stage2_intense"),
        default="stage2_robust",
        help="非 --quick 时的 Stage2 预设（默认 stage2_robust，与 v2 对齐）",
    )
    p5.add_argument("--lambda-min", type=float, default=0.25)
    p5.add_argument("--target-mean-cost", type=float, default=0.45)
    p5.add_argument("--n-episodes", type=int, default=10)
    p5.add_argument(
        "--baseline-lambda",
        type=float,
        default=0.35,
        help="三条 baseline 环境 λ（与 stage_2 compare 默认一致）",
    )
    p5.add_argument("--video-path", type=Path, default=None)
    p5.add_argument("--teacher-npz", type=Path, default=None)
    p5.add_argument(
        "--artifact-suffix",
        type=str,
        default="",
        help="写入文件名后缀（建议 _notchr），避免覆盖 teacher 奖励跑出的 ckpt/metrics/csv",
    )
    p5.add_argument(
        "--no-teacher-reward",
        action="store_true",
        help="训练子进程加 --no-teacher-reward（光流一致性 −λ·cost，不加载 teacher）",
    )
    p5.add_argument(
        "extra",
        nargs="*",
        help="透传给每次 train_stage2 的额外参数",
    )
    p5.set_defaults(func="standard")

    args = ap.parse_args(argv)
    if args.func == "collect":
        collect_summary(out_csv=args.out, glob_pattern=args.glob_pattern)
        return
    if args.func == "standard":
        run_standard_suite(
            device=args.device,
            quick=args.quick,
            dry_run=args.dry_run,
            skip_train=args.skip_train,
            skip_collect=args.skip_collect,
            skip_eval=args.skip_eval,
            stage2_preset=args.stage2_preset,
            lambda_min=args.lambda_min,
            target_mean_cost=args.target_mean_cost,
            n_episodes=args.n_episodes,
            baseline_lambda=args.baseline_lambda,
            video_path=getattr(args, "video_path", None),
            teacher_npz=getattr(args, "teacher_npz", None),
            extra_args=list(getattr(args, "extra", []) or []),
            artifact_suffix=getattr(args, "artifact_suffix", "") or "",
            no_teacher_reward=bool(getattr(args, "no_teacher_reward", False)),
        )
        return

    extra = list(getattr(args, "extra", []) or [])

    if args.func == "stack_k":
        ks = tuple(int(x.strip()) for x in args.stack_ks.split(",") if x.strip())
        sweep_stack_k(
            stack_ks=ks,
            ablation=args.ablation,
            device=args.device,
            quick=args.quick,
            dry_run=args.dry_run,
            stage2_preset=args.stage2_preset,
            lambda_min=args.lambda_min,
            target_mean_cost=args.target_mean_cost,
            extra_args=extra,
        )
    elif args.func == "ablation":
        abls = tuple(x.strip() for x in args.ablations.split(",") if x.strip())
        sweep_ablation(
            stack_k=args.stack_k,
            ablations=abls,
            device=args.device,
            quick=args.quick,
            dry_run=args.dry_run,
            stage2_preset=args.stage2_preset,
            lambda_min=args.lambda_min,
            target_mean_cost=args.target_mean_cost,
            extra_args=extra,
        )
    elif args.func == "grid":
        ks = tuple(int(x.strip()) for x in args.stack_ks.split(",") if x.strip())
        abls = tuple(x.strip() for x in args.ablations.split(",") if x.strip())
        sweep_grid(
            stack_ks=ks,
            ablations=abls,
            device=args.device,
            quick=args.quick,
            dry_run=args.dry_run,
            stage2_preset=args.stage2_preset,
            lambda_min=args.lambda_min,
            target_mean_cost=args.target_mean_cost,
            extra_args=extra,
        )


if __name__ == "__main__":
    main()
