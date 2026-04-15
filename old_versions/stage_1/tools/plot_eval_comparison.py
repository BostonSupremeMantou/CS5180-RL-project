#!/usr/bin/env python3
"""读取 eval_summary.csv，生成方法对比柱状图到 stage_1/outputs/。"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _sort_key(policy_id: str) -> tuple[int, str]:
    if policy_id == "always_full":
        return (0, policy_id)
    if policy_id.startswith("periodic_"):
        return (1, policy_id)
    if policy_id == "flow_only":
        return (2, policy_id)
    if policy_id in ("dqn_greedy", "dqn_stage1"):
        return (3, policy_id)
    if policy_id == "dqn_stage2":
        return (4, policy_id)
    if policy_id.startswith("dqn_stage3_sk"):
        try:
            k = int(policy_id.removeprefix("dqn_stage3_sk"))
        except ValueError:
            k = 999
        return (5, f"{k:04d}")
    return (9, policy_id)


def _bar_label(policy_id: str) -> str:
    if policy_id == "always_full":
        return "Always\nFULL"
    if policy_id.startswith("periodic_"):
        n = policy_id.removeprefix("periodic_")
        return f"Periodic\n(n={n})"
    if policy_id == "flow_only":
        return "Light\n(flow)"
    if policy_id == "dqn_greedy":
        return "DQN\nS1"
    if policy_id == "dqn_stage1":
        return "DQN\nStage1"
    if policy_id == "dqn_stage2":
        return "DQN\nStage2"
    if policy_id.startswith("dqn_stage3_sk"):
        k = policy_id.removeprefix("dqn_stage3_sk")
        return f"DQN\nS3 sk={k}"
    return policy_id.replace("_", "\n")


def load_eval_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    from stage_1.utils.paths import OUTPUTS_DIR

    ap = argparse.ArgumentParser(description="Plot eval_summary.csv bar charts")
    ap.add_argument(
        "--csv",
        type=Path,
        default=OUTPUTS_DIR / "eval_summary.csv",
        help="run_eval 写出的汇总 CSV",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=OUTPUTS_DIR,
        help="PNG 输出目录",
    )
    ap.add_argument(
        "--suptitle",
        default="Stage 1: baselines vs DQN (eval means)",
        help="三联图 eval_compare_all.png 的总标题",
    )
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        raise SystemExit("需要 matplotlib、numpy：pip install matplotlib numpy") from e

    if not args.csv.is_file():
        raise SystemExit(
            f"找不到 {args.csv}。请先运行：python -m stage_1.main eval --lambda-from-ckpt"
        )

    rows = load_eval_csv(args.csv)
    if not rows:
        raise SystemExit("CSV 为空")

    def _mean_consistency_row(r: dict[str, str]) -> float:
        if "mean_consistency" in r and r["mean_consistency"].strip():
            return float(r["mean_consistency"])
        return float(r.get("mean_iou", "0") or 0)

    def _mean_teacher_iou_row(r: dict[str, str]) -> float | None:
        if "mean_iou_teacher" not in r:
            return None
        s = (r.get("mean_iou_teacher") or "").strip()
        if not s or s.lower() == "nan":
            return None
        try:
            v = float(s)
        except ValueError:
            return None
        return v if math.isfinite(v) else None

    rows.sort(key=lambda r: _sort_key(r["policy_id"]))
    labels = [_bar_label(r["policy_id"]) for r in rows]
    x = np.arange(len(rows))
    rets = [float(r["mean_return"]) for r in rows]
    ious = [_mean_consistency_row(r) for r in rows]
    costs = [float(r["mean_cost"]) for r in rows]
    teacher_ious: list[float] | None = None
    tvals = [_mean_teacher_iou_row(r) for r in rows]
    if tvals and all(v is not None for v in tvals):
        teacher_ious = [float(v) for v in tvals]

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.tab10(np.linspace(0, 0.45, len(rows)))

    def one_bar(values: list[float], title: str, ylabel: str, fname: str) -> None:
        fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(rows)), 4.2))
        bars = ax.bar(x, values, color=colors, edgecolor="0.35", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.35)
        ymax = max(values) * 1.12 if max(values) > 0 else 1.0
        ymin = 0.0 if min(values) >= 0 else min(values) * 1.05
        ax.set_ylim(ymin, ymax)
        for b, v in zip(bars, values):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        fig.tight_layout()
        fig.savefig(out / fname, dpi=150)
        plt.close(fig)

    one_bar(rets, "Mean episode return", "return", "eval_compare_return.png")
    one_bar(ious, "Mean flow consistency (train signal)", "consistency", "eval_compare_iou.png")
    one_bar(costs, "Mean action cost", "cost", "eval_compare_cost.png")
    if teacher_ious is not None:
        one_bar(
            teacher_ious,
            "Mean IoU vs teacher box (eval)",
            "IoU",
            "eval_compare_teacher_iou.png",
        )

    # 三联或四联（含 teacher IoU 时多一列）
    if teacher_ious is not None:
        panels = (
            (rets, "Mean return", "return"),
            (ious, "Mean consistency", "consistency"),
            (costs, "Mean cost", "cost"),
            (teacher_ious, "Mean IoU vs teacher", "IoU"),
        )
        fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.8))
    else:
        panels = (
            (rets, "Mean return", "return"),
            (ious, "Mean consistency", "consistency"),
            (costs, "Mean cost", "cost"),
        )
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (vals, ttl, ylab) in zip(axes, panels):
        bars = ax.bar(x, vals, color=colors, edgecolor="0.35", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(ttl)
        ax.set_ylabel(ylab)
        ax.grid(True, axis="y", alpha=0.35)
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    fig.suptitle(args.suptitle, fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "eval_compare_all.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[ok] 已写入 {out.resolve()}:")
    names = [
        "eval_compare_return.png",
        "eval_compare_iou.png",
        "eval_compare_cost.png",
    ]
    if teacher_ious is not None:
        names.append("eval_compare_teacher_iou.png")
    names.append("eval_compare_all.png")
    for name in names:
        print(f"  {name}")


if __name__ == "__main__":
    main()
