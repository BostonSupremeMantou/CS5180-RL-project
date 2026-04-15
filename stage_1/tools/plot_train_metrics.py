#!/usr/bin/env python3
"""从 train_metrics.csv 生成训练曲线图，写入 stage_1/outputs/。"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _f(s: str) -> float:
    s = (s or "").strip()
    if not s:
        return float("nan")
    return float(s)


def load_metrics_csv(path: Path) -> dict[str, list[float]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("empty csv")
        cols: dict[str, list[float]] = {h: [] for h in reader.fieldnames}
        for row in reader:
            for h in reader.fieldnames:
                cols[h].append(_f(row.get(h, "")))
    return cols


def main() -> None:
    from stage_1.utils.paths import CHECKPOINTS_DIR, OUTPUTS_DIR

    ap = argparse.ArgumentParser(description="Plot Stage1 train_metrics.csv")
    ap.add_argument(
        "--csv",
        type=Path,
        default=CHECKPOINTS_DIR / "train_metrics.csv",
        help="训练指标 CSV",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=OUTPUTS_DIR,
        help="图片输出目录（默认 stage_1/outputs）",
    )
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("需要 matplotlib：pip install matplotlib") from e

    if not args.csv.is_file():
        raise SystemExit(f"找不到 CSV: {args.csv}")

    d = load_metrics_csv(args.csv)
    step = d["step"]
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    def _plot_xy(
        xs: list[float],
        series: list[tuple[str, list[float], str]],
        title: str,
        ylabel: str,
        fname: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for name, ys, color in series:
            ax.plot(xs, ys, label=name, color=color, linewidth=1.2)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=150)
        plt.close(fig)

    cons_col = "ep_consistency_ma" if "ep_consistency_ma" in d else "ep_iou_ma"
    cons_label = "ep_consistency_ma" if cons_col == "ep_consistency_ma" else "ep_iou_ma"
    # 1) 滑动平均 episode 指标
    ma_series: list[tuple[str, list[float], str]] = [
        ("ep_ret_ma", d["ep_ret_ma"], "C0"),
        (cons_label, d[cons_col], "C1"),
    ]
    ma_title = "Episode MA: return & flow consistency"
    if "ep_teacher_iou_ma" in d:
        ma_series.append(("ep_teacher_iou_ma", d["ep_teacher_iou_ma"], "C4"))
        ma_title = "Episode MA: return, flow consistency & teacher IoU"
    _plot_xy(
        step,
        ma_series,
        ma_title,
        "value",
        "train_episode_ma_ret_iou.png",
    )
    _plot_xy(
        step,
        [
            ("ep_comp_ma", d["ep_comp_ma"], "C2"),
            ("ep_full_frac_ma (%)", [x * 100 for x in d["ep_full_frac_ma"]], "C3"),
        ],
        "Episode MA: compute & FULL action %",
        "value",
        "train_episode_ma_comp_full.png",
    )

    # 2) 损失与梯度
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].semilogy(step, d["loss_mean"], label="loss_mean", alpha=0.85)
    axes[0].semilogy(step, d["td_abs_mean"], label="td_abs_mean", alpha=0.85)
    axes[0].set_ylabel("value (log)")
    axes[0].set_title("Loss & mean |TD|")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, which="both")
    axes[1].plot(step, d["grad_norm_mean"], color="C2", linewidth=1.0, label="grad_norm_mean")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("grad norm")
    axes[1].set_title("Mean grad norm (after clip)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "train_loss_td_grad.png", dpi=150)
    plt.close(fig)

    _plot_xy(
        step,
        [
            ("q_pred_mean", d["q_pred_mean"], "C4"),
            ("q_target_mean", d["q_target_mean"], "C5"),
        ],
        "Mean Q(s,a) vs target",
        "Q",
        "train_q_means.png",
    )

    # 3) 探索与 λ、学习率
    _plot_xy(
        step,
        [
            ("epsilon", d["epsilon"], "C0"),
            ("lambda_cost", d["lambda_cost"], "C1"),
        ],
        "Epsilon & lambda (Lagrangian)",
        "value",
        "train_epsilon_lambda.png",
    )
    _plot_xy(
        step,
        [("lr", d["lr"], "C2")],
        "Learning rate",
        "lr",
        "train_lr.png",
    )

    # 4) buffer
    _plot_xy(
        step,
        [("buffer_size", d["buffer_size"], "C6")],
        "Replay buffer size",
        "size",
        "train_buffer.png",
    )

    print(f"[ok] 已写入 {out.resolve()}:")
    for p in sorted(out.glob("train_*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
