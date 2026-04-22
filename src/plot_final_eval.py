#!/usr/bin/env python3
"""Plot intuitive final evaluation comparisons from eval_raw.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _policy_type(name: str) -> str:
    if name.startswith("rl_greedy_"):
        return "rl_agent"
    if name in ("flow_only", "periodic_5"):
        return "non_learning"
    if name == "always_full":
        return "baseline"
    return "other"


def _pretty_name(name: str) -> str:
    if name.startswith("rl_greedy_"):
        return name.replace("rl_greedy_", "").replace("_last", "")
    return name


def _draw_metric(
    ordered: pd.DataFrame,
    out_dir: Path,
    metric_mean: str,
    metric_std: str,
    ylabel: str,
    title: str,
    filename: str,
    higher_is_better: bool,
    colors: dict[str, str],
) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.8))
    x = np.arange(len(ordered))
    vals = ordered[metric_mean].to_numpy()
    errs = ordered[metric_std].fillna(0.0).to_numpy()
    types = ordered["type"].to_list()

    idx_rl = np.where(ordered["type"] == "rl_agent")[0]
    idx_non = np.where(ordered["type"] == "non_learning")[0]
    idx_base = np.where(ordered["type"] == "baseline")[0]
    if len(idx_rl):
        ax.axvspan(idx_rl.min() - 0.5, idx_rl.max() + 0.5, color=colors["rl_agent"], alpha=0.06)
    if len(idx_non):
        ax.axvspan(idx_non.min() - 0.5, idx_non.max() + 0.5, color=colors["non_learning"], alpha=0.06)
    if len(idx_base):
        ax.axvspan(idx_base.min() - 0.5, idx_base.max() + 0.5, color=colors["baseline"], alpha=0.06)

    # Soft bar shadows for quick visual separation.
    ax.bar(x + 0.035, vals, width=0.62, color="black", alpha=0.16, linewidth=0, zorder=1)

    bar_colors = [colors[t] for t in types]
    ax.bar(
        x,
        vals,
        width=0.62,
        color=bar_colors,
        alpha=0.72,
        edgecolor="black",
        linewidth=0.8,
        yerr=errs,
        capsize=4,
        ecolor="black",
        zorder=3,
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["label"], rotation=28, ha="right")
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    rank_order = np.argsort(-vals if higher_is_better else vals)
    top3 = set(rank_order[:3].tolist())
    for i, v in enumerate(vals):
        if i in top3:
            ax.text(i, v, "  ★", fontsize=10, va="bottom", ha="left")

    import matplotlib.patches as mpatches

    handles = [
        mpatches.Patch(color=colors["rl_agent"], alpha=0.72, label="RL agents"),
        mpatches.Patch(color=colors["non_learning"], alpha=0.72, label="Non-learning"),
        mpatches.Patch(color=colors["baseline"], alpha=0.72, label="Baseline"),
    ]
    ax.legend(handles=handles, loc="best", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot final eval comparisons from eval_raw.csv")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("final_results/eval/eval_raw.csv"),
        help="Path to evaluation raw CSV",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("final_results/eval/plots"),
        help="Directory for output plots",
    )
    args = parser.parse_args()

    if not args.input_csv.is_file():
        raise SystemExit(f"missing input csv: {args.input_csv}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input_csv)

    summary = df.groupby("policy", as_index=False).agg(
        episodes=("episode", "count"),
        return_mean=("return", "mean"),
        return_std=("return", "std"),
        iou_mean=("mean_iou_ref", "mean"),
        iou_std=("mean_iou_ref", "std"),
        cons_mean=("mean_consistency", "mean"),
        cons_std=("mean_consistency", "std"),
        cost_mean=("mean_cost", "mean"),
        cost_std=("mean_cost", "std"),
    )
    summary["type"] = summary["policy"].map(_policy_type)
    summary["label"] = summary["policy"].map(_pretty_name)

    rl = summary[summary["type"] == "rl_agent"].sort_values("return_mean", ascending=False)
    non = summary[summary["type"] == "non_learning"].sort_values("return_mean", ascending=False)
    base = summary[summary["type"] == "baseline"]
    ordered = pd.concat([rl, non, base], ignore_index=True)

    colors = {
        "rl_agent": "#1f77b4",
        "non_learning": "#2ca02c",
        "baseline": "#d62728",
        "other": "#7f7f7f",
    }

    _draw_metric(
        ordered,
        args.out_dir,
        "return_mean",
        "return_std",
        "Mean return",
        "Return (higher is better)",
        "01_return_comparison.png",
        True,
        colors,
    )
    _draw_metric(
        ordered,
        args.out_dir,
        "iou_mean",
        "iou_std",
        "Mean IoU vs baseline",
        "Evaluation comparison: baseline IoU (higher is better)",
        "02_iou_comparison.png",
        True,
        colors,
    )
    _draw_metric(
        ordered,
        args.out_dir,
        "cons_mean",
        "cons_std",
        "Mean consistency",
        "Consistency (higher is better)",
        "03_consistency_comparison.png",
        True,
        colors,
    )
    _draw_metric(
        ordered,
        args.out_dir,
        "cost_mean",
        "cost_std",
        "Mean cost",
        "Cost (lower is better)",
        "04_cost_comparison.png",
        False,
        colors,
    )

    ordered[
        [
            "policy",
            "type",
            "episodes",
            "return_mean",
            "return_std",
            "iou_mean",
            "iou_std",
            "cons_mean",
            "cons_std",
            "cost_mean",
            "cost_std",
        ]
    ].to_csv(args.out_dir / "plot_order_table.csv", index=False)

    readme = args.out_dir / "README.txt"
    readme.write_text(
        "Plots generated from eval_raw.csv\n"
        "Category color coding:\n"
        "- RL agents: blue\n"
        "- Non-learning: green\n"
        "- Baseline: red\n\n"
        "Files:\n"
        "01_return_comparison.png\n"
        "02_iou_comparison.png\n"
        "03_consistency_comparison.png\n"
        "04_cost_comparison.png\n"
        "plot_order_table.csv\n",
        encoding="utf-8",
    )

    print(f"saved plots in {args.out_dir}")


if __name__ == "__main__":
    main()
