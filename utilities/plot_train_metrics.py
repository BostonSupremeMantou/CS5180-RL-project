"""Read train_metrics.csv (RL agents) and write PNG learning curves."""

from __future__ import annotations

import csv
from pathlib import Path


def _f(s: str) -> float:
    s = (s or "").strip()
    if not s:
        return float("nan")
    return float(s)


def load_metrics_csv(path: Path) -> dict[str, list[float]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("empty csv header")
        cols: dict[str, list[float]] = {h: [] for h in reader.fieldnames}
        for row in reader:
            for h in reader.fieldnames:
                cols[h].append(_f(row.get(h, "")))
    return cols


def _finite_series(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    out_x: list[float] = []
    out_y: list[float] = []
    for x, y in zip(xs, ys, strict=False):
        if y == y:  # not nan
            out_x.append(x)
            out_y.append(y)
    return out_x, out_y


def plot_train_metrics_csv(csv_path: Path, out_dir: Path) -> list[Path]:
    """
    Write standard training plots. Skips panels when columns are missing or all NaN.
    Returns list of PNG paths written.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("matplotlib required: pip install matplotlib") from e

    if not csv_path.is_file():
        print(f"[plot_train_metrics] missing csv: {csv_path}", flush=True)
        return []

    d = load_metrics_csv(csv_path)
    step = d.get("step", [])
    if not step:
        print(f"[plot_train_metrics] no rows in {csv_path}", flush=True)
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def plot_xy(
        fname: str,
        title: str,
        ylabel: str,
        series: list[tuple[str, list[float], str]],
    ) -> None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        any_line = False
        for label, ys, color in series:
            xs, yy = _finite_series(step, ys)
            if not xs:
                continue
            ax.plot(xs, yy, label=label, color=color, linewidth=1.2)
            any_line = True
        if not any_line:
            plt.close(fig)
            return
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = out_dir / fname
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

    # Episode moving averages
    ma_series: list[tuple[str, list[float], str]] = []
    if "ep_ret_ma" in d:
        ma_series.append(("ep_ret_ma", d["ep_ret_ma"], "C0"))
    if "ep_consistency_ma" in d:
        ma_series.append(("ep_consistency_ma", d["ep_consistency_ma"], "C1"))
    if "ep_teacher_iou_ma" in d:
        ma_series.append(("ep_teacher_iou_ma", d["ep_teacher_iou_ma"], "C4"))
    title_ma = "Episode MA: return & consistency"
    if any(s[0] == "ep_teacher_iou_ma" for s in ma_series):
        title_ma = "Episode MA: return, consistency & baseline IoU"
    if ma_series:
        plot_xy("train_episode_ma_ret_iou.png", title_ma, "value", ma_series)

    comp_series: list[tuple[str, list[float], str]] = []
    if "ep_comp_ma" in d:
        comp_series.append(("ep_comp_ma", d["ep_comp_ma"], "C2"))
    if "ep_full_frac_ma" in d:
        scaled = [x * 100.0 for x in d["ep_full_frac_ma"]]
        comp_series.append(("ep_full_frac_ma (%)", scaled, "C3"))
    if comp_series:
        plot_xy(
            "train_episode_ma_comp_full.png",
            "Episode MA: compute & FULL action %",
            "value",
            comp_series,
        )

    # Loss / TD / grad
    if "loss_mean" in d and "td_abs_mean" in d and "grad_norm_mean" in d:
        lx, ly = _finite_series(step, d["loss_mean"])
        tx, ty = _finite_series(step, d["td_abs_mean"])
        gx, gy = _finite_series(step, d["grad_norm_mean"])
        if lx or tx or gx:
            fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
            if lx and ly:
                axes[0].semilogy(lx, ly, label="loss_mean", alpha=0.85)
            if tx and ty:
                axes[0].semilogy(tx, ty, label="td_abs_mean", alpha=0.85)
            axes[0].set_ylabel("value (log)")
            axes[0].set_title("Loss & mean |TD|")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3, which="both")
            if gx:
                axes[1].plot(gx, gy, color="C2", linewidth=1.0, label="grad_norm_mean")
            axes[1].set_xlabel("step")
            axes[1].set_ylabel("grad norm")
            axes[1].set_title("Mean grad norm (after clip)")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            fig.tight_layout()
            p = out_dir / "train_loss_td_grad.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            written.append(p)

    # Q means (only if logged — older Stage1 CSVs had these)
    if "q_pred_mean" in d and "q_target_mean" in d:
        plot_xy(
            "train_q_means.png",
            "Mean Q(s,a) vs target",
            "Q",
            [
                ("q_pred_mean", d["q_pred_mean"], "C4"),
                ("q_target_mean", d["q_target_mean"], "C5"),
            ],
        )

    eps_series: list[tuple[str, list[float], str]] = []
    if "epsilon" in d:
        eps_series.append(("epsilon", d["epsilon"], "C0"))
    if "lambda_cost" in d:
        eps_series.append(("lambda_cost", d["lambda_cost"], "C1"))
    if eps_series:
        plot_xy("train_epsilon_lambda.png", "Epsilon & lambda", "value", eps_series)
    if "lr" in d:
        plot_xy("train_lr.png", "Learning rate", "lr", [("lr", d["lr"], "C2")])
    if "buffer_size" in d:
        plot_xy(
            "train_buffer.png",
            "Replay buffer size",
            "size",
            [("buffer_size", d["buffer_size"], "C6")],
        )

    if written:
        print(f"[plot_train_metrics] wrote {len(written)} file(s) -> {out_dir.resolve()}", flush=True)
    return written


def main_cli() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Plot train_metrics.csv into PNG curves.")
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ns = ap.parse_args()
    plot_train_metrics_csv(ns.csv, ns.out_dir)


if __name__ == "__main__":
    main_cli()
