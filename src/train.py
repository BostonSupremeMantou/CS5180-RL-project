#!/usr/bin/env python3
"""Train entry: dispatches to agents/RL_agents/<group>/train.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.RL_agents.registry import RL_GROUP_IDS  # noqa: E402
from utilities import paths  # noqa: E402
from utilities.reference_iou import compute_ref_iou_early_stop_threshold  # noqa: E402
from utilities.train_early_stop import RefIoUEarlyStopTracker  # noqa: E402

_DEFAULT_EPILOG = (
    "Default artifact paths (override with --save / --metrics-csv / --plot-dir):\n"
    "  weights  -> weights/RL_agents/<group>/last.pt\n"
    "  metrics  -> outputs/RL_agents/<group>/train_metrics.csv\n"
    "  plots    -> outputs/RL_agents/<group>/plots/*.png (after training, unless --no-plots)\n"
    "Early stop: --early-stop-ref-iou uses always_full mean_iou_ref (needs baseline npz); "
    "--total-steps is always a hard cap."
)


def _learning_starts(args: argparse.Namespace) -> int:
    if args.learning_starts is not None:
        return int(args.learning_starts)
    return min(5_000, max(100, int(args.total_steps) // 4))


def _plot_after_training(
    group_id: str,
    metrics_csv: Path | None,
    *,
    plot_dir: Path | None,
    no_plots: bool,
) -> None:
    if no_plots or metrics_csv is None or not metrics_csv.is_file():
        return
    from utilities.plot_train_metrics import plot_train_metrics_csv

    out = plot_dir or paths.outputs_for("RL_agents", group_id, "plots")
    plot_train_metrics_csv(metrics_csv, out)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train an RL agent group.",
        epilog=_DEFAULT_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--group", choices=RL_GROUP_IDS, required=True)
    p.add_argument("--video", type=Path, default=None)
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--baseline-npz", type=Path, default=None)
    p.add_argument("--stack-k", type=int, default=4)
    p.add_argument("--ablation", default="none")
    p.add_argument(
        "--total-steps",
        type=int,
        default=200_000,
        help="max environment steps (budget); early stop may finish sooner",
    )
    p.add_argument(
        "--early-stop-ref-iou",
        action="store_true",
        help="stop when MA(episode_mean_iou_teacher) >= ratio*mean_iou_ref(always_full); needs baseline npz",
    )
    p.add_argument(
        "--early-stop-ref-ratio",
        type=float,
        default=0.9,
        help="early stop when MA(teacher IoU) >= ratio * ref (default 0.9)",
    )
    p.add_argument("--early-stop-measure-episodes", type=int, default=5)
    p.add_argument("--early-stop-ma-window", type=int, default=20)
    p.add_argument("--early-stop-min-steps", type=int, default=5000)
    p.add_argument(
        "--learning-starts",
        type=int,
        default=None,
        help="default min(5000, total_steps//4)",
    )
    p.add_argument("--device", default="cpu", help="device string for YOLO inside env")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", type=Path, default=None)
    p.add_argument("--metrics-csv", type=Path, default=None)
    p.add_argument("--n-step", type=int, default=3, help="g3_nstep only")
    p.add_argument("--ssf-penalty", type=float, default=0.0, help="g2: SSF reward penalty")
    p.add_argument("--epsilon-tail", type=float, default=None, help="g2: tail epsilon value")
    p.add_argument("--epsilon-tail-steps", type=int, default=0, help="g2: ramp steps after decay")
    p.add_argument(
        "--lambda-baseline-iou-floor",
        type=float,
        default=None,
        help="g2: raise lambda when episode baseline IoU below threshold",
    )
    p.add_argument(
        "--lambda-baseline-iou-below",
        type=float,
        default=0.42,
        help="g2: trigger floor when mean baseline IoU < this",
    )
    p.add_argument("--per-alpha", type=float, default=0.6, help="g4")
    p.add_argument("--per-beta-start", type=float, default=0.4, help="g4")
    p.add_argument("--softq-tau", type=float, default=0.5, help="g5")
    p.add_argument(
        "--gru-hidden",
        type=int,
        default=None,
        help="g7 GRU hidden size (default = hidden dim)",
    )
    p.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="where to save training PNGs (default outputs/RL_agents/<group>/plots)",
    )
    p.add_argument("--no-plots", action="store_true", help="do not write learning-curve PNGs")
    args = p.parse_args()

    vp = args.video or paths.default_video_path()
    wp = args.weights or paths.default_yolo_weights()
    bp = args.baseline_npz or paths.BASELINE_NPZ
    if not vp.is_file():
        raise SystemExit(f"missing video: {vp}")
    if not wp.is_file():
        raise SystemExit(f"missing yolo weights: {wp}")

    ls = _learning_starts(args)
    bp_use = bp if bp.is_file() else None

    ref_stop: RefIoUEarlyStopTracker | None = None
    if args.early_stop_ref_iou:
        thr = compute_ref_iou_early_stop_threshold(
            video_path=vp,
            yolo_weights=wp,
            baseline_npz=bp_use,
            stack_k=args.stack_k,
            ablation=args.ablation,
            device_s=args.device,
            seed=args.seed,
            measure_episodes=args.early_stop_measure_episodes,
            ratio=float(args.early_stop_ref_ratio),
        )
        if thr is None:
            print(
                "[cli] --early-stop-ref-iou ignored (need baseline npz with valid always_full mean_iou_ref)",
                flush=True,
            )
        else:
            print(
                f"[cli] early_stop_ref_iou on: threshold={thr:.6f}  "
                f"ma_window={args.early_stop_ma_window}  min_steps={args.early_stop_min_steps}",
                flush=True,
            )
            ref_stop = RefIoUEarlyStopTracker(
                thr,
                ma_window=args.early_stop_ma_window,
                min_steps=args.early_stop_min_steps,
            )

    common = dict(
        video_path=vp,
        yolo_weights=wp,
        baseline_npz=bp_use,
        stack_k=args.stack_k,
        ablation=args.ablation,
        total_steps=args.total_steps,
        learning_starts=ls,
        device_s=args.device,
        seed=args.seed,
        ref_iou_early_stop=ref_stop,
    )

    gid = args.group
    save = args.save or paths.weights_for("RL_agents", gid, "last.pt")
    metrics = args.metrics_csv or paths.outputs_for("RL_agents", gid, "train_metrics.csv")
    print(
        f"[cli] group={gid}\n  checkpoint -> {save}\n  metrics csv -> {metrics}",
        flush=True,
    )

    if gid == "g1_vanilla":
        from agents.RL_agents.g1_vanilla.train import train_g1

        train_g1(**common, save_path=save, metrics_csv=metrics)
        _plot_after_training(gid, metrics, plot_dir=args.plot_dir, no_plots=args.no_plots)
        return
    if gid == "g2_double_dueling":
        from agents.RL_agents.g2_double_dueling.train import train_g2

        train_g2(
            **common,
            save_path=save,
            metrics_csv=metrics,
            ssf_reward_penalty=float(args.ssf_penalty),
            epsilon_tail=args.epsilon_tail,
            epsilon_tail_steps=int(args.epsilon_tail_steps),
            lambda_baseline_iou_floor=args.lambda_baseline_iou_floor,
            lambda_baseline_iou_below=float(args.lambda_baseline_iou_below),
        )
        _plot_after_training(gid, metrics, plot_dir=args.plot_dir, no_plots=args.no_plots)
        return
    if gid == "g3_nstep":
        from agents.RL_agents.g3_nstep.train import train_g3

        train_g3(**common, save_path=save, metrics_csv=metrics, n_step=int(args.n_step))
        _plot_after_training(gid, metrics, plot_dir=args.plot_dir, no_plots=args.no_plots)
        return
    if gid == "g4_per":
        from agents.RL_agents.g4_per.train import train_g4

        train_g4(
            **common,
            save_path=save,
            metrics_csv=metrics,
            per_alpha=float(args.per_alpha),
            per_beta_start=float(args.per_beta_start),
        )
        _plot_after_training(gid, metrics, plot_dir=args.plot_dir, no_plots=args.no_plots)
        return
    if gid == "g5_softq":
        from agents.RL_agents.g5_softq.train import train_g5

        train_g5(**common, save_path=save, metrics_csv=metrics, softq_tau=float(args.softq_tau))
        _plot_after_training(gid, metrics, plot_dir=args.plot_dir, no_plots=args.no_plots)
        return
    if gid == "g6_c51":
        from agents.RL_agents.g6_c51.train import train_g6

        train_g6(**common, save_path=save, metrics_csv=metrics)
        _plot_after_training(gid, metrics, plot_dir=args.plot_dir, no_plots=args.no_plots)
        return
    if gid == "g7_gru":
        from agents.RL_agents.g7_gru.train import train_g7

        train_g7(
            **common,
            save_path=save,
            metrics_csv=metrics,
            gru_hidden_dim=args.gru_hidden,
        )
        _plot_after_training(gid, metrics, plot_dir=args.plot_dir, no_plots=args.no_plots)
        return

    raise SystemExit(f"group {gid!r} not wired")


if __name__ == "__main__":
    main()
