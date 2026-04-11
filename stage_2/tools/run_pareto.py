#!/usr/bin/env python3
"""固定 λ 训练多个 Stage2 策略并评估，得到 IoU–cost Pareto 散点（CSV + PNG）。"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_lambdas(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def train_pareto_points(
    *,
    lambdas: list[float],
    steps: int,
    stack_k: int,
    ablation: str,
    device: str,
    seed: int,
    out_dir: Path,
    imgsz: int,
    max_episode_steps: int,
) -> None:
    from stage_2.training.train_stage2 import train_stage2
    from stage_2.utils.paths import STAGE2_CHECKPOINTS

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for lam in lambdas:
        tag = f"{lam:.4f}".replace(".", "p")
        save = out_dir / f"pareto_lam_{tag}.pt"
        print(f"\n=== Pareto train λ={lam} -> {save.name} ===")
        train_stage2(
            stack_k=stack_k,
            ablation=ablation,
            total_steps=steps,
            learning_starts=max(300, steps // 25),
            train_freq=2,
            batch_size=64,
            gamma=0.99,
            lr=1e-4,
            lr_min=1e-6,
            weight_decay=0.0,
            use_cosine_lr=False,
            epsilon_start=1.0,
            epsilon_end=0.08,
            epsilon_decay_steps=max(steps - 500, 1000),
            target_update_every=500,
            target_tau=0.0,
            replay_capacity=60_000,
            lambda_cost=lam,
            max_episode_steps=max_episode_steps,
            imgsz=imgsz,
            device_s=device,
            seed=seed,
            save_path=save,
            log_every=max(500, steps // 20),
            metrics_csv=None,
            fixed_lambda=True,
            lambda_initial=lam,
            huber_beta=1.0,
            hidden_dim=320,
            dropout=0.05,
            obs_noise_std=0.02,
            policy_obs_noise_std=0.01,
            grad_updates=1,
            preset_name="pareto",
            stage2_preset_name="none",
        )


def eval_pareto_points(
    *,
    ckpt_dir: Path,
    n_episodes: int,
    device: str,
    imgsz: int,
    max_episode_steps: int,
    out_csv: Path,
    out_png: Path,
) -> None:
    import torch

    from stage_1.env.fish_tracking_env import FishTrackingEnv
    from stage_1.evaluation.baselines import rollout_episode
    from stage_1.models.dqn import load_q_from_checkpoint, select_action_greedy
    from stage_1.utils.paths import DEFAULT_TEACHER_NPZ, VIDEO_PATH, WEIGHTS_PATH
    from stage_2.env.wrappers import build_stage2_env

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("pip install matplotlib") from e

    paths = sorted(ckpt_dir.glob("pareto_lam_*.pt"))
    if not paths:
        raise SystemExit(f"未找到 {ckpt_dir}/pareto_lam_*.pt，请先 --train")

    dev = torch.device("cpu")
    seeds = [0, 1, 2, 3, 4][: max(1, n_episodes)]
    rows: list[dict[str, float | str]] = []

    for path in paths:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")
        lam_trained = float(ckpt.get("lambda_cost_final", ckpt.get("lambda_cost", 0.35)))
        stack_k = int(ckpt.get("obs_stack_k", 4))
        abl = str(ckpt.get("ablation", "none"))
        q = load_q_from_checkpoint(ckpt, dev)

        def pol(obs: np.ndarray, _s: int) -> int:
            return select_action_greedy(q, obs, dev)

        rets, ious, costs = [], [], []
        for ep in range(n_episodes):
            seed = seeds[ep % len(seeds)]
            kw = dict(
                video_path=VIDEO_PATH,
                teacher_npz=DEFAULT_TEACHER_NPZ,
                yolo_weights=WEIGHTS_PATH,
                lambda_cost=lam_trained,
                max_episode_steps=max_episode_steps,
                imgsz=imgsz,
                device=device,
            )
            be = FishTrackingEnv(**kw, random_start=True, seed=seed)
            env_ep = build_stage2_env(be, stack_k=stack_k, ablation=abl)
            st = rollout_episode(env_ep, pol, seed=seed)
            rets.append(st["return"])
            ious.append(st["mean_iou"])
            costs.append(st["mean_cost"])

        rows.append(
            {
                "checkpoint": path.name,
                "lambda_train": lam_trained,
                "mean_return": float(np.mean(rets)),
                "mean_iou": float(np.mean(ious)),
                "mean_cost": float(np.mean(costs)),
            }
        )
        print(
            f"{path.name}: λ_train={lam_trained:.4f} mean_iou={np.mean(ious):.4f} mean_cost={np.mean(costs):.4f}"
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    xs = [r["mean_cost"] for r in rows]
    ys = [r["mean_iou"] for r in rows]
    lbs = [f"λ={r['lambda_train']:.2f}" for r in rows]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(xs, ys, s=80, c=range(len(rows)), cmap="viridis", edgecolors="0.3")
    for x, y, lb in zip(xs, ys, lbs):
        ax.annotate(lb, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("mean action cost (eval)")
    ax.set_ylabel("mean IoU vs teacher")
    ax.set_title("Pareto-style frontier (fixed-λ policies, Stage2 stack)")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[ok] {out_csv}\n[ok] {out_png}")


def main() -> None:
    from stage_2.utils.paths import STAGE2_CHECKPOINTS, STAGE2_ROOT

    ap = argparse.ArgumentParser(description="Pareto: train fixed-λ Stage2 checkpoints + eval curve")
    ap.add_argument("--train", action="store_true", help="对每个 λ 训练一个 checkpoint")
    ap.add_argument("--eval", action="store_true", help="扫描 pareto_lam_*.pt 并写 CSV/图")
    ap.add_argument(
        "--lambdas",
        type=str,
        default="0.03,0.08,0.15,0.25,0.38,0.55",
        help="逗号分隔",
    )
    ap.add_argument("--steps", type=int, default=14_000, help="每个 Pareto 点的训练步数")
    ap.add_argument("--stack-k", type=int, default=4)
    ap.add_argument("--ablation", type=str, default="none")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-dir", type=Path, default=STAGE2_CHECKPOINTS / "pareto")
    ap.add_argument("--n-episodes", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-episode-steps", type=int, default=200)
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=STAGE2_ROOT / "outputs" / "pareto_frontier.csv",
    )
    ap.add_argument(
        "--out-png",
        type=Path,
        default=STAGE2_ROOT / "outputs" / "pareto_iou_vs_cost.png",
    )
    args = ap.parse_args()

    if not args.train and not args.eval:
        args.train = True
        args.eval = True

    lambs = _parse_lambdas(args.lambdas)
    if args.train:
        train_pareto_points(
            lambdas=lambs,
            steps=args.steps,
            stack_k=args.stack_k,
            ablation=args.ablation,
            device=args.device,
            seed=args.seed,
            out_dir=args.ckpt_dir,
            imgsz=args.imgsz,
            max_episode_steps=args.max_episode_steps,
        )
    if args.eval:
        eval_pareto_points(
            ckpt_dir=args.ckpt_dir,
            n_episodes=args.n_episodes,
            device=args.device,
            imgsz=args.imgsz,
            max_episode_steps=args.max_episode_steps,
            out_csv=args.out_csv,
            out_png=args.out_png,
        )


if __name__ == "__main__":
    main()
