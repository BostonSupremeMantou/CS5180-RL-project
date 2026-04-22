#!/usr/bin/env python3
"""eval: no-learning policies or greedy RL ckpt (new stack only, no old_versions)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utilities import paths  # noqa: E402
from utilities.env_wrappers import build_wrapped_env  # noqa: E402
from utilities.evaluate import rollout  # noqa: E402
from utilities.fish_tracking_env import FishTrackingEnv  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="eval policies")
    p.add_argument(
        "--policy",
        choices=("periodic", "flow_only", "always_full", "rl_greedy"),
        default="flow_only",
    )
    p.add_argument("--period", type=int, default=5)
    p.add_argument("--ckpt", type=Path, default=None, help="for rl_greedy")
    p.add_argument("--stack-k", type=int, default=4)
    p.add_argument("--ablation", default="none")
    p.add_argument("--video", type=Path, default=None)
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--baseline-npz", type=Path, default=None)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-csv", type=Path, default=None)
    args = p.parse_args()

    vp = args.video or paths.default_video_path()
    wp = args.weights or paths.default_yolo_weights()
    bp = args.baseline_npz or paths.BASELINE_NPZ
    if not vp.is_file():
        raise SystemExit(f"missing video {vp}")
    if not wp.is_file():
        raise SystemExit(f"missing weights {wp}")

    base = FishTrackingEnv(
        vp,
        wp,
        baseline_npz=bp if bp.is_file() else None,
        lambda_cost=0.35,
        max_episode_steps=500,
        imgsz=640,
        device=args.device,
        random_start=True,
        seed=args.seed,
    )
    env = build_wrapped_env(base, stack_k=args.stack_k, ablation=args.ablation)

    if args.policy == "periodic":
        from agents.no_learning_agents.periodic import periodic_policy

        policy = periodic_policy(args.period)
        tag = f"periodic_{args.period}"
    elif args.policy == "flow_only":
        from agents.no_learning_agents.flow_only import flow_only_policy

        policy = flow_only_policy
        tag = "flow_only"
    elif args.policy == "always_full":
        from baseline.always_full import always_full_policy

        policy = always_full_policy
        tag = "always_full"
    else:
        if args.ckpt is None or not args.ckpt.is_file():
            raise SystemExit("rl_greedy needs --ckpt")
        import torch

        from utilities.load_rl_policy import load_rl_q_network
        from utilities.nn.policy_select import select_action_greedy_any

        dev = torch.device("cpu")
        q = load_rl_q_network(args.ckpt, dev)
        tag = f"rl_greedy_{args.ckpt.stem}"

        def policy(obs: np.ndarray, _step: int) -> int:
            return select_action_greedy_any(q, obs, dev)

    rows: list[dict] = []
    for ep in range(args.episodes):
        stats = rollout(env, policy, seed=args.seed + ep)
        stats["episode"] = ep
        stats["policy"] = tag
        rows.append(stats)
        print(
            f"ep{ep} {tag} return={stats['return']:.2f} "
            f"mean_cost={stats['mean_cost']:.3f} mean_iou_ref={stats['mean_iou_ref']}"
        )

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = not args.out_csv.is_file()
        with args.out_csv.open("a", newline="", encoding="utf-8") as f:
            fieldnames = [
                "episode",
                "policy",
                "return",
                "mean_consistency",
                "mean_iou_ref",
                "mean_cost",
                "steps",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            for r in rows:
                w.writerow(
                    {
                        "episode": r["episode"],
                        "policy": r["policy"],
                        "return": f"{float(r['return']):.6f}",
                        "mean_consistency": f"{float(r['mean_consistency']):.6f}",
                        "mean_iou_ref": r["mean_iou_ref"],
                        "mean_cost": f"{float(r['mean_cost']):.6f}",
                        "steps": r["steps"],
                    }
                )
        print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
