from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from stage_1.evaluation.baselines import (
    policy_always_full,
    policy_optical_flow_only,
    policy_periodic,
    rollout_episode,
)
from stage_1.env.fish_tracking_env import FishTrackingEnv
from stage_1.models.dqn import load_q_from_checkpoint, select_action_greedy
from stage_1.utils.paths import (
    CHECKPOINTS_DIR,
    DEFAULT_TEACHER_NPZ,
    OUTPUTS_DIR,
    VIDEO_PATH,
    WEIGHTS_PATH,
)


def evaluate_policy(
    *,
    policy_id: str,
    policy_name: str,
    env_kwargs: dict,
    policy_fn,
    n_episodes: int,
    seeds: list[int],
    random_start: bool = False,
) -> dict[str, float | str | int]:
    print(f"\n=== {policy_name} ===")
    rets, ious, costs = [], [], []
    for ep in range(n_episodes):
        seed = seeds[ep % len(seeds)]
        env = FishTrackingEnv(**env_kwargs, random_start=random_start, seed=seed)
        stats = rollout_episode(env, policy_fn, seed=seed)
        rets.append(stats["return"])
        ious.append(stats["mean_iou"])
        costs.append(stats["mean_cost"])
        print(
            f"  ep{ep} seed={seed} return={stats['return']:.3f} "
            f"mean_iou={stats['mean_iou']:.4f} mean_cost={stats['mean_cost']:.4f} steps={stats['steps']}"
        )
    mr, mi, mc = float(np.mean(rets)), float(np.mean(ious)), float(np.mean(costs))
    print(f"  MEAN return={mr:.4f} mean_iou={mi:.4f} mean_cost={mc:.4f}")
    return {
        "policy_id": policy_id,
        "policy_name": policy_name,
        "mean_return": mr,
        "mean_iou": mi,
        "mean_cost": mc,
        "n_episodes": n_episodes,
    }


def write_eval_summary_csv(rows: list[dict[str, float | str | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "policy_id",
        "policy_name",
        "mean_return",
        "mean_iou",
        "mean_cost",
        "n_episodes",
        "lambda_cost",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def add_eval_cli_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n-episodes", type=int, default=3)
    p.add_argument("--lambda-cost", type=float, default=0.35)
    p.add_argument("--max-episode-steps", type=int, default=200)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="mps", help="Ultralytics 设备")
    p.add_argument("--periodic-n", type=int, default=5)
    p.add_argument("--dqn-ckpt", type=Path, default=CHECKPOINTS_DIR / "dqn_stage1.pt")
    p.add_argument("--skip-dqn", action="store_true", help="无 checkpoint 时跳过 DQN")
    p.add_argument(
        "--dqn-fixed-start",
        action="store_true",
        help="DQN 评估用固定起点；默认随机起点（与训练分布一致）",
    )
    p.add_argument(
        "--lambda-from-ckpt",
        action="store_true",
        help="DQN 段用 checkpoint 里的 lambda_cost_final 作为环境 λ（若无则仍用 --lambda-cost）",
    )
    p.add_argument(
        "--eval-csv",
        type=Path,
        default=OUTPUTS_DIR / "eval_summary.csv",
        help="各方法均值汇总 CSV（与画图脚本共用）",
    )
    p.add_argument(
        "--no-eval-csv",
        action="store_true",
        help="不写评估汇总 CSV",
    )


def run_eval_from_namespace(args: argparse.Namespace) -> None:

    if not DEFAULT_TEACHER_NPZ.is_file():
        raise SystemExit("缺少 teacher .npz，请先 preprocess")

    base_kw = dict(
        video_path=VIDEO_PATH,
        teacher_npz=DEFAULT_TEACHER_NPZ,
        yolo_weights=WEIGHTS_PATH,
        lambda_cost=args.lambda_cost,
        max_episode_steps=args.max_episode_steps,
        imgsz=args.imgsz,
        device=args.device,
    )

    seeds = [0, 1, 2, 3, 4][: max(1, args.n_episodes)]

    summary_rows: list[dict[str, float | str | int]] = []

    r = evaluate_policy(
        policy_id="always_full",
        policy_name="Baseline: always FULL_DETECT",
        env_kwargs=base_kw,
        policy_fn=policy_always_full,
        n_episodes=args.n_episodes,
        seeds=seeds,
        random_start=False,
    )
    summary_rows.append({**r, "lambda_cost": float(base_kw["lambda_cost"])})

    r = evaluate_policy(
        policy_id=f"periodic_{args.periodic_n}",
        policy_name=f"Baseline: periodic FULL every {args.periodic_n} (else REUSE)",
        env_kwargs=base_kw,
        policy_fn=policy_periodic(args.periodic_n),
        n_episodes=args.n_episodes,
        seeds=seeds,
        random_start=False,
    )
    summary_rows.append({**r, "lambda_cost": float(base_kw["lambda_cost"])})

    r = evaluate_policy(
        policy_id="flow_only",
        policy_name="Baseline: optical flow only (LIGHT_UPDATE)",
        env_kwargs=base_kw,
        policy_fn=policy_optical_flow_only,
        n_episodes=args.n_episodes,
        seeds=seeds,
        random_start=False,
    )
    summary_rows.append({**r, "lambda_cost": float(base_kw["lambda_cost"])})

    if args.skip_dqn or not args.dqn_ckpt.is_file():
        print("\n=== DQN (skipped) ===")
        if not args.no_eval_csv:
            write_eval_summary_csv(summary_rows, args.eval_csv)
            print(f"[ok] eval summary -> {args.eval_csv}")
        return

    try:
        ckpt = torch.load(args.dqn_ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.dqn_ckpt, map_location="cpu")

    lam = float(args.lambda_cost)
    if args.lambda_from_ckpt:
        lam_ck = ckpt.get("lambda_cost_final", ckpt.get("lambda_cost"))
        if lam_ck is not None:
            lam = float(lam_ck)
    dqn_kw = {**base_kw, "lambda_cost": lam}

    print(
        f"\n[DQN checkpoint] architecture={ckpt.get('architecture', '?')} "
        f"state_dim={ckpt.get('state_dim', '?')} λ_eval={lam}"
    )

    q = load_q_from_checkpoint(ckpt, torch.device("cpu"))
    dqn_random = not args.dqn_fixed_start

    def dqn_policy(obs: np.ndarray, _step: int) -> int:
        return select_action_greedy(q, obs, torch.device("cpu"))

    r = evaluate_policy(
        policy_id="dqn_greedy",
        policy_name=f"DQN greedy (random_start={dqn_random})",
        env_kwargs=dqn_kw,
        policy_fn=dqn_policy,
        n_episodes=args.n_episodes,
        seeds=seeds,
        random_start=dqn_random,
    )
    summary_rows.append({**r, "lambda_cost": float(lam)})

    if not args.no_eval_csv:
        write_eval_summary_csv(summary_rows, args.eval_csv)
        print(f"\n[ok] eval summary -> {args.eval_csv}")


def run_eval_from_argv(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Stage 1 baselines + DQN 评估")
    add_eval_cli_args(p)
    args = p.parse_args(argv)
    run_eval_from_namespace(args)


def main() -> None:
    import sys

    run_eval_from_argv(sys.argv[1:])


if __name__ == "__main__":
    main()
