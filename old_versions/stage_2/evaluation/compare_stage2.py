"""Baselines + Stage1 DQN + Stage2（堆叠）DQN 汇总，写 CSV 供柱状图脚本使用。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from stage_1.env.fish_tracking_env import FishTrackingEnv
from stage_1.evaluation.baselines import (
    policy_always_full,
    policy_optical_flow_only,
    policy_periodic,
    rollout_episode,
)
from stage_1.evaluation.run_eval import evaluate_policy, write_eval_summary_csv
from stage_1.models.dqn import load_q_from_checkpoint, select_action_greedy
from stage_1.utils.paths import CHECKPOINTS_DIR, VIDEO_PATH, WEIGHTS_PATH
from stage_2.env.wrappers import build_stage2_env
from stage_2.utils.paths import STAGE2_CHECKPOINTS


def add_compare_cli(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n-episodes", type=int, default=3)
    p.add_argument("--lambda-cost", type=float, default=0.35)
    p.add_argument("--max-episode-steps", type=int, default=200)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="mps")
    p.add_argument("--periodic-n", type=int, default=5)
    p.add_argument(
        "--stage1-ckpt",
        type=Path,
        default=CHECKPOINTS_DIR / "dqn_stage1.pt",
    )
    p.add_argument(
        "--stage2-ckpt",
        type=Path,
        default=STAGE2_CHECKPOINTS / "dqn_stage2.pt",
    )
    p.add_argument("--skip-stage1", action="store_true")
    p.add_argument("--skip-stage2", action="store_true")
    p.add_argument(
        "--lambda-from-ckpt",
        action="store_true",
        help="各 DQN 段用对应 checkpoint 的 lambda_cost_final",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="默认 stage_2/outputs/eval_all_methods.csv",
    )
    p.add_argument("--no-csv", action="store_true")
    p.add_argument(
        "--video-path",
        type=Path,
        default=None,
        help="评估用视频（默认 stage_1 默认 fish_video）",
    )
    p.add_argument(
        "--teacher-npz",
        type=Path,
        default=None,
        help="可选；存在时记录 mean_iou_teacher",
    )


def collect_compare_rows(args: argparse.Namespace) -> list[dict]:
    """Baselines + Stage1 + Stage2 DQN 评估行（不写 CSV）。"""
    vp = Path(args.video_path) if args.video_path is not None else VIDEO_PATH
    if not vp.is_file():
        raise SystemExit(f"找不到视频: {vp}")
    tn: Path | None = None
    if args.teacher_npz is not None:
        tnp = Path(args.teacher_npz)
        if tnp.is_file():
            tn = tnp
        else:
            print(f"[warn] --teacher-npz 不存在 ({tnp})，按无 teacher 评估")

    base_kw = dict(
        video_path=vp,
        yolo_weights=WEIGHTS_PATH,
        teacher_npz=tn,
        lambda_cost=args.lambda_cost,
        max_episode_steps=args.max_episode_steps,
        imgsz=args.imgsz,
        device=args.device,
    )
    seeds = [0, 1, 2, 3, 4][: max(1, args.n_episodes)]
    rows: list[dict] = []

    r = evaluate_policy(
        policy_id="always_full",
        policy_name="Baseline: always FULL_DETECT",
        env_kwargs=base_kw,
        policy_fn=policy_always_full,
        n_episodes=args.n_episodes,
        seeds=seeds,
        random_start=False,
    )
    rows.append({**r, "lambda_cost": float(base_kw["lambda_cost"])})

    r = evaluate_policy(
        policy_id=f"periodic_{args.periodic_n}",
        policy_name=f"Baseline: periodic FULL every {args.periodic_n} (else REUSE)",
        env_kwargs=base_kw,
        policy_fn=policy_periodic(args.periodic_n),
        n_episodes=args.n_episodes,
        seeds=seeds,
        random_start=False,
    )
    rows.append({**r, "lambda_cost": float(base_kw["lambda_cost"])})

    r = evaluate_policy(
        policy_id="flow_only",
        policy_name="Baseline: optical flow only (LIGHT_UPDATE)",
        env_kwargs=base_kw,
        policy_fn=policy_optical_flow_only,
        n_episodes=args.n_episodes,
        seeds=seeds,
        random_start=False,
    )
    rows.append({**r, "lambda_cost": float(base_kw["lambda_cost"])})

    dev = torch.device("cpu")

    if not args.skip_stage1 and args.stage1_ckpt.is_file():
        try:
            ckpt = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(args.stage1_ckpt, map_location="cpu")
        lam = float(args.lambda_cost)
        if args.lambda_from_ckpt:
            v = ckpt.get("lambda_cost_final", ckpt.get("lambda_cost"))
            if v is not None:
                lam = float(v)
        kw = {**base_kw, "lambda_cost": lam}
        print(
            f"\n[Stage1 ckpt] state_dim={ckpt.get('state_dim')} λ_eval={lam}"
        )
        q = load_q_from_checkpoint(ckpt, dev)

        def pol1(obs: np.ndarray, _s: int) -> int:
            return select_action_greedy(q, obs, dev)

        r = evaluate_policy(
            policy_id="dqn_stage1",
            policy_name="DQN Stage1 (MLP, single frame)",
            env_kwargs=kw,
            policy_fn=pol1,
            n_episodes=args.n_episodes,
            seeds=seeds,
            random_start=True,
        )
        rows.append({**r, "lambda_cost": lam})
    else:
        print("\n[Stage1 DQN skipped]")

    if not args.skip_stage2 and args.stage2_ckpt.is_file():
        try:
            ckpt2 = torch.load(args.stage2_ckpt, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt2 = torch.load(args.stage2_ckpt, map_location="cpu")
        if int(ckpt2.get("stage", 0)) != 2:
            print("\n[warn] stage2 ckpt 缺少 stage=2 标记，仍按堆叠观测加载")
        stack_k = int(ckpt2.get("obs_stack_k", 4))
        abl = str(ckpt2.get("ablation", "none"))
        lam2 = float(args.lambda_cost)
        if args.lambda_from_ckpt:
            v = ckpt2.get("lambda_cost_final", ckpt2.get("lambda_cost"))
            if v is not None:
                lam2 = float(v)
        kw2 = {**base_kw, "lambda_cost": lam2}
        print(
            f"\n[Stage2 ckpt] stack_k={stack_k} ablation={abl} state_dim={ckpt2.get('state_dim')} λ_eval={lam2}"
        )
        q2 = load_q_from_checkpoint(ckpt2, dev)

        def pol2(obs: np.ndarray, _s: int) -> int:
            return select_action_greedy(q2, obs, dev)

        print(f"\n=== DQN Stage2 (stack={stack_k}, ablation={abl}) ===")
        rets, cons, teach, costs = [], [], [], []
        for ep in range(args.n_episodes):
            seed = seeds[ep % len(seeds)]
            be = FishTrackingEnv(**kw2, random_start=True, seed=seed)
            env_ep = build_stage2_env(be, stack_k=stack_k, ablation=abl)
            stats = rollout_episode(env_ep, pol2, seed=seed)
            rets.append(stats["return"])
            cons.append(stats["mean_consistency"])
            teach.append(float(stats["mean_iou_teacher"]))
            costs.append(stats["mean_cost"])
            mt = stats["mean_iou_teacher"]
            ts = f" mean_iou_teacher={mt:.4f}" if not np.isnan(mt) else ""
            print(
                f"  ep{ep} seed={seed} return={stats['return']:.3f} "
                f"mean_consistency={stats['mean_consistency']:.4f} mean_cost={stats['mean_cost']:.4f}"
                f" steps={stats['steps']}{ts}"
            )
        mr = float(np.mean(rets))
        mc = float(np.mean(costs))
        mcons = float(np.mean(cons))
        mteach_vals = [x for x in teach if not np.isnan(x)]
        mteach = float(np.mean(mteach_vals)) if mteach_vals else float("nan")
        print(
            f"  MEAN return={mr:.4f} mean_consistency={mcons:.4f} mean_cost={mc:.4f}"
            + (f" mean_iou_teacher={mteach:.4f}" if mteach_vals else "")
        )
        rows.append(
            {
                "policy_id": "dqn_stage2",
                "policy_name": f"DQN Stage2 (stack={stack_k}, ablation={abl})",
                "mean_return": mr,
                "mean_consistency": mcons,
                "mean_iou_teacher": mteach,
                "mean_cost": mc,
                "n_episodes": args.n_episodes,
                "lambda_cost": lam2,
            }
        )
    else:
        print("\n[Stage2 DQN skipped]")

    return rows


def run_compare(args: argparse.Namespace) -> None:
    from stage_2.utils.paths import STAGE2_ROOT

    out_csv = args.out_csv
    if out_csv is None:
        out_csv = STAGE2_ROOT / "outputs" / "eval_all_methods.csv"

    rows = collect_compare_rows(args)
    if not args.no_csv:
        write_eval_summary_csv(rows, out_csv)
        print(f"\n[ok] 对比表 -> {out_csv}")


def main() -> None:
    p = argparse.ArgumentParser(description="Stage2 扩展评估对比")
    add_compare_cli(p)
    run_compare(p.parse_args())


if __name__ == "__main__":
    main()
