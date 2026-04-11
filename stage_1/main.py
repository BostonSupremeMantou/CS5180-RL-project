#!/usr/bin/env python3
"""
Stage 1 入口（在仓库根目录执行）:
  python -m stage_1.main preprocess
  python -m stage_1.main train  [--total-steps 15000 ...]
  python -m stage_1.main eval   [--n-episodes 3 ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _cmd_preprocess(args: argparse.Namespace) -> None:
    from stage_1.utils.paths import DEFAULT_TEACHER_NPZ, VIDEO_PATH, WEIGHTS_PATH
    from stage_1.utils.teacher_cache import build_teacher_for_video

    build_teacher_for_video(
        VIDEO_PATH,
        WEIGHTS_PATH,
        DEFAULT_TEACHER_NPZ,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
    )


def _cmd_train(args: argparse.Namespace) -> None:
    from stage_1.training.train_args import apply_train_preset
    from stage_1.training.train_dqn import train
    from stage_1.utils.paths import DEFAULT_TEACHER_NPZ

    if not DEFAULT_TEACHER_NPZ.is_file():
        raise SystemExit("缺少 teacher，请先: python -m stage_1.main preprocess")

    apply_train_preset(args)
    if args.no_cosine_lr:
        args.use_cosine_lr = False
    elif args.cosine_lr:
        args.use_cosine_lr = True
    elif not hasattr(args, "use_cosine_lr"):
        args.use_cosine_lr = False

    train(
        total_steps=args.total_steps,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        batch_size=args.batch_size,
        gamma=args.gamma,
        lr=args.lr,
        lr_min=args.lr_min,
        weight_decay=args.weight_decay,
        use_cosine_lr=args.use_cosine_lr,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        target_update_every=args.target_update_every,
        target_tau=args.target_tau,
        replay_capacity=args.replay_capacity,
        lambda_cost=args.lambda_cost,
        max_episode_steps=args.max_episode_steps,
        imgsz=args.imgsz,
        device_s=args.device,
        seed=args.seed,
        save_path=args.save,
        log_every=args.log_every,
        metrics_csv=args.metrics_csv,
        metrics_ma_episodes=args.metrics_ma_episodes,
        fixed_lambda=args.fixed_lambda,
        target_mean_cost=args.target_mean_cost,
        lambda_lr=args.lambda_lr,
        lambda_initial=args.lambda_initial,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        huber_beta=args.huber_beta,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        obs_noise_std=args.obs_noise_std,
        policy_obs_noise_std=args.policy_obs_noise_std,
        grad_updates=args.grad_updates,
        preset_name=str(args.preset),
    )


def _cmd_eval(args: argparse.Namespace) -> None:
    from stage_1.evaluation.run_eval import run_eval_from_namespace

    run_eval_from_namespace(args)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1: fish tracking + DQN")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("preprocess", help="生成 teacher .npz（离线全帧 YOLO）")
    pp.add_argument("--imgsz", type=int, default=640)
    pp.add_argument("--device", default="mps")
    pp.add_argument("--conf", type=float, default=0.25)
    pp.set_defaults(func=_cmd_preprocess)

    from stage_1.training.train_args import build_train_argparser

    _train_parent = build_train_argparser(add_help=False)
    tp = sub.add_parser(
        "train",
        parents=[_train_parent],
        conflict_handler="resolve",
        help="Dueling+Double DQN；默认 --preset robust（轻量请 --preset none）",
    )
    tp.set_defaults(func=_cmd_train)

    ep = sub.add_parser("eval", help="基线 + DQN 评估")
    from stage_1.evaluation.run_eval import add_eval_cli_args

    add_eval_cli_args(ep)
    ep.set_defaults(func=_cmd_eval)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
