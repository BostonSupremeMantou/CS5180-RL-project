"""训练 CLI 参数与 preset（无 torch，供 main 与 train_dqn 共用）。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from stage_1.utils.paths import CHECKPOINTS_DIR

TRAIN_PRESETS: dict[str, dict[str, Any]] = {
    "robust": {
        "total_steps": 65_000,
        "learning_starts": 1_500,
        "train_freq": 1,
        "batch_size": 96,
        "replay_capacity": 180_000,
        "grad_updates": 2,
        "epsilon_decay_steps": 55_000,
        "epsilon_end": 0.06,
        "target_update_every": 2_000,
        "target_tau": 0.008,
        "lr": 8e-5,
        "lr_min": 1e-6,
        "use_cosine_lr": True,
        "weight_decay": 1e-5,
        "hidden_dim": 320,
        "dropout": 0.08,
        "obs_noise_std": 0.035,
        "policy_obs_noise_std": 0.018,
        "log_every": 1_000,
    },
    "intense": {
        "total_steps": 130_000,
        "learning_starts": 2_500,
        "train_freq": 1,
        "batch_size": 128,
        "replay_capacity": 280_000,
        "grad_updates": 4,
        "epsilon_decay_steps": 110_000,
        "epsilon_end": 0.05,
        "target_update_every": 5_000,
        "target_tau": 0.005,
        "lr": 6e-5,
        "lr_min": 5e-7,
        "use_cosine_lr": True,
        "weight_decay": 2e-5,
        "hidden_dim": 416,
        "dropout": 0.1,
        "obs_noise_std": 0.045,
        "policy_obs_noise_std": 0.022,
        "log_every": 2_000,
    },
}


def apply_train_preset(ns: argparse.Namespace) -> None:
    name = getattr(ns, "preset", "none") or "none"
    if name == "none":
        return
    ov = TRAIN_PRESETS.get(name)
    if ov is None:
        raise ValueError(f"unknown preset {name}")
    for k, v in ov.items():
        setattr(ns, k, v)


def build_train_argparser(*, add_help: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=add_help)
    p.add_argument(
        "--preset",
        choices=("none", "robust", "intense"),
        default="robust",
        help="robust≈6.5万步+软更新+观测噪声；intense≈13万步+更宽网络",
    )
    p.add_argument("--total-steps", type=int, default=18_000)
    p.add_argument("--learning-starts", type=int, default=600)
    p.add_argument("--train-freq", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--cosine-lr", action="store_true", help="余弦退火（preset robust/intense 会开启）")
    p.add_argument("--no-cosine-lr", action="store_true", help="强制关闭余弦（覆盖 preset）")
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.08)
    p.add_argument("--epsilon-decay-steps", type=int, default=14_000)
    p.add_argument("--target-update-every", type=int, default=500)
    p.add_argument(
        "--target-tau",
        type=float,
        default=0.0,
        help=">0 时用 Polyak 软更新 target（推荐与 preset 同用）；0 则硬同步",
    )
    p.add_argument("--replay-capacity", type=int, default=60_000)
    p.add_argument("--grad-updates", type=int, default=1, help="每 env 步触发训练时，重复采样+反传次数")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument(
        "--obs-noise-std",
        type=float,
        default=0.0,
        help="Replay 采样后对状态加高斯噪声（仅训练 Q）",
    )
    p.add_argument(
        "--policy-obs-noise-std",
        type=float,
        default=0.0,
        help="ε-greedy 选动作前对观测加噪，增强探索鲁棒性",
    )
    p.add_argument("--lambda-cost", type=float, default=0.35)
    p.add_argument("--fixed-lambda", action="store_true")
    p.add_argument("--target-mean-cost", type=float, default=0.32)
    p.add_argument("--lambda-lr", type=float, default=0.03)
    p.add_argument("--lambda-initial", type=float, default=0.45)
    p.add_argument("--lambda-min", type=float, default=0.02)
    p.add_argument("--lambda-max", type=float, default=1.5)
    p.add_argument("--huber-beta", type=float, default=1.0)
    p.add_argument("--max-episode-steps", type=int, default=200)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", type=Path, default=CHECKPOINTS_DIR / "dqn_stage1.pt")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--metrics-csv", type=Path, default=None)
    p.add_argument("--metrics-ma-episodes", type=int, default=20)
    return p
