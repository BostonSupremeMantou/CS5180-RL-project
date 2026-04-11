"""Stage 2 训练 CLI：堆叠观测 + 消融 + 高强度 preset。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from stage_1.training.train_args import apply_train_preset, build_train_argparser
from stage_2.utils.paths import STAGE2_CHECKPOINTS

# 在 Stage 1 preset 之上再覆盖（时序输入更宽网络、更长步数）
STAGE2_PRESETS: dict[str, dict[str, Any]] = {
    "stage2_intense": {
        "total_steps": 155_000,
        "learning_starts": 3_000,
        "train_freq": 1,
        "batch_size": 128,
        "replay_capacity": 320_000,
        "grad_updates": 4,
        "epsilon_decay_steps": 130_000,
        "epsilon_end": 0.05,
        "target_update_every": 5_000,
        "target_tau": 0.005,
        "lr": 5e-5,
        "lr_min": 5e-7,
        "use_cosine_lr": True,
        "weight_decay": 2e-5,
        "hidden_dim": 512,
        "dropout": 0.12,
        "obs_noise_std": 0.04,
        "policy_obs_noise_std": 0.02,
        "log_every": 2_000,
    },
    "stage2_robust": {
        "total_steps": 72_000,
        "learning_starts": 1_800,
        "train_freq": 1,
        "batch_size": 96,
        "replay_capacity": 200_000,
        "grad_updates": 2,
        "epsilon_decay_steps": 60_000,
        "epsilon_end": 0.06,
        "target_update_every": 2_000,
        "target_tau": 0.008,
        "lr": 7e-5,
        "lr_min": 1e-6,
        "use_cosine_lr": True,
        "weight_decay": 1e-5,
        "hidden_dim": 384,
        "dropout": 0.1,
        "obs_noise_std": 0.035,
        "policy_obs_noise_std": 0.018,
        "log_every": 1_000,
    },
}


def apply_stage2_preset(ns: argparse.Namespace) -> None:
    name = getattr(ns, "stage2_preset", "none") or "none"
    if name == "none":
        return
    ov = STAGE2_PRESETS.get(name)
    if ov is None:
        raise ValueError(f"unknown stage2-preset {name}")
    for k, v in ov.items():
        setattr(ns, k, v)


def build_stage2_train_argparser(*, add_help: bool = True) -> argparse.ArgumentParser:
    p = build_train_argparser(add_help=add_help)
    p.set_defaults(
        save=STAGE2_CHECKPOINTS / "dqn_stage2.pt",
        preset="none",
    )
    for action in p._actions:
        if action.dest == "preset":
            action.help = "Stage 1 侧 preset；与 --stage2-preset 组合时先应用本参数再应用 stage2"
            break
    p.add_argument(
        "--stage2-preset",
        choices=("none", "stage2_robust", "stage2_intense"),
        default="stage2_intense",
        help="Stage 2 高强度/稳健覆盖（默认 stage2_intense）",
    )
    p.add_argument("--stack-k", type=int, default=4, help="堆叠帧数，1=仅消融无堆叠")
    p.add_argument(
        "--ablation",
        type=str,
        default="none",
        help="观测消融：none, no_frame_diff, no_velocity, no_iou, no_ssf, bbox_only, ...",
    )
    return p


def apply_all_presets(ns: argparse.Namespace) -> None:
    """先 Stage 1 preset（若 none 则跳过），再 Stage 2 preset。"""
    apply_train_preset(ns)
    apply_stage2_preset(ns)
