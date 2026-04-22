from __future__ import annotations

from collections.abc import Callable

import numpy as np

from stage_1.env.fish_tracking_env import (
    ACTION_FULL_DETECT,
    ACTION_LIGHT_UPDATE,
    ACTION_REUSE,
    FishTrackingEnv,
)

PolicyFn = Callable[[np.ndarray, int], int]


def policy_always_full(_obs: np.ndarray, _step: int) -> int:
    return ACTION_FULL_DETECT


def policy_periodic(period: int) -> PolicyFn:
    def _p(_obs: np.ndarray, step: int) -> int:
        return ACTION_FULL_DETECT if step % period == 0 else ACTION_REUSE

    return _p


def policy_optical_flow_only(_obs: np.ndarray, _step: int) -> int:
    """每步光流轻量更新（文档 baseline：光学流跟踪 / 不用 YOLO）。"""
    return ACTION_LIGHT_UPDATE


def rollout_episode(
    env: FishTrackingEnv,
    policy: PolicyFn,
    *,
    seed: int,
) -> dict:
    obs, _ = env.reset(seed=seed)
    total_r = 0.0
    consistencies: list[float] = []
    teacher_ious: list[float] = []
    costs: list[float] = []
    step_i = 0
    while True:
        a = policy(obs, step_i)
        obs, r, term, trunc, info = env.step(a)
        total_r += r
        consistencies.append(float(info["consistency_iou"]))
        it = float(info.get("iou_teacher", float("nan")))
        if not np.isnan(it):
            teacher_ious.append(it)
        costs.append(float(info["action_cost"]))
        step_i += 1
        if term or trunc:
            break
    mcons = float(np.mean(consistencies)) if consistencies else 0.0
    mteach = float(np.mean(teacher_ious)) if teacher_ious else float("nan")
    return {
        "return": total_r,
        "mean_consistency": mcons,
        "mean_iou_teacher": mteach,
        "mean_iou": mcons,
        "sum_cost": float(np.sum(costs)),
        "mean_cost": float(np.mean(costs)) if costs else 0.0,
        "steps": step_i,
    }
