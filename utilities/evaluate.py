"""roll out a policy and collect simple episode stats."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

PolicyFn = Callable[[np.ndarray, int], int]


def rollout(
    env: Any,
    policy: PolicyFn,
    *,
    seed: int,
) -> dict[str, float | int]:
    """works with legacy FishTrackingEnv from old_versions."""
    obs, _ = env.reset(seed=seed)
    total_r = 0.0
    consistencies: list[float] = []
    ref_ious: list[float] = []
    costs: list[float] = []
    step_i = 0
    while True:
        a = policy(obs, step_i)
        obs, r, term, trunc, info = env.step(a)
        total_r += float(r)
        consistencies.append(float(info["consistency_iou"]))
        it = float(info.get("iou_teacher", float("nan")))
        if not np.isnan(it):
            ref_ious.append(it)
        costs.append(float(info["action_cost"]))
        step_i += 1
        if term or trunc:
            break
    mcons = float(np.mean(consistencies)) if consistencies else 0.0
    mref = float(np.mean(ref_ious)) if ref_ious else float("nan")
    return {
        "return": total_r,
        "mean_consistency": mcons,
        "mean_iou_ref": mref,
        "mean_cost": float(np.mean(costs)) if costs else 0.0,
        "steps": step_i,
    }
