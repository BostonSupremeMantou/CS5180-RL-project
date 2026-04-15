"""Measure mean_iou_ref under always_full (same stack as training) for early-stop thresholds."""

from __future__ import annotations

from pathlib import Path

from utilities.env_wrappers import build_wrapped_env
from utilities.evaluate import rollout
from utilities.fish_tracking_env import FishTrackingEnv


def compute_ref_iou_early_stop_threshold(
    *,
    video_path: Path,
    yolo_weights: Path,
    baseline_npz: Path | None,
    stack_k: int,
    ablation: str,
    device_s: str,
    seed: int,
    measure_episodes: int,
    ratio: float,
) -> float | None:
    """Return ``ratio * mean(mean_iou_ref)`` over ``always_full`` rollouts, or ``None`` if unusable."""
    if baseline_npz is None or not Path(baseline_npz).is_file():
        return None
    from baseline.always_full import always_full_policy

    base = FishTrackingEnv(
        video_path,
        yolo_weights,
        baseline_npz=baseline_npz,
        lambda_cost=0.35,
        max_episode_steps=500,
        imgsz=640,
        device=device_s,
        random_start=True,
        seed=seed,
    )
    env = build_wrapped_env(base, stack_k=stack_k, ablation=ablation)
    vals: list[float] = []
    for ep in range(max(1, int(measure_episodes))):
        st = rollout(env, always_full_policy, seed=seed + ep)
        v = float(st["mean_iou_ref"])
        if v == v:
            vals.append(v)
    if not vals:
        return None
    ref_mean = float(sum(vals) / len(vals))
    return float(ratio) * ref_mean
