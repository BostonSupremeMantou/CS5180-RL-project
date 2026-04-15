from __future__ import annotations

"""Integration: real FishTrackingEnv when video + weights exist on disk."""

import pytest

pytest.importorskip("cv2")
pytest.importorskip("gymnasium")

from utilities import paths
from utilities.fish_tracking_env import FishTrackingEnv


@pytest.mark.skipif(
    not paths.default_video_path().is_file() or not paths.default_yolo_weights().is_file(),
    reason="default video or YOLO weights missing",
)
def test_fish_env_reset_one_step() -> None:
    env = FishTrackingEnv(
        paths.default_video_path(),
        paths.default_yolo_weights(),
        baseline_npz=paths.BASELINE_NPZ if paths.BASELINE_NPZ.is_file() else None,
        lambda_cost=0.35,
        max_episode_steps=3,
        imgsz=320,
        device="cpu",
        random_start=True,
        seed=0,
    )
    obs, info = env.reset(seed=0)
    assert obs.shape[0] == 10
    obs2, r, term, trunc, info2 = env.step(0)
    assert obs2.shape == obs.shape
    assert "consistency_iou" in info2
