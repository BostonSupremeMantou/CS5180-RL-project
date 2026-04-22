"""Minimal Gymnasium env for wrapper / rollout tests (no YOLO / video)."""

from __future__ import annotations

import numpy as np

import pytest

pytest.importorskip("gymnasium")
import gymnasium as gym
from gymnasium import spaces


class DummyTrackingEnv(gym.Env):
    """10-D state, 3 actions, short episodes; info keys match utilities.evaluate.rollout."""

    metadata: dict = {}

    def __init__(self, episode_len: int = 5) -> None:
        super().__init__()
        self.episode_len = int(episode_len)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
        self.lambda_cost = 0.35
        self._step_i = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._step_i = 0
        return np.linspace(0.0, 0.9, 10, dtype=np.float32), {}

    def step(self, action: int):
        self._step_i += 1
        obs = np.linspace(0.1, 1.0, 10, dtype=np.float32) * (self._step_i * 0.1)
        r = 0.05 * float(action)
        term = self._step_i >= self.episode_len
        info = {
            "consistency_iou": 0.4 + 0.01 * self._step_i,
            "iou_teacher": 0.5 + 0.01 * self._step_i,
            "action_cost": float([1.0, 0.25, 0.0][int(action) % 3]),
        }
        return obs, r, term, False, info
