"""stack + ablation on top of FishTrackingEnv."""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from utilities.ablation_masks import get_ablation_mask
from utilities.state import STATE_DIM


class AblationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, mask: np.ndarray) -> None:
        super().__init__(env)
        self.mask = np.asarray(mask, dtype=np.float32).reshape(STATE_DIM)

    @property
    def lambda_cost(self) -> float:
        return float(self.unwrapped.lambda_cost)

    @lambda_cost.setter
    def lambda_cost(self, v: float) -> None:
        self.unwrapped.lambda_cost = float(v)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return (obs.astype(np.float32, copy=False) * self.mask).astype(np.float32)


class StackedObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, stack_k: int) -> None:
        if stack_k < 1:
            raise ValueError("stack_k >= 1")
        super().__init__(env)
        self.stack_k = int(stack_k)
        d = STATE_DIM * self.stack_k
        self._buf: deque[np.ndarray] = deque(maxlen=self.stack_k)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(d,), dtype=np.float32
        )

    @property
    def lambda_cost(self) -> float:
        u = self.unwrapped
        return float(getattr(u, "lambda_cost"))

    @lambda_cost.setter
    def lambda_cost(self, v: float) -> None:
        self.unwrapped.lambda_cost = float(v)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._buf.clear()
        z = np.zeros(STATE_DIM, dtype=np.float32)
        for _ in range(self.stack_k - 1):
            self._buf.append(z.copy())
        self._buf.append(obs.astype(np.float32, copy=False).copy())
        return np.concatenate(list(self._buf), axis=0), info

    def observation(self, obs: np.ndarray) -> np.ndarray:
        self._buf.append(obs.astype(np.float32, copy=False).copy())
        return np.concatenate(list(self._buf), axis=0)


def stacked_state_dim(stack_k: int) -> int:
    return STATE_DIM if int(stack_k) <= 1 else STATE_DIM * int(stack_k)


def build_wrapped_env(
    base_env: gym.Env,
    *,
    stack_k: int,
    ablation: str,
) -> gym.Env:
    mask = get_ablation_mask(ablation)
    e: gym.Env = AblationWrapper(base_env, mask)
    if int(stack_k) > 1:
        e = StackedObsWrapper(e, int(stack_k))
    return e
