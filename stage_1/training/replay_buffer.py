from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, rng: np.random.Generator):
        self.capacity = int(capacity)
        self.rng = rng
        self._obs = np.zeros((capacity, state_dim), dtype=np.float32)
        self._next_obs = np.zeros((capacity, state_dim), dtype=np.float32)
        self._act = np.zeros((capacity,), dtype=np.int64)
        self._rew = np.zeros((capacity,), dtype=np.float32)
        self._done = np.zeros((capacity,), dtype=np.float32)
        self._idx = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        i = self._idx
        self._obs[i] = obs
        self._act[i] = action
        self._rew[i] = reward
        self._next_obs[i] = next_obs
        self._done[i] = 1.0 if done else 0.0
        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        idx = self.rng.integers(0, self._size, size=batch_size)
        return (
            self._obs[idx],
            self._act[idx],
            self._rew[idx],
            self._next_obs[idx],
            self._done[idx],
        )
