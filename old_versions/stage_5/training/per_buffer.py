"""优先经验回放（PER）：SumTree + 重要性采样权重。"""

from __future__ import annotations

import numpy as np


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


class SumTree:
    """线段树存优先级和；叶子与环形缓冲区下标一一对应。"""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)

    def total(self) -> float:
        return float(self.tree[0])

    def update(self, data_idx: int, priority: float) -> None:
        tree_idx = int(data_idx) + self.capacity - 1
        change = float(priority) - self.tree[tree_idx]
        self.tree[tree_idx] = float(priority)
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def _is_leaf(self, tree_idx: int) -> bool:
        return 2 * tree_idx + 1 >= len(self.tree)

    def retrieve(self, s: float) -> int:
        """返回 data_idx（0..capacity-1）。"""
        idx = 0
        while not self._is_leaf(idx):
            left = 2 * idx + 1
            right = left + 1
            if self.tree[left] >= s:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        return idx - (self.capacity - 1)


class PrioritizedReplayBuffer:
    """环形缓冲区 + SumTree；sample 返回 IS 权重。"""

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        rng: np.random.Generator,
        *,
        alpha: float = 0.6,
        eps_priority: float = 1e-6,
    ) -> None:
        self.capacity = _next_pow2(int(capacity))
        self.state_dim = int(state_dim)
        self.rng = rng
        self.alpha = float(alpha)
        self.eps_priority = float(eps_priority)
        self.tree = SumTree(self.capacity)
        self._obs = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self._next_obs = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self._act = np.zeros((self.capacity,), dtype=np.int64)
        self._rew = np.zeros((self.capacity,), dtype=np.float32)
        self._done = np.zeros((self.capacity,), dtype=np.float32)
        self._idx = 0
        self._size = 0
        self._max_prio = 1.0

    def __len__(self) -> int:
        return int(self._size)

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
        prio = max(self._max_prio, self.eps_priority) ** self.alpha
        self.tree.update(i, prio)
        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        beta: float,
    ) -> tuple[np.ndarray, ...]:
        n = max(1, self._size)
        segment = self.tree.total() / float(batch_size)
        idxs = np.zeros((batch_size,), dtype=np.int64)
        priorities = np.zeros((batch_size,), dtype=np.float64)
        for i in range(batch_size):
            a = float((i + self.rng.random()) * segment)
            idx = int(self.tree.retrieve(a))
            idx = min(max(idx, 0), self.capacity - 1)
            idxs[i] = idx
            ti = idx + self.capacity - 1
            priorities[i] = self.tree.tree[ti]
        probs = priorities / max(self.tree.total(), 1e-12)
        weights = (n * probs) ** (-float(beta))
        weights /= weights.max() + 1e-12
        o = self._obs[idxs]
        a = self._act[idxs]
        r = self._rew[idxs]
        no = self._next_obs[idxs]
        d = self._done[idxs]
        w = weights.astype(np.float32)
        return o, a, r, no, d, idxs, w

    def update_priorities(self, idxs: np.ndarray, td_errors: np.ndarray) -> None:
        for i, td in zip(idxs, td_errors, strict=False):
            prio = (float(abs(td)) + self.eps_priority) ** self.alpha
            self._max_prio = max(self._max_prio, prio)
            self.tree.update(int(i), prio)
