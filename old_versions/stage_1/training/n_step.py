"""n-step 回报写入 1-step ReplayBuffer（供 Stage2/4 Double DQN 使用）。"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from stage_1.training.replay_buffer import ReplayBuffer


class NStepReplayBridge:
    """将环境逐步 (s,a,r,s',d) 转为 n 步元组再 push 到底层 buffer。"""

    def __init__(self, n: int, gamma: float, replay: ReplayBuffer) -> None:
        self.n = max(1, int(n))
        self.gamma = float(gamma)
        self.replay = replay
        self._q: deque[
            tuple[np.ndarray, int, float, np.ndarray, bool]
        ] = deque()

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        if self.n == 1:
            self.replay.push(obs, action, reward, next_obs, done)
            return
        self._q.append((obs, action, reward, next_obs, bool(done)))
        if done:
            while self._q:
                h = min(self.n, len(self._q))
                self._emit(h)
        elif len(self._q) >= self.n:
            self._emit(self.n)

    def _emit(self, h: int) -> None:
        items = [self._q[i] for i in range(h)]
        g = sum(self.gamma**i * items[i][2] for i in range(h))
        o0, a0, _, _, _ = items[0]
        _, _, _, next_h, d_h = items[-1]
        self.replay.push(o0, a0, g, next_h, d_h)
        self._q.popleft()

    def flush_terminal(self) -> None:
        """训练非正常截断时把队列里剩余片段写入 replay（与 episode 末 flush 同逻辑）。"""
        if self.n == 1:
            return
        while self._q:
            self._emit(min(self.n, len(self._q)))
