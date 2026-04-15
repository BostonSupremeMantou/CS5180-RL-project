"""Early stop when episode teacher-IoU moving average crosses a reference threshold."""

from __future__ import annotations

from collections import deque

import numpy as np


class RefIoUEarlyStopTracker:
    """Stop when MA(episode ``episode_mean_iou_teacher``) >= threshold (precomputed)."""

    def __init__(
        self,
        threshold: float | None,
        *,
        ma_window: int = 20,
        min_steps: int = 5000,
    ) -> None:
        self.threshold = threshold
        self.ma_window = max(1, int(ma_window))
        self.min_steps = max(0, int(min_steps))
        self._win: deque[float] = deque(maxlen=self.ma_window)

    def on_episode_end(self, step: int, episode_mean_iou_teacher: float) -> bool:
        if self.threshold is None:
            return False
        v = float(episode_mean_iou_teacher)
        if v != v:
            return False
        self._win.append(v)
        if step < self.min_steps or len(self._win) < self.ma_window:
            return False
        ma = float(np.mean(self._win))
        thr = float(self.threshold)
        if ma >= thr:
            print(
                f"[early_stop_ref_iou] teacher_iou_ma={ma:.6f} >= {thr:.6f} "
                f"(window={self.ma_window}) step={step}",
                flush=True,
            )
            return True
        return False
