"""every n steps hit FULL, else REUSE — cheap schedule baseline."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from utilities.actions import FULL_DETECT, REUSE

PolicyFn = Callable[[np.ndarray, int], int]


def periodic_policy(period: int) -> PolicyFn:
    if period < 1:
        raise ValueError("period must be >= 1")

    def _p(_obs: np.ndarray, step: int) -> int:
        return FULL_DETECT if step % period == 0 else REUSE

    return _p
