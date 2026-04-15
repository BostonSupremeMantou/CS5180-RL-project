"""Always choose FULL_DETECT — reference baseline for max detector usage."""

from __future__ import annotations

import numpy as np

from utilities.actions import FULL_DETECT


def always_full_policy(_obs: np.ndarray, _step: int) -> int:
    return FULL_DETECT
