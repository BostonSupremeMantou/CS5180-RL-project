"""always LIGHT — no YOLO calls, flow-style update every step."""

from __future__ import annotations

import numpy as np

from utilities.actions import LIGHT_UPDATE


def flow_only_policy(_obs: np.ndarray, _step: int) -> int:
    return LIGHT_UPDATE
