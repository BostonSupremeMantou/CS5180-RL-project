from __future__ import annotations

import numpy as np

from utilities.state import STATE_DIM

IDX_FRAME_DIFF = 0
IDX_BOX = slice(1, 5)
IDX_CONF = 5
IDX_VEL = slice(6, 8)
IDX_IOU = 8
IDX_SSF = 9


def _ones() -> np.ndarray:
    return np.ones(STATE_DIM, dtype=np.float32)


def get_ablation_mask(name: str) -> np.ndarray:
    n = name.strip().lower().replace("-", "_")
    if n in ("none", "full", ""):
        return _ones()
    m = _ones()
    if n == "no_frame_diff":
        m[IDX_FRAME_DIFF] = 0.0
    elif n == "no_velocity":
        m[IDX_VEL] = 0.0
    elif n == "no_iou":
        m[IDX_IOU] = 0.0
    elif n == "no_ssf":
        m[IDX_SSF] = 0.0
    elif n == "no_conf":
        m[IDX_CONF] = 0.0
    elif n == "bbox_only":
        m[:] = 0.0
        m[IDX_BOX] = 1.0
    elif n == "no_motion_context":
        m[IDX_FRAME_DIFF] = 0.0
        m[IDX_VEL] = 0.0
    else:
        raise ValueError(f"unknown ablation {name!r}")
    return m
