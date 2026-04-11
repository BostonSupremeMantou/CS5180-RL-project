from __future__ import annotations

import numpy as np


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """a, b: (4,) xyxy. 无效框（w/h<=0）返回 0。"""
    ax1, ay1, ax2, ay2 = a.astype(np.float64)
    bx1, by1, bx2, by2 = b.astype(np.float64)
    if ax2 <= ax1 or ay2 <= ay1 or bx2 <= bx1 or by2 <= by1:
        return 0.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def xyxy_to_cxcywh_norm(xyxy: np.ndarray, w: int, h: int) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    cx = ((x1 + x2) * 0.5) / max(w, 1)
    cy = ((y1 + y2) * 0.5) / max(h, 1)
    bw = (x2 - x1) / max(w, 1)
    bh = (y2 - y1) / max(h, 1)
    return np.array([cx, cy, bw, bh], dtype=np.float32)


def clip_xyxy(xyxy: np.ndarray, w: int, h: int) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    x1 = float(np.clip(x1, 0, w - 1))
    y1 = float(np.clip(y1, 0, h - 1))
    x2 = float(np.clip(x2, 0, w))
    y2 = float(np.clip(y2, 0, h))
    if x2 <= x1:
        x2 = min(x1 + 1, w)
    if y2 <= y1:
        y2 = min(y1 + 1, h)
    return np.array([x1, y1, x2, y2], dtype=np.float32)
