from __future__ import annotations

import numpy as np

from utilities import geometry


def test_iou_overlap() -> None:
    a = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32)
    b = np.array([5.0, 5.0, 15.0, 15.0], dtype=np.float32)
    iou = geometry.iou_xyxy(a, b)
    assert 0.0 < iou < 1.0


def test_iou_disjoint() -> None:
    a = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    b = np.array([5.0, 5.0, 6.0, 6.0], dtype=np.float32)
    assert geometry.iou_xyxy(a, b) == 0.0


def test_clip_xyxy() -> None:
    box = np.array([-5.0, -5.0, 1000.0, 1000.0], dtype=np.float32)
    out = geometry.clip_xyxy(box, w=100, h=100)
    assert out[0] >= 0 and out[2] <= 100


def test_xyxy_to_cxcywh_norm() -> None:
    box = np.array([0.0, 0.0, 100.0, 50.0], dtype=np.float32)
    v = geometry.xyxy_to_cxcywh_norm(box, w=100, h=100)
    assert v.shape == (4,)
    assert 0.0 <= float(v[0]) <= 1.0
