from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ultralytics import YOLO


def load_yolo(weights: Path, device: str | None = None) -> "YOLO":
    from ultralytics import YOLO

    return YOLO(str(weights))


def predict_best_box(
    model: "YOLO",
    frame_bgr: np.ndarray,
    *,
    imgsz: int,
    device: str,
    conf: float,
) -> tuple[np.ndarray | None, float]:
    results = model.predict(
        frame_bgr,
        imgsz=imgsz,
        device=device,
        conf=conf,
        verbose=False,
    )
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return None, 0.0
    confs = r0.boxes.conf.cpu().numpy()
    bi = int(np.argmax(confs))
    xyxy = r0.boxes.xyxy[bi].cpu().numpy().astype(np.float32)
    score = float(confs[bi])
    return xyxy, score
