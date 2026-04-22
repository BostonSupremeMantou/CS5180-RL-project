#!/usr/bin/env python3
"""离线全帧 YOLO 生成 teacher 框序列（奖励与评估参考）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np
from tqdm import tqdm

from stage_1.utils.detector import load_yolo, predict_best_box
from stage_1.utils.geometry import clip_xyxy
from stage_1.utils.paths import STAGE1_ROOT, VIDEO_PATH, WEIGHTS_PATH


def build_teacher_for_video(
    video_path: Path,
    weights: Path,
    out_npz: Path,
    *,
    imgsz: int = 640,
    device: str = "cpu",
    conf: float = 0.25,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames: list[np.ndarray] = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    n = len(frames)
    if n == 0:
        raise RuntimeError("视频无帧")

    model = load_yolo(weights, device=device)
    boxes = np.zeros((n, 4), dtype=np.float32)
    valid = np.zeros((n,), dtype=np.bool_)
    confs = np.zeros((n,), dtype=np.float32)

    prev: np.ndarray | None = None
    for i in tqdm(range(n), desc="teacher FULL_DETECT"):
        b, sc = predict_best_box(
            model, frames[i], imgsz=imgsz, device=device, conf=conf
        )
        if b is not None:
            b = clip_xyxy(b, w, h)
            prev = b.copy()
            boxes[i] = b
            valid[i] = True
            confs[i] = sc
        elif prev is not None:
            boxes[i] = prev
            valid[i] = False
            confs[i] = 0.0
        else:
            valid[i] = False
            confs[i] = 0.0

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        boxes=boxes,
        valid=valid.astype(np.uint8),
        confs=confs,
        width=w,
        height=h,
        n_frames=n,
        video_path=str(video_path.resolve()),
        weights=str(weights.resolve()),
    )
    print(f"[ok] teacher 已写入 {out_npz} ({n} 帧)")


def load_teacher_npz(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {
        "boxes": z["boxes"],
        "valid": z["valid"].astype(bool),
        "confs": z["confs"],
        "width": int(z["width"]),
        "height": int(z["height"]),
        "n_frames": int(z["n_frames"]),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="生成 teacher 轨迹 .npz")
    p.add_argument("--video", type=Path, default=VIDEO_PATH)
    p.add_argument("--weights", type=Path, default=WEIGHTS_PATH)
    p.add_argument("--out", type=Path, default=STAGE1_ROOT / "data/teacher/teacher_fish_video.npz")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="mps")
    p.add_argument("--conf", type=float, default=0.25)
    args = p.parse_args()
    if not args.video.is_file():
        raise SystemExit(f"缺少视频: {args.video}")
    if not args.weights.is_file():
        raise SystemExit(f"缺少权重: {args.weights}")
    build_teacher_for_video(
        args.video,
        args.weights,
        args.out,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
    )


if __name__ == "__main__":
    main()
