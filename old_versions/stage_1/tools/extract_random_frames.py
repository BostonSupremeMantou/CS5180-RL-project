#!/usr/bin/env python3
"""Extract n random frames from a video (two-pass: count/sort indices, then save).

Typical workflow for detection:
  1) python tools/extract_random_frames.py --video ... --out data/fish_frames_200 --n 200
  2) python tools/simple_annotator.py --source data/fish_frames_200 --output data/annotations
  3) train YOLO on images/ + labels/ (Ultralytics dataset yaml pointing at those folders).
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path

import cv2


def count_frames_ffprobe(video: Path) -> int | None:
    try:
        r = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                str(video),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        s = (r.stdout or "").strip()
        if s and s != "N/A":
            return int(s)
    except FileNotFoundError:
        pass
    return None


def count_frames_cv2_sequential(video: Path) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return 0
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n


def main() -> None:
    p = argparse.ArgumentParser(description="Randomly sample frames from a video")
    p.add_argument("--video", type=str, required=True)
    p.add_argument(
        "--out",
        type=str,
        default="data/random_frames",
        help="Output directory for JPEGs",
    )
    p.add_argument("--n", type=int, default=200, help="Number of frames to save")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = p.parse_args()

    video = Path(args.video)
    if not video.is_file():
        print(f"not found: {video}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = count_frames_ffprobe(video)
    if total is None or total <= 0:
        total = count_frames_cv2_sequential(video)
    if total <= 0:
        print("no frames in video", file=sys.stderr)
        sys.exit(1)

    k = min(args.n, total)
    rng = random.Random(args.seed)
    pick = set(rng.sample(range(total), k))

    cap = cv2.VideoCapture(str(video))
    idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in pick:
            path = out_dir / f"{video.stem}_frame_{idx:06d}.jpg"
            cv2.imwrite(str(path), frame)
            saved += 1
        idx += 1
    cap.release()

    if saved != k:
        print(
            f"warning: wanted {k} frames but saved {saved} "
            f"(decoded {idx} frames; recount may differ from container metadata)",
            file=sys.stderr,
        )

    print(f"total_frames={total} requested={args.n} saved={saved} -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
