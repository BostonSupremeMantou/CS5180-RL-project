#!/usr/bin/env python3
"""Resize video spatially with ffmpeg (e.g. 50% = scale 0.5)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", type=Path, required=True)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="Linear scale factor (0.5 = half width and half height)",
    )
    p.add_argument("--crf", type=int, default=23, help="libx264 CRF, lower = higher quality")
    args = p.parse_args()
    if not args.input.is_file():
        sys.exit(f"not found: {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    vf = f"scale=iw*{args.scale}:ih*{args.scale}"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(args.input),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-crf",
        str(args.crf),
        "-preset",
        "medium",
        "-an",
        str(args.output),
    ]
    subprocess.run(cmd, check=True)
    print(f"ok -> {args.output.resolve()}")


if __name__ == "__main__":
    main()
