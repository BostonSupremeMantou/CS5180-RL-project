#!/usr/bin/env python3
"""缩小 YOLO 数据集中的图片；标签为归一化坐标时可直接复制，无需改写。"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, required=True, help="含 jpg/png 的图片目录")
    p.add_argument("--labels", type=Path, required=True, help="YOLO txt 标签目录")
    p.add_argument("--out", type=Path, required=True, help="输出根目录，将建 images/ 与 labels/")
    p.add_argument("--scale", type=float, default=0.5, help="宽高的线性比例，0.5 = 一半边长")
    args = p.parse_args()

    if not args.images.is_dir() or not args.labels.is_dir():
        sys.exit("images 或 labels 目录不存在")

    out_img = args.out / "images"
    out_lbl = args.out / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(f for f in args.images.iterdir() if f.suffix.lower() in exts)
    if not files:
        sys.exit(f"{args.images} 下没有图片")

    for img_path in files:
        im = cv2.imread(str(img_path))
        if im is None:
            print(f"skip unreadable: {img_path}", file=sys.stderr)
            continue
        h, w = im.shape[:2]
        nw = max(1, int(round(w * args.scale)))
        nh = max(1, int(round(h * args.scale)))
        small = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
        out_im_path = out_img / img_path.name
        cv2.imwrite(str(out_im_path), small)

        stem = img_path.stem
        lb = args.labels / f"{stem}.txt"
        if lb.is_file():
            shutil.copy2(lb, out_lbl / lb.name)
        else:
            (out_lbl / f"{stem}.txt").write_text("", encoding="utf-8")

    print(f"[ok] {len(files)} 张图 -> {out_img.resolve()} (约 {args.scale:.0%} 边长)")
    print(f"[ok] 标签已复制到 {out_lbl.resolve()}（归一化框，不需改坐标）")


if __name__ == "__main__":
    main()
