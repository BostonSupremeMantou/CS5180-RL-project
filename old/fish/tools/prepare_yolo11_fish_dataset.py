#!/usr/bin/env python3
"""Build a standard Ultralytics YOLO detection dataset layout from flat images + labels.

Output layout (Ultralytics convention):
    <out>/
      dataset.yaml
      images/train/*.jpg
      images/val/*.jpg
      labels/train/*.txt
      labels/val/*.txt

Default source: data/fish_annotations_half (200 half-res fish images + YOLO txt).
Default out:   data/fish_yolo11

Example:
    python tools/prepare_yolo11_fish_dataset.py
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import yaml

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src-images",
        type=Path,
        default=repo / "data" / "fish_annotations_half" / "images",
    )
    p.add_argument(
        "--src-labels",
        type=Path,
        default=repo / "data" / "fish_annotations_half" / "labels",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=repo / "data" / "fish_yolo11",
        help="Dataset root (will contain images/train, labels/train, …)",
    )
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.src_images.is_dir() or not args.src_labels.is_dir():
        sys.exit(f"Missing source dirs:\n  {args.src_images}\n  {args.src_labels}")

    images = sorted(
        f for f in args.src_images.iterdir() if f.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        sys.exit(f"No images in {args.src_images}")

    rng = random.Random(args.seed)
    indices = list(range(len(images)))
    rng.shuffle(indices)
    n_val = max(1, int(round(len(images) * args.val_ratio)))
    val_idx = set(indices[:n_val])

    if args.out.exists():
        shutil.rmtree(args.out)

    for split in ("train", "val"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest_train: list[str] = []
    manifest_val: list[str] = []

    for i, img_path in enumerate(images):
        split = "val" if i in val_idx else "train"
        stem = img_path.stem
        lbl_path = args.src_labels / f"{stem}.txt"
        if not lbl_path.is_file():
            sys.exit(f"Missing label for {img_path.name}: {lbl_path}")

        shutil.copy2(img_path, args.out / "images" / split / img_path.name)
        shutil.copy2(lbl_path, args.out / "labels" / split / f"{stem}.txt")

        if split == "train":
            manifest_train.append(stem)
        else:
            manifest_val.append(stem)

    dataset_yaml = {
        "path": str(args.out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["fish"],
    }
    yaml_path = args.out / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")

    split_info = args.out / "split_manifest.txt"
    split_info.write_text(
        f"seed={args.seed}\nval_ratio={args.val_ratio}\n\n"
        f"train ({len(manifest_train)}):\n"
        + "\n".join(sorted(manifest_train))
        + "\n\n"
        f"val ({len(manifest_val)}):\n"
        + "\n".join(sorted(manifest_val))
        + "\n",
        encoding="utf-8",
    )

    print(f"[ok] Wrote {yaml_path}")
    print(f"[ok] train={len(manifest_train)} val={len(manifest_val)} total={len(images)}")
    print(f"[ok] Root: {args.out.resolve()}")


if __name__ == "__main__":
    main()
