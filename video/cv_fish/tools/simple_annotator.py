#!/usr/bin/env python3
"""
A minimal interactive bounding-box annotation tool.

Features:
- Input can be an image directory or a video file.
- Draw boxes with mouse drag.
- Save labels in YOLO format and images in parallel folders.

Keyboard:
- n: next frame/image
- p: previous frame/image
- d: delete last box
- c: clear all boxes
- s: save current labels and image
- q: quit
Quick start for your current workflow (200 random fish frames):
    python3 tools/simple_annotator.py

It defaults to:
    --source data/fish_frames_200
    --output data/fish_annotations
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int


class Annotator:
    def __init__(
        self,
        source: Path,
        output_dir: Path,
        class_id: int = 0,
        class_name: str = "object",
        start_index: int = 0,
        stride: int = 1,
    ) -> None:
        self.source = source
        self.output_dir = output_dir
        self.images_dir = output_dir / "images"
        self.labels_dir = output_dir / "labels"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        self.class_id = class_id
        self.class_name = class_name
        self.start_index = max(0, start_index)
        self.stride = max(1, stride)

        self.items = self._collect_items(source)
        if not self.items:
            raise ValueError(f"No valid images/frames found in: {source}")

        self.idx = min(self.start_index, len(self.items) - 1)
        self.cur_image = None
        self.cur_name = ""
        self.boxes: List[Box] = []

        self.dragging = False
        self.start_pt = (0, 0)
        self.temp_pt = (0, 0)

        self.window_name = "Simple Annotator"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

    def _collect_items(self, source: Path) -> List[Tuple[str, object]]:
        if source.is_dir():
            images = sorted(
                [p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
            )
            return [("image", p) for p in images]

        if source.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
            cap = cv2.VideoCapture(str(source))
            items: List[Tuple[str, object]] = []
            frame_idx = 0
            keep_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % self.stride == 0:
                    name = f"{source.stem}_{keep_idx:06d}.jpg"
                    items.append(("frame", (name, frame)))
                    keep_idx += 1
                frame_idx += 1
            cap.release()
            return items

        return []

    def _load_current(self) -> None:
        kind, data = self.items[self.idx]
        if kind == "image":
            path = data
            assert isinstance(path, Path)
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Failed to read image: {path}")
            self.cur_image = img
            self.cur_name = path.stem
        else:
            name, frame = data
            self.cur_image = frame.copy()
            self.cur_name = Path(name).stem
        self.boxes = self._load_existing_boxes()

    def _label_path(self) -> Path:
        return self.labels_dir / f"{self.cur_name}.txt"

    def _image_path(self) -> Path:
        return self.images_dir / f"{self.cur_name}.jpg"

    def _load_existing_boxes(self) -> List[Box]:
        label_path = self._label_path()
        if not label_path.exists() or self.cur_image is None:
            return []

        h, w = self.cur_image.shape[:2]
        boxes: List[Box] = []
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cid, xc, yc, bw, bh = map(float, parts)
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            boxes.append(Box(x1, y1, x2, y2, int(cid)))
        return boxes

    def _save(self) -> None:
        if self.cur_image is None:
            return
        h, w = self.cur_image.shape[:2]
        lines = []
        for b in self.boxes:
            x1, x2 = sorted((max(0, b.x1), min(w - 1, b.x2)))
            y1, y2 = sorted((max(0, b.y1), min(h - 1, b.y2)))
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            xc = x1 + bw / 2.0
            yc = y1 + bh / 2.0
            lines.append(
                f"{b.class_id} {xc / w:.6f} {yc / h:.6f} {bw / w:.6f} {bh / h:.6f}"
            )
        self._label_path().write_text("\n".join(lines) + ("\n" if lines else ""))
        cv2.imwrite(str(self._image_path()), self.cur_image)

    def _draw(self) -> None:
        if self.cur_image is None:
            return
        canvas = self.cur_image.copy()
        for b in self.boxes:
            cv2.rectangle(canvas, (b.x1, b.y1), (b.x2, b.y2), (0, 255, 0), 2)
            cv2.putText(
                canvas,
                f"{self.class_name}:{b.class_id}",
                (b.x1, max(15, b.y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        if self.dragging:
            cv2.rectangle(canvas, self.start_pt, self.temp_pt, (0, 200, 255), 2)

        info = (
            f"[{self.idx + 1}/{len(self.items)}] {self.cur_name} "
            f"boxes={len(self.boxes)} class={self.class_id} "
            "| n-next p-prev d-del c-clear s-save q-quit"
        )
        cv2.putText(
            canvas,
            info,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(self.window_name, canvas)

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if self.cur_image is None:
            return
        h, w = self.cur_image.shape[:2]
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))

        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start_pt = (x, y)
            self.temp_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.temp_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            x1, y1 = self.start_pt
            x2, y2 = (x, y)
            if abs(x2 - x1) >= 3 and abs(y2 - y1) >= 3:
                self.boxes.append(Box(x1, y1, x2, y2, self.class_id))

    def _next(self) -> None:
        self._save()
        self.idx = min(self.idx + 1, len(self.items) - 1)
        self._load_current()

    def _prev(self) -> None:
        self._save()
        self.idx = max(self.idx - 1, 0)
        self._load_current()

    def run(self) -> None:
        self._load_current()
        while True:
            self._draw()
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue
            if key == ord("q"):
                self._save()
                break
            if key == ord("n"):
                self._next()
            elif key == ord("p"):
                self._prev()
            elif key == ord("d"):
                if self.boxes:
                    self.boxes.pop()
            elif key == ord("c"):
                self.boxes.clear()
            elif key == ord("s"):
                self._save()

        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple YOLO annotation tool")
    parser.add_argument(
        "--source",
        type=str,
        default="data/fish_frames_200",
        help="Image directory or video path (default: data/fish_frames_200)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/fish_annotations",
        help="Output directory containing images/ and labels/",
    )
    parser.add_argument("--class-id", type=int, default=0, help="Class id to annotate")
    parser.add_argument(
        "--class-name", type=str, default="fish", help="Class display name"
    )
    parser.add_argument(
        "--start-index", type=int, default=0, help="Start item index for annotation"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every n-th frame when source is video",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.source)
    if not source_path.exists():
        print(
            f"Source not found: {source_path}\n"
            "Tip: run frame sampling first, e.g.\n"
            "  python3 tools/extract_random_frames.py "
            "--video first_attempt/fish_video.mp4 --out data/fish_frames_200 --n 200",
            file=sys.stderr,
        )
        sys.exit(1)

    if source_path.is_dir():
        image_count = len(
            [p for p in source_path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
        )
        print(f"[annotator] source images: {image_count} from {source_path}")
        if image_count == 0:
            print("No images found in source directory.", file=sys.stderr)
            sys.exit(1)

    annotator = Annotator(
        source=source_path,
        output_dir=Path(args.output),
        class_id=args.class_id,
        class_name=args.class_name,
        start_index=args.start_index,
        stride=args.stride,
    )
    annotator.run()


if __name__ == "__main__":
    main()
