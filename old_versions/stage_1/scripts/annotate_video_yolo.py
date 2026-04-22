#!/usr/bin/env python3
"""逐帧 YOLO 检测并写出标注视频到 stage_1/outputs/。

在仓库根目录执行:
  .venv/bin/python -m stage_1.scripts.annotate_video_yolo
或:
  python stage_1/scripts/annotate_video_yolo.py
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except OSError:
        pass

HERE = Path(__file__).resolve().parent
STAGE1_ROOT = HERE.parent
PROJECT_ROOT = HERE.parents[1]
DEFAULT_WEIGHTS = STAGE1_ROOT / "data" / "weights" / "best.pt"
DEFAULT_VIDEO = STAGE1_ROOT / "data" / "videos" / "fish_video.mp4"


def _allow_system_python() -> bool:
    v = os.environ.get("RL_PROJECT_ALLOW_SYSTEM_PYTHON", "").strip().lower()
    return v in ("1", "true", "yes")


def _reexec_with_project_venv_if_needed() -> None:
    if _allow_system_python():
        return
    vpy = PROJECT_ROOT / ".venv" / "bin" / "python"
    if not vpy.is_file():
        return
    try:
        if Path(sys.executable).resolve() == vpy.resolve():
            return
    except OSError:
        pass
    script = Path(__file__).resolve()
    print(f"[annotate] 切换到项目 .venv:\n  {vpy}", flush=True)
    os.execv(str(vpy), [str(vpy), "-u", str(script), *sys.argv[1:]])


def _is_io_timeout(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    return isinstance(exc, OSError) and getattr(exc, "errno", None) == 60


def _load_yolo(YOLO, weights: Path):
    def _yolo_from(path: str):
        print("[annotate] 加载权重…", flush=True)
        hb = _import_heartbeat("torch.load")
        try:
            return YOLO(path)
        finally:
            hb.set()

    try:
        return _yolo_from(str(weights))
    except Exception as e:
        if not _is_io_timeout(e):
            raise
    fd, name = tempfile.mkstemp(suffix=".pt", prefix="yolo_ckpt_")
    os.close(fd)
    tmp = Path(name)
    try:
        shutil.copy2(weights, tmp, follow_symlinks=True)
        return _yolo_from(str(tmp))
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _import_heartbeat(label: str) -> threading.Event:
    stop = threading.Event()

    def _tick() -> None:
        sec = 0
        while not stop.wait(10.0):
            sec += 10
            print(f"[annotate] 仍在加载 {label}… {sec}s", flush=True)

    threading.Thread(target=_tick, daemon=True).start()
    return stop


def _load_backends():
    hb = _import_heartbeat("torch")
    try:
        import torch
    finally:
        hb.set()
    print(f"[annotate] torch {torch.__version__}", flush=True)
    hb = _import_heartbeat("cv2")
    try:
        import cv2
    finally:
        hb.set()
    hb = _import_heartbeat("ultralytics")
    try:
        from ultralytics import YOLO
    finally:
        hb.set()
    return cv2, YOLO


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--allow-system-python", action="store_true")
    pre_args, _ = pre.parse_known_args()
    if pre_args.allow_system_python:
        os.environ["RL_PROJECT_ALLOW_SYSTEM_PYTHON"] = "1"

    _reexec_with_project_venv_if_needed()

    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-system-python", action="store_true")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--source", type=Path, default=DEFAULT_VIDEO)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out-name", default="fish_video_detected.mp4")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--progress-every", type=int, default=30)
    args = ap.parse_args()

    weights = args.weights.resolve()
    src = args.source.resolve()
    out = (STAGE1_ROOT / "outputs" / args.out_name).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if not weights.is_file():
        raise SystemExit(f"找不到权重: {weights}")
    if not src.is_file():
        raise SystemExit(f"找不到视频: {src}")

    cv2, YOLO = _load_backends()
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit("无法打开视频")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3:
        fps = 30.0
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps, (w0, h0))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(out), fourcc, fps, (w0, h0))
    if not writer.isOpened():
        cap.release()
        raise SystemExit("无法创建 VideoWriter")

    model = _load_yolo(YOLO, weights)
    n = 0
    t0 = time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            results = model.predict(
                frame,
                imgsz=args.imgsz,
                device=args.device,
                conf=args.conf,
                verbose=False,
            )
            plotted = results[0].plot()
            if plotted.shape[0] != h0 or plotted.shape[1] != w0:
                plotted = cv2.resize(plotted, (w0, h0))
            writer.write(plotted)
            n += 1
            pe = max(1, args.progress_every)
            if n % pe == 0 and n_total > 0:
                print(f"[annotate] {n}/{n_total}", flush=True)
    finally:
        cap.release()
        writer.release()

    dt = time.monotonic() - t0
    print(f"[ok] {n} 帧 {dt:.1f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
