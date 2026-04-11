#!/usr/bin/env python3
"""逐帧做目标检测，并把画框后的视频写到 cv_implementation/outputs/。

说明：使用官方 .pt 权重时，底层仍需要 PyTorch + Ultralytics + OpenCV，
无法「再少一个」而不换模型格式；本脚本只是把逻辑缩成「读帧—检测—写视频」，
并避免 stream/tqdm/临时目录等额外复杂度。

若 ``import torch`` 单次超过约 10 分钟仍无「torch x.x.x」行，多为杀毒全盘扫
描、项目放在 iCloud/网络盘、或磁盘极慢；可把仓库放到本机 SSD，或为 ``.venv``
加排除后重装 torch。

加载 ``best.pt`` 若出现 ``TimeoutError`` / ``Errno 60``，多为云盘占位文件未完全
本地化；脚本会尝试先复制到临时文件再加载。请尽量用项目 ``.venv`` 运行，与训练
环境一致：直接 ``python cv_implementation/scripts/annotate_video_yolo.py`` 时会**自动改由**
``.venv/bin/python`` 重新启动（若存在 .venv）。若必须用当前解释器，请加
``--allow-system-python`` 或设置环境变量 ``RL_PROJECT_ALLOW_SYSTEM_PYTHON=1``。

也可先 ``source ./activate_venv.sh``，或：
`` .venv/bin/python cv_implementation/scripts/annotate_video_yolo.py ``。
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

# 略减部分 BLAS 线程争抢（对 import 速度帮助有限，但无害）
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_WEIGHTS = REPO_ROOT / "runs" / "fish_yolo11n" / "weights" / "best.pt"
DEFAULT_VIDEO = REPO_ROOT / "videos" / "fish_video.mp4"


def _allow_system_python() -> bool:
    v = os.environ.get("RL_PROJECT_ALLOW_SYSTEM_PYTHON", "").strip().lower()
    return v in ("1", "true", "yes")


def _reexec_with_project_venv_if_needed() -> None:
    """若仓库存在 .venv 且当前不是其 python，则 exec 换解释器（避免误用 conda base）。"""
    if _allow_system_python():
        return
    repo = REPO_ROOT.parent
    vpy = repo / ".venv" / "bin" / "python"
    if not vpy.is_file():
        return
    try:
        if Path(sys.executable).resolve() == vpy.resolve():
            return
    except OSError:
        pass
    script = Path(__file__).resolve()
    print(
        f"[annotate] 检测到未使用项目 .venv，自动切换解释器后重新启动:\n  {vpy}",
        flush=True,
    )
    os.execv(str(vpy), [str(vpy), "-u", str(script), *sys.argv[1:]])


def _warn_if_not_project_venv() -> None:
    repo = REPO_ROOT.parent
    vpy = repo / ".venv" / "bin" / "python"
    if not vpy.is_file():
        return
    try:
        if Path(sys.executable).resolve() == vpy.resolve():
            return
    except OSError:
        pass
    print("[warn] 当前未使用项目 .venv，与训练环境可能不一致。", flush=True)
    print(f"[warn] 正在使用: {sys.executable}", flush=True)
    print(
        f"[warn] 建议在仓库根目录执行: {vpy} cv_implementation/scripts/annotate_video_yolo.py",
        flush=True,
    )


def _is_io_timeout(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    return isinstance(exc, OSError) and getattr(exc, "errno", None) == 60


def _load_yolo(YOLO, weights: Path):
    """加载权重；若云盘/iCloud 直接读 .pt 触发 errno=60，则先复制到本机临时文件。"""

    def _yolo_from(path: str):
        print(
            "[annotate] 正在加载权重（torch.load 读盘；云盘/iCloud 可能需数分钟）…",
            flush=True,
        )
        hb = _import_heartbeat("权重 torch.load")
        try:
            return YOLO(path)
        finally:
            hb.set()

    try:
        return _yolo_from(str(weights))
    except Exception as e:
        if not _is_io_timeout(e):
            raise
    print(
        "[annotate] 直接读取权重超时/失败（常见于 iCloud、网络盘）。"
        "正在复制到本机临时文件再加载…",
        flush=True,
    )
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
    """导入阻塞时主线程无法 print，用后台线程每 10s 提示仍在工作。"""
    stop = threading.Event()

    def _tick() -> None:
        sec = 0
        while not stop.wait(10.0):
            sec += 10
            print(
                f"[annotate] 仍在导入 {label}… 已约 {sec}s "
                f"（在读盘加载大量模块，未死机；请暂勿 Ctrl+C）",
                flush=True,
            )

    threading.Thread(target=_tick, daemon=True).start()
    return stop


def _load_backends():
    """按需导入重依赖；冷启动读盘多，几分钟都有可能。"""
    print(
        "[annotate] 导入 PyTorch…（常见 1～5 分钟，视磁盘/杀毒而定；"
        "下方会每 10s 打一行心跳）",
        flush=True,
    )
    hb = _import_heartbeat("torch")
    try:
        import torch
    finally:
        hb.set()

    print(f"[annotate] torch {torch.__version__}", flush=True)

    print("[annotate] 导入 OpenCV (cv2)…", flush=True)
    hb = _import_heartbeat("cv2")
    try:
        import cv2
    finally:
        hb.set()

    print("[annotate] 导入 ultralytics.YOLO…", flush=True)
    hb = _import_heartbeat("ultralytics")
    try:
        from ultralytics import YOLO
    finally:
        hb.set()

    return cv2, YOLO


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--allow-system-python",
        action="store_true",
        help="禁止自动切换到项目 .venv（仍用当前 python/conda）",
    )
    pre_args, _ = pre.parse_known_args()
    if pre_args.allow_system_python:
        os.environ["RL_PROJECT_ALLOW_SYSTEM_PYTHON"] = "1"

    _reexec_with_project_venv_if_needed()

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--allow-system-python",
        action="store_true",
        help="禁止自动切换到项目 .venv",
    )
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--source", type=Path, default=DEFAULT_VIDEO)
    ap.add_argument("--imgsz", type=int, default=1376)
    ap.add_argument("--device", default="mps")
    ap.add_argument(
        "--out-name",
        default="fish_video_detected.mp4",
        help="写到 cv_implementation/outputs/ 目录",
    )
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--progress-every", type=int, default=30, help="每 N 帧打印一次进度")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    weights = args.weights.resolve()
    src = args.source.resolve()
    out = (REPO_ROOT / "outputs" / args.out_name).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[annotate] weights={weights}", flush=True)
    print(f"[annotate] source={src}", flush=True)
    print(f"[annotate] output={out}", flush=True)

    if not weights.is_file():
        raise SystemExit(f"找不到权重: {weights}")
    if not src.is_file():
        raise SystemExit(f"找不到视频: {src}")

    _warn_if_not_project_venv()

    cv2, YOLO = _load_backends()

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit("无法打开视频（cv2.VideoCapture）")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3:
        fps = 30.0
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_total < 0:
        n_total = 0

    print(
        f"[annotate] 视频 {w0}x{h0} @ {fps:.3f} fps"
        + (f", 约 {n_total} 帧" if n_total > 0 else ""),
        flush=True,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps, (w0, h0))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(out), fourcc, fps, (w0, h0))
    if not writer.isOpened():
        cap.release()
        raise SystemExit("无法创建输出 VideoWriter（可换 --out-name 为 .avi 试 MJPG）")

    t_load = time.monotonic()
    model = _load_yolo(YOLO, weights)
    print(f"[annotate] 模型已加载 ({time.monotonic() - t_load:.1f}s)", flush=True)

    stop_hb = threading.Event()

    def _heartbeat() -> None:
        t0 = time.monotonic()
        while not stop_hb.wait(20.0):
            print(f"[annotate] 仍在处理… {int(time.monotonic() - t0)}s", flush=True)

    hb = threading.Thread(target=_heartbeat, daemon=True)
    hb.start()

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
            if n == 1:
                stop_hb.set()
                if args.debug:
                    r0 = results[0]
                    nb = 0 if r0.boxes is None else len(r0.boxes)
                    print(f"[debug] 首帧检测框数量: {nb}", flush=True)

            pe = max(1, args.progress_every)
            if n % pe == 0:
                if n_total > 0:
                    pct = 100 * n // n_total
                    print(f"[annotate] 已处理 {n}/{n_total} 帧 ({pct}%)", flush=True)
                else:
                    print(f"[annotate] 已处理 {n} 帧", flush=True)
    finally:
        stop_hb.set()
        cap.release()
        writer.release()

    dt = time.monotonic() - t0
    fps_eff = n / dt if dt > 0 else 0.0
    print(f"[ok] 完成: {n} 帧, {dt:.1f}s, 约 {fps_eff:.2f} fps", flush=True)
    print(f"[ok] 输出视频: {out}", flush=True)


if __name__ == "__main__":
    main()
