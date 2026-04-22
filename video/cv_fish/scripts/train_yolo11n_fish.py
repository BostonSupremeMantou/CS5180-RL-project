#!/usr/bin/env python3
"""Train YOLO11n on fish (data/fish_yolo11).

Prepare data first:
    python tools/prepare_yolo11_fish_dataset.py

On macOS, Ultralytics can appear “stuck” during:
- label scan (first run builds labels/train.cache — wait for 100%, do not Ctrl+C)
- matplotlib font scan, thop import — patched below.
- On macOS, RAM dataset cache (``cache=True``) has been seen to trigger a bus error
  right after the dataloader speed probe; default is ``cache=False``.
"""

from __future__ import annotations

import argparse
import csv
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

_patch_done = False


def _apply_ultralytics_mac_patches() -> None:
    global _patch_done
    if _patch_done:
        return
    _patch_done = True

    try:
        import torch.xpu as xpu

        def _noop(*_a, **_k):
            return None

        xpu.manual_seed_all = _noop  # type: ignore[assignment]
        if hasattr(xpu, "manual_seed"):
            xpu.manual_seed = _noop  # type: ignore[assignment]
    except Exception:
        pass

    def _manual_seed_cpu_only(seed: int | float):
        return torch.default_generator.manual_seed(int(seed))

    torch.manual_seed = _manual_seed_cpu_only  # type: ignore[assignment]

    from ultralytics import settings  # noqa: E402
    from ultralytics.utils import USER_CONFIG_DIR  # noqa: E402

    settings.update({"sync": False})

    def _check_font_fast(font: str = "Arial.ttf"):
        name = Path(font).name
        p = USER_CONFIG_DIR / name
        if p.is_file():
            return p
        for q in (
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
            Path("/Library/Fonts/Arial.ttf"),
            Path("/System/Library/Fonts/Helvetica.ttc"),
        ):
            if q.is_file():
                return q
        return p

    import ultralytics.data.utils as du  # noqa: E402
    import ultralytics.engine.trainer as tr  # noqa: E402
    import ultralytics.utils.callbacks.base as cb  # noqa: E402
    import ultralytics.utils.checks as chk  # noqa: E402
    import ultralytics.utils.torch_utils as tu  # noqa: E402

    chk.check_font = _check_font_fast  # type: ignore[assignment]
    du.check_font = _check_font_fast  # type: ignore[assignment]

    def _no_integration(_t):
        return

    cb.add_integration_callbacks = _no_integration  # type: ignore[assignment]
    tr.callbacks.add_integration_callbacks = _no_integration  # type: ignore[attr-defined]

    def _no_flops(_m, imgsz=640):
        return 0.0

    tu.get_flops = _no_flops  # type: ignore[assignment]

    def _read_results_csv_no_polars(self):
        """Avoid importing polars in save_model() on macOS/Python 3.13."""
        try:
            csv_path = Path(self.csv)
            if not csv_path.is_file():
                return {}
            with csv_path.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                return {}
            cols = rows[0].keys()
            out = {k: [] for k in cols}
            for r in rows:
                for k in cols:
                    v = r.get(k, "")
                    if v is None or v == "":
                        out[k].append(v)
                        continue
                    try:
                        out[k].append(float(v))
                    except (TypeError, ValueError):
                        out[k].append(v)
            return out
        except Exception:
            return {}

    tr.BaseTrainer.read_results_csv = _read_results_csv_no_polars  # type: ignore[assignment]

    print(
        "[ok] Mac patches: sync off, check_font, get_flops, integration callbacks, no-polars results csv.",
        flush=True,
    )


_apply_ultralytics_mac_patches()

from ultralytics import YOLO  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DATASET_YAML = REPO_ROOT / "data" / "fish_yolo11" / "dataset.yaml"
PREPARE = REPO_ROOT / "tools" / "prepare_yolo11_fish_dataset.py"
HEARTBEAT_SEC = 30


def default_device() -> str:
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def remove_label_caches(dataset_root: Path) -> None:
    """Remove *.cache under labels/ if you interrupted a previous run (optional safety)."""
    labels = dataset_root / "labels"
    if not labels.is_dir():
        return
    for c in labels.rglob("*.cache"):
        try:
            c.unlink()
            print(f"[ok] removed incomplete cache: {c}", flush=True)
        except OSError:
            pass


def build_args(
    device: str, epochs: int, imgsz: int, batch: int, *, cache: bool = False
) -> dict:
    return {
        "data": str(DATASET_YAML.resolve()),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "workers": 0,
        "cache": cache,
        "rect": True,
        "plots": True,
        "pretrained": True,
        "deterministic": False,
        "verbose": True,
        "close_mosaic": 0,
        "project": str(REPO_ROOT / "runs"),
        "name": "fish_yolo11n",
        "exist_ok": True,
        "amp": False if device == "mps" else True,
        "augment": False,
        "auto_augment": None,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "fliplr": 0.0,
        "flipud": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
    }


def _heartbeat(stop: threading.Event) -> None:
    t0 = time.monotonic()
    while not stop.wait(HEARTBEAT_SEC):
        print(
            f"[heartbeat] {int(time.monotonic() - t0)}s (scanning labels or first batch can take minutes)",
            flush=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare-dataset", action="store_true")
    ap.add_argument("--clear-label-cache", action="store_true", help="删除 labels 下 *.cache 后重扫")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument(
        "--imgsz",
        type=int,
        default=1376,
        help="目标输入长边尺寸（默认 1376，接近原图 1366，且更易满足 stride 对齐）",
    )
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=default_device())
    ap.add_argument(
        "--cache",
        action="store_true",
        help="将训练集载入 RAM（Ultralytics cache）；小数据集通常不需要，macOS+MPS 上曾引发 bus error",
    )
    args = ap.parse_args()

    if args.prepare_dataset:
        subprocess.run([sys.executable, str(PREPARE)], cwd=str(REPO_ROOT), check=True)

    if not DATASET_YAML.is_file():
        sys.exit(f"缺少 {DATASET_YAML}\n运行: python tools/prepare_yolo11_fish_dataset.py")

    ds_root = DATASET_YAML.parent
    if args.clear_label_cache:
        remove_label_caches(ds_root)

    train_kw = build_args(
        args.device, args.epochs, args.imgsz, args.batch, cache=args.cache
    )
    print(train_kw, flush=True)
    print(
        "\n提示：首次训练会在「Scanning ... labels/train」停留几分钟，"
        "是在生成 train.cache（约 160 个文件）。**不要 Ctrl+C**，等进度到 100%。\n",
        flush=True,
    )

    stop = threading.Event()
    hb = threading.Thread(target=_heartbeat, args=(stop,), daemon=True)
    hb.start()
    try:
        model = YOLO(str(REPO_ROOT / "models" / "yolo11n.pt"))
        model.train(**train_kw)
    finally:
        stop.set()

    print(
        "[ok] done. Weights:", REPO_ROOT / "runs/fish_yolo11n/weights/best.pt", flush=True
    )


if __name__ == "__main__":
    main()
