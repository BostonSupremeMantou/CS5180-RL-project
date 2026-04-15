"""Disk layout for data, weights, and run outputs. Edit here if you move folders.

Training defaults (see src/train.py):
  - Checkpoints: weights/RL_agents/<group>/last.pt
  - CSV logs:    outputs/RL_agents/<group>/train_metrics.csv
  - Plot PNGs:   outputs/RL_agents/<group>/plots/train_*.png (from CSV unless --no-plots)
"""

from __future__ import annotations

import sys
from pathlib import Path

# repo root (parent of utilities/)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

BASELINE_DIR: Path = PROJECT_ROOT / "baseline"
BASELINE_NPZ: Path = BASELINE_DIR / "baseline.npz"

VIDEO_DIR: Path = PROJECT_ROOT / "video"
WEIGHTS_DIR: Path = PROJECT_ROOT / "weights"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
SMOKE_TESTS_DIR: Path = OUTPUTS_DIR / "smoke_tests"
OLD_VERSIONS: Path = PROJECT_ROOT / "old_versions"

_LEGACY_VIDEO: Path = OLD_VERSIONS / "stage_1" / "data" / "videos" / "fish_video.mp4"
_LEGACY_WEIGHTS: Path = OLD_VERSIONS / "stage_1" / "data" / "weights" / "best.pt"


def default_video_path() -> Path:
    v = VIDEO_DIR / "fish_video.mp4"
    if v.is_file():
        return v
    if _LEGACY_VIDEO.is_file():
        return _LEGACY_VIDEO
    return v


def default_yolo_weights() -> Path:
    w = WEIGHTS_DIR / "baseline" / "best.pt"
    if w.is_file():
        return w
    if _LEGACY_WEIGHTS.is_file():
        return _LEGACY_WEIGHTS
    return w


def weights_for(kind: str, *sub: str) -> Path:
    """kind is 'baseline' | 'non_learning_agents' | 'RL_agents'; sub is extra path parts."""
    p = WEIGHTS_DIR / kind
    for s in sub:
        p = p / s
    return p


def outputs_for(kind: str, *sub: str) -> Path:
    """kind: baseline | non_learning_agents | RL_agents | smoke_tests | final_results | ..."""
    p = OUTPUTS_DIR / kind
    for s in sub:
        p = p / s
    return p


def ensure_legacy_import_path() -> None:
    """so `import stage_1...` works when running from repo root."""
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    ov = str(OLD_VERSIONS)
    if ov not in sys.path:
        sys.path.insert(0, ov)
