"""stage_1 根目录与默认数据路径（不依赖工作目录）。"""

from __future__ import annotations

from pathlib import Path

STAGE1_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = STAGE1_ROOT / "data"
VIDEO_PATH = DATA_DIR / "videos" / "fish_video.mp4"
WEIGHTS_PATH = DATA_DIR / "weights" / "best.pt"
TEACHER_DIR = DATA_DIR / "teacher"
DEFAULT_TEACHER_NPZ = TEACHER_DIR / "teacher_fish_video.npz"
YOLO_BASE = STAGE1_ROOT / "models" / "yolo11n.pt"
RUNS_DIR = STAGE1_ROOT / "runs"
OUTPUTS_DIR = STAGE1_ROOT / "outputs"
CHECKPOINTS_DIR = STAGE1_ROOT / "checkpoints"


def project_root() -> Path:
    """含 .venv 的仓库根（RL_project）。"""
    for p in (STAGE1_ROOT, *STAGE1_ROOT.parents):
        if (p / ".venv" / "bin" / "python").is_file():
            return p
    return STAGE1_ROOT.parent
