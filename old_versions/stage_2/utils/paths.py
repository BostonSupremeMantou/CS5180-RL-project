"""Stage 2 根目录与 checkpoint 路径。"""

from __future__ import annotations

from pathlib import Path

STAGE2_ROOT = Path(__file__).resolve().parents[1]
STAGE2_CHECKPOINTS = STAGE2_ROOT / "checkpoints"
