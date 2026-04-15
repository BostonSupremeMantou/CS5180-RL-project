"""Stage 3 根目录与默认输出路径（与 stage_1/2 解耦）。"""

from __future__ import annotations

from pathlib import Path

STAGE3_ROOT = Path(__file__).resolve().parents[1]
STAGE3_OUTPUTS = STAGE3_ROOT / "outputs"
STAGE3_CHECKPOINTS = STAGE3_ROOT / "checkpoints"
