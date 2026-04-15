"""Stage 4 根目录与默认输出路径。"""

from __future__ import annotations

from pathlib import Path

STAGE4_ROOT = Path(__file__).resolve().parents[1]
STAGE4_OUTPUTS = STAGE4_ROOT / "outputs"
STAGE4_CHECKPOINTS = STAGE4_ROOT / "checkpoints"
