"""Stage 5 根目录与默认输出路径。"""

from __future__ import annotations

from pathlib import Path

STAGE5_ROOT = Path(__file__).resolve().parents[1]
STAGE5_OUTPUTS = STAGE5_ROOT / "outputs"
STAGE5_CHECKPOINTS = STAGE5_ROOT / "checkpoints"
