"""load offline full-detection box track (.npz)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_baseline_npz(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {
        "boxes": z["boxes"],
        "valid": z["valid"].astype(bool),
        "confs": z["confs"],
        "width": int(z["width"]),
        "height": int(z["height"]),
        "n_frames": int(z["n_frames"]),
    }
