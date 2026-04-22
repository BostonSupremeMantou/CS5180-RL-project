from __future__ import annotations

import numpy as np
import pytest

from utilities.ablation_masks import get_ablation_mask
from utilities.state import STATE_DIM


def test_ablation_none_is_ones() -> None:
    m = get_ablation_mask("none")
    assert m.shape == (STATE_DIM,)
    assert np.allclose(m, 1.0)


@pytest.mark.parametrize(
    "name",
    ["no_frame_diff", "no_velocity", "no_iou", "no_ssf", "no_conf", "bbox_only", "no_motion_context"],
)
def test_ablation_masks_no_crash(name: str) -> None:
    m = get_ablation_mask(name)
    assert m.shape == (STATE_DIM,)
    assert float(m.sum()) < STATE_DIM + 0.1


def test_ablation_unknown() -> None:
    with pytest.raises(ValueError, match="unknown ablation"):
        get_ablation_mask("not_a_real_mask")
