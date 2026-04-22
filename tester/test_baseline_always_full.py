from __future__ import annotations

import numpy as np

from baseline.always_full import always_full_policy
from utilities.actions import FULL_DETECT


def test_always_full_action() -> None:
    assert always_full_policy(np.zeros(10), 0) == FULL_DETECT
