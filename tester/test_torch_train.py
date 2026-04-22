from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import torch
import torch.nn as nn

from utilities.torch_train import augment_obs_batch, polyak_update


def test_augment_zero_std_noop() -> None:
    rng = np.random.default_rng(0)
    x = np.ones((4, 8), dtype=np.float32)
    y = augment_obs_batch(x, 0.0, rng)
    assert np.array_equal(x, y)


def test_augment_nonzero_changes() -> None:
    rng = np.random.default_rng(0)
    x = np.zeros((8, 3), dtype=np.float32)
    y = augment_obs_batch(x, 0.5, rng)
    assert y.shape == x.shape
    assert not np.allclose(x, y)


def test_polyak_update() -> None:
    online = nn.Linear(4, 2, bias=False)
    target = nn.Linear(4, 2, bias=False)
    target.weight.data.fill_(0.0)
    online.weight.data.fill_(1.0)
    polyak_update(target, online, tau=0.1)
    assert torch.allclose(target.weight, torch.full_like(target.weight, 0.1))
