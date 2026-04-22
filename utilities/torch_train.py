"""tiny torch helpers shared by RL trainers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def augment_obs_batch(obs: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    if std <= 0:
        return obs
    return (obs + rng.normal(0.0, std, size=obs.shape)).astype(np.float32)


def polyak_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), online.parameters(), strict=True):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)
