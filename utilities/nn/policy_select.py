"""Epsilon / greedy helpers that work for scalar Q and for C51 (via expected_q)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def select_action_greedy_any(q_net: nn.Module, obs: np.ndarray, device: torch.device) -> int:
    with torch.no_grad():
        t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(device)
        if hasattr(q_net, "expected_q"):
            return int(q_net.expected_q(t).argmax(dim=1).item())
        return int(q_net(t).argmax(dim=1).item())


def select_action_epsilon_greedy_any(
    q_net: nn.Module,
    obs: np.ndarray,
    epsilon: float,
    n_actions: int,
    device: torch.device,
    rng: np.random.Generator,
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, n_actions))
    return select_action_greedy_any(q_net, obs, device)
