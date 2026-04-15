from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from utilities.state import STATE_DIM


class TrackingDQN(nn.Module):
    def __init__(
        self,
        state_dim: int = STATE_DIM,
        hidden: int = 256,
        n_actions: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        trunk_layers: list[nn.Module] = [
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
        ]
        if dropout and dropout > 0:
            trunk_layers.append(nn.Dropout(dropout))
        trunk_layers.extend(
            [
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(inplace=True),
            ]
        )
        if dropout and dropout > 0:
            trunk_layers.append(nn.Dropout(dropout))
        self.trunk = nn.Sequential(*trunk_layers)
        self.v_head = nn.Linear(hidden, 1)
        self.adv_head = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        v = self.v_head(h)
        a = self.adv_head(h)
        return v + a - a.mean(dim=1, keepdim=True)


def select_action_greedy(q_net: nn.Module, obs: np.ndarray, device: torch.device) -> int:
    with torch.no_grad():
        t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        return int(q_net(t).argmax(dim=1).item())


def select_action_epsilon_greedy(
    q_net: nn.Module,
    obs: np.ndarray,
    epsilon: float,
    n_actions: int,
    device: torch.device,
    rng: np.random.Generator,
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, n_actions))
    return select_action_greedy(q_net, obs, device)
