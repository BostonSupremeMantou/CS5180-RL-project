from __future__ import annotations

import torch
import torch.nn as nn

from utilities.state import STATE_DIM


class VanillaDQN(nn.Module):
    """Plain MLP Q(s,·) logits (matches current stacked state size, not legacy 8-D ckpts)."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        hidden: int = 256,
        n_actions: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        layers: list[nn.Module] = [
            nn.Linear(self.state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
        ]
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend(
            [
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(inplace=True),
            ]
        )
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.trunk(x))
