"""GRU + Dueling：将 stack_k 展平观测拆成序列（§6.4.3）。"""

from __future__ import annotations

import torch
import torch.nn as nn


class GRUDuelingDQN(nn.Module):
    """
    输入 (B, state_dim)，其中 state_dim = stack_k * STATE_DIM(10)。
    视作 (B, stack_k, 10) 喂入 GRU，再 Dueling 头。
    """

    def __init__(
        self,
        state_dim: int,
        stack_k: int,
        raw_dim: int,
        hidden: int,
        gru_hidden: int,
        n_actions: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if state_dim != stack_k * raw_dim:
            raise ValueError(f"state_dim {state_dim} != stack_k*raw_dim {stack_k}*{raw_dim}")
        self.stack_k = int(stack_k)
        self.raw_dim = int(raw_dim)
        self.gru = nn.GRU(raw_dim, gru_hidden, batch_first=True)
        trunk_layers: list[nn.Module] = [
            nn.Linear(gru_hidden, hidden),
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
        b = x.shape[0]
        seq = x.view(b, self.stack_k, self.raw_dim)
        y, _ = self.gru(seq)
        h = y[:, -1, :]
        t = self.trunk(h)
        v = self.v_head(t)
        a = self.adv_head(t)
        return v + a - a.mean(dim=1, keepdim=True)
