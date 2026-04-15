"""C51 distributional head (Dueling + 51 atoms)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class C51TrackingDQN(nn.Module):
    """Dueling C51: logits shaped (batch, n_actions, n_atoms)."""

    def __init__(
        self,
        state_dim: int,
        hidden: int,
        n_actions: int = 3,
        n_atoms: int = 51,
        v_min: float = -30.0,
        v_max: float = 30.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.n_atoms = int(n_atoms)
        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))
        self.v_min = float(v_min)
        self.v_max = float(v_max)
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
        self.v_head = nn.Linear(hidden, n_atoms)
        self.adv_head = nn.Linear(hidden, n_actions * n_atoms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        v = self.v_head(h).view(-1, 1, self.n_atoms)
        a = self.adv_head(h).view(-1, self.n_actions, self.n_atoms)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q

    def probs(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=-1)

    def expected_q(self, x: torch.Tensor) -> torch.Tensor:
        p = self.probs(x)
        z = self.support.view(1, 1, -1)
        return (p * z).sum(dim=-1)


def project_c51(
    next_probs: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    support: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Project the bootstrap distribution onto the fixed atom support (Bellemare et al.)."""
    vmin = support[0]
    vmax = support[-1]
    n_atoms = int(support.numel())
    dz = (vmax - vmin) / max(n_atoms - 1, 1)
    bsz = rewards.shape[0]
    device = rewards.device
    m = torch.zeros(bsz, n_atoms, device=device, dtype=next_probs.dtype)
    mask = 1.0 - dones
    for j in range(n_atoms):
        tzj = rewards + mask * float(gamma) * support[j]
        tzj = torch.clamp(tzj, vmin, vmax)
        b = (tzj - vmin) / dz
        lo = b.floor().long().clamp(0, n_atoms - 1)
        up = torch.clamp(lo + 1, max=n_atoms - 1)
        one = up <= lo
        denom = (up.float() - lo.float()).clamp(min=1e-6)
        w_lo = torch.where(one, torch.ones_like(b), (up.float() - b) / denom)
        w_up = torch.where(one, torch.zeros_like(b), (b - lo.float()) / denom)
        pj = next_probs[:, j]
        m.scatter_add_(1, lo.unsqueeze(1), (pj * w_lo).unsqueeze(1))
        m.scatter_add_(1, up.unsqueeze(1), (pj * w_up).unsqueeze(1))
    m = m / (m.sum(dim=1, keepdim=True) + 1e-8)
    return m
