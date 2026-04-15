from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from stage_1.env.fish_tracking_env import STATE_DIM


class VanillaDQN(nn.Module):
    """旧版 8 维 MLP，仅用于加载历史 checkpoint。"""

    def __init__(self, state_dim: int = 8, hidden: tuple[int, int] = (128, 128), n_actions: int = 3):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(state_dim, h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TrackingDQN(nn.Module):
    """
    Dueling 结构 + 训练时使用 Double DQN 选 bootstrap 动作。
    LayerNorm 减轻各维量纲差异；适合学习「何时 FULL / LIGHT / REUSE」。
    """

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
        self.hidden_dim = hidden
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


def build_q_network(
    state_dim: int,
    architecture: str,
) -> nn.Module:
    if architecture == "vanilla":
        return VanillaDQN(state_dim=state_dim)
    if architecture == "dueling_double":
        return TrackingDQN(state_dim=state_dim)
    raise ValueError(f"unknown architecture: {architecture}")


def load_q_from_checkpoint(ckpt: dict, device: torch.device) -> nn.Module:
    arch = ckpt.get("architecture", "vanilla")
    sd = int(ckpt.get("state_dim", 8))
    if arch == "dueling_double":
        hidden = int(ckpt.get("hidden_dim", 256))
        dropout = float(ckpt.get("dropout", 0.0))
        q = TrackingDQN(state_dim=sd, hidden=hidden, dropout=dropout)
        q.load_state_dict(ckpt["q_state_dict"])
    else:
        q = build_q_network(sd, arch)
        q.load_state_dict(ckpt["q_state_dict"])
    q.to(device)
    q.eval()
    return q
