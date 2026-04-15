"""Build a Q-network from a saved checkpoint (shared by all RL groups)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from utilities.state import STATE_DIM


def load_rl_q_network(ckpt_path: Path | str, device: torch.device) -> nn.Module:
    ckpt = torch.load(Path(ckpt_path), map_location=device, weights_only=False)
    arch = str(ckpt.get("architecture", "dueling_double"))
    sd = int(ckpt["state_dim"])
    hidden = int(ckpt.get("hidden_dim", 256))
    dropout = float(ckpt.get("dropout", 0.0))

    q: nn.Module
    if arch == "vanilla":
        from utilities.nn.vanilla import VanillaDQN

        q = VanillaDQN(state_dim=sd, hidden=hidden, dropout=dropout)
    elif arch in ("c51", "s5_c51"):
        from utilities.nn.c51 import C51TrackingDQN

        q = C51TrackingDQN(state_dim=sd, hidden=hidden, dropout=dropout)
    elif arch in ("gru", "s5_gru"):
        from utilities.nn.gru import GRUDuelingDQN

        stack_k = int(ckpt.get("obs_stack_k", 4))
        gh = int(ckpt.get("gru_hidden_dim", hidden))
        q = GRUDuelingDQN(
            state_dim=sd,
            stack_k=stack_k,
            raw_dim=STATE_DIM,
            hidden=hidden,
            gru_hidden=gh,
            dropout=dropout,
        )
    else:
        from utilities.nn.dueling import TrackingDQN

        q = TrackingDQN(state_dim=sd, hidden=hidden, dropout=dropout)

    q.load_state_dict(ckpt["q_state_dict"])
    q.to(device)
    q.eval()
    return q
