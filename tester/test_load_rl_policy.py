from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
import torch

from utilities.load_rl_policy import load_rl_q_network
from utilities.nn.dueling import TrackingDQN
from utilities.nn.vanilla import VanillaDQN


def test_load_dueling_roundtrip(tmp_path: Path) -> None:
    sd = 14
    q = TrackingDQN(state_dim=sd, hidden=16)
    ck = tmp_path / "t.pt"
    torch.save(
        {
            "q_state_dict": q.state_dict(),
            "target_state_dict": q.state_dict(),
            "state_dim": sd,
            "architecture": "dueling_double",
            "hidden_dim": 16,
            "dropout": 0.0,
            "obs_stack_k": 2,
        },
        ck,
    )
    dev = torch.device("cpu")
    q2 = load_rl_q_network(ck, dev)
    x = torch.randn(1, sd)
    assert q2(x).shape == (1, 3)


def test_load_vanilla(tmp_path: Path) -> None:
    q = VanillaDQN(state_dim=10, hidden=8)
    ck = tmp_path / "v.pt"
    torch.save(
        {
            "q_state_dict": q.state_dict(),
            "state_dim": 10,
            "architecture": "vanilla",
            "hidden_dim": 8,
            "dropout": 0.0,
        },
        ck,
    )
    q2 = load_rl_q_network(ck, torch.device("cpu"))
    assert q2(torch.randn(1, 10)).shape == (1, 3)
