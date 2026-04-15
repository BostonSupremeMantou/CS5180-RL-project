from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from utilities.nn.c51 import C51TrackingDQN, project_c51
from utilities.nn.dueling import TrackingDQN, select_action_epsilon_greedy, select_action_greedy
from utilities.nn.gru import GRUDuelingDQN
from utilities.nn.policy_select import select_action_epsilon_greedy_any
from utilities.nn.vanilla import VanillaDQN
from utilities.state import STATE_DIM


def test_vanilla_forward() -> None:
    q = VanillaDQN(state_dim=12, hidden=32)
    x = torch.randn(4, 12)
    y = q(x)
    assert y.shape == (4, 3)


def test_dueling_forward_and_greedy() -> None:
    q = TrackingDQN(state_dim=12, hidden=32)
    x = torch.randn(2, 12)
    assert q(x).shape == (2, 3)
    rng = np.random.default_rng(0)
    a = select_action_epsilon_greedy(q, np.zeros(12, dtype=np.float32), 0.0, 3, torch.device("cpu"), rng)
    assert a in (0, 1, 2)
    g = select_action_greedy(q, np.zeros(12, dtype=np.float32), torch.device("cpu"))
    assert g in (0, 1, 2)


def test_c51_expected_q_and_project() -> None:
    q = C51TrackingDQN(state_dim=8, hidden=32, n_atoms=11, v_min=-5.0, v_max=5.0)
    x = torch.randn(3, 8)
    eq = q.expected_q(x)
    assert eq.shape == (3, 3)
    logits = q(x)
    probs = torch.softmax(logits[torch.arange(3), :, :], dim=-1)
    na = eq.argmax(dim=1)
    next_probs = probs[torch.arange(3), na]
    r = torch.zeros(3)
    d = torch.zeros(3)
    m = project_c51(next_probs, r, d, q.support, gamma=0.99)
    assert m.shape == (3, 11)


def test_gru_requires_matching_dims() -> None:
    q = GRUDuelingDQN(
        state_dim=STATE_DIM * 3,
        stack_k=3,
        raw_dim=STATE_DIM,
        hidden=32,
        gru_hidden=32,
    )
    x = torch.randn(2, STATE_DIM * 3)
    assert q(x).shape == (2, 3)


def test_gru_bad_dims_raises() -> None:
    with pytest.raises(ValueError, match="state_dim"):
        GRUDuelingDQN(
            state_dim=99,
            stack_k=3,
            raw_dim=STATE_DIM,
            hidden=32,
            gru_hidden=32,
        )


def test_policy_select_c51_greedy() -> None:
    q = C51TrackingDQN(state_dim=6, hidden=16, n_atoms=11)
    rng = np.random.default_rng(0)
    a = select_action_epsilon_greedy_any(q, np.zeros(6, dtype=np.float32), 0.0, 3, torch.device("cpu"), rng)
    assert a in (0, 1, 2)
