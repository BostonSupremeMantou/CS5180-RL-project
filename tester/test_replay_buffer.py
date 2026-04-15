from __future__ import annotations

import numpy as np

from utilities.replay_buffer import ReplayBuffer


def test_replay_push_sample_roundtrip() -> None:
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(100, state_dim=4, rng=rng)
    for i in range(10):
        o = np.array([i, i + 1, i + 2, i + 3], dtype=np.float32)
        no = o + 1.0
        buf.push(o, i % 3, float(i), no, i == 9)
    assert len(buf) == 10
    b_o, b_a, b_r, b_no, b_d = buf.sample(5)
    assert b_o.shape == (5, 4)
    assert b_a.shape == (5,)
    assert set(np.unique(b_a)).issubset({0, 1, 2})


def test_replay_overwrite_ring() -> None:
    rng = np.random.default_rng(1)
    buf = ReplayBuffer(5, state_dim=2, rng=rng)
    for i in range(12):
        buf.push(
            np.array([i, i], dtype=np.float32),
            0,
            1.0,
            np.array([i + 0.5, i], dtype=np.float32),
            False,
        )
    assert len(buf) == 5
