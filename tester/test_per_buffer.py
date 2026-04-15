from __future__ import annotations

import numpy as np

from utilities.per_buffer import PrioritizedReplayBuffer, SumTree


def test_sum_tree_update_total() -> None:
    st = SumTree(8)
    st.update(0, 1.0)
    st.update(1, 2.0)
    assert abs(st.total() - 3.0) < 1e-9


def test_per_push_sample_update() -> None:
    rng = np.random.default_rng(0)
    buf = PrioritizedReplayBuffer(32, state_dim=3, rng=rng, alpha=0.6)
    for i in range(20):
        o = np.array([i, i, i], dtype=np.float32)
        buf.push(o, i % 2, float(i), o + 0.1, i == 19)
    assert len(buf) == 20
    b_o, b_a, b_r, b_no, b_d, idxs, iw = buf.sample(8, beta=0.5)
    assert b_o.shape[0] == 8
    assert iw.shape[0] == 8
    td = np.abs(np.random.randn(8).astype(np.float32))
    buf.update_priorities(idxs, td)
    assert len(buf) == 20
