from __future__ import annotations

import numpy as np

from utilities.n_step import NStepReplayBridge
from utilities.replay_buffer import ReplayBuffer


def test_n_step_one_equals_direct_push() -> None:
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(50, state_dim=2, rng=rng)
    bridge = NStepReplayBridge(1, 0.99, buf)
    o0 = np.array([1.0, 0.0], dtype=np.float32)
    o1 = np.array([2.0, 0.0], dtype=np.float32)
    bridge.add(o0, 1, 0.5, o1, False)
    assert len(buf) == 1
    _bo, ba, br, _bno, _bd = buf.sample(1)
    assert int(ba[0]) == 1
    assert abs(float(br[0]) - 0.5) < 1e-5


def test_n_step_two_done_discount() -> None:
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(100, state_dim=2, rng=rng)
    gamma = 0.9
    bridge = NStepReplayBridge(2, gamma, buf)
    o0 = np.zeros(2, dtype=np.float32)
    o1 = np.ones(2, dtype=np.float32)
    o2 = np.ones(2, dtype=np.float32) * 2.0
    bridge.add(o0, 0, 1.0, o1, False)
    bridge.add(o1, 1, 2.0, o2, True)
    # Terminal flush can emit a second partial n-step tuple.
    assert len(buf) >= 1
    expected_g = 1.0 + gamma * 2.0
    found = False
    for _ in range(80):
        _bo, ba, br, _bno, bd = buf.sample(min(8, len(buf)))
        for i in range(len(br)):
            if abs(float(br[i]) - expected_g) < 1e-3 and int(ba[i]) == 0:
                found = True
                assert float(bd[i]) > 0.5
                break
        if found:
            break
    assert found, "expected 2-step discounted return in buffer"


def test_n_step_flush_terminal() -> None:
    rng = np.random.default_rng(42)
    buf = ReplayBuffer(100, state_dim=2, rng=rng)
    bridge = NStepReplayBridge(3, 0.99, buf)
    for t in range(2):
        o = np.array([float(t), 0.0], dtype=np.float32)
        bridge.add(o, 0, 1.0, o + 1.0, False)
    bridge.flush_terminal()
    assert len(buf) >= 1
