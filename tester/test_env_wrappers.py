from __future__ import annotations

import pytest

pytest.importorskip("gymnasium")

from utilities.env_wrappers import StackedObsWrapper, build_wrapped_env, stacked_state_dim
from utilities.state import STATE_DIM

from tester.fixtures_dummy_env import DummyTrackingEnv


def test_stacked_state_dim() -> None:
    assert stacked_state_dim(1) == STATE_DIM
    assert stacked_state_dim(4) == STATE_DIM * 4


def test_stacked_obs_shape() -> None:
    base = DummyTrackingEnv(episode_len=10)
    w = StackedObsWrapper(base, stack_k=4)
    obs, _ = w.reset(seed=0)
    assert obs.shape == (STATE_DIM * 4,)


def test_build_wrapped_stack4() -> None:
    base = DummyTrackingEnv()
    env = build_wrapped_env(base, stack_k=4, ablation="none")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (STATE_DIM * 4,)
    obs2, r, term, trunc, info = env.step(0)
    assert obs2.shape == (STATE_DIM * 4,)
    assert "consistency_iou" in info


def test_stack_k_lt_1_raises() -> None:
    base = DummyTrackingEnv()
    with pytest.raises(ValueError, match="stack_k"):
        StackedObsWrapper(base, stack_k=0)
