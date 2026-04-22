from __future__ import annotations

import pytest

pytest.importorskip("gymnasium")

from utilities.evaluate import rollout

from tester.fixtures_dummy_env import DummyTrackingEnv


def test_rollout_keys_and_steps() -> None:
    env = DummyTrackingEnv(episode_len=4)
    stats = rollout(env, lambda obs, s: int(s % 3), seed=0)
    assert set(stats.keys()) == {
        "return",
        "mean_consistency",
        "mean_iou_ref",
        "mean_cost",
        "steps",
    }
    assert stats["steps"] == 4
    assert stats["return"] != 0.0 or stats["steps"] > 0
