from __future__ import annotations

from utilities.train_early_stop import RefIoUEarlyStopTracker


def test_tracker_never_fires_when_disabled() -> None:
    tr = RefIoUEarlyStopTracker(None, ma_window=3, min_steps=0)
    assert not tr.on_episode_end(100, 0.99)


def test_tracker_fires_when_ma_above_threshold() -> None:
    tr = RefIoUEarlyStopTracker(0.5, ma_window=3, min_steps=10)
    assert not tr.on_episode_end(5, 0.4)
    assert not tr.on_episode_end(8, 0.4)
    assert not tr.on_episode_end(11, 0.4)
    assert tr.on_episode_end(12, 0.7)


def test_tracker_skips_nan_episodes() -> None:
    tr = RefIoUEarlyStopTracker(0.66, ma_window=2, min_steps=0)
    assert not tr.on_episode_end(1, float("nan"))
    assert not tr.on_episode_end(2, 0.6)
    assert not tr.on_episode_end(3, float("nan"))
    assert not tr.on_episode_end(4, 0.6)
    assert not tr.on_episode_end(5, 0.7)
    assert tr.on_episode_end(6, 0.7)


def test_tracker_respects_min_steps() -> None:
    tr = RefIoUEarlyStopTracker(0.0, ma_window=2, min_steps=100)
    assert not tr.on_episode_end(50, 1.0)
    assert not tr.on_episode_end(50, 1.0)
    assert tr.on_episode_end(101, 1.0)
