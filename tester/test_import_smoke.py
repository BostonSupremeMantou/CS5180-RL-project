from __future__ import annotations

import pytest


def test_import_plot_train_metrics() -> None:
    pytest.importorskip("matplotlib")
    import utilities.plot_train_metrics  # noqa: F401


def test_import_detector_if_torch_available() -> None:
    pytest.importorskip("torch")
    import utilities.detector  # noqa: F401


def test_import_fish_env_if_cv2() -> None:
    pytest.importorskip("cv2")
    pytest.importorskip("gymnasium")
    import utilities.fish_tracking_env  # noqa: F401


def test_import_train_entrypoints_if_torch() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")
    from agents.RL_agents.g1_vanilla import train as _t1  # noqa: F401
    from agents.RL_agents.g2_double_dueling import train as _t2  # noqa: F401
