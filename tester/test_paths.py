from __future__ import annotations

from pathlib import Path

import pytest

from utilities import paths


def test_project_root_exists() -> None:
    assert paths.PROJECT_ROOT.is_dir()
    assert (paths.PROJECT_ROOT / "utilities").is_dir()


def test_default_video_path_type() -> None:
    p = paths.default_video_path()
    assert isinstance(p, Path)


def test_default_yolo_weights_type() -> None:
    p = paths.default_yolo_weights()
    assert isinstance(p, Path)


def test_weights_outputs_for_layout() -> None:
    w = paths.weights_for("RL_agents", "g2_double_dueling", "last.pt")
    assert w.name == "last.pt"
    assert "RL_agents" in str(w)
    o = paths.outputs_for("RL_agents", "g2_double_dueling", "train_metrics.csv")
    assert o.name == "train_metrics.csv"


def test_smoke_tests_dir_under_outputs() -> None:
    assert paths.SMOKE_TESTS_DIR.name == "smoke_tests"
    assert paths.SMOKE_TESTS_DIR.parent == paths.OUTPUTS_DIR


def test_baseline_npz_constant() -> None:
    assert paths.BASELINE_NPZ.suffix == ".npz"
