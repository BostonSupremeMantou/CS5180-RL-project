from __future__ import annotations

import csv
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from utilities.plot_train_metrics import load_metrics_csv, plot_train_metrics_csv


def _write_sample_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "step": 100,
            "epsilon": 0.5,
            "lambda_cost": 0.35,
            "lr": 0.001,
            "buffer_size": 400,
            "loss_mean": 0.1,
            "td_abs_mean": 0.2,
            "grad_norm_mean": 0.3,
            "ep_ret_ma": 1.0,
            "ep_consistency_ma": 0.8,
            "ep_teacher_iou_ma": 0.75,
            "ep_comp_ma": 2.0,
            "ep_full_frac_ma": 0.1,
            "episodes_done": 1,
        },
        {
            "step": 200,
            "epsilon": 0.4,
            "lambda_cost": 0.36,
            "lr": 0.0009,
            "buffer_size": 800,
            "loss_mean": 0.08,
            "td_abs_mean": 0.15,
            "grad_norm_mean": 0.25,
            "ep_ret_ma": 1.2,
            "ep_consistency_ma": 0.82,
            "ep_teacher_iou_ma": 0.76,
            "ep_comp_ma": 2.1,
            "ep_full_frac_ma": 0.12,
            "episodes_done": 2,
        },
    ]
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: str(v) for k, v in r.items()})


def test_load_metrics_csv(tmp_path: Path) -> None:
    p = tmp_path / "m.csv"
    _write_sample_csv(p)
    d = load_metrics_csv(p)
    assert d["step"] == [100.0, 200.0]
    assert len(d["loss_mean"]) == 2


def test_plot_train_metrics_writes_pngs(tmp_path: Path) -> None:
    csv_p = tmp_path / "train_metrics.csv"
    out_d = tmp_path / "plots"
    _write_sample_csv(csv_p)
    written = plot_train_metrics_csv(csv_p, out_d)
    assert len(written) >= 3
    assert all(p.suffix == ".png" for p in written)
