from __future__ import annotations

from pathlib import Path

import pytest

from utilities.train_console import print_episode_done, print_run_header, print_training_done


def test_print_run_header_no_crash(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    save = tmp_path / "x.pt"
    met = tmp_path / "m.csv"
    print_run_header(
        "test_group",
        total_steps=1000,
        learning_starts=100,
        state_dim=40,
        stack_k=4,
        ablation="none",
        save_path=save,
        metrics_csv=met,
        extra_lines="  foo=1",
    )
    out = capsys.readouterr().out
    assert "test_group" in out
    assert "1000" in out


def test_print_episode_done_no_crash(capsys: pytest.CaptureFixture[str]) -> None:
    print_episode_done(
        "g2",
        step=500,
        total_steps=1000,
        episodes_done=3,
        ep_return=12.3,
        ep_len=50,
        buf_size=400,
        epsilon=0.2,
        lambda_cost=0.4,
        mean_consistency=0.88,
    )
    assert "ep#3" in capsys.readouterr().out


def test_print_training_done_no_crash(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = tmp_path / "a.pt"
    p.write_text("x")
    print_training_done("g1", save_path=p, final_lambda=0.35)
    assert "finished" in capsys.readouterr().out
