from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def test_train_cli_help() -> None:
    r = subprocess.run(
        [sys.executable, str(_ROOT / "src" / "train.py"), "--help"],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0, r.stderr
    assert len(r.stdout) > 50


def test_evaluate_cli_help() -> None:
    pytest.importorskip("gymnasium")
    r = subprocess.run(
        [sys.executable, str(_ROOT / "src" / "evaluate.py"), "--help"],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0, r.stderr
    assert len(r.stdout) > 50
