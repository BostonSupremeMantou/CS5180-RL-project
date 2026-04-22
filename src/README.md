# src

I treat this folder as my **CLI front door**—small `__main__` modules I can run with `python -m src.<name>`.

## What’s inside

- **`train.py`** — Kicks off RL training for a `--group` into `weights/RL_agents/...`.
- **`evaluate.py`** — Rolls out a policy (RL greedy, flow-only, etc.) and logs CSV metrics.
- **`final_eval.py`** — Batch / sweep evaluation helpers for comparing lots of runs.
- **`plot_final_eval.py`** — Reads eval CSVs and writes paper-style figures (often into `final_report/plots/`).
- **`run_full_suite.py`** — Orchestrates multiple train/eval steps (I use `--smoke` when I’m impatient).

## How I use it

From the repo root I always set `export PYTHONPATH=.` first, then e.g. `.venv/bin/python -m src.train --help` to remind myself of flags.
