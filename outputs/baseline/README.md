# outputs/baseline

I route **baseline-policy experiment logs** into this subfolder so they don’t get mixed with RL runs.

## What shows up

- CSV exports from `src.evaluate` when I’m comparing against `always_full` or similar.
- Occasionally a stray plot if I pointed a plotting script here.

## Note

This is **not** the same as repo-root `baseline/` (that’s code + `baseline.npz`). This folder is purely **generated outputs**.
