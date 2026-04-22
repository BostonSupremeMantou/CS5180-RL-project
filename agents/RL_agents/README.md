# RL_agents

This is where I stash **per-group training code** for my DQN-style fish-tracking agents.

## What’s inside

- **`registry.py`** — Maps group strings like `g2_double_dueling` to import paths and training entrypoints.
- **`__init__.py`** — Re-exports `RL_GROUP_IDS` and `get_rl_spec()` for convenience.
- **`g*/`** folders — Each group (`g1_vanilla`, `g2_double_dueling`, `g3_nstep`, …) typically has:
  - `train.py` — Main training loop for that variant.
  - `spec.py` — `GROUP_ID` + a short status line.
  - `checkpoint.py` *(sometimes)* — Loader glue if the default path needs extras.

## How I use it

I train with `python -m src.train --group g2_double_dueling` (for example). Checkpoints land under `weights/RL_agents/<group>/` by convention—not inside this folder.
