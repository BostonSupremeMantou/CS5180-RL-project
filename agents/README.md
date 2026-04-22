# agents

I keep all the **policies** here—the stuff that actually decides what the tracker does each step.

## What’s inside

- **`no_learning_agents/`** — Hand-written baselines (e.g. always flow, periodic full-detect). No training loop; just rules.
- **`RL_agents/`** — One subpackage per algorithm group (`g1_vanilla`, `g2_double_dueling`, …). Each group has a `train.py`, optional `checkpoint.py`, and a tiny `spec.py` with `GROUP_ID` so `src.train` can find me.

## How I use it

`src.train` / `src.evaluate` import these modules by group id. If I add a new RL variant, I usually copy an existing `g*` folder, wire it in `registry.py`, and point `weights/RL_agents/<group>/` at the saved checkpoints.
