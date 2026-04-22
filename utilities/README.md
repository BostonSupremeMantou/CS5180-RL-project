# utilities

This is my **shared toolbox**—everything the env, trainers, and eval scripts import so I don’t duplicate logic.

## What’s inside (high level)

- **`fish_tracking_env.py`** — Gym-style env: video frames, YOLO teacher, reward, action costs.
- **`paths.py`** — Defaults for video, weights, baseline `.npz`, output roots.
- **`env_wrappers.py`** — Stacking / ablation masks on top of the base env.
- **`replay_buffer.py`**, **`n_step.py`**, **`per_buffer.py`** — Experience replay bits.
- **`nn/`** — Q-network modules (vanilla MLP, dueling, distributional heads, GRU, etc.).
- **`torch_train.py`**, **`train_console.py`**, **`train_early_stop.py`** — Training helpers and logging.
- **`evaluate.py`**, **`plot_train_metrics.py`** — Eval + matplotlib glue used by CLIs and tests.

## How I use it

I import from `utilities.*` everywhere else in the repo. If I’m debugging, I often start here before touching `agents/`.
