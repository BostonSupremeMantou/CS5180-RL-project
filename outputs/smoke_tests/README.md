# outputs/smoke_tests

I point quick sanity runs here—**tiny step counts**, maybe one RL group and a mini eval—so I know the pipeline works without burning a day of GPU.

## What’s inside

- **`RL_agents/`** — Mini `train_metrics.csv` files from smoke training.
- **`full_eval/`** — Sometimes a shrunken `eval_raw.csv` from a partial sweep.

## How I use it

When something breaks, I grep this folder first. If smoke passes but the real run fails, I know the bug is probably scale- or data-related, not import wiring.
