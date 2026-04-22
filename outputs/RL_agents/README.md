# outputs/RL_agents

Under here I keep **per-group training artifacts**—each subdirectory matches a `g*` group id (`g2_double_dueling`, etc.).

## Typical layout inside a group folder

- **`train_metrics.csv`** — Step-level logs (loss, epsilon, moving-average episode stats).
- **`plots/`** — Auto-generated PNGs from `utilities/plot_train_metrics.py` or similar.

## Heads-up

I don’t commit huge CSVs from every machine—this folder is usually **local-only** or gitignored. The structure matters more than the exact files you’ll see in a fresh clone.
