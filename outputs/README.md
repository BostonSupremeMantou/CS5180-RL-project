# outputs

This is my **junk drawer for runs**—metrics CSVs, logs, training plots—anything I regenerate often and don’t want mixed into git-tracked “final” results.

## How I mentally partition it

I mirror the same rough shape as **`weights/`**:

- **`baseline/`** — CSVs / logs from always-full or teacher-style evals.
- **`non_learning_agents/`** — Outputs for `flow_only`, `periodic`, and friends.
- **`RL_agents/<group>/`** — Per-group `train_metrics.csv`, `plots/`, etc. after I train.
- **`smoke_tests/`** — Tiny runs (`run_full_suite --smoke`) so I don’t pollute the real folders.
- **`final_results/`** — Sometimes I stage cross-policy tables here before I copy the polished version to top-level `final_results/`.

## Heads-up

Most of this is **gitignored or disposable** on my machine—treat it as scratch space, not the canonical paper numbers (those I promote to `final_results/`).
