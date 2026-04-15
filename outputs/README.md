# outputs

same shape as `weights/`:

- `baseline/` — reference / upper bound (e.g. `always_full` eval CSVs)
- `non_learning_agents/` — other rule policies (`flow_only`, `periodic`, …) csv / logs
- `RL_agents/<group>/` — train + eval per RL group
- `smoke_tests/` — short sanity runs only (e.g. `run_full_suite --smoke` defaults here)
- `final_results/` — one place for tables / plots that compare everyone
