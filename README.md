# RL fish tracking (reorg)

simple layout:

- `baseline/` — offline full-detection track (`baseline.npz`) + future cv extras
- `video/` — fish videos
- `weights/` — `baseline/`, `non_learning_agents/`, `RL_agents/<group>/`
- `outputs/` — same split + `final_results/` for big tables
- `utilities/` — shared helpers (paths, actions, eval, …)
- `agents/` — plug-in policies (`no_learning_agents/`, `RL_agents/<group>/`)
- `src/` — thin runners (`train`, `evaluate`, `final_eval`)
- `old_versions/` — archive only (new code does not import it)

run from repo root:

```bash
export PYTHONPATH=.
.venv/bin/python -m src.train --group g2_double_dueling --total-steps 50000 --device cpu
.venv/bin/python -m src.evaluate --policy flow_only --episodes 2 --device cpu
.venv/bin/python -m src.evaluate --policy rl_greedy --ckpt weights/RL_agents/g2_double_dueling/last.pt --stack-k 4 --device cpu
```

defaults: video/weights fall back to `old_versions/stage_1/data/...` until you put files under `video/` and `weights/baseline/`. baseline boxes: `baseline/baseline.npz`.

**wired today:** `g2_double_dueling` (dueling + double + uniform replay + stack). other groups still stubs.
