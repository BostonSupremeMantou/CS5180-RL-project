# tester

I keep my **pytest suite** here—fast unit tests plus a couple of optional integration checks.

## How I run it

From the **repo root**:

```bash
source .venv/bin/activate   # optional but I usually do it
pip install -r requirements.txt
pytest tester
```

## What I’m actually testing

- **Fast stuff**: replay buffer, n-step bridge, PER SumTree, geometry helpers, ablation masks, dummy env + wrappers, rollout code, torch helpers, NN forward passes, RL registry, baseline policy behavior, CLI `--help`, and “CSV → PNG” plotting (needs `matplotlib`).
- **Optional integration**: `test_fish_env_optional.py` only runs if `utilities.paths` can find a real video + YOLO weights on disk—I skip it on CI sometimes.

## Handy variants

```bash
pytest tester -v
pytest tester/test_replay_buffer.py -v
```
