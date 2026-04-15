# tester

Run from **repository root**:

```bash
source .venv/bin/activate   # optional
pip install -r requirements.txt
pytest tester
```

- **Fast tests**: buffers, n-step bridge, PER SumTree, geometry, ablation masks, dummy env + wrappers, `rollout`, torch helpers, NN forwards, registry, baseline policy, CLI `--help`, plot CSV → PNG (needs `matplotlib`).
- **Optional integration**: `test_fish_env_optional.py` runs only if `utilities.paths.default_video_path()` and `default_yolo_weights()` exist.

Verbose: `pytest tester -v`  
One file: `pytest tester/test_replay_buffer.py -v`
