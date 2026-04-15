# baseline (offline full-detection track)

- `baseline.npz` — per-frame boxes from the heavy detector run (was `teacher_fish_video.npz` in old_versions).
- `always_full.py` — baseline policy that always runs `FULL_DETECT` (used by `src/evaluate.py` and `src/run_full_suite.py`).
- put YOLO / full-conv **cv weights** under `../weights/baseline/` when ready.
- annotations and training configs can live here as folders later (`annotations/`, `configs/`).
