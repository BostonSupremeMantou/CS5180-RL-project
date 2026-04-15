# video

Default clip for the RL env: **`fish_video.mp4`** (symlink → `cv_fish/videos/fish_video.mp4`).

Configured in `utilities/paths.py` as `VIDEO_DIR`.

## `cv_fish/` (baseline CV bundle)

Copied from `old_versions/old/fish/` — YOLO11n fish training and helpers:

| path | purpose |
|------|---------|
| `cv_fish/videos/` | source `fish_video.mp4` |
| `cv_fish/models/` | base / fine-tuned weights (`yolo11n.pt`, `fish_detection_yolo11n.pt`) |
| `cv_fish/runs/fish_yolo11n/` | training curves, `weights/best.pt`, `last.pt`, val batches |
| `cv_fish/scripts/` | train / annotate entry scripts |
| `cv_fish/tools/` | dataset prep, downscale, frame extract, simple annotator |

You can point `weights/baseline/best.pt` at `cv_fish/runs/fish_yolo11n/weights/best.pt` if you want the same checkpoint as this bundle.
