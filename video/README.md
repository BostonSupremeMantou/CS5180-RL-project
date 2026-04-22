# video

This is where I keep **fish footage** (and a bundled CV subfolder) so `FishTrackingEnv` can load frames without me hard-coding absolute paths everywhere.

## Default clip

I symlink **`fish_video.mp4`** at this level to `cv_fish/videos/fish_video.mp4` so both old and new code paths stay happy. The directory itself is wired through `utilities/paths.py` as `VIDEO_DIR`.

## `cv_fish/` — what that bundle is

I copied this subtree from an older experiment repo. Rough map:

| Path | What I use it for |
|------|---------------------|
| `cv_fish/videos/` | Source `fish_video.mp4` |
| `cv_fish/models/` | Base + fine-tuned YOLO weights (`yolo11n.pt`, `fish_detection_yolo11n.pt`) |
| `cv_fish/runs/fish_yolo11n/` | Training artifacts—curves, `weights/best.pt`, validation batches |
| `cv_fish/scripts/` | One-off train / annotate entrypoints |
| `cv_fish/tools/` | Dataset prep, downscale, frame extract, simple annotator |

## Tip

If I want the exact same checkpoint the CV bundle used, I can point `weights/baseline/best.pt` at `cv_fish/runs/fish_yolo11n/weights/best.pt`.
