# video/cv_fish

I copied this whole subtree out of **`old_versions/old/fish`** so I could keep **YOLO training junk** next to the actual clip I run in RL.

## What I actually touch

- **Train:** `scripts/train_yolo11n_fish.py`
- **Annotate a clip:** `scripts/annotate_video_yolo.py`
- **Dataset / one-off helpers:** `tools/` — stuff like `prepare_yolo11_fish_dataset.py`, downscale utilities, frame extraction, `simple_annotator.py`

## Where the juicy artifacts live

`runs/fish_yolo11n/` has the Ultralytics-style logs plus **`weights/best.pt`** (that’s the checkpoint I usually symlink into `weights/baseline/`).

## Video path reminder

The default env clip is **`../fish_video.mp4`**, which points at `videos/fish_video.mp4` inside this bundle.
