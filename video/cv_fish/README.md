# cv_fish — baseline fish detection (from `old_versions/old/fish`)

This tree was copied here so all video + CV assets live under `video/`.

- **Train:** `scripts/train_yolo11n_fish.py`
- **Annotate video:** `scripts/annotate_video_yolo.py`
- **Dataset / misc tools:** `tools/` (`prepare_yolo11_fish_dataset.py`, `downscale_*`, `extract_random_frames.py`, `simple_annotator.py`)
- **Artifacts:** `runs/fish_yolo11n/` (Ultralytics-style logs and `weights/best.pt`)

The repo default env video is the sibling symlink `../fish_video.mp4` → `videos/fish_video.mp4`.
