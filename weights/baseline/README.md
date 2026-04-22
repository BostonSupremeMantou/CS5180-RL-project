# weights/baseline

I put **detector / teacher checkpoints** here—the YOLO (or similar) weights my env uses for full-quality boxes.

## What I expect

- Something like **`best.pt`** or **`last.pt`** after I train or download a model.
- I symlink or copy from `video/cv_fish/runs/.../weights/` when I’m lazy and that run already exists.

## Not to be confused with

Repo-root **`baseline/`** holds `baseline.npz` and tiny Python policies. **This** folder is only the **torch weights**.
