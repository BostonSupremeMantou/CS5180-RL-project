# baseline

I use this folder for the **offline “teacher” track**—full detection quality without my RL policy in the loop.

## What I keep here

- **`baseline.npz`** *(when I generate it)* — Per-frame boxes from running the heavy detector on the whole clip (used as a reference trajectory in the env).
- **`always_full.py`** — A simple policy that always picks `FULL_DETECT`; handy for upper-bound evals via `src.evaluate`.

## Where the weights live

I don’t put big `.pt` files *inside* `baseline/` itself—I park YOLO / detector weights under **`weights/baseline/`** (see that README). This folder is more about **numpy traces + tiny Python**.

## Room to grow

If I ever add hand labels or YAML configs, I’ll probably add `annotations/` or `configs/` subfolders here so everything “static supervision” stays in one place.
