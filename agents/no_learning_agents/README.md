# no_learning_agents

I put **non-RL policies** here—simple Python callables the env can run without a neural net.

## What’s inside

- **`flow_only.py`** — Cheap optical-flow style behavior (no full YOLO every frame).
- **`periodic.py`** — Runs full detection on a fixed schedule (e.g. every *k* frames).
- **`__init__.py`** — Package marker; the real wiring lives in `src/evaluate.py` / policy registry elsewhere.

## How I use it

I pick these with `python -m src.evaluate --policy flow_only` (or `periodic`, etc.). They’re great sanity checks before I trust an RL checkpoint.
