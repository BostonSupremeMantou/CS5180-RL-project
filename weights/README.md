# weights

I use this tree as the **canonical place for binaries**—YOLO checkpoints, saved Q-networks, anything too fat for git.

## Layout (mirrors how I think about policies)

- **`baseline/`** — Teacher / full-detector weights (e.g. `best.pt`). There’s a tiny README inside too.
- **`non_learning_agents/`** — Reserved if I ever ship hand-tuned artifacts here (usually empty).
- **`RL_agents/<group>/`** — After training I expect something like `last.pt` per group. See `RL_agents/README.md` there.

## Note

Most `*.pt` / `*.onnx` files are **gitignored** on purpose—I don’t commit them, I just drop them locally or on the machine where I train.
