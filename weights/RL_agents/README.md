# weights/RL_agents

After I train a group (`g2_double_dueling`, `g3_nstep`, …), I save **PyTorch checkpoints** under the matching subfolder.

## What I expect per group

- Usually **`last.pt`** — contains `q_state_dict`, maybe `target_state_dict`, metadata like `state_dim`, `group`, `obs_stack_k`, etc.
- The training code in `agents/RL_agents/<group>/train.py` decides the exact dict keys.

## How I evaluate

```bash
python -m src.evaluate --policy rl_greedy \
  --ckpt weights/RL_agents/g2_double_dueling/last.pt \
  --stack-k 4
```

## Note

These files are **big** and normally **gitignored**—you’ll see `.gitkeep` placeholders in a fresh checkout until I actually train.
