# Stage 2 — 时序观测 + Pareto + 消融

在 Stage 1 的 `FishTrackingEnv` 上叠加 **观测堆叠**（默认最近 4 帧拼接 → 40 维）与可选 **状态消融**，训练脚本默认 **`--stage2-preset stage2_intense`**（高强度：更长步数、更大 replay、512 hidden、软 target 等）。

## 依赖

与 Stage 1 相同（`requirements.txt`），并需已完成：

```bash
python -m stage_1.main preprocess --device mps
```

## 主训练（Stage2 高强度）

```bash
# 默认：stack-k=4, ablation=none, stage2_intense, 保存到 stage_2/checkpoints/dqn_stage2.pt
python -m stage_2.training.train_stage2 --device mps \
  --metrics-csv stage_2/checkpoints/train_metrics_stage2.csv

# 轻量试跑
python -m stage_2.training.train_stage2 --preset none --stage2-preset none \
  --total-steps 3000 --stack-k 4 --device mps

# 仅做消融（无堆叠）
python -m stage_2.training.train_stage2 --stack-k 1 --ablation no_iou --stage2-preset stage2_robust
```

常用参数见 `--help`（与 Stage 1 训练参数兼容，并多 `--stack-k`、`--ablation`、`--stage2-preset`）。

## 评估：纳入 Stage1 + Stage2 对比

```bash
python -m stage_2.main compare --lambda-from-ckpt --device mps --n-episodes 5
```

写出 `stage_2/outputs/eval_all_methods.csv`。画图：

```bash
python -m stage_1.tools.plot_eval_comparison \
  --csv stage_2/outputs/eval_all_methods.csv \
  --out-dir stage_2/outputs
```

## Pareto（固定 λ 多点训练 + IoU–cost 图）

```bash
# 训练多个 checkpoint + 评估 + CSV/PNG（每个点默认 14k 步，可调 --steps）
python -m stage_2.tools.run_pareto --train --eval --device mps

# 仅评估已有 pareto_lam_*.pt
python -m stage_2.tools.run_pareto --eval
```

输出：`stage_2/outputs/pareto_frontier.csv`、`pareto_iou_vs_cost.png`。

## 消融名称

`none`, `no_frame_diff`, `no_velocity`, `no_iou`, `no_ssf`, `no_conf`, `bbox_only`, `no_motion_context`（定义见 `stage_2/env/ablation_masks.py`）。

## 目录

| 路径 | 说明 |
|------|------|
| `env/` | `AblationWrapper`, `StackedObsWrapper`, `build_stage2_env` |
| `training/train_stage2.py` | 训练入口 |
| `evaluation/compare_stage2.py` | 多方法对比 |
| `tools/run_pareto.py` | Pareto 扫描 |
| `checkpoints/` | `dqn_stage2.pt`、`pareto/` |
