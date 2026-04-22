# Stage 2 — 时序观测 + Pareto + 消融

在 Stage 1 的 `FishTrackingEnv` 上叠加 **观测堆叠**（默认最近 4 帧拼接 → 40 维）与可选 **状态消融**，训练脚本默认 **`--stage2-preset stage2_intense`**。

## 依赖

与 Stage 1 相同（`requirements.txt`）。**不要求** teacher：训练用 `--video-path`（默认与 Stage1 相同 fish 视频），`--teacher-npz` 可选。

## 主训练（Stage2 高强度）

```bash
python -m stage_2.training.train_stage2 --device mps \
  --metrics-csv stage_2/checkpoints/train_metrics_stage2.csv

python -m stage_2.training.train_stage2 --preset none --stage2-preset none \
  --total-steps 3000 --stack-k 4 --device mps

python -m stage_2.training.train_stage2 --stack-k 1 --ablation no_iou --stage2-preset stage2_robust
```

`--ablation no_iou` 表示去掉 **光流对齐** 那一维观测（原 10 维中第 9 个分量，0-based 索引 8）。

## 评估：纳入 Stage1 + Stage2 对比

```bash
python -m stage_2.main compare --lambda-from-ckpt --device mps --n-episodes 5
# 可选：新视频 + 对照 teacher
python -m stage_2.main compare --video-path /path/to/x.mp4 --teacher-npz /path/to/t.npz
```

写出 `stage_2/outputs/eval_all_methods.csv`（列含 **mean_consistency**、**mean_iou_teacher** 等）。画图：

```bash
python -m stage_1.tools.plot_eval_comparison \
  --csv stage_2/outputs/eval_all_methods.csv \
  --out-dir stage_2/outputs
```

## Pareto（固定 λ 多点训练 + 曲线）

```bash
python -m stage_2.tools.run_pareto --train --eval --device mps
python -m stage_2.tools.run_pareto --eval
```

输出：`pareto_frontier.csv`（**mean_consistency** vs cost）、`pareto_iou_vs_cost.png`（纵轴为 flow consistency）。

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
