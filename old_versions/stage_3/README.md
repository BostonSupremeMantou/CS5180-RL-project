# Stage 3 — §6.1 实验编排（stack_k / 消融 / 网格）

在 **不改动 Stage1/2 核心算法** 的前提下，用子进程批量调用 `stage_2.training.train_stage2`，系统化跑 **old_research_targets.md 第 6.1 节** 中的扫描实验，并汇总指标。

## 目录

| 路径 | 说明 |
|------|------|
| `main.py` | CLI 入口（转发到 `tools/sweep_experiments.py`） |
| `tools/sweep_experiments.py` | `sweep-stack-k` / `sweep-ablation` / `sweep-grid` / `collect` |
| `checkpoints/` | `dqn_stage3_sk{k}_abl{name}.pt` |
| `outputs/` | `train_metrics_*.csv`、`sweep_manifest.csv`、`sweep_summary.csv` |

## 依赖

与 Stage 1/2 相同；在仓库根目录执行。

## 标准实验（§6.1 主表，结果写入 `stage_3/outputs/`）

与 Stage1 v2 对齐：**`lambda_min=0.25`**、**`target_mean_cost=0.45`**；**`stack_k ∈ {1,2,4,8}`**，**`ablation=none`**。

```bash
# 正式长度（每组 stage2_robust，耗时长）
python -m stage_3.main standard --device mps

# 快速复现管线（每组 ~8k 步；仍写出完整 CSV）
python -m stage_3.main standard --quick --device mps

# 仅评估已有 ckpt、跳过训练
python -m stage_3.main standard --skip-train --skip-collect --device mps
```

**输出文件（均在 `stage_3/outputs/`）**

| 文件 | 含义 |
|------|------|
| `sweep_train_summary.csv` | 各组 `train_metrics_sk*_ablnone.csv` **末行**汇总 |
| `eval_standard_suite.csv` | 三条 baseline + 各 `stack_k` DQN 的 **10-episode** 贪心评估（`lambda_from_ckpt`） |
| `train_metrics_sk{k}_ablnone.csv` | 每组训练曲线原始 log |
| `sweep_manifest.csv` | 每组训练结束追加一行（含末步指标展开） |

---

## 命令（均在 `RL_project/` 根下）

```bash
# 只看将执行哪些训练（不跑）
python -m stage_3.main sweep-stack-k --dry-run --ablation none

# 快速扫 stack_k（默认 1,2,4,8；约 8k 步/组，preset none）
python -m stage_3.main sweep-stack-k --quick --ablation none --device mps

# 快速扫消融（默认 stack_k=4；消融列表见下）
python -m stage_3.main sweep-ablation --quick --device mps

# 小网格：只跑部分 k × 消融
python -m stage_3.main sweep-grid --quick --stack-ks 1,4 --ablations none,no_iou --device mps

# 正式长度（例如 stage2_robust，单组约 72k 步，耗时长）
python -m stage_3.main sweep-stack-k --stage2-preset stage2_robust --ablation none --device mps

# 汇总 outputs 下所有 train_metrics_*.csv 末行 -> sweep_summary.csv
python -m stage_3.main collect --out stage_3/outputs/sweep_summary.csv
```

也可直接调用：

```bash
python -m stage_3.tools.sweep_experiments sweep-stack-k --quick --dry-run
```

## 默认消融名

与 `stage_2/env/ablation_masks.py` 一致：`none`, `no_iou`, `no_velocity`, `no_frame_diff`, `bbox_only`（可用 `--ablations` 覆盖）。

## 与 Pareto（§6.1 另一条线）

多固定 λ 的 Pareto 仍用 **`python -m stage_2.tools.run_pareto`**；Stage3 专注 **stack_k / 消融** 与 **训练曲线末行汇总**。二者结果可一并写入报告表格。

## 注意

- **`sweep-grid` 全默认** 会跑 `4×5=20` 组；全长度 preset 时总步数极大，务必先用 `--quick` 或缩小 `--stack-ks` / `--ablations`。
- 子进程会继承当前 **默认 teacher** 行为（与 Stage2 `main` 一致：存在 `data/teacher/teacher_fish_video.npz` 则用于奖励）。
