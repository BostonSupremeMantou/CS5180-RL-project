# Stage 5 — `research_targets.md` §6.4

在 **`FishTrackingEnv` + Stage2 堆叠（`stack_k=4`）+ `stage2_robust` 超参** 下，实现四种与 §6.4 对齐的增量；**不重复** Stage1–4 训练。评估阶段 **直接加载** 已有 `stage_4/checkpoints/dqn_stage4_baseline62_teacher.pt`（及 `_notchr`）作为对照行。

| 模式 | `--stage5-mode` | 说明 |
|------|-----------------|------|
| 6.4.1 | `per` | 优先经验回放 + TD-error 优先级与 IS 权重 |
| 6.4.4 | `softq` | 离散最大熵式备份：`r + γ·τ·logsumexp(Q_target/τ)`（单目标网络 + target） |
| 6.4.2 | `c51` | Dueling C51（51 原子，支撑 `[-30,30]`） |
| 6.4.3 | `gru` | GRU 融合 4×10 维历史（替代纯展平 MLP 主干） |

实现位置：`stage_5/training/per_buffer.py`、`stage_5/models/c51_net.py`、`stage_5/models/gru_dueling.py`、`stage_5/training/train_stage5.py`。训练仍用 **CPU 上的 Q 网络**（与 `train_stage2` 一致），环境内 YOLO 用 `--device`（如 `mps`）。

## 命令（仓库根目录）

```bash
# 8 次训练：4 模式 ×（teacher | notchr），均为 stage2_robust 长度
python -m stage_5.main standard --device mps

# 评估：baselines + 复用 Stage4 baseline62 + 本目录全部 dqn_stage5_*_{teacher,notchr}.pt
python -m stage_5.main evaluate --device mps --n-episodes 10

# 仅训练某一种（示例）
python -m stage_5.training.train_stage5 --preset none --stage2-preset stage2_robust \
  --stack-k 4 --ablation none --stage5-mode per \
  --save stage_5/checkpoints/dqn_stage5_per_teacher.pt \
  --metrics-csv stage_5/outputs/train_metrics_per_teacher.csv --device mps
```

## 产物

| 路径 | 含义 |
|------|------|
| `stage_5/checkpoints/dqn_stage5_{per,softq,c51,gru}_teacher.pt` | 有 teacher 奖励 |
| `stage_5/checkpoints/dqn_stage5_{per,softq,c51,gru}_notchr.pt` | `--no-teacher-reward` |
| `stage_5/outputs/train_metrics_*.csv` | 训练曲线（列与 Stage2 风格一致） |
| `stage_5/outputs/eval_suite_teacher.csv` | 贪心评估（含 teacher 列） |
| `stage_5/outputs/eval_suite_notchr.csv` | 无 teacher 环境评估 |

画图（与 Stage4 相同工具）：

```bash
python -m stage_1.tools.plot_eval_comparison \
  --csv stage_5/outputs/eval_suite_teacher.csv \
  --out-dir stage_5/outputs/plots_teacher --suptitle "Stage5 §6.4 eval (teacher)"
```

## 注意

- Stage5 **固定 1-step TD**（`--n-step` 若传入非 1 会警告并仍按 1-step 逻辑训练）。
- **PER** 与 **C51 / softq / gru** 互斥：仅 `per` 使用 SumTree 缓冲；其余为均匀 replay。
- 若尚未训练 Stage4 baseline62，评估表中仍会写 baselines 与 Stage5 权重；**Stage4 行会缺失**（文件不存在时跳过）。
