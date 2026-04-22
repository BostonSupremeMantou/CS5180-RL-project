# Stage 4 — §6.2 中等成本拓展

对应 **`old_research_targets.md` 第 6.2 节**：在 **同一 Dueling + Double DQN + Stage2 堆叠架构** 下，可选启用：

| 机制 | CLI | 说明 |
|------|-----|------|
| 距上次 FULL 惩罚 | `--ssf-reward-penalty` | 在 `FishTrackingEnv` 中 `reward -= coeff * ssf_norm`（与状态中归一化步数一致） |
| teacher IoU 低时抬 λ | `--lambda-teacher-iou-floor` / `--lambda-teacher-iou-below` | episode 末若平均 teacher IoU 低于阈值，将 λ 至少拉到 floor（需 **teacher 参与奖励**，勿与 `--no-teacher-reward` 一起期望生效） |
| n-step TD | `--n-step` | `n>1` 时用 n 步回报写入 replay，bootstrap 用 `γ^n` |
| ε 尾部退火 | `--epsilon-tail` + `--epsilon-tail-steps` | 主 ε 衰减结束后，再线性退火到 `epsilon_tail` |

实现位置：**`stage_1/env/fish_tracking_env.py`**、**`stage_1/training/n_step.py`**、**`stage_2/training/train_stage2.py`**、**`stage_2/training/stage2_train_args.py`**。也可直接：

```bash
python -m stage_2.training.train_stage2 --stack-k 4 --ssf-reward-penalty 0.05 --device mps
```

## 编排命令（`stage_4/`）

在仓库根目录执行：

```bash
# 快速：每组 ~8k 步，跑 §6.2 预设多组 variant
python -m stage_4.main standard --quick --device mps

# 正式长度（与 stage2_robust 对齐）
python -m stage_4.main standard --device mps

# 有 / 无 teacher 两套（文件名后缀区分，避免覆盖）
python -m stage_4.main standard --device mps --artifact-suffix _teacher --stage2-preset stage2_robust
python -m stage_4.main standard --device mps --artifact-suffix _notchr --no-teacher-reward --stage2-preset stage2_robust

# 标准贪心评估（10 episode，λ 从 ckpt）
python -m stage_4.main evaluate --artifact-suffix _teacher --device mps --n-episodes 10
python -m stage_4.main evaluate --artifact-suffix _notchr --no-teacher-eval --device mps --n-episodes 10

# 无 teacher 奖励（仅光流一致性），与 notchr 管线一致
python -m stage_4.main standard --quick --no-teacher-reward --device mps

# 仅汇总已有 train_metrics_*.csv
python -m stage_4.main collect --artifact-suffix _teacher
```

**输出**

| 路径 | 含义 |
|------|------|
| `stage_4/checkpoints/dqn_stage4_<variant>.pt` | 各 variant 权重 |
| `stage_4/outputs/train_metrics_<variant>.csv` | 训练曲线 |
| `stage_4/outputs/sweep_manifest_62.csv` | 每组末行指标展开 |
| `stage_4/outputs/sweep_train_summary_62.csv` | `collect` / `standard` 末尾汇总 |

`VARIANTS_62` 定义在 `stage_4/tools/sweep_stage4.py`（含 `baseline62`、`ssf05`、`nstep3`、`epstail`、`ioufloor`、`combo_ssf_n3`），可按报告需要增删行。

## 评估

评估协议与 Stage2/3 相同，可将 `--save` 指向 `stage_4/checkpoints/dqn_stage4_<tag>.pt` 后使用 `stage_2.main compare` 或 `stage_3.tools.standard_eval`（需 stack_k 与 ckpt 一致）。
