# Stage 1 — Compute-aware fish tracking + Dueling / Double DQN

资源已从 `old/fish` 迁入本目录（`data/`、`models/`、`runs/` 元数据等）。删除仓库根目录的 `old/` 后，只需保留 `stage_1/` 即可复现实验。

## 目录说明

| 路径 | 作用 |
|------|------|
| `data/videos/` | 输入视频（默认 `fish_video.mp4`） |
| `data/weights/` | 检测权重 `best.pt` / `last.pt` |
| `data/teacher/` | 离线 teacher 轨迹（`preprocess` 生成 `.npz`） |
| `models/yolo11n.pt` | Ultralytics 预训练起点（可选再训练） |
| `env/` | Gymnasium 环境 `FishTrackingEnv` |
| `models/dqn.py` | `TrackingDQN`（Dueling+Double 训练逻辑）；`VanillaDQN` 仅旧 ckpt |
| `training/` | Replay buffer、训练循环 |
| `evaluation/` | 三条 baseline + DQN 评估 |
| `scripts/` | YOLO 训练、视频画框 |
| `tools/` | 数据集准备、降采样、抽帧、简易标注 |

## 环境

在**仓库根目录** `RL_project/` 下执行（需已 `pip install -r requirements.txt` 并建议使用 `.venv`）：

```bash
python -m stage_1.main preprocess --device mps   # 或 cpu
python -m stage_1.main train --device mps   # 默认 --preset robust；轻量加 --preset none
# 每 500 步打印 [metrics]；写入 CSV 便于画曲线：
python -m stage_1.main train --metrics-csv stage_1/checkpoints/train_metrics.csv --log-every 250
# 默认：Dueling+Double DQN + Huber + 拉格朗日自适应 λ（目标平均 cost≈0.32）。固定 λ：
python -m stage_1.main train --fixed-lambda --lambda-cost 0.35
# DQN 评估默认随机起点；与 checkpoint 中 λ 对齐：
python -m stage_1.main eval --lambda-from-ckpt
python -m stage_1.main eval --skip-dqn            # 仅 baseline
python -m stage_1.main eval                       # 含 DQN（需 checkpoints/dqn_stage1.pt）
```

单独入口：

```bash
python -m stage_1.training.train_dqn --help
python -m stage_1.evaluation.run_eval --help
python stage_1/scripts/annotate_video_yolo.py --device mps
```

再训练 YOLO（需自备 `data/fish_annotations_half/` 等，见 `tools/prepare_yolo11_fish_dataset.py`）：

```bash
cd stage_1 && python scripts/train_yolo11n_fish.py --prepare-dataset
```

## MDP 摘要

- **状态**：帧差、归一化框 (cx,cy,w,h)、置信度、速度 (vx,vy)、**与 teacher 的 IoU**、**距上次 FULL 的步数（归一化）** — 共 10 维，便于学习何时检测。
- **动作**：`FULL_DETECT` / `LIGHT_UPDATE`（LK 光流平移框）/ `REUSE`。
- **奖励**：`IoU - λ * cost(action)`；**λ 训练时默认按「目标平均 cost」自适应**（拉格朗日），也可用 `--fixed-lambda` 固定。

## 与 Stage 2 的关系

加深算法与实验见仓库 `stage_2/README.md` 与根目录 `rl_cv_project.md`。
