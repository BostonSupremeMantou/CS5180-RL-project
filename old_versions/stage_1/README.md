# Stage 1 — Compute-aware fish tracking + Dueling / Double DQN

资源已从 `old/fish` 迁入本目录（`data/`、`models/`、`runs/` 元数据等）。删除仓库根目录的 `old/` 后，只需保留 `stage_1/` 即可复现实验。

## 目录说明

| 路径 | 作用 |
|------|------|
| `data/videos/` | 输入视频（默认 `fish_video.mp4`） |
| `data/weights/` | 检测权重 `best.pt` / `last.pt` |
| `data/teacher/` | **可选**：离线 teacher（`preprocess` 生成），仅用于评估时对照 IoU |
| `models/yolo11n.pt` | Ultralytics 预训练起点（可选再训练） |
| `env/` | Gymnasium 环境 `FishTrackingEnv` |
| `models/dqn.py` | `TrackingDQN`（Dueling+Double）；`VanillaDQN` 仅旧 ckpt |
| `training/` | Replay buffer、训练循环 |
| `evaluation/` | 三条 baseline + DQN 评估 |
| `scripts/` | YOLO 训练、视频画框 |
| `tools/` | 数据集准备、降采样、抽帧、简易标注 |

## 环境

在**仓库根目录** `RL_project/` 下执行（需已 `pip install -r requirements.txt` 并建议使用 `.venv`）：

```bash
# 可选：仅为「评估时对照」生成 teacher（训练默认不需要）
python -m stage_1.main preprocess --device mps

# 训练：默认视频 data/videos/fish_video.mp4，无 teacher；可加 --teacher-npz 仅多记录对照指标
python -m stage_1.main train --device mps
python -m stage_1.main train --video-path /path/to/new.mp4 --device mps
python -m stage_1.main train --metrics-csv stage_1/checkpoints/train_metrics.csv --log-every 250

python -m stage_1.main eval --lambda-from-ckpt
python -m stage_1.main eval --video-path /path/to/new.mp4
python -m stage_1.main eval --teacher-npz stage_1/data/teacher/teacher_fish_video.npz
python -m stage_1.main eval --skip-dqn
```

单独入口：

```bash
python -m stage_1.training.train_dqn --help
python -m stage_1.evaluation.run_eval --help
python stage_1/scripts/annotate_video_yolo.py --device mps
```

## MDP 摘要

- **状态（10 维，新视频可算）**：帧差；归一化框；**上次 FULL 的置信度**（LIGHT 时衰减）；速度；**当前框与光流参考框的 IoU**；距上次 FULL 的归一化步数。
- **动作**：`FULL_DETECT`（跑 YOLO）/ `LIGHT_UPDATE`（LK 光流平移框）/ `REUSE`。
- **奖励**：`IoU(pred, flow_ref) − λ·cost`；**不依赖 teacher**。若提供 teacher，仅在 `info` / 评估 CSV 中多一列 **mean_iou_teacher** 作对照。
- **λ**：训练默认自适应（目标平均 cost）；`--fixed-lambda` 固定。

## 与 Stage 2 的关系

加深算法与实验见仓库 `stage_2/README.md` 与根目录 `rl_cv_project.md`。
