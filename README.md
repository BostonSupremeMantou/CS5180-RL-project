# RL Fish Tracking (reorganized)

Reinforcement learning for **adaptive compute** in fish tracking: each step the policy picks one of:

- **`FULL_DETECT`** — high quality, high cost  
- **`LIGHT_UPDATE`** — moderate quality and cost  
- **`REUSE`** — cheap, drift risk  

Each top-level directory has a short **`README.md`** describing what I keep there.

## Table of contents

1. [Repository structure](#1-repository-structure)  
2. [Environment setup](#2-environment-setup)  
3. [Data and checkpoints](#3-data-and-checkpoints)  
4. [Training](#4-training)  
5. [Evaluation](#5-evaluation)  
6. [Plots](#6-plots)  
7. [Final report (PDF)](#7-final-report-pdf)  
8. [Minimal end-to-end script](#8-minimal-end-to-end-script)  
9. [Notes](#9-notes)  

## 1. Repository structure

| Path | Purpose |
|------|---------|
| `baseline/` | Offline full-detection reference (e.g. `baseline.npz`) |
| `video/` | Fish videos for the env and evaluation |
| `weights/` | Checkpoints (`weights/baseline/`, `weights/non_learning_agents/`, `weights/RL_agents/<group>/`) |
| `outputs/` | Run logs, metrics CSVs, training plots (often gitignored) |
| `final_results/` | Curated eval tables and figures |
| `utilities/` | Shared helpers (paths, env, replay, plotting, …) |
| `agents/` | Policies: `no_learning_agents/`, `RL_agents/<group>/` |
| `src/` | Runnable entrypoints (`train`, `evaluate`, `final_eval`, plotting) |
| `final_report/` | Paper source (`final_report.tex`) and paper plots *(entire directory is gitignored; keep locally or publish separately)* |
| `old_versions/` | Legacy tree; not wired into the current pipeline |

## 2. Environment setup

Run commands from the **repository root** (your clone path may differ):

```bash
cd path/to/RL_project
```

Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Module path (required for imports):

```bash
export PYTHONPATH=.
```

Optional: `source ./activate_venv.sh` activates `.venv` from the repo root.

## 3. Data and checkpoints

| Asset | Default location |
|-------|-------------------|
| Videos | `video/` |
| Offline reference boxes | `baseline/baseline.npz` |
| RL checkpoint (example) | `weights/RL_agents/<group>/last.pt` |

If those paths are missing, some defaults may fall back under `old_versions/stage_1/data/...`.

## 4. Training

### Example (`g2_double_dueling`)

```bash
export PYTHONPATH=.
.venv/bin/python -m src.train \
  --group g2_double_dueling \
  --total-steps 50000 \
  --device cpu
```

### Common flags

| Flag | Meaning |
|------|---------|
| `--group` | Algorithm id (`g2_double_dueling`, `g3_nstep`, `g4_per`, …) |
| `--total-steps` | Total environment steps |
| `--device` | `cpu` or `cuda` |

Checkpoints usually appear under `weights/RL_agents/<group>/`.

## 5. Evaluation

### Non-learning baseline

```bash
export PYTHONPATH=.
.venv/bin/python -m src.evaluate \
  --policy flow_only \
  --episodes 3 \
  --device cpu
```

### RL checkpoint (greedy)

```bash
export PYTHONPATH=.
.venv/bin/python -m src.evaluate \
  --policy rl_greedy \
  --ckpt weights/RL_agents/g2_double_dueling/last.pt \
  --stack-k 4 \
  --episodes 3 \
  --device cpu
```

### Main outputs

| File | Content |
|------|---------|
| `final_results/eval/eval_raw.csv` | Per-episode metrics |
| `final_results/eval/eval_summary.csv` | Aggregated summary |

## 6. Plots

```bash
export PYTHONPATH=.
.venv/bin/python src/plot_final_eval.py \
  --input-csv final_results/eval/eval_raw.csv \
  --out-dir final_report/plots
```

Output directory: **`final_report/plots/`** (bar comparisons, Pareto / scatter views, variability and multi-metric diagnostics). Narrative order: `final_report/plots/README_ordered.md`.

## 7. Final report (PDF)

```bash
cd final_report
pdflatex -interaction=nonstopmode -halt-on-error final_report.tex
pdflatex -interaction=nonstopmode -halt-on-error final_report.tex
```

Artifact: **`final_report/final_report.pdf`**.

## 8. Minimal end-to-end script

```bash
cd path/to/RL_project
export PYTHONPATH=.

# 1) Train
.venv/bin/python -m src.train --group g2_double_dueling --total-steps 50000 --device cpu

# 2) Evaluate baseline and RL
.venv/bin/python -m src.evaluate --policy flow_only --episodes 3 --device cpu
.venv/bin/python -m src.evaluate --policy rl_greedy \
  --ckpt weights/RL_agents/g2_double_dueling/last.pt --stack-k 4 \
  --episodes 3 --device cpu

# 3) Plots
.venv/bin/python src/plot_final_eval.py \
  --input-csv final_results/eval/eval_raw.csv \
  --out-dir final_report/plots

# 4) Paper PDF
cd final_report
pdflatex -interaction=nonstopmode -halt-on-error final_report.tex
pdflatex -interaction=nonstopmode -halt-on-error final_report.tex
```

## 9. Notes

- The pipeline is most heavily exercised for **`g2_double_dueling`**; other groups may need extra validation.
- Plot order for the write-up: **`final_report/plots/README_ordered.md`**.
