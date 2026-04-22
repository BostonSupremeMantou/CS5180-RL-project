# RL + CV Project: Compute-Aware Fish Tracking

## 1. Project Proposal

### Title

Adaptive Computation Allocation for Visual Tracking using Reinforcement
Learning

### Motivation

In simple visual environments, running full detection on every frame is
wasteful. This project explores how reinforcement learning can
dynamically decide when to perform expensive computation versus
lightweight updates.

### Objective

Reduce computation cost while maintaining tracking accuracy.

### Roadmap

Work is split into two stages. **Stage 1** is the minimum viable project
(end-to-end tracker + environment + DQN + baselines + metrics). **Stage
2** deepens the RL and evaluation without requiring changes to the YOLO
backbone: algorithm upgrades, richer MDP modeling, optional constrained
objectives, and stronger experiments.

------------------------------------------------------------------------

## Stage 1 — Core Deliverable

### 2. Problem Formulation

#### State

-   **10 features**, all computable on a **new video** without a precomputed
    teacher: frame difference; normalized box; **last YOLO confidence** (with
    decay under LIGHT); velocity $(v_x,v_y)$; **IoU(pred, flow-warp reference)**
    where the reference warps the **pre-action** box with the same optical flow
    used in LIGHT; **steps since last FULL** (normalized).

#### Action Space

-   FULL_DETECT
-   LIGHT_UPDATE
-   REUSE

#### Reward Function

reward = IoU(pred, flow\_reference) − λ · compute\_cost  
Optional offline teacher: **not** used in state or reward; only for **eval
logging** (mean IoU vs teacher) when `--teacher-npz` is provided.

### 3. Environment Design

-   Input: video frames
-   Output: fish bounding box
-   Step:
    -   Apply action
    -   Update tracking
    -   Compute reward

### 4. Baselines

-   Always FULL_DETECT
-   Periodic detection (every N frames)
-   Optical flow tracking

### 5. Model (Stage 1)

#### RL Algorithm

-   **Dueling + Double DQN** (replay buffer, target network with optional
    Polyak soft updates, ε-greedy exploration, Huber TD loss). **Adaptive
    λ** (Lagrangian-style update toward a target mean cost) or **fixed λ**.
-   Legacy: **Vanilla MLP** path exists only for loading old 8-dim checkpoints.

#### Network

-   Input: **10-d** state (see above); output: Q-values for 3 actions. Stage~2:
    **$10k$-d** stacked input ($k$ frames).

### 6. Training Pipeline

1.  (Optional) `preprocess` to build teacher `.npz` for **eval-only** IoU vs full-detect
2.  Baseline trackers + Gym environment
3.  Train DQN on `--video-path` (default fish video); `--teacher-npz` optional
4.  Evaluate: **mean flow consistency**, mean cost; optional **mean IoU vs teacher**

### 7. Evaluation Metrics

-   **Mean flow consistency** (training-aligned proxy)
-   **Optional** mean IoU vs teacher (requires `--teacher-npz` at eval)
-   Compute cost and return

### 8. Expected Results (Stage 1)

-   Lower mean cost than always FULL\_DETECT at a chosen $\lambda$
-   Consistency and (if logged) teacher IoU used as complementary signals

------------------------------------------------------------------------

## Stage 2 — Deeper RL & Stronger Experiments

Stage 2 builds on the same environment and actions. Pick a **primary**
track (algorithm and/or modeling) plus at least one **experimental**
depth item so the write-up goes beyond “DQN runs.”

### 9. Algorithm Upgrades

-   **Double DQN**: reduce Q overestimation when selecting bootstrap
    actions.
-   **Dueling architecture**: separate state value and action advantages;
    often helps when many actions are similar under many states.
-   **Optional**: prioritized experience replay (PER) or a simplified
    variant for sparse high-impact transitions (e.g., after missed
    detections).

Deliverable (achieved in Stage~1 code): **Double + Dueling** with learning
curves; optional **PER** not implemented. Final paper should compare **Stage~1
vs Stage~2** and baselines rather than “vanilla vs dueling” (dueling is default).

### 10. Partial Observability (Optional but High Signal)

Tracking is not fully Markov if the state is only instantaneous
hand-crafted features. **DRQN** (or a short **stacked state** of the last
k steps fed to an MLP/LSTM) models temporal context.

Deliverable (achieved): justify POMDP motivation; compare **single-step
(10-d)** vs **stacked history ($k$ frames → MLP)** in Stage~2. **DRQN/LSTM**
optional extension if time permits.

### 11. Constrained Compute Objective (Optional)

Beyond a fixed λ in the reward, treat average compute as a **budget**:
maximize accuracy subject to (or in tradeoff with) a per-episode or
per-frame cost constraint. Practical approach: **Lagrangian λ** updated
online (primal-dual style) so the policy adapts to a target cost.

Deliverable: **adaptive λ** is default in Stage~1 training; **fixed-λ**
policies appear in the Stage~2 **Pareto sweep** script (multiple checkpoints).
Report should relate both to the accuracy–compute trade-off.

### 12. Optional Extras (If Time)

-   **Risk-sensitive shaping**: extra penalty for long tracks without a
    successful FULL_DETECT or for large IoU drops (emphasize reliability,
    not only mean return).
-   **Generalization**: train on a subset of videos, evaluate on held-out
    videos; light **domain randomization** on brightness/blur/noise during
    training.

### 13. Experimental Depth (Recommended for All Stage 2 Runs)

-   **State ablations**: remove or zero out components (e.g., no frame
    diff, no velocity, **no\_iou** = no flow-alignment channel) and report
    impact on return and consistency.
-   **Pareto / tradeoff curves**: sweep $\lambda$; plot **mean consistency vs
    mean cost** (optional teacher IoU in a separate figure if teacher is available).
-   **Failure cases**: short qualitative analysis when the agent sticks
    to REUSE too long or oscillates actions.

------------------------------------------------------------------------

## 14. File Structure

project/ │── data/ │── env/ │── models/ │── training/ │── evaluation/
│── utils/ │── main.py

------------------------------------------------------------------------

## 15. Further Extensions (Beyond Stage 2)

-   Multi-agent version
-   Tile-based extension
-   Real-time deployment
