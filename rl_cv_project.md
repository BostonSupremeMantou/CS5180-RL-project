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

-   Implemented as **10 features**: frame difference; normalized box; confidence;
    velocity; **IoU vs offline teacher**; **steps since last FULL** (normalized).
    (Design doc originally listed four groups; code merges these.)

#### Action Space

-   FULL_DETECT
-   LIGHT_UPDATE
-   REUSE

#### Reward Function

reward = accuracy - λ \* compute_cost

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

-   Input: **10-d** state vector (see environment); output: Q-values for 3 actions.

### 6. Training Pipeline

1.  Preprocess video
2.  Implement baseline tracker
3.  Build Gym-like environment
4.  Train DQN agent
5.  Evaluate performance

### 7. Evaluation Metrics

-   Tracking accuracy (IoU)
-   Compute cost
-   Accuracy vs compute tradeoff

### 8. Expected Results (Stage 1)

-   Reduced computation vs always FULL_DETECT
-   Comparable accuracy to full detection under a chosen λ

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
    diff, no velocity) and report impact on return and IoU.
-   **Pareto / tradeoff curves**: sweep λ or compute budget; plot
    accuracy vs cost, not only a single operating point.
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
