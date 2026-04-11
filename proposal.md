\begin{center}
\LARGE\textbf{Adaptive Computation Allocation for Visual Tracking using Reinforcement Learning}
\end{center}

\vspace{1em}

## 1. Problem Description

Object detectors such as YOLO achieve strong accuracy but incur substantial cost when run on every frame of a video. In relatively stable scenes (e.g., underwater fish tracking), full detection each timestep is often redundant. *Adaptive computation allocation* applies expensive inference only when needed and relies on cheaper updates otherwise (Bolukbasi et al., 2017; Bengio et al., 2016). This project studies this trade-off in a **real video** setting: a fixed fish detector (YOLO) provides full detections when invoked, while a reinforcement learning agent learns **when** to call full detection versus lighter alternatives.

**Formal formulation.** The problem is modeled as a **Markov Decision Process (MDP)** (possibly **partially observed** in extensions; see Stage 2).

- **State space $\mathcal{S}$:** A **10-dimensional** vector (Stage~1) derived each step: frame-difference statistic; normalized box $(c_x,c_y,w,h)$; detector confidence; velocity $(v_x,v_y)$; **IoU** between the current prediction and the **offline teacher** box at the same frame; **normalized time since last FULL\_DETECT** (encourages learning when to refresh). Stage~2 stacks the last $k$ such vectors (default $k=4$) for a $10k$-dimensional input to the same Q-head family. The environment consumes raw frames; the agent does not change detector weights.

- **Action space $\mathcal{A}$:** Three discrete actions: **FULL_DETECT** (run the full YOLO forward pass on the frame or region policy), **LIGHT_UPDATE** (e.g., optical flow or prediction-based box update without full detection), **REUSE** (keep the previous estimate with minimal computation). Costs and implementation details of LIGHT_UPDATE are fixed by the environment so the agent learns a scheduling policy.

- **Rewards $\mathcal{R}$:** Balance tracking quality against compute. A practical form is $R_t = \mathrm{IoU}_t - \lambda \, c(a_t)$, where $\mathrm{IoU}_t$ compares the predicted box to a reference (e.g., pseudo-ground truth from an offline always-detect run or held-out labels), $c(a_t)$ is the cost of action $a_t$, and $\lambda \geq 0$ controls the accuracy–compute trade-off. Discounted cumulative return is used for episodic training over video clips.

**Why this is non-trivial.** (1) **Temporal coupling:** Skipping detection saves cost but can cause drift; the agent must anticipate when reuse is safe. (2) **Quality–cost trade-off:** Different $\lambda$ values induce different Pareto points; the policy must not collapse to trivial always-REUSE or always-DETECT without learning. (3) **Partial observability (optional):** Instantaneous hand-crafted features may not summarize occlusion or sudden motion; Stage 2 can address this with recurrent value functions.

**Domain.** Gymnasium-compatible environment stepping over preprocessed video frames. A pretrained or finetuned YOLO model serves as the **fixed** full detector (no backbone redesign required for the core proposal). Existing training and annotation scripts in the repository support dataset and detector preparation.

## 2. Work Plan: Stage 1 and Stage 2

**Stage 1 (core deliverable, as implemented).** End-to-end pipeline: offline **teacher** generation (`preprocess`: full-frame YOLO → `.npz`), three **non-learning baselines**, Gymnasium environment `FishTrackingEnv` with a **10-dimensional** state, and **Dueling + Double DQN** (MLP with LayerNorm; Huber TD loss; replay buffer; target network with optional **Polyak** soft updates; $\varepsilon$-greedy). **Lagrangian-style adaptive $\lambda$** during training (updates toward a target mean per-step cost) with optional **fixed-$\lambda$** mode. Training **presets** (`robust` / `intense`) and periodic **metrics CSV** for learning curves. *(Legacy path: an 8-dim vanilla MLP checkpoint format remains loadable for backward compatibility.)*

**Stage 2 (as implemented).** Same video, detector, and action set. **Temporal context** via **stacking the last $k$ single-step observations** (default $k{=}4$, 40-dim input) into the same Dueling MLP head (wider hidden layers; `stage2_intense` preset). Optional **state ablation** masks (zeroing feature groups such as frame-difference or teacher-IoU). **Pareto-style experiment:** train separate policies at several **fixed** training-$\lambda$ values (shorter budget per point) and plot mean IoU vs.\ mean action cost. **Evaluation** script aggregates baselines, Stage~1 DQN, and Stage~2 DQN. *Not implemented in code:* full **DRQN / LSTM**; **prioritized replay (PER)**—listed as optional extensions for the report.

## 3. Algorithms and Baselines

**DQN family (main method).** Experience replay and a **target network** following Mnih et al.\ (2015); training uses **Double DQN** for bootstrap action selection and a **Dueling** architecture (state value + action advantages; Wang et al., 2016). Stage~1 default: **10-d** `TrackingDQN`; Stage~2: same head on **stacked** observations. $\varepsilon$-greedy exploration.

**Baselines (non-learned).** (1) **Always FULL_DETECT**—accuracy upper bound, cost upper bound. (2) **Periodic detection**—full detect every $N$ frames, else reuse. (3) **Optical-flow-only LIGHT_UPDATE**—isolates the value of learned scheduling.

**Implementation.** PyTorch; Ultralytics YOLO for FULL_DETECT; Gymnasium; entry points `python -m stage_1.main` (preprocess / train / eval), `python -m stage_2.training.train_stage2`, and `python -m stage_2.main compare`.

## 4. Expected Results and Evaluation

**Metrics.** (1) **Tracking accuracy:** mean IoU (and related statistics) vs. a reference. (2) **Compute cost:** proxy such as fraction of frames with FULL_DETECT, wall-clock, or FLOPs-style counts tied to actions. (3) **Trade-off:** curves of accuracy vs. cost when varying $\lambda$ or budget.

**Expected outcome.** The learned policy should **reduce** average compute relative to always FULL_DETECT while maintaining **comparable** IoU for a suitable choice of $\lambda$, and improve over fixed periodic schedules in some regimes.

**Risks and fallback plans.** (1) *Sparse or noisy rewards:* teacher IoU signal and $\lambda$ tuning; optional reward shaping (future). (2) *Training instability:* Huber loss, gradient clipping, replay capacity, Polyak target updates, and training presets already used in code. (3) *Limited time:* Stage~1 + one Stage~2 axis (stacked observations + Pareto) is the delivered scope; PER/DRQN remain report-level extensions if not run.

By the deadline, the project demonstrates: (1) a defined MDP over **real video** with a **fixed** YOLO detector, (2) **working Dueling+Double DQN** and the listed baselines, (3) quantitative **accuracy–compute** comparisons and **Pareto** points, and (4) **Stage~2** stacked-observation policy compared to Stage~1 under the same evaluation protocol.

## 5. References

- T. Bolukbasi, J. Wang, O. Dekel, and V. Saligrama. Adaptive neural networks for efficient inference. *ICML*, 2017.
- E. Bengio, P.-L. Bacon, J. Pineau, and D. Precup. Conditional computation in neural networks for faster models. *ICLR*, 2016 (arXiv:1511.06297, 2015).
- V. Mnih et al. Human-level control through deep reinforcement learning. *Nature*, 2015.
- Z. Wang, T. Schaul, M. Hessel, H. van Hasselt, M. Lanctot, and N. de Freitas. Dueling network architectures for deep reinforcement learning. *ICML*, 2016.
