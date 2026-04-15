\begin{center}
\LARGE\textbf{Adaptive Computation Allocation for Visual Tracking using Reinforcement Learning}
\end{center}

\vspace{1em}

## 1. Problem Description

Object detectors such as YOLO achieve strong accuracy but incur substantial cost when run on every frame of a video. In relatively stable scenes (e.g., underwater fish tracking with a **fixed camera**), full detection each timestep is often redundant. *Adaptive computation allocation* applies expensive inference only when needed and relies on cheaper updates otherwise (Bolukbasi et al., 2017; Bengio et al., 2016). This project studies this trade-off in a **real video** setting: a fixed fish detector (YOLO) runs **only when the RL policy chooses FULL\_DETECT**; otherwise the environment applies optical-flow-style updates or reuse.

**Formal formulation.** The problem is modeled as a **Markov Decision Process (MDP)** (possibly **partially observed** in extensions; see Stage 2).

- **State space $\mathcal{S}$:** A **10-dimensional** vector computable **without** offline per-frame labels on the deployment path: frame-difference statistic; normalized box $(c_x,c_y,w,h)$; **last detector confidence** (from the most recent FULL pass, with decay under LIGHT); velocity $(v_x,v_y)$; **IoU between the current prediction and a flow-only reference** (optical-flow warp of the box *before* the current action—cheap and always defined); **normalized time since last FULL\_DETECT**. Stage~2 stacks the last $k$ such vectors (default $k=4$) for a $10k$-dimensional input. The agent does not change detector weights.

- **Action space $\mathcal{A}$:** Three discrete actions: **FULL\_DETECT** (run YOLO on the current frame), **LIGHT\_UPDATE** (LK optical flow shift of the box), **REUSE** (keep the previous estimate with minimal computation). Costs are fixed by the environment.

- **Rewards $\mathcal{R}$:** **Hybrid by data availability.** If an offline teacher `.npz` is loaded (training default when the file exists), $R_t=\mathrm{IoU}(\mathrm{pred},\mathrm{teacher}_t)-\lambda\,c(a_t)$ so the policy is explicitly aligned to the full-detection trajectory on that video. **Without** a teacher file, $R_t=\mathrm{IoU}(\mathrm{pred},\mathrm{Flow}(\cdot))-\lambda\,c(a_t)$ as a deployable self-supervised signal. Teacher never enters the state vector.

**Why this is non-trivial.** (1) **Temporal coupling:** Skipping detection saves cost but can cause drift; the policy must anticipate when reuse is unsafe. (2) **Proxy reward:** Flow consistency is not identical to human or YOLO-always quality; $\lambda$ and shaping must be tuned. (3) **Partial observability (Stage 2):** Stacking frames improves context without RNNs in the current code.

**Domain.** Gymnasium-compatible environment over video frames. Pretrained YOLO serves as the **fixed** full detector. Optional `preprocess` still generates a teacher file for **optional** eval-only comparison.

## 2. Work Plan: Stage 1 and Stage 2

**Stage 1 (core deliverable, as implemented).** Pipeline: training on a chosen `--video-path`; if `--teacher-npz` is omitted but the default teacher file exists, it is loaded for **teacher-aligned reward** (otherwise flow-consistency reward); three **non-learning baselines**; `FishTrackingEnv` with the **10-d** state above; **Dueling + Double DQN** (Huber TD loss; replay; target net with optional Polyak; $\varepsilon$-greedy); **adaptive $\lambda$** or **fixed $\lambda$**; presets and **metrics CSV**. Legacy **8-dim vanilla** checkpoints remain loadable for old experiments.

**Stage 2 (as implemented).** Same actions and reward definition. **Temporal context** via **stacking** the last $k$ observations (default $k=4$). Optional **ablation** masks. **Pareto-style** training at several fixed $\lambda$ values; evaluation CSV/plots use **mean flow consistency** vs.\ mean cost (teacher IoU column optional when teacher is provided at eval time). *Not implemented:* DRQN/LSTM; PER.

## 3. Algorithms and Baselines

**DQN family.** Replay, **target network** (Mnih et al., 2015), **Double DQN** bootstrap, **Dueling** head (Wang et al., 2016). Stage~1: **10-d** input; Stage~2: stacked **$10k$-d** input.

**Baselines.** (1) Always FULL\_DETECT; (2) periodic FULL; (3) optical-flow-only LIGHT.

**Entry points.** `python -m stage_1.main` (preprocess / train / eval), `python -m stage_2.training.train_stage2`, `python -m stage_2.main compare`, `python -m stage_2.tools.run_pareto`.

## 4. Expected Results and Evaluation

**Metrics.** (1) **Mean flow consistency** (same quantity as in the training reward, averaged over an episode). (2) **Optional** mean IoU vs.\ offline teacher when `--teacher-npz` is supplied at eval. (3) **Compute:** mean action cost, fraction of FULL steps.

**Expected outcome.** Policies should **reduce** average cost vs.\ always FULL while maintaining reasonable consistency; when a teacher file is used in training, teacher IoU is the **primary** reward signal; without it, flow consistency is the target and teacher IoU at eval remains **diagnostic**.

**Risks.** Proxy reward may not match desired tracking quality; fixed-camera motion cues help but do not remove the need for validation. Fallback: add sparse labels or hybrid training phases in future work.

By the deadline, the project demonstrates: (1) an MDP over **real video** with **RL-gated** YOLO, (2) **Dueling+Double DQN** and baselines, (3) **accuracy–compute** reporting using **consistency + cost** (and optional teacher), and (4) Stage~2 stacked policies and Pareto-style sweeps.

## 5. References

- T. Bolukbasi, J. Wang, O. Dekel, and V. Saligrama. Adaptive neural networks for efficient inference. *ICML*, 2017.
- E. Bengio, P.-L. Bacon, J. Pineau, and D. Precup. Conditional computation in neural networks for faster models. *ICLR*, 2016 (arXiv:1511.06297, 2015).
- V. Mnih et al. Human-level control through deep reinforcement learning. *Nature*, 2015.
- Z. Wang, T. Schaul, M. Hessel, H. van Hasselt, M. Lanctot, and N. de Freitas. Dueling network architectures for deep reinforcement learning. *ICML*, 2016.
