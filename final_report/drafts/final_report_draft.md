# Adaptive Compute Allocation in Fish Tracking via Reinforcement Learning

**Can He**

## Abstract

This project studies reinforcement learning (RL) for adaptive compute allocation in video tracking. Instead of running a full detector on every frame, the agent selects one of three actions at each step: `FULL_DETECT`, `LIGHT_UPDATE`, or `REUSE`. The objective is to maximize tracking quality while minimizing compute cost. I formulate this as a discrete-action Markov decision process and evaluate several value-based RL agents (Vanilla DQN, Double+Dueling DQN, n-step, PER, SoftQ-style, C51, and GRU-based DQN) against non-learning baselines (`always_full`, `flow_only`, and `periodic_5`). Results on held-out episodes show that learned policies can achieve high IoU relative to an offline reference trajectory at much lower average cost than `always_full`. Among all tested policies, the GRU-based agent achieves the best return and strongest quality-cost balance, while `periodic_5` remains a strong non-learning baseline. The study demonstrates that RL-based decision policies can produce competitive tracking quality with substantial compute savings in fixed-camera fish tracking.

## 1. Introduction

A common systems problem in practical vision pipelines is that the highest-quality operation is often too expensive to apply at every timestep. In tracking, this appears as a tension between accuracy and compute: full detection is reliable but costly, while lightweight propagation is cheaper but more vulnerable to drift. This project focuses on fish tracking in fixed-camera videos, where temporal smoothness suggests that full detection can be selectively invoked.

The core question is:

> Can an RL policy learn when to spend compute (full detect) and when to save compute (light update or reuse), while preserving high tracking quality?

This problem is naturally sequential and long-horizon. A cheap action now may cause degraded state estimates later; conversely, paying for a corrective full detect may prevent future compounding error. RL is therefore a suitable framework for optimizing long-term quality-cost trade-offs.

## 2. Background and Related Work

### 2.1 Value-based RL foundations

Q-learning established temporal-difference optimization for action values in Markov decision processes (Watkins and Dayan 1992). Deep Q-Networks (DQN) extended this to neural function approximation with replay and target networks (Mnih et al. 2015), enabling stable learning in high-dimensional and correlated settings.

### 2.2 DQN improvements relevant to compute control

Several extensions are directly relevant to this project:

- **Double DQN** reduces overestimation bias in bootstrap targets (van Hasselt, Guez, and Silver 2015).
- **Dueling networks** improve value estimation when many actions have similar effects in a state (Wang et al. 2016).
- **Prioritized replay** focuses learning on transitions with high learning signal (Schaul et al. 2015).
- **Distributional RL (C51)** models return distributions instead of only expectations (Bellemare, Dabney, and Munos 2017).
- **Rainbow** demonstrates that these improvements can be complementary (Hessel et al. 2018).

For partial observability, recurrent value models such as DRQN provide sequence-aware state representations (Hausknecht and Stone 2015), which is important when action impact depends on recent temporal context.

### 2.3 Evaluation and implementation standards

Tracking literature emphasizes protocol consistency for credible comparison (Leal-Taixe et al. 2015). On the RL side, correct environment semantics (`terminated` vs. `truncated`) matter for valid bootstrapping targets (Farama Foundation). Reproducible implementation ecosystems (Stable-Baselines3, CleanRL, Dopamine, RLlib, OpenAI Baselines) provide practical references for robust experimentation.

## 3. Problem Formulation

### 3.1 MDP setup

The environment is a fish tracking task with discrete action space:

\[
\mathcal{A}=\{\texttt{FULL\_DETECT},\texttt{LIGHT\_UPDATE},\texttt{REUSE}\}.
\]

`FULL_DETECT` incurs the highest cost and typically best quality correction, `LIGHT_UPDATE` has moderate cost and quality, and `REUSE` has lowest immediate cost but highest drift risk.

### 3.2 Observation and reward

The state is an engineered feature representation (single-step or short temporal stack, depending on model variant). Reward follows a quality-minus-cost principle, aligning optimization with both tracking fidelity and compute usage. Quality is measured against an offline reference trajectory, and cost is action-dependent.

This setup captures the practical objective: maintain high alignment to reference while minimizing expensive detector calls.

## 4. Methods

I evaluate seven learned policies:

- `rl_greedy_g1_vanilla_last` (Vanilla DQN)
- `rl_greedy_g2_double_dueling_last` (Double + Dueling baseline)
- `rl_greedy_g3_nstep_last`
- `rl_greedy_g4_per_last`
- `rl_greedy_g5_softq_last`
- `rl_greedy_g6_c51_last`
- `rl_greedy_g7_gru_last`

and three non-learning baselines:

- `always_full`
- `flow_only`
- `periodic_5`

The DQN-family variants isolate the impact of specific algorithmic additions (Double+Dueling backbone, n-step returns, PER, distributional targets, and recurrence).

## 5. Experimental Setup

Evaluation uses shared episode protocol and metrics from:

- `final_results/eval/eval_raw.csv`
- `final_results/eval/eval_summary.csv`

Each policy is evaluated over 3 episodes with:

- return,
- mean IoU against reference (`mean_iou_ref`),
- temporal consistency (`mean_consistency`),
- average action cost (`mean_cost`),
- steps.

Plot-based comparisons are available in:

- `01_return_comparison.png`
- `02_iou_comparison.png`
- `03_consistency_comparison.png`
- `04_cost_comparison.png`

## 6. Results

### 6.1 Aggregate outcomes

From `eval_summary.csv`:

- `g7_gru`: return **307.38**, IoU **0.9434**, cost **0.2048**
- `periodic_5`: return **307.10**, IoU **0.9362**, cost **0.2002**
- `always_full`: IoU ~ **1.0000**, cost **1.0000**
- `g1_vanilla`: IoU **0.9041**, cost **0.1404**
- `flow_only`: IoU **0.6964**, cost **0.2500**

### 6.2 Interpretation

**Clear Pareto structure.** `always_full` serves as a high-quality, high-cost anchor; `flow_only` is low-information and underperforms in quality. Learned policies occupy intermediate and often favorable regions.

**Best learned balance.** `g7_gru` provides the strongest learned quality-cost trade-off, suggesting recurrence helps stabilize decisions under partial observability.

**Strong heuristic baseline.** `periodic_5` remains highly competitive, indicating that RL gains should be interpreted against non-trivial scheduling heuristics rather than only extreme baselines.

**Variant sensitivity.** More complex methods are not always better in this small evaluation: `g6_c51` has high cost/variance in this run, and `g5_softq` is weaker in IoU/return. This reinforces the need for larger-scale tuning and repeated seeds.

## 7. Discussion

The experimental evidence supports RL as a viable control layer for adaptive detector invocation in fish tracking. The results are practically meaningful because cost reductions are substantial relative to always-full operation, while quality remains high for top learned methods.

At the same time, the current scope has limitations: only three evaluation episodes per policy, single-domain setting, and no confidence intervals or significance tests. These limits do not invalidate the findings but constrain the strength of claims about general superiority among closely ranked methods.

## 8. Conclusion

This project demonstrates that adaptive compute control in fish tracking can be effectively learned via RL. Among tested agents, GRU-based DQN achieves the best learned quality-cost performance in this evaluation. The study also highlights two practical lessons: (1) robust non-learning schedules are essential baselines, and (2) algorithmic complexity must be validated empirically rather than assumed beneficial.

Future work will expand evaluation breadth (more seeds/videos), add statistical tests, and report explicit Pareto fronts to support decision-oriented deployment.

## References

- Bellemare, M. G., Dabney, W., and Munos, R. 2017. *A Distributional Perspective on Reinforcement Learning*. arXiv:1707.06887.
- Farama Foundation. *Gymnasium Env API*. https://gymnasium.farama.org/api/env/
- Farama Foundation. *Gymnasium Terminated/Truncated Step API*. https://farama.org/Gymnasium-Terminated-Truncated-Step-API
- Hausknecht, M., and Stone, P. 2015. *Deep Recurrent Q-Learning for Partially Observable MDPs*. arXiv:1507.06527.
- Hessel, M., et al. 2018. *Rainbow: Combining Improvements in Deep Reinforcement Learning*. AAAI.
- Leal-Taixe, L., Milan, A., Reid, I., Roth, S., and Schindler, K. 2015. *MOTChallenge 2015: Towards a Benchmark for Multi-Target Tracking*. arXiv:1504.01942.
- Mnih, V., et al. 2015. *Human-level control through deep reinforcement learning*. Nature.
- Schaul, T., Quan, J., Antonoglou, I., and Silver, D. 2015. *Prioritized Experience Replay*. arXiv:1511.05952.
- Sutton, R. S., McAllester, D., Singh, S., and Mansour, Y. 1999. *Policy Gradient Methods for Reinforcement Learning with Function Approximation*. NeurIPS.
- Tesauro, G. 1995. *Temporal Difference Learning and TD-Gammon*.
- van Hasselt, H., Guez, A., and Silver, D. 2015. *Deep Reinforcement Learning with Double Q-learning*. arXiv:1509.06461.
- Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M., and de Freitas, N. 2016. *Dueling Network Architectures for Deep Reinforcement Learning*. arXiv:1511.06581.
- Watkins, C. J. C. H., and Dayan, P. 1992. *Q-learning*. Machine Learning.

### Additional implementation and protocol resources used

- CleanRL: https://github.com/vwxyzjn/cleanrl
- Dopamine: https://github.com/google/dopamine
- OpenAI Baselines: https://github.com/openai/baselines
- Ray RLlib: https://github.com/ray-project/ray/tree/master/rllib
- Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3
- Stable-Baselines3 Docs: https://stable-baselines3.readthedocs.io/en/master/
- OpenAI Spinning Up: https://spinningup.openai.com/
- TrackEval MOTChallenge protocol docs: https://github.com/JonathonLuiten/TrackEval/tree/master/docs/MOTChallenge-Official
- COCO format docs: https://cocodataset.org/#format-data
