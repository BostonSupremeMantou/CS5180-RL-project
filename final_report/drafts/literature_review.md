# Literature Review: Reinforcement Learning for Adaptive Compute Allocation in Video Tracking

## 1. Problem setting: tracking under a quality–compute trade-off

Modern object detectors can be highly accurate, but running a heavy detector on every frame of a long video is often computationally wasteful—especially in fixed-camera settings where motion is locally smooth and the appearance changes slowly. This motivates **adaptive compute allocation**: at each time step, a controller decides whether to pay for a high-quality but expensive operation (e.g., a full detector), or to apply a cheaper approximation (e.g., optical-flow-like propagation), or to reuse a previous estimate. In our fish tracking project, this is explicitly formulated as a sequential decision problem with a discrete action set: **FULL\_DETECT**, **LIGHT\_UPDATE**, and **REUSE**. The evaluation goal is naturally multi-objective: we care about tracking quality (e.g., IoU against a reference trajectory) and about computation cost (e.g., average action cost or the fraction of FULL\_DETECT steps).

A central design choice is how to operationalize “quality” when online ground truth is not available. Our project uses an offline “teacher” or reference trajectory produced by full detection, and then measures or shapes performance via **IoU(predicted box, reference box)**. This fits a broad theme in practical reinforcement learning: when real supervision is expensive or unavailable at run time, learning is often anchored to a reliable proxy signal that can be precomputed (offline) and then reused in training and evaluation. The resulting control problem is still non-trivial because tracking errors accumulate over time: choosing too many cheap actions can cause drift, which later requires corrective expensive actions. This temporal coupling makes the problem a good match for temporal-difference learning methods that optimize long-horizon returns.

## 2. Foundations: value-based learning and temporal-difference methods

The theoretical backbone for many value-based RL methods is **Q-learning** (Watkins & Dayan, 1992), which defines the optimal action-value function \(Q^\*(s,a)\) via the Bellman optimality equation and updates estimates using bootstrapping from successor states. In compute allocation for tracking, the action-value function can be interpreted as the long-term benefit of choosing FULL\_DETECT vs. LIGHT\_UPDATE vs. REUSE in a given observation state, accounting for both near-term cost and downstream quality recovery. While our environment is best described as partially observed (a tracker’s state typically depends on recent history), practical solutions often approximate the Markov property through feature design and/or short history windows.

The success of temporal-difference learning in complex domains is exemplified by TD-Gammon (Tesauro, 1995), which demonstrated that learning from incremental prediction errors—rather than waiting for terminal outcomes—can scale to long sequential decision problems. Although TD-Gammon is not a tracking system, its core lesson generalizes: in long horizons with delayed consequences, bootstrapping and incremental learning can discover strategies that balance immediate and future considerations. For tracking, this translates to learning when to “spend” compute now to prevent compounding drift later.

Policy-gradient methods provide another foundational perspective. Sutton et al. (1999) formalized policy gradients with function approximation, enabling direct optimization of a parameterized policy. In compute allocation, a policy-gradient approach could directly optimize a stochastic decision rule over discrete actions. However, in our project and many practical discrete-action control tasks, value-based methods are attractive because they provide an explicit estimate of action utilities and integrate naturally with replay buffers and off-policy learning—tools that are particularly useful when data collection is expensive or when we want to reuse trajectories.

## 3. Deep Q-Networks and the modern DQN family

The Deep Q-Network (DQN) (Mnih et al., 2015) is the seminal result that brought Q-learning to high-dimensional function approximation at scale, using a neural network to approximate \(Q(s,a)\), plus two stabilizing mechanisms: **experience replay** and a **target network**. Although our project uses low-dimensional engineered features rather than raw pixels, the DQN recipe remains relevant: it is a robust baseline for discrete-action problems and supports variants tailored to data efficiency and stability. In tracking, replay is particularly helpful because the distribution of states is highly correlated in time; replay reduces correlation and increases sample reuse.

### 3.1 Double DQN: reducing overestimation bias

One known issue with vanilla DQN is overestimation caused by the max operator in bootstrap targets. **Double Q-learning** addresses this by decoupling action selection from action evaluation. In **Double DQN** (van Hasselt et al., 2015), the next action is selected using the online network but evaluated using the target network. In our setting, this can reduce systematic optimism that might otherwise lead the controller to underpay for costly FULL\_DETECT steps early, only to suffer larger downstream corrections. The more stable value estimates provided by Double DQN are therefore directly aligned with our project’s aim of achieving reliable compute–quality trade-offs.

### 3.2 Dueling networks: separating state value from action advantage

Compute allocation tasks often have states where many actions behave similarly (e.g., when the object is static and the tracker is confident). **Dueling network architectures** (Wang et al., 2016) explicitly factorize the Q-function into a state value \(V(s)\) and an action advantage \(A(s,a)\). This can improve learning efficiency by allowing the model to learn “how good the state is” even when action differences are subtle. In our tracking controller, many consecutive frames may be easy, and the principal decision is *when* to trigger FULL\_DETECT; dueling architectures are therefore a natural match.

### 3.3 Prioritized Experience Replay (PER): focusing updates on informative transitions

In temporal control problems, not all transitions are equally informative. Failures (e.g., drift events) or rare corrective sequences may contain disproportionately important learning signals. **Prioritized Experience Replay** (Schaul et al., 2015) samples transitions with probability proportional to a priority measure (often based on TD error), while correcting the induced bias with importance sampling weights. For our compute allocation problem, PER can emphasize transitions around tracker degradation and recovery, where the controller’s decision is most consequential. This aligns with the intuition that the agent should learn strongly from cases where cheap actions cause future penalties or where timely FULL\_DETECT prevents large quality drops.

### 3.4 Rainbow DQN: integrating complementary improvements

Rainbow (Hessel et al., 2018) systematically studies a set of DQN improvements and shows that combining them yields strong performance on Atari. The value for our project is twofold. First, Rainbow provides a conceptual map of which extensions matter (e.g., double Q-learning, dueling, multi-step targets, PER, distributional RL). Second, it suggests a pragmatic methodology: rather than inventing an entirely new algorithm, we can start from a stable DQN variant and layer improvements that address known failure modes (bias, sample inefficiency, poor exploration, etc.). Our own agent taxonomy (e.g., Double+Dueling as a base, with optional PER, n-step, distributional heads, or recurrence) mirrors Rainbow’s spirit of modular improvements.

### 3.5 Distributional RL (C51): modeling return distributions

While standard Q-learning models only the expected return, **distributional RL** models the full return distribution. The C51 algorithm (Bellemare et al., 2017) represents the value distribution over a fixed support and performs a projection step during Bellman updates. In compute allocation for tracking, return distributions can be heavy-tailed due to rare but severe drift or recovery events. A distributional view can, in principle, provide richer learning signals and possibly enable risk-sensitive behavior. Even when we ultimately act greedily with respect to the expectation, learning the distribution can improve representation and stability.

## 4. Partial observability and recurrent value functions

Tracking is naturally a partially observed problem because the true state of the world (e.g., precise object pose and occlusion status) is not fully captured by a single frame’s summary features. A standard engineering workaround is to stack a fixed-length history. A more flexible solution is to use recurrence. The **Deep Recurrent Q-Network (DRQN)** (Hausknecht & Stone, 2015) extends DQN with an RNN (e.g., LSTM/GRU) to integrate information over time. In our project’s taxonomy, a GRU-based variant is used to consume a \(k\times d\) sequence of feature vectors and output dueling Q-values, retaining the Double DQN bootstrap structure. This is a principled way to handle temporal dependencies such as “time since last FULL\_DETECT” or motion consistency patterns that are better inferred from sequences than from single-step features.

## 5. Evaluation protocols for tracking and for RL environments

### 5.1 Tracking benchmarks and protocols

A persistent challenge in tracking research is that evaluation can vary widely across datasets and scripts, making results hard to compare. MOTChallenge (Leal-Taixé et al., 2015) is a canonical effort to standardize multi-object tracking benchmarks and evaluation. Although our fish tracking setting may not use the exact MOT metrics (e.g., MOTA/HOTA/IDF1), MOTChallenge’s broader contribution is methodological: define a consistent protocol, publish evaluation code, and avoid “moving target” comparisons. This is directly relevant to our final report: our own evaluation should be specified as a protocol (video list, episode length, deterministic vs. stochastic action selection, and fixed cost model), and the quality metric (IoU against a reference) should be computed consistently.

Relatedly, the **COCO format** and evaluation tooling are widely used standards for detection datasets and for metric computation, offering a common language for representing bounding boxes and annotations. Even if we do not run COCO mAP, aligning data representation and box conventions with common standards reduces ambiguity and makes it easier to integrate external tools.

### 5.2 RL environment APIs and termination semantics

Reproducible RL also depends on correct environment interfaces. The Gymnasium API specifies the `reset`/`step` contracts and introduces the modern `terminated` vs. `truncated` distinction (Farama Foundation). This matters for value bootstrapping: whether we should bootstrap from the next state depends on whether an episode ended due to a true terminal condition or a time limit. In our tracking setup, episodes often end due to fixed video segment length (a truncation), and correct handling prevents biased value targets. Explicitly referencing this standard strengthens the engineering credibility of the project and clarifies implementation details in the report.

## 6. Practical implementations and reproducibility-oriented codebases

Beyond academic papers, mature open-source implementations provide de facto “engineering standards” for RL experiments. Stable-Baselines3 is a widely used PyTorch library offering reliable baselines and common utilities (logging, vectorized environments, evaluation loops). CleanRL emphasizes minimal and reproducible single-file implementations, which can be especially valuable for debugging algorithmic details and ensuring that experimental claims are traceable. Dopamine (Google) provides a research-oriented DQN framework with careful implementations of DQN-family algorithms and ablations. RLlib addresses the systems side of RL at scale and exemplifies how to structure training, sampling, and evaluation in distributed settings. OpenAI Baselines serves as a historical reference that influenced many later implementations and still provides useful comparisons and design patterns.

Community resources also play a practical role in applied RL. The OpenAI Spinning Up materials offer pedagogical explanations and reference implementations that help ensure correct reasoning about objectives and updates. Discussions in broader communities (e.g., r/reinforcementlearning) can surface common pitfalls, implementation bugs, and replication notes, though such sources should be used cautiously and primarily as pointers rather than as authoritative evidence.

## 7. Synthesis: how the literature informs our project design

Taken together, the literature supports a clear design logic for RL-based compute allocation in tracking:

1. **Value-based control is well suited to discrete compute decisions.** Q-learning and TD methods provide a natural way to learn long-horizon trade-offs between immediate compute costs and future quality degradation (Watkins & Dayan, 1992; Tesauro, 1995).
2. **The DQN recipe provides a practical baseline for stable learning.** Experience replay and target networks mitigate instability in off-policy learning (Mnih et al., 2015), which is important when video trajectories are correlated.
3. **Known DQN improvements match known failure modes in tracking control.** Double DQN reduces bias (van Hasselt et al., 2015), dueling networks improve representation when action differences are small (Wang et al., 2016), PER focuses learning on rare but critical transitions (Schaul et al., 2015), and distributional RL can better represent uncertain outcomes (Bellemare et al., 2017). Rainbow demonstrates that these improvements are modular and complementary (Hessel et al., 2018).
4. **Partial observability motivates sequence modeling.** DRQN-style recurrence provides a principled mechanism to aggregate temporal evidence (Hausknecht & Stone, 2015), complementing history stacking.
5. **Protocol clarity is essential for credible trade-off claims.** Tracking benchmarks stress standardized evaluation (Leal-Taixé et al., 2015), and RL environment standards clarify termination and bootstrapping semantics (Gymnasium API; terminated/truncated rationale). These references justify the evaluation and implementation choices in our report and help avoid subtle but consequential methodological errors.

In summary, our project’s approach—treating adaptive detector invocation as a discrete-action RL problem and using DQN-family algorithms with well-motivated enhancements—fits squarely within established RL methodology while addressing a practically important systems objective: reducing compute without sacrificing tracking quality.

---

## References

- Bellemare, M. G., Dabney, W., & Munos, R. (2017). *A Distributional Perspective on Reinforcement Learning*. arXiv. https://arxiv.org/abs/1707.06887
- Farama Foundation. *Gymnasium Env API*. https://gymnasium.farama.org/api/env/
- Farama Foundation. *Gymnasium Terminated/Truncated Step API*. https://farama.org/Gymnasium-Terminated-Truncated-Step-API
- Hausknecht, M., & Stone, P. (2015). *Deep Recurrent Q-Learning for Partially Observable MDPs*. arXiv. https://arxiv.org/abs/1507.06527
- Hessel, M., et al. (2018). *Rainbow: Combining Improvements in Deep Reinforcement Learning*. AAAI. https://ojs.aaai.org/index.php/AAAI/article/view/11796
- Leal-Taixé, L., Milan, A., Reid, I., Roth, S., & Schindler, K. (2015). *MOTChallenge 2015: Towards a Benchmark for Multi-Target Tracking*. arXiv. https://arxiv.org/abs/1504.01942
- Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. Nature. https://www.nature.com/articles/nature14236
- OpenAI. *Spinning Up in Deep Reinforcement Learning*. https://spinningup.openai.com/
- r/reinforcementlearning. *Community discussions and resource sharing*. https://www.reddit.com/r/reinforcementlearning/
- Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2015). *Prioritized Experience Replay*. arXiv. https://arxiv.org/abs/1511.05952
- Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (1999). *Policy Gradient Methods for Reinforcement Learning with Function Approximation*. NeurIPS. https://papers.nips.cc/paper_files/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf
- Tesauro, G. (1995). *Temporal Difference Learning and TD-Gammon*. https://www.bkgm.com/articles/tesauro/tdl.html
- TrackEval. *MOTChallenge Official Evaluation Protocol Documentation*. https://github.com/JonathonLuiten/TrackEval/tree/master/docs/MOTChallenge-Official
- van Hasselt, H., Guez, A., & Silver, D. (2015). *Deep Reinforcement Learning with Double Q-learning*. arXiv. https://arxiv.org/abs/1509.06461
- Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M., & de Freitas, N. (2016). *Dueling Network Architectures for Deep Reinforcement Learning*. arXiv. https://arxiv.org/abs/1511.06581
- Watkins, C. J. C. H., & Dayan, P. (1992). *Q-learning*. Machine Learning. https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf

### Source code (for analysis / implementation reference)

- CleanRL. https://github.com/vwxyzjn/cleanrl
- Dopamine. https://github.com/google/dopamine
- OpenAI Baselines. https://github.com/openai/baselines
- Ray RLlib. https://github.com/ray-project/ray/tree/master/rllib
- Stable-Baselines3 (library). https://github.com/DLR-RM/stable-baselines3
- Stable-Baselines3 (documentation). https://stable-baselines3.readthedocs.io/en/master/

### Engineering / industry deep dives

- CleanRL documentation. https://docs.cleanrl.dev/
- Google Research Football environment overview. https://research.google/pubs/google-research-football-a-novel-reinforcement-learning-environment/
- Hugging Face Deep RL Course. https://huggingface.co/learn/deep-rl-course/unit0/introduction
