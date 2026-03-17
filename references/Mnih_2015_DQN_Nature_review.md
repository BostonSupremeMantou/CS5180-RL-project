# Mnih et al., Nature 2015 — Human-level control through deep reinforcement learning

**Citation:** V. Mnih et al. Human-level control through deep reinforcement learning. *Nature*, 2015.

---

## What the paper is about

- **Problem:** RL had been largely limited to low-dimensional or handcrafted features and fully observed spaces; the goal is to learn control **directly from high-dimensional sensory input** (e.g. raw pixels).
- **Method:** **Deep Q-Network (DQN)**—a deep convolutional network approximates the optimal action-value function Q(s, a). Standard Q-learning updates are combined with **experience replay** (sampling uniformly from a replay buffer to break temporal correlation) and a **target network** (target Q copied periodically from the current network to reduce correlation with the target and stabilize learning with nonlinear function approximation).
- **Experiments:** Atari 2600, 49 games, same architecture and hyperparameters; only pixels and score as input; performance exceeds prior methods and reaches human-level on many games.
- **Formulas:** Q-learning loss, replay buffer usage, and target network update interval C are all given explicitly in the paper.

---

## What you can use (for your project)

- **Baseline for Centralized DQN:** Your single-agent, full-state, joint-action (REUSE/FULL per block) DQN can cite Mnih et al. (2015) as the standard DQN reference. In implementation, use **experience replay + target network** (and double DQN as in your proposal) for stability.
- **Network and Q parameterization:** They use a CNN for images; you can use an MLP or small network on “block feature vector + budget.” The idea “deep network approximating Q(s, a)” is the same; you can write “following Mnih et al.”
- **Training stability:** Their discussion of instability (sequential correlation, changing data distribution, Q–target correlation) and the replay + target fix can be cited in your method or experiments to justify replay, target network, and double DQN.
- **Evaluation and metrics:** They report learning curves (average return, average Q). You can report “average return, average drift, number of FULL updates per step” and note that learning curves and stability analysis follow common DQN practice (e.g. Mnih et al.).
- **What not to reuse directly:** Their setting is single-agent Atari; yours is multi-block, multi-step, synthetic environment, and budget-constrained. The environment and action space differ, but the algorithm skeleton is the same.
