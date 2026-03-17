# Tan, ICML 1993 — Multi-Agent Reinforcement Learning: Independent vs. Cooperative Agents

**Citation:** M. Tan. Multi-agent reinforcement learning: Independent vs. cooperative agents. *ICML*, 1993.

---

## What the paper is about

- **Problem:** In multi-agent settings, how do **independent** learners (no communication) compare to **cooperative** ones (sharing information / experience / policies), and what is the cost of cooperation?
- **Three forms of cooperation:** (1) Sharing instantaneous information (e.g. sensation); (2) sharing episodes (sequences of s, a, r); (3) sharing learned policies.
- **Setting:** Multiple RL agents (Q-learning with Boltzmann exploration) in a grid world capturing prey; each hunter has local observation (limited field of view) and can learn independently or with cooperation.
- **Findings:** Cooperation can speed learning or improve final performance, but only if the shared information is sufficient and usable; in joint tasks, cooperative agents can outperform independent ones but may learn more slowly at first; if cooperation requires observing other agents, state space can grow exponentially in the number of agents.
- **Algorithm:** Each agent runs **one-step Q-learning** and updates its own Q(s, a) independently; cooperation is implemented by sharing observations, trajectories, or policies within this Q-learning framework.

---

## What you can use (for your project)

- **Source for IQL:** Your “Independent Q-learning (IQL)” can be directly attributed to Tan (1993): one agent per block, each doing Q-learning, treating other blocks as part of the environment—the classic “independent agents” setup.
- **Independent vs centralized:** Tan compares independent vs cooperative; you compare IQL (independent, local observation) vs Centralized DQN (full observation, joint action). You can frame this as the same conceptual line extended to budget-constrained, block-level allocation.
- **Partial observability and state space:** Tan discusses how observing other agents can blow up state space. You can use this to justify giving each block only local features plus shared budget in IQL, and why Centralized DQN needs a factorized representation when the number of blocks is large.
- **Experiment design:** They use multiple seeds and report before/after convergence; you can similarly use multiple seeds, learning curves, and a final performance table.
- **What not to reuse directly:** Their domain is hunter–prey grid with no budget constraint; yours is block-level REUSE/FULL with a per-step budget K. The problem differs, but the IQL definition and motivation align for citation.
