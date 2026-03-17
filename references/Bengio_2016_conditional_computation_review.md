# Bengio et al., ICLR 2016 — Conditional Computation in Neural Networks for Faster Models

**Citation:** E. Bengio, P.-L. Bacon, J. Pineau, and D. Precup. Conditional computation in neural networks for faster models. *ICLR*, 2016 (arXiv:1511.06297, 2015).

---

## What the paper is about

- **Problem:** Training and evaluating deep models is costly; the goal is to reduce computation while preserving accuracy via **conditional computation** (activating only part of the network).
- **Idea:** Decide **per input** which units (or blocks) in each layer participate (input-dependent activation). Only a subset of the network is used for each example, like a learned, data-dependent dropout.
- **Formalism:** “Which units to activate per layer” is modeled as an **MDP**: state = activations from the previous layer, action = binary mask over units in the current layer, cost = main network loss (e.g. negative log-likelihood). Single-step MDP: observe input → choose mask → incur loss.
- **Method:** A **sigmoid-Bernoulli policy** per layer (participation probabilities from previous-layer activations + parameters) is learned with **REINFORCE**. Sparsity and variance regularizers encourage few activations and diversity across examples.
- **Keywords:** Conditional computation, REINFORCE; contrast with dropout (data-independent vs their data-dependent activation).

---

## What you can use (for your project)

- **MDP framing:** They explicitly model “which units to compute” as an MDP (state, action, cost). You can cite “conditional / sparse computation formulated as MDP + RL” as related work for your block-level MDP (REUSE / FULL).
- **Reward / cost design:** They minimize both **prediction error** and **number of active units** (compute). Your reward trades off a quality proxy (drift) and compute cost; the same trade-off idea fits well in related work.
- **Policy form:** They use one policy per layer with Bernoulli actions; you use per-block REUSE/FULL (discrete binary). You can compare granularity: spatial blocks (yours) vs units within layers (theirs).
- **Algorithm contrast:** They use policy gradient (REINFORCE); you use value-based DQN. In the method section you can note that “similar problems have been tackled with policy gradient; we use DQN for easier comparison with heuristics.”
- **What not to reuse directly:** Their setting is unit-level within layers and single-step MDP; yours is multi-step, block-level, and budget-constrained, so the environment and constraints differ.
