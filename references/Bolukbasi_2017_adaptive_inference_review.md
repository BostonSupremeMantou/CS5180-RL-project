# Bolukbasi et al., ICML 2017 — Adaptive Neural Networks for Efficient Inference

**Citation:** T. Bolukbasi, J. Wang, O. Dekel, and V. Saligrama. Adaptive neural networks for efficient inference. *ICML*, 2017.

---

## What the paper is about

- **Problem:** Deep networks are expensive at test time; the goal is to reduce per-example inference cost with little or no loss in accuracy.
- **Idea:** Do not redesign or compress the network; instead, **adaptively use** the existing network per example.
  - **Scheme 1 (early exit):** Before each expensive layer, learn a binary policy: either continue to the next layer or exit early and predict with a simple classifier. Easy examples exit early; hard ones go through the full network.
  - **Scheme 2 (network selection):** Several pre-trained networks with different cost/accuracy trade-offs are arranged in a DAG. At each node, an exit policy decides whether to use the current model’s prediction or to forward the example to a more expensive downstream network. Evaluation starts from the cheapest model and steps up only when needed.
- **Method:** The learning of adaptive early-exit or network-selection policies is formulated as **policy learning** and then **reduced to layer-by-layer weighted binary classification**, solved with classifiers rather than explicit RL.
- **Experiments:** On ImageNet, early exit gives 20–30% speedup; the network cascade gives ~2.8× speedup over ResNet50 with &lt;1% top-5 accuracy loss.
- **Related work:** The paper links to “conditional computation” and “spatially adaptive networks” and cites Bengio et al.

---

## What you can use (for your project)

- **Motivation and related work:** Cite “adaptive inference” and “per-example choice of computation” as the broader context for your block-level adaptive computation allocation.
- **Easy vs. hard analogy:** They use “easy examples → less compute; hard examples → full compute.” You can analogize to “low-change blocks → REUSE; high-change blocks → FULL” when motivating the problem.
- **Policy form:** They use per-layer binary decisions (continue / exit); you use per-block binary decisions (REUSE / FULL). You can contrast **spatial** sparse computation (yours) vs **depth**-wise (theirs).
- **Systems / deployment:** They mention edge–cloud and fog (easy on device, hard to cloud). You can cite this when discussing real-time video inference and resource limits.
- **What not to reuse directly:** They use classifiers for the policy, not DQN; your formulation is MDP + DQN/heuristics, so the method is distinct.
