# Final Report References Collection

> Topic context: RL-based fish tracking / lightweight-vs-full detection control.

## 1) Academic_Papers_Classic_Foundational (4)

1. **Q-learning** — Watkins, C. J. C. H., & Dayan, P. (1992). *Q-learning*.  
   Link: https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf
2. **TD-Gammon / TD learning impact** — Tesauro, G. (1995). *Temporal Difference Learning and TD-Gammon*.  
   Link: https://www.bkgm.com/articles/tesauro/tdl.html
3. **Policy Gradient Theorem (REINFORCE family)** — Sutton, R. S., et al. (1999). *Policy Gradient Methods for Reinforcement Learning with Function Approximation*.  
   Link: https://papers.nips.cc/paper_files/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf
4. **Deep Q-Network (DQN)** — Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. Nature.  
   Link: https://www.nature.com/articles/nature14236

## 2) Academic_Papers_Contemporary (5)

1. **Double DQN** — van Hasselt, H., Guez, A., & Silver, D. (2015). *Deep Reinforcement Learning with Double Q-learning*.  
   Link: https://arxiv.org/abs/1509.06461
2. **Prioritized Experience Replay** — Schaul, T., et al. (2015). *Prioritized Experience Replay*.  
   Link: https://arxiv.org/abs/1511.05952
3. **Dueling Network Architecture** — Wang, Z., et al. (2016). *Dueling Network Architectures for Deep Reinforcement Learning*.  
   Link: https://arxiv.org/abs/1511.06581
4. **Rainbow DQN** — Hessel, M., et al. (2018). *Rainbow: Combining Improvements in Deep Reinforcement Learning*. AAAI.  
   Link: https://ojs.aaai.org/index.php/AAAI/article/view/11796
5. **Distributional RL (C51)** — Bellemare, M. G., Dabney, W., & Munos, R. (2017). *A Distributional Perspective on Reinforcement Learning*.  
   Link: https://arxiv.org/abs/1707.06887

## 3) Community_Discourse (3)

1. **OpenAI Spinning Up (community-used practical guide)**  
   Link: https://spinningup.openai.com/
2. **Farama Gymnasium migration discussion (API behavior + common pitfalls)**  
   Link: https://gymnasium.farama.org/main/introduction/migration_guide/
3. **r/reinforcementlearning (resource sharing, replication discussions)**  
   Link: https://www.reddit.com/r/reinforcementlearning/

## 4) Engineering_Blogs_Industry_Deep_Dives (4)

1. **Stable-Baselines3 documentation (industry-used RL engineering docs)**  
   Link: https://stable-baselines3.readthedocs.io/en/master/
2. **Google Research Football RL benchmark engineering notes**  
   Link: https://research.google/pubs/google-research-football-a-novel-reinforcement-learning-environment/
3. **Hugging Face Deep RL Course (engineering-focused implementation walkthroughs)**  
   Link: https://huggingface.co/learn/deep-rl-course/unit0/introduction
4. **CleanRL docs (single-file reproducible RL implementations)**  
   Link: https://docs.cleanrl.dev/

## 5) Optional_Bonus_Academic_Paper (2)

1. **Deep Recurrent Q-Network (partial observability, sequence modeling)** — Hausknecht, M., & Stone, P. (2015).  
   Link: https://arxiv.org/abs/1507.06527
2. **MOT benchmark foundation** — Leal-Taixé, L., et al. (2015). *MOTChallenge 2015: Towards a Benchmark for Multi-Target Tracking*.  
   Link: https://arxiv.org/abs/1504.01942

## 6) Source_Code_To_Analyze (5)

1. **Stable-Baselines3** (reliable PyTorch RL baselines)  
   Link: https://github.com/DLR-RM/stable-baselines3
2. **CleanRL** (minimal, readable, reproducible RL training scripts)  
   Link: https://github.com/vwxyzjn/cleanrl
3. **Dopamine (Google)** (DQN-family research framework)  
   Link: https://github.com/google/dopamine
4. **RLlib (Ray)** (distributed RL systems engineering)  
   Link: https://github.com/ray-project/ray/tree/master/rllib
5. **OpenAI Baselines** (historical RL reference implementations)  
   Link: https://github.com/openai/baselines

## 7) Standards_Protocol_Specs (4)

1. **Gymnasium Env API spec (`reset/step` contract)**  
   Link: https://gymnasium.farama.org/api/env/
2. **Gymnasium terminated/truncated Step API rationale**  
   Link: https://farama.org/Gymnasium-Terminated-Truncated-Step-API
3. **MOTChallenge evaluation protocol (TrackEval MOTChallenge official docs)**  
   Link: https://github.com/JonathonLuiten/TrackEval/tree/master/docs/MOTChallenge-Official
4. **COCO dataset format & evaluation API (for detection/tracking metric alignment)**  
   Link: https://cocodataset.org/#format-data