# Reference Views — 四篇参考文献摘要与可借鉴点

针对 proposal 中的自适应计算分配（块级 REUSE/FULL、预算约束、合成环境、DQN + 启发式基线）项目，对每篇文献做简要总结，并标出可直接用于本项目的部分。

---

## 1. Bolukbasi et al., ICML 2017 — Adaptive Neural Networks for Efficient Inference

### 写了什么

- **问题**：深度网络在测试时计算成本高，希望在不（或少量）损失精度的前提下减少单样本推理时间。
- **思路**：不重设计或压缩网络，而是**按样本自适应地使用**已有网络。
  - **方案一（early exit）**：在每层前学一个二分类策略——当前样本是“继续到下一层”还是“从当前层提前退出、用简单分类器直接预测”。简单样本早退出，难样本走完全程。
  - **方案二（network selection）**：多个预训练网络（不同 cost/accuracy）组成 DAG，每节点学一个 exit policy：是用当前模型预测，还是把样本送到更贵的下游网络。从最便宜的模型开始，按需跳到更贵的模型。
- **方法**：把“学自适应 early exit / 选网络”写成**策略学习**，再**约简为逐层加权二分类**（layer-by-layer weighted binary classification），用分类器而非显式 RL 求解。
- **实验**：ImageNet 上 early exit 带来 20–30% 加速，network cascade 相对 ResNet50 约 2.8× 加速、top-5 损失 <1%。
- **相关**：文中提到与“conditional computation”“spatially adaptive networks”相关，并引用 Bengio et al. 2015。

### 你可以用到的点

- **动机与 related work**：在引言/相关工作里引用“自适应推理、按样本选择计算量”，作为你 block 级自适应计算分配的大背景。
- **“easy vs. hard 样本”类比**：他们用“简单样本少算、难样本多算”；你可类比为“变化小的块 REUSE、变化大的块 FULL”，用于写问题动机。
- **策略形式**：他们是逐层二值决策（继续/退出）；你是逐块二值决策（REUSE/FULL），可对比说明你的是**空间上**的稀疏计算，他们是**深度上**的。
- **系统/部署**：文中提到 edge–cloud、fog 部署（简单样本在设备、难样本上云），若你写“实时视频推理、资源受限”时可顺带引用。
- **不直接用的**：他们用分类器做策略、不是 DQN；你用的是 MDP + DQN/启发式，方法上可区分开。

---

## 2. Bengio et al., ICLR 2016 — Conditional Computation in Neural Networks for Faster Models

### 写了什么

- **问题**：深度模型训练和评估都费时，希望用**条件计算**（只激活网络的一部分）在保证精度的前提下减少计算。
- **思路**：按**输入**决定每层哪些单元/块参与计算（input-dependent activation），即“看什么输入就开哪部分网络”，类似可学习的、与数据相关的 dropout。
- **形式化**：把“学每层激活/关闭哪些单元”写成 **MDP**：状态 = 前一层的激活向量，动作 = 该层的二值 mask（哪些单元开），代价 = 主网络损失（如负对数似然）。单步 MDP：看输入 → 选 mask → 得到 loss。
- **方法**：每层一个 **sigmoid-Bernoulli 策略**（由上一层激活 + 参数得到参与概率），用 **REINFORCE** 学策略参数；加稀疏/方差正则，鼓励“少激活、且不同样本激活不同单元”。
- **关键词**：Conditional computation, REINFORCE, 与 dropout 对比（dropout 是 data-independent，他们是 data-dependent）。

### 你可以用到的点

- **MDP 建模**：他们明确把“选哪些单元算”写成 MDP（状态、动作、代价），你可以直接引用“条件计算/稀疏计算用 MDP + RL 建模”作为你块级 MDP（REUSE/FULL）的理论同类工作。
- **奖励/代价设计**：他们同时最小化**预测误差**和**参与单元数**（计算量）；你的是质量代理（drift）与计算成本权衡，可类比写进 related work。
- **策略形式**：他们是每层一个策略、Bernoulli；你是每块 REUSE/FULL，也是离散二值决策，可对比“空间块 vs 层内单元”的粒度差异。
- **算法**：他们用 policy gradient（REINFORCE）；你用 DQN，可在方法部分说明“同类问题有人用 policy gradient，本项目用 value-based DQN 便于与启发式对比”。
- **不直接用的**：他们是层内单元级、单步 MDP；你是多步、块级、带预算约束的 MDP，环境和设定不同。

---

## 3. Tan, ICML 1993 — Multi-Agent Reinforcement Learning: Independent vs. Cooperative Agents

### 写了什么

- **问题**：在多智能体设定下，**独立学习**（不通信）和**合作学习**（共享信息/经验/策略）谁更好、合作代价多大。
- **三种合作方式**：(1) 共享即时信息（sensation）；(2) 共享 episode（s, a, r 序列）；(3) 共享学到的策略。
- **设定**：多个 RL agent（Q-learning，Boltzmann 探索）在网格里抓 prey；每个 hunter 局部观测（有限视野），可独立或合作。
- **结论**：合作能加速学习或提高最终表现，但依赖“信息是否充分、是否用得起来”；联合任务下合作 agent 能超过独立 agent，但前期可能学得慢；若合作需观测其他 agent，状态空间会随 agent 数指数增长。
- **算法**：每个 agent 用 **one-step Q-learning**，独立更新自己的 Q(s, a)；合作时在 Q-learning 框架下共享观测/轨迹/策略。

### 你可以用到的点

- **IQL 的出处**：你 proposal 里的 “Independent Q-learning (IQL)” 可直接引用 Tan (1993)：每个 block 一个 agent、各自 Q-learning、把其他 block 当环境的一部分，即“independent agents”的经典设定。
- **独立 vs 中心化**：Tan 比较独立 vs 合作；你比较 IQL（独立、局部观测）vs Centralized DQN（全观测、联合动作），可写成“同一脉络下的扩展：从独立多 agent 到带预算约束的块级分配”。
- **局部观测与状态空间**：Tan 讨论“合作时观测别的 agent 会导致状态空间爆炸”；你可用来解释为何 IQL 只给每个 block 局部特征 + 共享 budget，以及为何中心化 DQN 在块数多时要 factorized。
- **实验设计**：他们用多 seed、收敛前后分别测；你可类比“多 seeds、学习曲线 + 最终性能表”。
- **不直接用的**：他们是 hunter–prey 网格、无预算约束；你是块级 REUSE/FULL、有每步预算 K，问题不同，但 IQL 的定义和动机可一致引用。

---

## 4. Mnih et al., Nature 2015 — Human-level control through deep reinforcement learning

### 写了什么

- **问题**：RL 以往多在低维、 handcrafted 特征或全观测空间；如何从**高维感官输入（如像素）**直接学控制策略。
- **方法**：**Deep Q-Network (DQN)**——用深度卷积网络近似最优动作价值 Q(s,a)；Q-learning 更新 + **experience replay**（从 replay buffer 均匀采样，打破时序相关）+ **target network**（目标 Q 周期性从当前网络拷贝，减少与目标的相关性），使用神经网络做 Q 近似时稳定。
- **实验**：Atari 2600，49 款游戏，同一网络/超参；仅用像素和分数，在多数游戏上超过之前方法并接近人类水平。
- **公式**：Q-learning 损失、replay、target 更新步数 C 等，文中都有明确写出。

### 你可以用到的点

- **Centralized DQN 的基准实现**：你的“单 agent、全状态、联合动作（每块 REUSE/FULL）”的 DQN 可直接引用 Mnih et al. 2015 作为 DQN 的标准来源；实现时采用 **experience replay + target network**（以及你已写的 double DQN）以保证稳定。
- **网络与 Q 形式**：他们用 CNN 处理图像；你用 MLP 或小网络处理“块特征向量 + budget”即可，不必照搬 CNN，但“用深度网络近似 Q(s,a)”这一点一致，可写“following Mnih et al.”。
- **训练稳定性**：文中讨论的不稳定原因（序列相关、数据分布变化、Q 与 target 相关）以及 replay + target 的解决办法，你可以在方法或实验里简要引用，说明你为何用 replay、target、double DQN。
- **实验与指标**：他们画学习曲线（平均回报、平均 Q）；你可同样画“平均 return、平均 drift、每步 FULL 数”等，并在文中说明“学习曲线与稳定性分析参考 DQN 常见做法（e.g. Mnih et al.）”。
- **不直接用的**：他们是单 agent、Atari；你是多块/多步、合成环境、预算约束，环境和动作空间设计不同，但算法骨架一致。

---

## 小结表

| 文献 | 核心内容 | 你可直接用的 |
|------|----------|----------------|
| Bolukbasi et al. 2017 | 自适应 early exit / 选网络，按样本省计算 | 动机、related work、easy/hard 类比、系统场景 |
| Bengio et al. 2016 | 条件计算，MDP + REINFORCE 学激活 | MDP 建模思路、奖励含“质量+计算”、与 DQN 方法对比 |
| Tan 1993 | 多 agent 独立 vs 合作，IQL 设定 | IQL 定义与引用、独立 vs 中心化对比、局部观测与状态爆炸 |
| Mnih et al. 2015 | DQN：replay + target，高维输入 | Centralized DQN 实现依据、稳定性设计、实验指标写法 |
