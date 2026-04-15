# Research Targets — RL + CV 自适应计算分配（鱼追踪）

本文档汇总本项目的**研究目标**、**阶段定位**、**已用实验说明的结论**、**已知局限**与**可拓展方向**，便于写报告、答辩或与 `proposal.md` / `rl_cv_project.md` 对照。

---

## 1. 核心研究目标

1. **问题**：在固定机位视频上，每帧跑全量检测（如 YOLO）成本高；许多帧可用光流或复用上一估计。
2. **方法**：将「何时做 FULL、何时 LIGHT/REUSE」建模为 **MDP**，用 **Dueling + Double DQN** 学习策略。
3. **成功标准（工程上）**：
   - 在**可接受的追踪质量**（与全检测或 teacher 轨迹的可比指标）下，
   - **降低平均动作代价**（`mean_cost`）或等价计算量，
   - 并与 **非学习 baseline**（always FULL、周期 FULL、纯光流）可对照。

**注意**：「质量」在实现中有两种信号——**有 teacher 时**可用 `mean_iou_teacher`；**无 teacher 的部署路径**应以 **flow consistency + cost** 为主，teacher 仅作可选诊断。

---

## 2. Stage 1 与 Stage 2 的定位

| 维度 | Stage 1 | Stage 2 |
|------|---------|---------|
| **底层环境** | `FishTrackingEnv`（同一套物理转移与奖励逻辑） | 相同 |
| **动作空间** | 3 动作：FULL / LIGHT / REUSE | **相同** |
| **单步语义状态** | 10 维（`STATE_DIM`） | 每帧仍是这 10 维（经可选消融掩码） |
| **策略网络输入** | 10 维 | **堆叠**最近 `k` 帧（默认 `k=4` → **40 维**），部分可观测性由历史缓解 |
| **算法骨架** | DQN 族 + replay + target | 同左，超参可有 `stage2_*` preset |

**结论（已由实验支持）**：在相同视频、对齐的 λ 与评估协议下，**Stage 2 在 Stage 1 上叠加时序观测是可行的**；贪心评估中相对 teacher 的 IoU 与 FULL 使用率可优于单帧 Stage 1（见 `stage_2/outputs/eval_all_methods_align.csv` 一类对比表）。

---

## 3. 奖励与数据依赖（与「新视频 / teacher」的关系）

- **部署（新视频、无 teacher）**：状态仍可由视频与在线检测算得；奖励可走 **光流一致性 − λ·cost**，**不要求**先 `preprocess` 生成 teacher。
- **训练期若要对齐全检测轨迹**：可对**该训练视频**生成离线 teacher `.npz`，奖励改为 **IoU(pred, teacher) − λ·cost`**（teacher **不进 state**）。
- **仅想报告与 teacher 的差距**：评估时传入 `--teacher-npz` 即可；与是否用 teacher 训练独立。

---

## 4. 评估与已知现象（写进 Limitation / Discussion）

### 4.1 评估协议要点

- **随机起点**（默认 DQN 段）：每 episode 用固定 seed 列表在合法帧区间抽**起始帧**，再跑至多 `max_episode_steps`（默认 200）或至视频末尾；**不是**「从随机起点必跑满整段视频」。
- **固定起点**（`--dqn-fixed-start`）：从环境 `_min_start` 起，便于复现、对照。
- **贪心**：与训练末期 **ε-greedy** 分布不同，可能出现 **train–deploy gap**（例如贪心下长期不 FULL、teacher IoU 偏低，而训练滑动均值仍较好）。

### 4.2 已观察到的构造层面问题（非「eval 实现错误」）

- **自适应 λ** 若只追随平均 cost，可能压到 **`lambda_min`**，策略强烈偏向省 FULL。
- **光流 consistency** 与 **teacher IoU** 可能脱节：短期 LIGHT/REUSE 仍可能维持较高 consistency，但 teacher 质量在贪心轨迹上崩溃。
- **缓解手段（已实现或可选）**：提高 `lambda_min` / `target_mean_cost`；评估侧 **`--dqn-epsilon-eval`**（默认 0）模拟偶发纠偏；后续可在奖励中加入「久未 FULL」惩罚或 λ 与质量下界联动（见下文拓展）。

---

## 5. 数据与泛化（当前项目边界）

- **当前主要证据**：单条固定机位鱼视频上的训练、评估与 Stage1/2 对比。
- **无第二条实拍视频时**：仍可通过 **多 seed、多 episode、固定/随机起点、Stage1 vs Stage2、Pareto 多 λ** 加厚实验；在文中明确 **「未在额外视频上验证跨场景泛化」** 作为局限与未来工作即可。

---

## 6. 拓展方向（按投入大致排序）

### 6.1 低成本（偏实验与写作）

- **实现编排**：仓库新增 **`stage_3/`**（见 `stage_3/README.md`），提供 `sweep-stack-k` / `sweep-ablation` / `sweep-grid` / `collect`，用子进程批量调用 Stage2 训练并汇总 `train_metrics_*.csv`。
- **系统扫 `stack_k`**（如 1 / 2 / 4 / 8）与 **已有 ablation**（`no_iou` 等），固定其余超参，制表 + 简短讨论。
- **完善 Pareto**：`stage_2.tools.run_pareto` 多固定 λ，报告 **cost–consistency（及可选 teacher）** 前沿。
- **补充 baseline**：不同周期 FULL、或基于置信度的简单规则。
- **报告**：Related work、可复现命令、指标列含义、局限与伦理（误检对生态监测的影响等，若课程需要）。

### 6.2 中等成本（仍属同一架构）

- **奖励 / λ**：对 `steps_since_full` 加惩罚，或当滑动 teacher IoU 低于阈值时阻止 λ 继续下降（需有 teacher 或代理信号）。
- **N-step TD**（如 n=3）或训练末期 **ε 退火**，减轻贪心 gap（实现量适中）。
- **实现与编排**：仓库新增 **`stage_4/`**（见 `stage_4/README.md`）；训练开关在 **`stage_2.training.train_stage2`**（`--ssf-reward-penalty`、`--lambda-teacher-iou-floor`、`--n-step`、`--epsilon-tail` 等），环境侧惩罚在 **`FishTrackingEnv`**。

### 6.3 高成本：策略梯度 / Actor–Critic（与 DQN 对照的研究主线）

在 **同一 `FishTrackingEnv`、同一离散 3 动作、同一评估协议** 下，增加 **Policy Gradient / Actor–Critic** 类方法，与现有 **Dueling + Double DQN** 做公平对照。本节作为 **research 延伸** 的优先落点；实现可逐步落地（不必与 Stage1–4 同步完成）。

#### 6.3.1 PPO（首选）

- **适用性**：离散 3 动作；有 **GPU/MPS** 时训练稳定、资料与参考实现多，**叙事清晰**（on-policy + clipped surrogate + 可选 value baseline）。
- **优点**：与 value-based DQN **互补**；易报告「同环境、不同学习范式」的差异（return、mean_cost、teacher IoU、FULL 比例等）。
- **代价**：需 **新训练循环**（rollout 收集、多 epoch 小批更新）、与现有 **replay / target 网络** 接口不同；**超参**（clip ε、horizon、epochs、GAE λ、entropy 系数）需小网格。
- **公平对比**：与 DQN 对齐 **总环境步数或总采样步数**、**`random_start` / `max_episode_steps`**、**eval 贪心协议**（及可选 `--teacher-npz`）；建议固定 **同一条训练视频 + 同 seed 列表** 再比分布，避免「步数不等」带来的虚假优劣。

#### 6.3.2 SAC / TD3（次选 / 对照）

- **适用性**：经典设计偏 **连续动作**；本任务为 **离散 3 动作**。可做 **Gumbel-Softmax / 直接三 logits 的随机策略 + 重参数化式熵** 等离散化版本，但 **工程与调参性价比一般**，更适合作为 **「连续松弛或附录一条曲线」**，而非主文唯一替代算法。
- **何时值得做**：课程或审稿明确要求 **随机策略 + 熵正则**、或希望与 **SAC 的稳定性叙事** 挂钩时再上；否则在文中 **一句 limitation + 引用** 即可。
- **公平对比**：若实现离散 SAC，仍须与 **6.3.1** 及 DQN **同一 eval 脚本输出列**（`mean_return`、`mean_cost`、`mean_consistency`、`mean_iou_teacher`），便于与 `plot_eval_comparison` 一类表格合并。

#### 6.3.3 仍可作为并行 Future Work（非算法主线的短句）

- **约束 RL**（显式 cost 或质量约束下的策略优化，可与 PPO 的拉格朗日/惩罚项结合叙述）。
- **更强 backbone 或多档算力** 下的策略迁移。
- **第二条视频或合成扰动** 上的重复管线（若有数据再开展）。

### 6.4 同一项目上的「较新」价值函数增量（相对当前 DQN 栈）

**与仓库对齐的基线**：`FishTrackingEnv`、离散 3 动作、`stage_1.models.dqn` 的 **Dueling + Double**、Stage 2 的 **帧堆叠 `stack_k`**（或等价展平输入）、`stage_2.training.train_stage2` 里已可选 **`--n-step`**（§6.2 已占用的拼图不再重复当作「新方向」）。本节只列 **仍走 replay + Q 网络、改网络或采样机制**、且 **直接回答本任务现象**（长尾漂移、贪心 gap、部分可观测）的增量；**不写离线 DT / 世界模型** 等与当前管线距离过大的条目。

#### 6.4.1 优先经验回放（PER）

- **对应问题**：纠错型转移（例如刚 REUSE 后被迫 FULL）稀疏，均匀采样下 **有效梯度比例低**。
- **改哪里**：在现有 `ReplayBuffer` 上增加 **TD-error 优先采样 + 重要性采样校正**；环境、`STATE_DIM`、奖励定义 **不变**。
- **对照**：与当前 **均匀采样 + Dueling Double**（及可选 `n_step=3`）固定 **buffer 大小、总梯度步、eval 协议** 比 **return / mean_cost / consistency / teacher IoU**。

#### 6.4.2 分布价值头（C51 或 QR-DQN / IQN）

- **对应问题**：§4 已讨论 **长尾**（偶发大漂移、久未 FULL 后 teacher IoU 断崖）；标量 **Q(s,a)** 只拟合 **期望**，难显式优化 **尾部风险**。
- **改哪里**：替换或并联 **输出分布/分位** 的头，仍用 **离散 argmax / 评估时贪心**；可报告 **某分位下的策略** 或 **CVaR 风格** 的「保守用 FULL」与 cost 的折中。
- **对照**：与 **同结构宽度的 Dueling Double** 比，避免把「参数变多」误当成算法增益。

#### 6.4.3 DRQN：用循环网络替代「手工 `stack_k` 展平」

- **对应问题**：Stage 2 用 **历史帧拼成 40 维向量** 缓解 POMDP；更常见做法是 **LSTM/GRU 吃单帧 10 维序列**，隐藏状态携带历史。
- **改哪里**：`forward` 改为 **序列批** 或 **burn-in + 学习段**（文献中 DRQN 标准套路）；**不增加** FULL/LIGHT/REUSE 语义，只改 **时序压缩方式**。
- **对照**：与 **当前最优 `stack_k` preset**（如 §6.1 扫出的 k）对齐步数与 eval，比较 **同算力下** IoU–cost 与训练稳定性。

#### 6.4.4 离散最大熵 Q（可选，轻量对照 §6.3.2）

- **对应问题**：§4.1 **train–deploy gap**（训练期 ε、评估贪心）；在 **不离线改 PPO/SAC 整套** 的前提下，对 Q 目标加 **熵 bonus**，鼓励早期探索、略平滑贪心极限。
- **改哪里**：在现有 Double 目标上增加 **策略熵项**（三动作 softmax）；实现量 **小于** 全量 SAC，叙事上可与「缓解贪心塌缩」挂钩。

**公平对比备忘**：仍对齐 **`random_start` / `max_episode_steps` / `--teacher-npz` 列**；新跑法汇总进与 `eval_all_methods_*.csv` **同列** 的表，便于 `plot_eval_comparison` 一类脚本合并。

---

## 7. 与仓库内文档的关系

- **`proposal.md`**：偏正式问题陈述与公式，随实现迭代需与代码一致（尤其奖励 hybrid、默认 teacher 行为）。
- **`rl_cv_project.md`**：路线图与 Stage 说明；若与当前实现不一致，以**代码与 `README`** 为准，或逐步同步。
- **`stage_1/README.md` / `stage_2/README.md`**：命令行入口与目录说明。

本文件 **`research_targets.md`** 侧重「**要证明什么、已经证明了什么、还能往哪扩**」，不替代上述技术细节文档。

---

## 8. 关键产物路径（便于写报告时引用）

| 内容 | 路径示例 |
|------|-----------|
| Stage 1 训练指标 | `stage_1/checkpoints/train_metrics_robust.csv`、`train_metrics_robust_v2.csv` |
| Stage 1 权重 | `stage_1/checkpoints/dqn_stage1_robust.pt`、`dqn_stage1_robust_v2.pt` |
| Stage 2 对齐训练 | `stage_2/checkpoints/dqn_stage2_align.pt`、`train_metrics_stage2_align.csv` |
| Stage1+2+基线对比 | `stage_2/outputs/eval_all_methods_align.csv` |
| **Stage3 标准套件（§6.1）** | `stage_3/outputs/sweep_train_summary.csv`、`eval_standard_suite.csv`；权重 `stage_3/checkpoints/dqn_stage3_sk*_ablnone.pt` |
| **Stage4 §6.2 拓展** | `stage_4/README.md`；`python -m stage_4.main standard`；权重 `stage_4/checkpoints/dqn_stage4_*.pt`，汇总 `stage_4/outputs/sweep_train_summary_62.csv` |
| **§6.3 PPO / SAC（研究主线，实现待定）** | 与 `FishTrackingEnv` 及现有 eval 列对齐；落地后在此表增补 ckpt、`eval_*.csv`、训练日志路径 |
| **Stage5 §6.4（PER / Soft Q / C51 / GRU）** | `stage_5/README.md`；`python -m stage_5.main standard` / `evaluate`；权重 `stage_5/checkpoints/dqn_stage5_<mode>_teacher.pt` 与 `_notchr`；评估 `stage_5/outputs/eval_suite_teacher.csv`、`eval_suite_notchr.csv`（**复用** `stage_4/checkpoints/dqn_stage4_baseline62_*.pt` 写入对照行，不重训 Stage1–4） |
| 训练曲线脚本 | `stage_1/tools/plot_train_metrics.py`；评估对比图见 `stage_1.tools.plot_eval_comparison` 等 |

---

*文档随项目迭代可继续增补：例如每次正式跑完新 preset，在本节 8 增加一行路径与一句结论。*
