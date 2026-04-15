# Research Targets — Agent 清单与核心算法伪码

本文档汇总仓库**参考实现**（`old_versions/` 下训练脚本与 checkpoint 命名）中出现的 **所有可评估策略 / Agent**。**不按工程目录分期**，而按**底层学习算法**分组。伪码描述**与环境交互 + 学习更新**的核心逻辑，省略日志、CSV、设备细节。将全检测离线轨迹记作 **REF**（原实现中的 teacher 对齐对象）。

---

## 1. 非学习 Agent（规则策略）

| ID | 名称 | 说明 |
|----|------|------|
| `always_full` | Always FULL | 每步执行 FULL_DETECT |
| `periodic_n` | Periodic FULL | 每 `n` 步一次 FULL，其余步依实现（见 `old_versions/stage_1/evaluation/baselines.py`） |
| `flow_only` | Flow-only | 每步 LIGHT_UPDATE |

### 1.1 伪码：Always FULL

```text
对每个时间步 t:
  a_t ← FULL_DETECT
  执行 env.step(a_t)
```

### 1.2 伪码：Periodic FULL（周期 n）

```text
对每个时间步 t:
  若 (t mod n) == 0:  a_t ← FULL_DETECT
  否则:              a_t ← REUSE   // 或实现约定的非 FULL 动作
  执行 env.step(a_t)
```

### 1.3 伪码：Flow-only

```text
对每个时间步 t:
  a_t ← LIGHT_UPDATE
  执行 env.step(a_t)
```

---

## 2. 按底层算法分组的总览

**共同环境**：`FishTrackingEnv`；**动作**：`{FULL_DETECT, LIGHT_UPDATE, REUSE}`。除 **G6（C51）** 与 **G5（SoftQ）** 外，标量 Q 的 bootstrap 均采用 **Double DQN**（在线网选 `argmax_a Q(s',a)`，目标网取值）。

**观测**：可为单帧 **10 维**；或最近 **k** 帧拼接的 **10k 维** `φ(·)`（常见 `k=4`）。**REF**：若加载离线全检测轨迹，奖励中常含与 REF 的 IoU；否则可为光流一致性等。

| 算法组 | 底层核心 | 参考实现中的命名 / 权重示例 |
|--------|-----------|---------------------------|
| **G0** | 无学习，固定规则 | 见 §1 |
| **G1** | **Vanilla DQN**：浅层 MLP，**无 Dueling**；目标端 **max 式** bootstrap（非 Double） | 历史 `vanilla` / `architecture=vanilla` ckpt |
| **G2** | **Dueling + Double DQN + 均匀回放 + 1-step TD**（Huber） | `train_dqn`（10 维）；`train_stage2`（堆叠 `stack_k`、可选消融）；`dqn_stage3_sk{1,2,4,8}_ablnone` 等（**仅** `stack_k`/超参不同）；`dqn_stage4_baseline62_*`、`ssf05_*`、`epstail_*`、`ioufloor_*`（**仅**奖励塑形、ε 或 λ 调度不同，**TD 与网络族仍属 G2**） |
| **G3** | **G2 骨干 + n-step 回报**（replay 中存 n 步折扣和，`γ_bootstrap = γ^n`） | `dqn_stage4_nstep3_*`、`dqn_stage4_combo_ssf_n3_*`（后者 = n-step + ssf 塑形） |
| **G4** | **G2 标量 Q + SumTree PER + 重要性采样权重** + Double 目标 | `dqn_stage5_per_*` |
| **G5** | **G2 网络形状 + 软熵式 bootstrap**：`y = r + γ(1−d)·τ·logsumexp(Q_θ⁻/τ)` | `dqn_stage5_softq_*` |
| **G6** | **C51**：每动作在支撑上的分布；**投影贝尔曼** + 交叉熵；动作用期望 Q 贪心 | `dqn_stage5_c51_*` |
| **G7** | **GRU** 读入 `(k×10)` 序列 → 隐状态 → **Dueling 标量 Q** + **G2 同款** Double 1-step | `dqn_stage5_gru_*` |

实现索引：`old_versions/stage_1/training/train_dqn.py`（G1/G2 单帧）、`old_versions/stage_2/training/train_stage2.py`（G2/G3）、`old_versions/stage_4/tools/sweep_stage4.py`（G2/G3 变体标签）、`old_versions/stage_5/training/train_stage5.py`（G4–G7）。

---

## 3. 共享符号

- `Q_θ`, `Q_θ⁻`：在线 / 目标网络；`ε`：ε-greedy。
- `r_t`：一步奖励；`γ`：折扣。
- **Double DQN 标量目标**：`a' ← argmax_a Q_θ(s', a)`，`y ← r + γ(1−done) Q_θ⁻(s', a')`。
- **Huber**：对 `(q−y)` 的 SmoothL1。

---

## 4. G1：Vanilla DQN（8 维等）

兼容旧权重；**无 Dueling**。

```text
初始化 Q_θ, Q_θ⁻ ← Q_θ, 均匀 Replay B
for 每个环境步:
  以 ε 在 Q_θ(s) 上 greedy 或均匀随机选 a
  (s', r, done) ← env.step(a); B.push(s,a,r,s',done); s ← s'
  若该更新步:
    采样 batch
    y_i ← r_i + γ(1−d_i) · max_a' Q_θ⁻(s'_i, a')
    最小化 Huber(Q_θ(s_i,a_i), y_i)；周期性 θ⁻ ← θ
```

---

## 5. G2：Dueling + Double DQN + 均匀回放 + 1-step

**观测**可为 `s`（10 维）或 `s_vec = φ`（10k 维拼接）；其余相同。

```text
初始化 Dueling Q_θ, Q_θ⁻, 均匀 B
for step = 1 … T:
  a ← ε-greedy(Q_θ, 当前观测, ε_step)
  (s', r, done) ← env.step(a)
  B.push(当前观测, a, r, 下一观测, done)
  更新观测；若 done 则 reset；可选按 episode 平均 cost 更新 λ

  若 step ≥ learning_starts 且该更新步:
    均匀采样 (s_i, a_i, r_i, s'_i, d_i)
    a'_i ← argmax_a Q_θ(s'_i, a)
    y_i ← r_i + γ(1−d_i) · Q_θ⁻(s'_i, a'_i)
    L ← mean_i Huber(Q_θ(s_i,a_i), y_i)
    θ ← Adam(∇L)；梯度裁剪
    Polyak 软更新或每 K 步硬拷贝 θ⁻ ← θ
```

### 5.1 与 G2 同构、仅训练外围不同的变体（仍属 G2）

以下 **不改变** 上段「均匀采样 + Double + 1-step」的数学形式，只改环境奖励、探索率或 λ 调度：

- **ssf 塑形**：`r ← r − coeff · ssf_norm`（距上次 FULL 的归一化步数）。
- **ε 尾部退火**：主线性衰减结束后再线性降到 `ε_tail`。
- **λ 地板**：episode 末若 `mean IoU(REF) < θ_below`，将 `λ` 抬至至少 `λ_floor`。

```text
// 仅概念：一步奖励进入 replay 之前
r_t ← r_base(·) − [可选] coeff · ssf_norm_t
// episode 末可选
若 IoU_ep(REF) < θ_below: λ ← max(λ, λ_floor)
```

---

## 6. G3：n-step +（G2 同款）Double Dueling

在写入 `B` 前经 **n-step 桥**：存 `(s_0, a_0, G, s_n, d_n)`，其中 `G = Σ_{i=0}^{n−1} γ^i r_i`，`γ_bootstrap = γ^n`。

```text
维护长度 ≤ n 的单步队列
每凑满 n 步或 episode 结束:
  emit (s_0, a_0, G, s_n, d_n) 到均匀 B

训练步:
  y ← r_batch + γ^n (1−d_batch) · Q_θ⁻(s'_batch, argmax_a Q_θ(s'_batch,a))
  L ← Huber(Q_θ(s,a), y)
```

---

## 7. G4：PER + Double Dueling（标量 1-step）

**采样** `P(i) ∝ (p_i)^α`；**IS 权重** `w_i ∝ (N·P(i))^{−β}`。

```text
初始化 PER 缓冲（SumTree）
环境交互同 G2，转移写入 PER

训练步:
  按优先级采样得 (s_i,a_i,r_i,s'_i,d_i, idx_i, w_i)
  y_i ← Double DQN 标量目标（同第 5 节）
  L ← mean_i w_i · Huber(Q_θ(s_i,a_i), y_i)
  用 |TD_i| 更新 idx_i 的优先级
```

---

## 8. G5：软熵式 bootstrap（Dueling 标量 Q）

**探索**仍为 ε-greedy 于 `Q_θ`；**备份**用目标网的 log-sum-exp。

```text
q_sa ← Q_θ(s,a)
a' ← argmax_a' Q_θ(s', a')
V_soft ← τ · logsumexp_a' ( Q_θ⁻(s', a') / τ )
y ← r + γ(1−d) · V_soft
L ← Huber(q_sa, y)
```

---

## 9. G6：C51（分布式 + 投影）

**动作**：`a ← argmax_a E[Q|s,a]`（期望 Q 由原子支撑加权）。

```text
p_sa ← softmax(logits_θ(s,a))
a' ← argmax_a' E[Q_θ(s', a')]
p_next ← softmax(logits_θ⁻(s', a'))
m ← Project_C51(p_next, r, d, support, γ)
L ← − Σ_j m_j log p_sa,j
```

---

## 10. G7：GRU + Dueling 标量 Q + Double 1-step

将 **10k 维**向量 reshape 为 `(k, 10)`，GRU 取最后隐状态，再接 Dueling MLP；**TD 与第 5 节相同**。

```text
h ← GRU_last( reshape(s_vec, (k,10)) )
Q_θ(s,·) ← DuelingHead(MLP(h))
其后与第 5 节的 Double + Huber 更新一致
```

---

## 11. 评估时「玩家」与 REF

- **同一视频列表、同一贪心（或约定随机）、同一 λ 协议**下，报告 **mean IoU vs. REF** 与 **mean cost**。
- **REF 不参与动作**：仅对齐与/或奖励；与 `project_proposal.md` 表述一致。

---

## 12. 命名对照（可自行统一）

- **REF** ↔ 代码中 `teacher_npz`、`episode_mean_iou_teacher` 等。
- **胜负规则**（如 IoU ≥ τ 下最小 cost）见 `project_proposal.md` 第 4 节；本文只刻画**算法分组**。
