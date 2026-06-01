---
title: Part B（续）：TD-MPC 与规划机制对比
description: TD-MPC 的时序差分混合规划方案、与 DreamerV3 的对比，以及三种规划机制的综合总结。
lecture: 3
---

# Part B（续）：TD-MPC 与规划机制对比

## 机制三：TD-MPC，两者的桥梁

TD-MPC（Temporal Difference Model Predictive Control）[Hansen et al., 2022] 同时拥有 MPC 的前瞻规划能力和 Actor-Critic 的时序差分学习效率。

**核心设计**：

| 组件 | 作用 |
|------|------|
| 潜在一致性损失 | 训练隐式动力学模型：$\hat{z}_{t+1} = f(z_t, a_t)$ 应与编码器输出的 $\text{sg}(z_{t+1})$ 一致 |
| 时序差分目标 | 用 Bellman 方程更新 Q 函数（action-value function，动作价值函数，$Q(s,a)$ 表示"在状态 $s$ 执行动作 $a$、此后遵循策略所能获得的期望累计折扣奖励"）：$Q(z_t, a_t) = r_t + \gamma \cdot Q(z_{t+1}, \pi(z_{t+1}))$，其中 $\gamma$（折扣因子）使未来奖励指数衰减 |
| CEM 规划 | 在每步决策时，用 CEM 在潜在空间中搜索最优动作序列 |

三个组件联合训练：一致性损失塑造潜在空间，TD 目标训练 Q 函数，Q 函数再引导 CEM 搜索。

**stop-gradient 的作用**：一致性损失中的 `sg(z_{t+1})` 表示停止梯度。如果编码器的两端都可以被梯度更新，模型可能学到一个"恒等函数"，把所有状态映射到同一个点，使得一致性损失为零，但毫无意义。stop-gradient 固定住目标端，防止这种**模式坍塌**（mode collapse，模型找到一个退化解：把所有不同的输入都映射到同一个输出，使损失最小化但毫无意义）。

> **📖 Bellman 方程**：$Q(s_t, a_t) = r_t + \gamma \cdot \max_{a'} Q(s_{t+1}, a')$。把无限步的累积奖励问题，转化为只看"一步奖励 + 下一步 Q 值"的形式。**自举（bootstrapping）**：用模型自身的估计（如 $Q(s_{t+1}, a')$）来作为训练目标，"用自己预测自己"。TD 学习用 Bellman 方程做 bootstrapping，使学习可以在每步都进行，无需等到 episode 结束。

TD 学习用 Bellman 方程，以"当前奖励 + 下一步 Q 值估计"代替完整展开，将有效规划深度从"模型精确步数"缩短到"1步 + Q函数的自举"。

**与 DreamerV3 的对比**：

| 维度 | DreamerV3 | TD-MPC2 |
|------|-----------|---------|
| 世界模型形式 | 显式生成（重建像素/观测） | 隐式（只保证价值预测准确） |
| 规划方式 | 潜在空间 Actor-Critic | CEM + TD |
| 适用任务范围 | 视觉复杂任务，需要丰富观测 | 状态观测任务，高效连续控制 |
| 可解释性 | 可以可视化重建 | 潜在空间无直接语义 |

---

## 三种规划机制对比

| 维度 | CEM-MPC | Dreamer Actor-Critic | TD-MPC |
|------|---------|---------------------|--------|
| 规划方式 | 随机搜索 | 策略梯度（可微） | 随机搜索 + TD |
| 是否需要像素重建 | 否 | 是 | 否 |
| 长程规划能力 | 受 $H$ 限制 | 靠 Critic bootstrap | TD + MPC 结合 |
| 计算成本 | 高（大 $N$） | 中（想象展开） | 低-中 |
| 高维动作空间 | 效率低 | 梯度直接优化 | Q函数引导搜索 |
| 模型漏洞风险 | 中（短视） | 高（policy可钻漏洞） | 中（TD抑制累积误差） |
| 典型场景 | 简单连续控制 | 视觉复杂任务 | 高效连续控制 |

---

## 本讲小结

- **七个架构族**代表了突破 GRU 记忆瓶颈的不同方向：RNN/RSSM 计算最轻、Transformer 长程依赖最强、Diffusion 视觉最真实、JEPA 最专注语义、RWM 最专注部署稳定性、Genie 从视频自动发现动作、WAM 把世界预测和动作规划统一。
- **三种学习范式**决定模型的知识边界：观察型学视觉规律但无法控制，交互型学动作因果但数据昂贵，反事实型学价值推理但可解释性弱。WAM 代表第四范式：视频作为 dense physical supervision 联合训练世界和动作。
- **三种规划机制**决定模型如何被用于决策：CEM 最直白但在高维空间效率低，Actor-Critic 最优雅但有模型漏洞风险，TD-MPC 最务实地兼顾了两者。
- Dreamer = 交互型范式 + RSSM + 潜在 Actor-Critic，是本系列课程的核心参考系统。
- TD-MPC = 反事实型范式 + CEM + TD，将在 P04 中亲手实现并与 Dreamer 对比。

---

## 下一讲

构建和运行世界模型之后，下一个问题是：怎么判断它好不好？L04 针对每种架构给出专属的评估指标，Dreamer 的 FID 和奖励相关性、MuZero 的 MCTS 访问熵、TD-MPC 的潜在一致性损失、STORM 的长程 PSNR，以及一个所有模型都会遇到的通用失效模式：**horizon drift（时程漂移）**。

---

## 延伸阅读

本讲涉及的关键论文，按出现顺序排列：

**基础架构**
- [Ha & Schmidhuber (2018): World Models](https://arxiv.org/abs/1803.10122)：V/M/C 三模块框架与梦中训练的原始论文
- [Hafner et al. (2019): PlaNet / RSSM](https://arxiv.org/abs/1811.04551)：确定性+随机双路径潜在动力学模型
- [Hafner et al. (2019/2020/2023/2025): Dreamer V1/V2/V3/V4](https://arxiv.org/abs/1912.01603)：RSSM 与潜在 Actor-Critic 系列；V4 见 [arxiv 2509.24527](https://arxiv.org/abs/2509.24527)

**Transformer 架构**
- [Micheli et al. (2022): IRIS](https://arxiv.org/abs/2209.00588)：VQ-VAE 离散化 + GPT 自回归世界模型，Atari 100k 1.046 HNS
- [Zhang et al. (2023): STORM](https://arxiv.org/abs/2310.09615)：分类 VAE + 单 token Transformer，126.7% HNS，4.3h 训练

**Diffusion 架构**
- [Alonso et al. (2024): Diamond](https://arxiv.org/abs/2405.12399)：扩散世界模型，首次在 Atari 上 FVD 低于真实游戏帧

**规划机制**
- [Schrittwieser et al. (2020): MuZero](https://arxiv.org/abs/1911.08265)：隐式世界模型 + MCTS，围棋与 Atari 超人水平
- [Hansen et al. (2022): TD-MPC](https://arxiv.org/abs/2203.04955)、[TD-MPC2 (2024)](https://arxiv.org/abs/2310.16828)：CEM + TD 混合规划

**JEPA 系列**
- [Assran et al. (2023): I-JEPA](https://arxiv.org/abs/2301.08243)、[Bardes et al. (2024): V-JEPA](https://arxiv.org/abs/2404.08471)：语义空间预测，无像素重建

**Genie / 可交互生成**
- [Bruce et al. (2024): Genie](https://arxiv.org/abs/2402.15391)：从无标注视频自动发现 latent action，11B 参数

**RWM / 机器人部署**
- [Li et al. (2026): RWM-U](https://arxiv.org/abs/2504.16680)：离线 MBRL + 集成不确定性，四足/仿人机器人验证
- [NeurIPS 2025: Self-Forcing](https://arxiv.org/abs/2506.08009)：训练时引入自预测反馈，缓解 teacher forcing 差距

**WAM / 联合学习**
- [Bi et al. (2025): Motus](https://arxiv.org/abs/2512.13030)：统一 latent action 世界模型，从异构视频数据跨具身迁移
- [NVIDIA (2026): WAM](https://arxiv.org/abs/2509.20328)：预训练视频模型作为 zero-shot policy
- [NVIDIA (2025): Cosmos](https://arxiv.org/abs/2501.03575)：通用物理 AI 世界基础模型，开源开权重
- [Hu et al. (2023): GAIA-1](https://arxiv.org/abs/2309.17080)：自动驾驶生成世界模型，视频+文本+动作联合建模
