---
title: TD-MPC 的专属指标
description: 针对潜在 MPC 架构的一致性损失、表示崩塌诊断、latent space 可视化与规划效率评估。
lecture: 4
---

# TD-MPC 的专属指标

## TD-MPC（潜在 MPC）

*你在 P04 用 CEM 射击法实现了 TD-MPC，并与 P03 的 Dreamer Actor-Critic 做了奖励对比。*

TD-MPC 的核心是：编码器 `h = enc(o)` 在不同时间步产生的表示必须**彼此一致**，MPC 才能在潜在空间中进行有效规划。

### 潜在一致性损失（Latent Consistency Loss）

$$\mathcal{L}_{\text{consist}} = \|\text{sg}(h_{t+1}) - f(h_t, a_t)\|^2$$

其中 `f` 是动力学函数，`sg` 表示停止梯度（stop-gradient）。这个损失衡量"动力学函数预测的下一状态"与"编码器直接编码的下一观测"之间的距离。

**诊断规则（表示崩塌）**：如果去掉 `sg` 后一致性损失反而更低，说明编码器已退化为恒等映射，把所有状态压缩到同一个点。正式诊断：在验证集上检查潜在向量的协方差矩阵秩，秩接近 1 即为崩塌（`sg` 机制详见 L03 Part B TD-MPC 一节）。

**诊断规则（训练震荡）**：一致性损失在训练中震荡（不稳定地上下波动）而非单调下降，学习率过高，或编码器和动力学函数的梯度尺度不匹配。尝试将学习率降低一个数量级，或对编码器使用独立（更小的）学习率。

### 潜在空间可视化（Latent Space Visualization）

这是一个强大的定性诊断工具，可以在量化指标之外提供直觉性的洞察。

**实验方法**：在二维连续控制任务（如 Pendulum 或 HalfCheetah）中，收集一批状态-动作轨迹，用编码器将所有观测映射到潜在空间，然后用 **t-SNE**（t-distributed Stochastic Neighbor Embedding，一种非线性降维算法，将高维向量映射到二维平面，使原本在高维中相近的点在二维中也尽量相近，常用于可视化高维表示的聚类结构）将高维潜在向量降到二维，可视化结果。

**健康的 TD-MPC latent space 应该满足**：
- 把"物理上接近的状态"（如摆杆角度相似的状态）映射到"latent space 中接近的点"，局部同构性
- 不同动作从同一状态出发产生的轨迹在 latent space 中方向一致，动作可预测性
- 随着 rollout 步数增加，轨迹在 latent space 中平滑移动，而不是随机跳跃

**诊断规则**：t-SNE 图中，同类状态（如"摆杆直立"状态）散布在各处而非聚集，latent space 几何结构混乱，MPC 在这个空间里做的规划没有物理意义。

<figure>
<img src="/tdmpc/tdmpc-overview.png" alt="TD-MPC 架构总览：编码器、隐式动力学、Q 函数与 CEM 规划" style="width:90%;display:block;margin:0 auto">
<figcaption>Hansen et al. (2022) TD-MPC 架构：编码器、隐式动力学函数、Q 函数、CEM 四模块协同工作。潜在一致性损失（sg(z_{t+1}) 与 d(z_t, a_t) 对齐）确保动力学函数与编码器保持一致，防止表示坍缩。这是 TD-MPC 专属指标的诊断起点。</figcaption>
</figure>

### 规划效率（Plan Efficiency）

定义为：从随机初始化策略出发，达到某个**目标奖励阈值**（如最优策略的 80%）所需的 MPC 规划步数。步数越少，规划效率越高。

**诊断规则**：规划效率低（需要大量步数才能收敛），CEM 的精英样本比例（elite ratio）设置过低，或规划时域（horizon）过短，导致短视规划错过长程奖励。
