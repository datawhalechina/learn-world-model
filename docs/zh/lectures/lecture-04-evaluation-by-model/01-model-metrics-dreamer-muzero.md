---
title: Dreamer 的专属指标
description: 针对 RNN/RSSM 架构的专属评估指标与诊断规则，包括奖励相关性、FID 与 KL 崩塌预警。
lecture: 4
---

# Dreamer 的专属指标

## 为什么要"按模型评估"？

**反例**：用 FID 衡量 MuZero。MuZero 根本不生成像素图像，它的世界模型是隐式的，FID 对它毫无意义。同理，用"Token 预测损失"衡量 Dreamer，只会让人误以为 Dreamer 是个语言模型。

不同世界模型在不同环节崩溃：

| 架构类型 | 最容易崩溃的地方 |
|---------|----------------|
| RNN/RSSM（Dreamer）| 编码器退化，想象奖励失真，KL 崩塌 |
| 隐式模型（MuZero）| 价值估计偏差，表示稳定性弱，搜索树退化 |
| 潜在 MPC（TD-MPC）| 潜在表示不一致，表示崩塌，规划效率低下 |
| Transformer 动力学（STORM）| teacher forcing 与 free-running 差距，长时域 token 漂移 |
| 扩散世界模型（Diamond）| 物理一致性崩溃，物体凭空消失 |

---

## Dreamer（RNN/RSSM）

*你在 P03 亲手实现了 Dreamer 的完整流水线，编码器 → RSSM 预测 → 潜在 Actor-Critic → 执行动作。*

Dreamer 是一个强化学习算法，不是生成模型。Dreamer V1/V2/V3 论文的核心评估问题只有一个：**策略能不能在真实环境中获得高奖励？** 所有指标都服务于这个问题，而不是服务于"图像重建有多好看"。

### 策略奖励曲线（Episode Return）

这是 Dreamer 系列论文的首要指标。Hafner et al. 2019 在 DMControl 上报告每个任务的 episode return；Hafner et al. 2020 在 Atari 200M 帧上报告 episode return；Hafner et al. 2023 横跨 7 个领域用单一超参数报告 episode return。三篇论文的核心 figure 都是**训练步数 vs. 累计奖励曲线**，与 model-free 基线（如 **SAC**，Soft Actor-Critic，一种基于最大熵框架的 off-policy Actor-Critic 算法，在连续控制任务上是强力的无模型基线；**DrQ-v2**，Data-regularized Q，在 SAC 基础上加入数据增强和 n 步回报，是像素输入连续控制的代表性基线）和 model-based 基线（如 **MBPO**，Model-Based Policy Optimization，用学到的世界模型生成短程合成轨迹来提高样本效率的基线算法；**PlaNet**，Planning with Latent Dynamics，RSSM 的前身，只做 MPC 规划不做 Actor-Critic）对比数据效率。

**诊断规则**：训练曲线长期停滞或下降有两种来源。第一，奖励预测失真（世界模型在撒谎），Actor-Critic 在错误的想象奖励上优化；第二，RSSM 动力学预测漂移，想象轨迹越来越偏离真实环境分布。区分两者的办法是同时观察奖励相关性（见下节）。

**如何在 P03 实验中追踪**：每隔固定训练步数（如每 10k 步）暂停训练，在真实环境中跑若干完整 episode，记录平均 episode return，画出曲线。这条曲线是判断 Dreamer 是否正常训练的最终标准。

### 奖励相关性（Reward Correlation）

Dreamer 在"想象空间"中滚出轨迹并预测奖励，这些"想象奖励"需要和环境返回的真实奖励高度相关，Actor-Critic 才能学到有用的策略。

$$\rho = \text{Pearson}(r_{\text{imagined}},\, r_{\text{real}})$$

> **📖 皮尔逊相关系数（Pearson correlation coefficient）**：衡量两个变量线性相关程度的标准指标，取值范围 $[-1, 1]$。$\rho = 1$ 表示完全正相关（一个增大另一个也增大，且比例固定）；$\rho = 0$ 表示无线性相关；$\rho = -1$ 表示完全负相关。公式为 $\rho = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$，其中 $\text{Cov}$ 是协方差，$\sigma$ 是标准差。这里用它衡量"想象奖励"和"真实奖励"这两条序列的变化趋势是否一致。

实践中，取一批轨迹（如 1000 步），计算想象奖励序列和真实奖励序列的皮尔逊相关系数，目标值 `ρ ≥ 0.8`。

**诊断规则**：`ρ` 持续低于 0.5，RSSM 的随机状态 `z_t` 对奖励信息编码不足，可尝试增大潜在维度或延长 KL 退火周期。

**实验建议**：可视化想象轨迹 vs 真实轨迹的奖励曲线。具体做法是：从同一初始状态出发，分别让 Dreamer 想象展开 20 步，同时在真实环境中执行同样的动作序列，然后把两条奖励曲线画在同一张图上。如果两条曲线趋势相符（即使不完全重合），说明世界模型在诚实地反映环境；如果想象奖励持续高于真实奖励且趋势相反，说明"世界模型在撒谎"，policy 学到的是在虚假优化目标上的技巧，这正是 model exploitation 问题的根源。

### 编码器健康度诊断：重建 FID（Fréchet Inception Distance）

FID 不是 Dreamer 论文的报告指标，Dreamer 的 ELBO 目标包含重建损失，但论文从不用 FID 来衡量策略好坏。FID 在这里是一个**辅助诊断工具**：如果编码器退化，图像重建质量会下降，FID 可以提前捕捉到这个信号，让你在 episode return 崩塌之前介入。

FID 用 **Inception-v3**（Google 2015 年提出的深度卷积图像分类网络，在 ImageNet 上预训练，其中间层特征向量被广泛用作图像质量的感知度量代理）提取真实帧与重建帧的深层特征，计算两个特征分布之间的 **Fréchet 距离**（[Heusel et al., 2017](https://arxiv.org/abs/1706.08500)）。FID 数值越低越好：FID 高意味着重建帧的特征分布偏离真实帧，编码器表征已退化。

<details>
<summary>📖 FID 计算细节（展开）</summary>

① 用 Inception-v3 的中间层对大量真实图像和生成图像各提取一个特征向量；② 分别拟合两组特征的多元高斯分布（均值 $\mu$，协方差矩阵 $\Sigma$）；③ 计算两个高斯分布之间的 Fréchet 距离（又称 Wasserstein-2 距离，把一个分布"搬运"成另一个所需的最小工作量，比 KL 散度对形状差异更敏感）：

$$\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}\!\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$

其中 `Tr(·)` 是矩阵的迹（对角线元素之和）。选用 Inception-v3 而非像素 MSE，是因为它的特征空间近似人类对图像的感知判断。
</details>

**诊断规则**：FID 在训练中途突然上升，编码器发生了表示崩塌（representation collapse），卷积权重退化为常数输出。**缓解**：降低编码器学习率，或在编码器后加 LayerNorm。

**FID 正常不等于 Dreamer 正常**。视觉重建还不错，但想象奖励已经悄悄失真的情况确实存在，因为 RSSM 的随机状态 `z_t` 可以在重建质量达标的同时，对奖励信息编码不足。FID 正常只说明编码器没有退化，还必须结合奖励相关性一起判断。

### 想象轨迹熵（Imagined Trajectory Entropy）

这是一个容易被忽视但非常重要的早期预警指标。RSSM 的随机状态 `z_t` 是从高斯分布中采样的，理论上每一步应该有一定的方差，反映环境的内在随机性和模型的不确定性。

$$H_{\text{traj}} = \mathbb{E}_t\!\left[H\!\left(q(z_t \mid h_t, o_t)\right)\right] = \mathbb{E}_t\!\left[\tfrac{1}{2}\sum_i \left(1 + \log \sigma^2_i\right)\right]$$

**诊断规则**：如果 RSSM 在想象展开时对所有状态都返回非常相似的 `z_t`（方差 `σ²` 全部接近 0），说明随机变量退化为确定性，这是 **KL 崩塌（KL collapse）** 的早期信号。KL 崩塌时，KL 散度项在损失函数里变得几乎为零，编码器不再向 `z_t` 注入任何额外信息，整个 RSSM 退化为纯 RNN。

**缓解策略**：
- 使用 **KL 退火**（KL annealing，训练初期将损失函数中 KL 散度项的权重系数从 0 缓慢线性升至目标值，例如前 10k 步权重从 0 升到 1，给编码器时间先学会重建，再逐渐被迫向随机潜变量编码信息）：训练初期将 KL 权重从 0 缓慢升至目标值，给编码器时间先学会重建，再学会编码随机性
- 设置 **KL 自由比特**（free bits，在计算 KL 损失时，对 KL 值低于阈值 $\lambda$（如 1 nat）的维度不施加任何梯度，即 $\max(0, \text{KL} - \lambda)$，强制模型至少为每个潜变量维度保留 $\lambda$ nats 的信息量，防止所有维度同时坍缩为零）：强制 KL 项至少达到某个最小值，防止过早崩塌

