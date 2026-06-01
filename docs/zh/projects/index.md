---
title: 项目
description: 五个递进式项目，从 VAE 编码器到三模型评估仪表盘，把世界模型的每一块亲手搭起来。
---

# 项目

讲义负责把世界模型的原理讲清楚，项目负责让它们在你自己的机器上跑起来。五个项目按依赖顺序递进，参考实现沿着一条主线展开：**Dreamer（RSSM）→ TD-MPC → STORM**。前一个项目的产物是后一个的输入：做完 P01 的编码器，P02 才有潜在表示可以建模，依此类推。

**建议顺序：** P01 → P02 → P03 → P04 → P05。与讲义交替推进效果最好：L01 → L02 → P01 → P02 → L03 → P03 → P04 → L04 → P05 → L05。

## 五个项目一览

| # | 项目 | 前置 | 交付物 |
|---|------|------|--------|
| [P01](./project-01-vae-encoder/) | 训练 VAE 编码器 | [L01](../lectures/lecture-01-internal-simulation/)、[L02](../lectures/lecture-02-encode-and-dynamics/) Part A | 把 64×64 图像压成潜变量 z 的 VAE；重建损失曲线；潜变量滑块演示 |
| [P02](./project-02-latent-dynamics/) | 构建潜在动力学模型 | P01、L02 Part B | GRU → RSSM 预测下一帧潜变量；1 步 vs 5 步预测误差对比图 |
| [P03](./project-03-dreamer-pipeline/) | 完整 Dreamer 流水线 | P02、[L03](../lectures/lecture-03-architecture-patterns/) Part A | 端到端：编码 → RSSM → 潜在 Actor-Critic → 动作；奖励曲线 + FID/ρ/熵 自评 |
| [P04](./project-04-td-mpc/) | 实现 TD-MPC 规划 | P03、L03 Part B | CEM-MPC + 潜在一致性损失；与 Dreamer 奖励曲线对比 |
| [P05](./project-05-storm-dashboard/) | STORM + 三模型评估仪表盘 | P03、P04、L03、[L04](../lectures/lecture-04-evaluation-by-model/) | 把 GRU 换成 Transformer（STORM 风格）；Dreamer / TD-MPC / STORM 并排仪表盘 |

## 各项目要点

### [P01 · 训练 VAE 编码器](./project-01-vae-encoder/)

世界模型的第一步，是把高维观测压成紧凑的潜变量。你将实现一个 VAE，把 64×64 的图像编码成低维 z 再解码回来，用 ELBO 同时约束重建质量和潜空间结构。跑通之后，拖动一个滑块改变单个潜维度，看它对应图像里的哪种变化。

### [P02 · 构建潜在动力学模型](./project-02-latent-dynamics/)

有了潜变量，下一步是预测它如何随时间演化。从一个 GRU 起步，再扩展到 RSSM，把确定性状态和随机状态分开建模。重点是测量多步预测的误差增长：1 步预测往往很准，5 步之后漂移开始显现，这正是后续所有评估反复回到的核心问题。

### [P03 · 完整 Dreamer 流水线](./project-03-dreamer-pipeline/)

把编码器和动力学模型接成一条完整链路：编码观测，用 RSSM 在潜空间里展开想象，在想象出的轨迹上训练 Actor-Critic，最后输出动作。你将跑通一个端到端的 Dreamer，并用 FID、奖励相关性 ρ、访问熵给自己的模型打分。

### [P04 · 实现 TD-MPC 规划](./project-04-td-mpc/)

Dreamer 用学到的策略直接行动，TD-MPC 则在潜空间里做在线规划。你将实现 CEM-MPC，配合潜在一致性损失，让规划在想象中保持稳定，再把它的奖励曲线和 P03 的 Dreamer 摆在一起对比，体会“规划”与“策略”各自的取舍。

### [P05 · STORM + 三模型评估仪表盘](./project-05-storm-dashboard/)

最后，把 RSSM 里的 GRU 换成 Transformer，得到一个 STORM 风格的世界模型。你将搭一个并排仪表盘，让 Dreamer、TD-MPC、STORM 在同一套任务上同台竞技，用 [L04](../lectures/lecture-04-evaluation-by-model/) 的指标体系读出它们各自的强项与失效模式。
