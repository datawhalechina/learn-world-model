---
title: 世界模型的严格分类
---

# 世界模型的严格分类：从技术枚举到认知功能

“世界模型”这个词在不同社区里使用得很宽：自监督视觉、视频生成、3D 重建、物理仿真、强化学习和具身智能都可能说自己在做 world model。为了避免概念膨胀，本课程采用一个更严格的判定框架。

一个系统越接近完整的世界模型，越应该同时回答三个问题：

1. **它是否学习或维护了世界状态的内部表征？**
2. **它是否能根据当前状态和可能的行动预测未来状态？**
3. **它的预测是否能被智能体用于规划、控制或决策？**

这三个问题对应三层由弱到强的定义。它们能清理现有文献里的混用，但还不是本课程最终想要的分类方式。更有解释力的分类不应该只问“这个模型属于哪篇论文的技术路线”，而应该问：**它让智能体获得了哪一种关于世界的操作能力？**

## Level 1: 表征型世界模型组件

表征型模型学习世界的压缩表示，但通常不直接展开未来，也不直接服务行动。它们更准确地说是“世界模型组件”，而不是完整世界模型。

典型例子包括：

- **DINO / CLIP-like visual representation**：学习视觉语义和物体结构的表征。
- **MAE / masked reconstruction**：通过重建被遮挡的信息学习图像或视频的潜变量。
- **Autoencoder / VAE encoder**：把高维观测压缩到 latent state。

这类模型回答了第一个问题：它们有内部状态表征。但如果没有显式的动态预测和行动接口，就不应直接称为完整世界模型。

## Level 2: 预测型世界模型

预测型模型不仅表示当前世界，还能预测“接下来会发生什么”。预测可以发生在像素空间、特征空间、物体空间或 3D 空间。

典型例子包括：

- **JEPA**：在 latent space 中预测目标表征，避免直接重建所有像素细节。
- **Video diffusion / video prediction**：预测未来视频帧或可能的视觉轨迹。
- **Scene flow / optical flow models**：预测 3D 点、像素或物体的运动。
- **Dynamics model**：学习状态转移函数 `s_{t+1} = f(s_t, a_t)`。

这类模型回答了前两个问题：它们有状态表征，也有未来预测。但如果预测没有进入行动选择或规划闭环，它们仍然只是预测型世界模型，不是智能体世界模型。

## Level 3: 智能体世界模型

智能体世界模型把表征、预测和行动闭环连在一起。它不仅问“世界会怎样变化”，还问“如果我这样做，未来会怎样，我应该选哪个动作”。

典型例子包括：

- **Dreamer**：在 latent dynamics 中 imagination rollout，并用这些内部轨迹训练 actor 和 critic。
- **MuZero-style models**：学习可用于搜索的隐状态转移、奖励和价值预测。
- **Learned simulators for control**：模型本身支持规划、MPC 或策略优化。

这是本课程最严格意义上的世界模型：它服务于行动，并允许智能体在内部模拟未来。

## Level 0: 外部模拟器和环境

还有一类系统经常被误称为世界模型：手写模拟器或规则环境。它们确实“包含一个世界”，但通常不是智能体自己学习出来的内部模型。

典型例子包括：

- **MuJoCo / Brax / Isaac Gym**：物理模拟器，是外部环境模型。
- **Atari / 贪吃蛇 / Minecraft 环境**：规则或程序生成的交互环境。
- **游戏引擎**：可运行的世界，但不是 learned world model。

这些系统对训练世界模型非常重要，因为它们提供数据和评估环境。但除非智能体把它们的规律学习进自己的内部模型，否则它们不属于 learned world model。

## 二维分类：对象 × 能力

为了吸收常见“九宫格”式分类的直觉，我们可以保留两个正交维度：

| 表征对象 | 重建 | 预测 | 行动闭环 |
| --- | --- | --- | --- |
| 特征 / 潜变量 | MAE, autoencoder | JEPA, latent dynamics | Dreamer latent imagination |
| 对象 / 3D | NeRF, 3D Gaussian Splatting | scene flow, object dynamics | model-based manipulation |
| 像素 / 视频 | image/video reconstruction | video diffusion, video prediction | visual model predictive control |
| 状态 / 物理量 | state estimator | learned dynamics | MPC, model-based RL |

这个表的关键点是：**对象维度不决定它是不是完整世界模型，能力维度才决定严格等级**。NeRF 可以是非常好的 3D 世界表征，但如果它只是静态重建，就仍然停留在表征层；Dreamer 之所以更接近完整世界模型，是因为它把 latent prediction 接到了行动学习上。

## 超越九宫格：按可操作能力分类

“重建世界”“预测下一步”“能跑起来”是有用的直觉入口，但它们仍然停留在表面行为。一个更强的分类应该从智能体能做什么开始。

本课程采用五种世界模型能力作为主轴。

### 1. 压缩模型：What is here?

压缩模型把高维观测变成可计算、可记忆、可比较的内部状态。它解决的是“我现在看见的世界可以被表示成什么”的问题。

典型对象：

- DINO / MAE / CLIP-style representation
- autoencoder / VAE encoder
- object-centric representation

核心能力：

- 从像素到 latent state
- 从局部观测到稳定对象
- 从噪声细节到任务相关变量

局限：

- 它知道“这里有什么”，但未必知道“接下来会怎样”。

### 2. 动力学模型：What happens next?

动力学模型学习状态随时间变化的规律。它解决的是“如果世界继续演化，下一步是什么”的问题。

典型对象：

- JEPA / latent dynamics
- video prediction / video diffusion
- scene flow / object dynamics

核心能力：

- temporal prediction
- latent rollout
- uncertainty over futures

局限：

- 它能预测未来，但未必知道“我的动作会改变什么”。

### 3. 行动条件模型：What if I act?

行动条件模型把动作纳入世界演化。它不只是预测自然发生的未来，而是预测“如果我做某个动作，会发生什么”。

典型对象：

- model-based RL dynamics model
- robotics forward model
- controllable video generation
- action-conditioned latent transition

核心能力：

- `s_{t+1} = f(s_t, a_t)`
- action-conditioned counterfactual prediction
- imagined trajectories under candidate policies

局限：

- 它能回答单步或短程反事实，但未必能长期规划。

### 4. 价值耦合模型：What matters?

价值耦合模型把世界预测与目标、奖励、偏好或生存约束绑定。它解决的是“哪些未来更好，哪些未来更危险”的问题。

典型对象：

- Dreamer-style actor-critic in imagination
- MuZero-style reward/value prediction
- learned cost models for control
- preference-conditioned world models

核心能力：

- reward / value prediction
- planning over imagined futures
- credit assignment through latent rollouts

局限：

- 它知道哪些未来更有价值，但未必能持续修正自己的世界假设。

### 5. 自校正模型：How do I improve my model of the world?

自校正模型把预测误差、探索和模型更新形成闭环。它不只是使用世界模型，而是主动改进世界模型。

典型对象：

- active inference
- curiosity-driven model learning
- uncertainty-guided exploration
- lifelong world-model learning
- scientific discovery agents

核心能力：

- detect model error
- choose experiments that reduce uncertainty
- update beliefs after intervention
- maintain a growing world model across tasks

这是更高阶的世界模型：它不仅模拟世界，还能意识到自己的模拟哪里不可靠，并主动收集信息修正它。

## 最终分类：五层能力阶梯

因此，本课程最终不采用“某某模型是不是世界模型”的二元标签，而采用一个能力阶梯：

| 层级 | 核心问题 | 最小能力 | 典型例子 | 严格称谓 |
| --- | --- | --- | --- | --- |
| L1 压缩 | What is here? | 表征当前状态 | DINO, MAE, NeRF encoder | world-model component |
| L2 动力学 | What happens next? | 预测未来状态 | JEPA, video prediction, scene flow | predictive world model |
| L3 行动条件 | What if I act? | 预测动作后果 | robotics forward model, action-conditioned dynamics | controllable world model |
| L4 价值耦合 | What matters? | 用预测服务规划和决策 | Dreamer, MuZero | agentic world model |
| L5 自校正 | How do I improve? | 主动探索并更新模型 | active inference, curiosity, lifelong agents | self-improving world model |

这个分类比“重建 / 预测 / 能跑起来”更强，因为它刻画的是智能体能力，而不是模型表面形态。一个系统可以在像素、对象、3D、语言或物理状态上实现这些能力；模态不是本质，**可操作的反事实能力**才是本质。

## 与现有九宫格的关系

常见九宫格把模型按“重建、预测、能跑起来”和“特征、对象、像素”排列。这个视角没有错，但它混合了两个不同问题：

- 它在什么空间里建模？特征、对象、3D、像素、物理状态。
- 它能对这个空间做什么？压缩、预测、行动条件反事实、规划、自校正。

本课程保留“建模空间”作为横向维度，但把“能力阶梯”作为纵向主轴。这样可以解释为什么：

- DINO 和 MAE 很重要，但通常只是 L1。
- JEPA 和视频预测进入 L2。
- action-conditioned dynamics 进入 L3。
- Dreamer / MuZero 进入 L4。
- 能主动设计实验、修正自身假设的智能体才进入 L5。
- MuJoCo 和游戏引擎是外部模拟器，不自动等于智能体内部世界模型。

## 本课程采用的最终口径

本课程中，“世界模型”默认指 **智能体内部的、可预测的、可用于行动选择的世界动态模型**。在宽泛讨论中，我们会承认表征模型、重建模型、视频预测模型和外部模拟器都与世界模型相关，但会明确区分：

- **world-model component**：学习了世界的某种表征或局部规律。
- **predictive world model**：能在内部预测未来状态或观测。
- **controllable world model**：能预测动作造成的反事实后果。
- **agentic world model**：能把内部预测用于规划、控制和决策。
- **self-improving world model**：能通过探索和误差修正持续改进自身。
- **external simulator**：提供可交互世界，但不是智能体学到的内部模型。

因此，“DINO 是世界模型”“NeRF 是世界模型”“MuJoCo 是世界模型”这些说法只有在宽泛语境下成立。严格地说，DINO 和 NeRF 更像世界模型组件，MuJoCo 是外部模拟器，而 Dreamer / MuZero 这类系统才是本课程核心意义上的世界模型。
