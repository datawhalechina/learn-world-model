---
title: 世界模型的严格分类
---

# 世界模型的严格分类：从技术枚举到认知功能

## 最关键的区分：历史脉络不是分类体系

一个严谨的教程不能混淆两个都很有用、但性质完全不同的问题：

1. **历史脉络**：世界模型研究是怎样一步步演化到今天的？
2. **功能能力**：某个具体模型到底具备哪一种世界建模能力？

“四个时代”属于第一个问题。它是一条历史叙事线：从早期 RNN 式预测，到 Ha 和 Schmidhuber 对 world model 的明确表述，到 Dreamer 式 latent imagination，再到 JEPA 式表征/预测路线。这个框架很适合帮助读者建立时间感，但它不是世界模型最深层的分类方式。

L1-L5 属于第二个问题。它按照模型的操作能力分类：模型只是压缩当前状态，还是能预测未来；只是预测自然演化，还是能回答“如果我这样行动会怎样”；只是预测后果，还是能把预测和价值、规划、策略改进耦合起来；最后，它是否能在交互中发现错误并自我修正。

所以本教程采用一个双轴框架：

| 轴 | 回答的问题 | 在课程中的作用 |
| --- | --- | --- |
| 历史时代 | “这个领域是怎样发展到这里的？” | 教学导入和脉络梳理 |
| L1-L5 能力阶梯 | “这个系统到底是哪一种世界模型？” | 主分类体系 |

这样可以避免一个常见误区：把每个著名技术都当成一种并列的“世界模型类型”。DINO、MAE、JEPA、NeRF、视频预测、Dreamer、MuZero 和 simulator 并不是一个扁平列表里的同类项。它们处在不同能力层级，也常常只解决世界模型问题的不同子模块。

> **📖 DINO、MAE、CLIP、ViT**：这四个名字会在本分类体系中反复出现，作为 L1 表征学习的典型例子，有必要先在这里讲清楚。**ViT**（Vision Transformer，[Dosovitskiy et al., 2021](https://arxiv.org/abs/2010.11929)）把 Transformer 的自注意力机制（在 L03 详细介绍）直接用于图像：把图像切成固定大小的 patch，每个 patch 当作一个 token，在 patch 序列上做自注意力，而不是用卷积。**DINO**（self-DIstillation with NO labels，[Caron et al., 2021](https://arxiv.org/abs/2104.14294)，Meta AI）用纯自监督的方式训练 ViT，不需要任何标签：学生网络在同一张图片的不同增强视角上，学习去匹配一个缓慢更新的教师网络（通过 EMA，也就是 JEPA 中用到的同一种机制）的输出，训练出来的特征会按物体语义自动聚类，即使模型从未被告知“这是什么物体”。**CLIP**（Contrastive Language-Image Pretraining，[Radford et al., 2021](https://arxiv.org/abs/2103.00020)，OpenAI）联合训练一个图像编码器和一个文本编码器，让匹配的图文对表征互相靠近，不匹配的图文对表征互相远离，从而得到视觉和语言两个模态对齐的表征。**MAE**（Masked Autoencoder，掩码自编码器，[He et al., 2021](https://arxiv.org/abs/2111.06377)）的训练方式是遮住图像中很大一部分 patch（通常 75%），只用剩下可见的 patch 重建被遮住的像素，思路和 BERT 遮住文本中的词类似。这四者都是表征学习方法：它们产出的是图像的压缩、语义化编码，这也是它们在本分类体系中停留在 L1 而不是更高层级的原因，它们本身并不预测未来，也不以动作为条件。

## 四个时代简述

**时代一：理论奠基（1950s-2017）**。循环神经网络（RNN）、卡尔曼滤波器、隐马尔可夫模型，这七十年里，研究者们在控制论、语音识别、机器人学等不同领域独立构建"预测未来状态"的工具，但这些工作从未被统一冠以"世界模型"之名。

**时代二：Ha 与 Schmidhuber 的"梦中学习"（2018）**。Ha 与 Schmidhuber 的《[World Models](https://arxiv.org/abs/1803.10122)》用一个三模块框架统一了这些散落的思想：**V**（视觉模块）把每帧画面压缩成一个低维向量；**M**（记忆模块，MDN-RNN）以历史向量和动作为输入，预测这个向量如何演化；**C**（控制器）直接以当前向量和 M 的隐状态输出动作。把控制器完全放进记忆模块幻想出的虚拟环境里训练，再把策略迁移到真实游戏，这一思路让世界模型第一次进入主流视野。

**时代三：Dreamer 与潜在空间（2019）**。Hafner 等人的 [Dreamer V1](https://arxiv.org/abs/1912.01603) 引入了 RSSM（Recurrent State Space Model，完整机制见第二讲），把状态拆成确定性历史路径和随机不确定性路径两条并行分支。与 Ha & Schmidhuber 的方法不同，Dreamer 完全不在像素空间重建图像：预测、规划、奖励学习全部直接在潜在空间完成，在 Atari 和连续控制任务上大幅超越了此前的无模型方法。

**时代四：视频即世界（2023 至今）**。JEPA（Joint Embedding Predictive Architecture，LeCun 团队，[2022](https://openreview.net/forum?id=BZ5a1r-kVsf)）彻底放弃像素重建，只在语义嵌入空间里做预测："我不需要画出你的脸，我只需要知道你是谁。"

四个时代的演化逻辑一以贯之：从"如何在序列中预测状态"（时代一），到"如何在梦境中训练策略"（时代二），到"如何在潜在空间里压缩感知"（时代三），再到"如何只保留语义、丢掉噪声"（时代四）。每一步都是对上一步瓶颈的直接回应。

## 四个时代如何映射到 L1-L5

历史时代和能力阶梯可以融合，而且并不矛盾：

| 历史框架 | 典型贡献 | 能力解释 |
| --- | --- | --- |
| 早期 recurrent prediction | 学习紧凑隐状态，并预测序列变化 | L1-L2 |
| Ha 和 Schmidhuber 的 world models | 把表征、记忆/动力学、控制器分离出来 | L1-L3，并开始出现 agent 接口 |
| Dreamer 式 latent imagination | 在学到的 latent dynamics 中规划和学习策略 | L3-L4 |
| JEPA 式表征/预测 | 学习抽象预测表征，而不是重建每个像素 | L1-L2，也可以成为 L3-L5 的底座 |

这张表也说明了为什么不能对单个方法过度命名。JEPA 很重要，但 JEPA 本身并不自动等于 agentic world model。Dreamer 更接近 agentic 意义上的世界模型，因为它把学到的 dynamics 用于策略学习。MuJoCo 这样的 simulator 虽然“能跑起来”，但它不是智能体内部学到的世界模型；只有当智能体内化、近似或利用其动力学进行内部推演时，它才进入本教程讨论的 world model 范畴。

因此，本教程严格使用 **world model** 这个词：一个系统必须包含内部模型，并支持预测、反事实评估、规划或自我修正。只提供表征的系统，除非被整合进更大的预测或行动闭环，否则应称为 **world-model component**，而不是完整世界模型。

“世界模型”这个词在不同社区里使用得很宽：自监督视觉、视频生成、3D 重建、物理仿真、强化学习和具身智能都可能说自己在做 world model。为了避免概念膨胀，本课程采用一个更严格的判定框架。

一个系统越接近完整的世界模型，越应该同时回答三个问题：

1. **它是否学习或维护了世界状态的内部表征？**
2. **它是否能根据当前状态和可能的行动预测未来状态？**
3. **它的预测是否能被智能体用于规划、控制或决策？**

这三个问题对应三层由弱到强的定义。它们能清理现有文献里的混用，但还不是本课程最终想要的分类方式。更有解释力的分类不应该只问“这个模型属于哪篇论文的技术路线”，而应该问：**它让智能体获得了哪一种关于世界的操作能力？**

## L1-L5：按可操作能力分类

“重建世界”“预测下一步”“能跑起来”是有用的直觉入口，但它们仍然停留在表面行为。一个更强的分类应该从智能体能做什么开始，本课程采用五级能力阶梯作为主分类体系。每一级都给出核心问题、形式化表达、典型例子和局限。

### L1 压缩模型：What is here?

压缩模型把高维观测变成可计算、可记忆、可比较的内部状态。它解决的是“我现在看见的世界可以被表示成什么”的问题，通常不直接展开未来，也不直接服务行动，更准确地说是“世界模型组件”而非完整世界模型。

形式化：

$$z_t = \text{Encoder}(o_t)$$

其中 $o_t$ 是时刻 $t$ 的高维观测（像素、点云等），$z_t$ 是压缩后的内部状态。

典型对象：

- DINO / MAE / CLIP-style representation
- autoencoder / VAE encoder
- object-centric representation

核心能力：从像素到 latent state，从局部观测到稳定对象，从噪声细节到任务相关变量。

局限：它知道“这里有什么”，但未必知道“接下来会怎样”。

### L2 动力学模型：What happens next?

动力学模型不仅表示当前世界，还学习状态随时间变化的规律，预测“如果世界继续演化，下一步是什么”。预测可以发生在像素空间、特征空间、物体空间或 3D 空间。

形式化：

$$z_{t+1} = \text{Predictor}(z_{\le t})$$

JEPA 是这一层级的代表，它进一步把预测限制在可见与被遮挡 patch 之间：$z_{\text{masked}} = \text{Predictor}(z_{\text{visible}}, \Delta)$，其中 $\Delta$ 是被遮挡区域的位置编码。视频世界模型走的是像素空间的版本：$I_{t+1} = \text{Generator}(I_{\le t}, c)$，用历史帧 $I_{\le t}$ 和可选的语义提示 $c$ 直接生成下一帧。

典型对象：

- JEPA / latent dynamics
- video prediction / video diffusion
- scene flow / object dynamics

核心能力：temporal prediction、latent rollout、uncertainty over futures。

局限：它能预测未来，但未必知道“我的动作会改变什么”。

### L3 行动条件模型：What if I act?

行动条件模型把动作纳入世界演化。它不只是预测自然发生的未来，而是预测“如果我做某个动作，会发生什么”。

形式化：

$$s_{t+1} = f(s_t, a_t)$$

其中 $a_t$ 是智能体在时刻 $t$ 执行的动作。这一个条件的存在，让世界模型从“旁观者”变成了“参与者”。

典型对象：

- model-based RL dynamics model
- robotics forward model
- controllable video generation
- action-conditioned latent transition

核心能力：action-conditioned counterfactual prediction、imagined trajectories under candidate policies。

局限：它能回答单步或短程反事实，但未必能长期规划，也未必知道哪些后果更值得追求。

### L4 价值耦合模型：What matters?

价值耦合模型把世界预测与目标、奖励、偏好或生存约束绑定，解决的是“哪些未来更好，哪些未来更危险”的问题，是本课程最严格意义上的 agentic 世界模型：预测被直接用于规划、控制和决策。

形式化：

$$a^* = \arg\max_{a} \; \text{Value}\big(\text{Rollout}_{\text{WM}}(s, a)\big)$$

即在世界模型内部并行展开多个候选动作的想象轨迹，用价值函数或评估器打分，选出最优动作。Dreamer 用学到的 actor-critic 在 latent imagination 中完成这一步；MuZero 用搜索完成同样的事情。

典型对象：

- Dreamer-style actor-critic in imagination
- MuZero-style reward/value prediction
- learned cost models for control
- preference-conditioned world models

核心能力：reward / value prediction、planning over imagined futures、credit assignment through latent rollouts。

局限：它知道哪些未来更有价值，但未必能持续修正自己的世界假设，模型本身的误差会被规划过程放大。

### L5 自校正模型：How do I improve my model of the world?

自校正模型把预测误差、探索和模型更新形成闭环。它不只是使用世界模型，而是主动改进世界模型：检测模型误差，选择能降低不确定性的实验，在干预之后更新信念，并在多个任务上维护一个不断增长的世界模型。这是更高阶的世界模型，它不仅模拟世界，还能意识到自己的模拟哪里不可靠。

典型对象：

- active inference
- curiosity-driven model learning
- uncertainty-guided exploration
- lifelong world-model learning
- scientific discovery agents

核心能力：detect model error、choose experiments that reduce uncertainty、update beliefs after intervention、maintain a growing world model across tasks。

## 最终分类：五层能力阶梯

因此，本课程最终不采用“某某模型是不是世界模型”的二元标签，而采用一个能力阶梯：

| 层级 | 核心问题 | 形式化 | 典型例子 | 严格称谓 |
| --- | --- | --- | --- | --- |
| L1 压缩 | What is here? | $z_t = \text{Encoder}(o_t)$ | DINO, MAE, NeRF encoder | world-model component |
| L2 动力学 | What happens next? | $z_{t+1} = \text{Predictor}(z_{\le t})$ | JEPA, video prediction, scene flow | predictive world model |
| L3 行动条件 | What if I act? | $s_{t+1} = f(s_t, a_t)$ | robotics forward model, action-conditioned dynamics | controllable world model |
| L4 价值耦合 | What matters? | $a^* = \arg\max_a \text{Value}(\text{Rollout}_{\text{WM}}(s,a))$ | Dreamer, MuZero | agentic world model |
| L5 自校正 | How do I improve? | 主动探索并更新模型（无单一闭式表达） | active inference, curiosity, lifelong agents | self-improving world model |

这个分类比“重建 / 预测 / 能跑起来”更强，因为它刻画的是智能体能力，而不是模型表面形态。一个系统可以在像素、对象、3D、语言或物理状态上实现这些能力；模态不是本质，**可操作的反事实能力**才是本质。

还有一类系统经常被误称为世界模型：MuJoCo、Brax、Isaac Gym 这样的物理模拟器，Atari、贪吃蛇、Minecraft 这样的规则或程序生成环境，以及各类游戏引擎。它们确实“包含一个世界”，也对训练世界模型非常重要，因为它们提供数据和评估环境。但它们通常不是智能体自己学习出来的内部模型；除非智能体把它们的规律学习进自己的内部模型，否则它们属于 **external simulator**，不进入 L1-L5 阶梯。

## 二维分类：对象 × 能力

网上流传的“世界模型九宫格”梗图（横轴是重建、预测下一步、能跑起来，纵轴是特征/潜变量、对象/3D、像素/视频）说的正是本节的直觉：DINO、JEPA、Dreamer 各占一格，NeRF、Scene Flow、MuJoCo 各占一格。这张梗图和[思想基石](./01-foundations)里“预测什么 / 是否接受动作 / 服务什么目的”那张三问表本质上是同一件事的两种排布方式。它把模型按两个正交维度排列：

- **横向：建模空间**，也就是表征对象是什么，特征/潜变量、对象/3D、像素/视频、状态/物理量。
- **纵向：能力层级**，也就是模型能对这个空间做什么，重建、预测、行动闭环，对应上一节的 L1-L4。

| 表征对象 | 重建（L1） | 预测（L2） | 行动闭环（L3-L4） | 代表公式 |
| --- | --- | --- | --- | --- |
| 特征 / 潜变量 | MAE, autoencoder | JEPA, latent dynamics | Dreamer latent imagination | $z_{\text{masked}} = \text{Predictor}(z_{\text{visible}}, \Delta)$ |
| 对象 / 3D | NeRF, 3D Gaussian Splatting | scene flow, object dynamics | model-based manipulation | $I = \text{Renderer}(\Theta, c)$ |
| 像素 / 视频 | image/video reconstruction | video diffusion, video prediction | visual model predictive control | $I_{t+1} = \text{Generator}(I_{\le t}, c)$ |
| 状态 / 物理量 | state estimator | learned dynamics | MPC, model-based RL | $s_{t+1} = f(s_t, a_t)$ |

其中 NeRF / 3D Gaussian Splatting 一行的公式里，$\Theta$ 是场景表征（NeRF 权重或 3DGS 高斯点集合），$c$ 是查询条件（视角、时间戳或位姿），$I$ 是渲染出的图像。

这个表的关键点是：**对象维度不决定它是不是完整世界模型，能力维度才决定严格等级**。NeRF 可以是非常好的 3D 世界表征，但如果它只是静态重建，就仍然停留在 L1 表征层；Dreamer 之所以更接近完整世界模型，是因为它把 latent prediction 接到了行动学习上，进入了 L3-L4。这也解释了为什么 DINO 和 MAE 通常只是 L1，JEPA 和视频预测进入 L2，action-conditioned dynamics 进入 L3，Dreamer / MuZero 进入 L4，而只有能主动设计实验、修正自身假设的智能体才进入 L5，这一层在九宫格里没有对应的格子，因为它已经超出了"建模空间"这个横向维度能描述的范围。

## 应用：从 L1-L5 阶梯到规划架构的七条路径

前面的阶梯回答的是“这个系统具备哪种世界建模能力”，但对做机器人和自动驾驶规划的人来说，更实际的问题是“世界模型到底怎样被用进规划管线里”。下面这七条路径来自赵知宁（HKUST UAV Group）对当前文献的梳理，每一条都可以映射回 L1-L5 的某个层级，从而把抽象阶梯和具体架构选择连起来。

| 路径 | 一句话概括 | 对应能力层级 |
| --- | --- | --- |
| 1. 表征预训练 | 用无标签视频预训练编码器，推理时只用编码器接规划器 | L1，为更高层级打地基 |
| 2. 慢快分层 | VLM 低频出高层指令，端到端规划器高频执行 | L1-L3 组合，指令是外部条件而非内部预测 |
| 3. VLA + 因果链（CoC） | 单一自回归流里先输出思维链再输出动作，监督“如何”映射而不只是监督结果 | L3，把行动条件预测和决策过程本身都纳入训练信号 |
| 4. 稠密监督 | 训练时并行做未来场景预测和动作/轨迹预测，两路损失联合优化，推理时只用动作分支 | L2 提供的稠密梯度反哺 L3 的规划头 |
| 5. 生成式目标 / 视觉轨迹 | 先“想象”目标图像或视觉轨迹，再用逆动力学模型把想象转成动作 | L2 生成视觉思维链，L3 把它转化为行动 |
| 6. 通过世界模型选动作（评估器） | 把世界模型当黑盒模拟器，并行 rollout 多个候选动作，用像素、点云、latent 或 VLM 打分，选最优后按 receding horizon 执行 | L4，直接对应上一节的 $a^* = \arg\max_a \text{Value}(\text{Rollout}_{\text{WM}}(s,a))$ |
| 7. 闭环仿真 | 世界模型离线制造分布外场景：模仿学习里扰动轨迹再由专家规划器纠偏来扩充训练集，强化学习里在想象中做高吞吐采样训练 policy | L4-L5，用世界模型持续改进训练数据和策略本身 |

几个值得注意的地方：

- 路径一到路径三本质上是同一个思路的三种紧密程度：路径一是训练完世界模型后把预测头换掉，路径二是保留两个独立系统协作，路径三是把预测和决策揉进同一个自回归流。紧密程度越高，训练和测试时的行为越一致，但工程复杂度也越高。
- 路径四和路径五都在提供“稠密监督”，区别在于监督信号的形式：路径四直接预测未来场景本身，路径五先生成一个视觉目标或轨迹再交给逆动力学模型转成动作。两者都试图缓解动作标签过于稀疏的问题。
- 路径六和路径七的关键差异是**在线**还是**离线**使用世界模型：路径六在部署时实时调用世界模型评估候选动作，路径七在训练前用世界模型离线生成或扩充数据，训练完成后世界模型不再参与推理。
- 语言世界模型（路径二、三近似路径）和视频世界模型（路径五近似路径）哪个更适合做“想象”，目前没有定论。语言模型信息密度高但依赖大量人工标注的因果链，且可能过滤掉规划所需的细节；视频模型保留原始像素，标注成本低，但计算开销更大，也更容易学到与规划无关的背景噪声。

这七条路径共同说明：**世界模型对规划的价值不只是“预测得准”，而是能不能被恰当地接入训练信号、动作选择或数据生成这三个环节之一**。这也是为什么 L4（价值耦合）和 L5（自校正）在实际系统里往往不是单一模型的属性，而是某个 L1-L3 世界模型搭配某种接入方式共同构成的。

## 本课程采用的最终口径

本课程中，“世界模型”默认指 **智能体内部的、可预测的、可用于行动选择的世界动态模型**。在宽泛讨论中，我们会承认表征模型、重建模型、视频预测模型和外部模拟器都与世界模型相关，但会明确区分：

- **world-model component**：学习了世界的某种表征或局部规律。
- **predictive world model**：能在内部预测未来状态或观测。
- **controllable world model**：能预测动作造成的反事实后果。
- **agentic world model**：能把内部预测用于规划、控制和决策。
- **self-improving world model**：能通过探索和误差修正持续改进自身。
- **external simulator**：提供可交互世界，但不是智能体学到的内部模型。

因此，“DINO 是世界模型”“NeRF 是世界模型”“MuJoCo 是世界模型”这些说法只有在宽泛语境下成立。严格地说，DINO 和 NeRF 更像世界模型组件，MuJoCo 是外部模拟器，而 Dreamer / MuZero 这类系统才是本课程核心意义上的世界模型。
