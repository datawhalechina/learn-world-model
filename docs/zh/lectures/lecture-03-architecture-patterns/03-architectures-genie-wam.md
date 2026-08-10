---
title: "Part C（续，选读）：Genie"
description: Genie 的 latent action 发现机制，介于观察型预训练与无需动作标签的交互式生成之间。
lecture: 3
---

# Part C（续，选读）：Genie

## 架构六：Genie 从视频隐式发现动作

**代表系统**：Genie (Google DeepMind, 2024)、Genie 2 (2024)

前五个架构族都有一个共同假设：训练数据要么包含动作标签（交互型），要么完全不需要动作（观察型）。Genie 打破了这个二分法：**从无标注互联网视频中，自动发现隐式的 latent action。**

训练数据是大量人类玩游戏、操作物体的视频片段，没有任何动作标签。Genie 同时训练三个模块：视频 tokenizer（**ST-ViT**，Spatiotemporal Vision Transformer，时空视觉 Transformer，将视频片段在时间和空间两个维度上同时做 patch 分割和编码，输出时空离散 token）将帧序列压缩为时空离散 token；latent action model（**LAM**，latent action model，潜在动作模型，从相邻帧对中学习推断帧间变化的类型）从相邻帧对中推断离散的 latent action code；dynamics model 以 latent action 为条件预测下一帧 token 序列。推理时，用户可以指定一个 latent action，模型据此生成下一帧，整个过程完全可交互。

> **📖 latent action**：不是键盘上的"向左"或关节空间的力矩，而是一个纯粹从视频帧差异中归纳出的离散编码。它捕捉的是"相邻帧之间发生了什么类型的变化"，而非具体的物理动作。两段视频如果场景变化模式相似（如"某物体向右移动"），它们的 latent action code 就应该相同，无论实际拍摄的是游戏还是机器人操作。

<figure>
<img src="/genie/genie-architecture.png" alt="Genie 架构：ST-ViT tokenizer、LAM latent action model 和 MaskGIT dynamics model 三模块" style="width:100%;display:block;margin:0 auto">
<figcaption>Bruce et al. (2024) Genie 的三模块设计：ST-ViT 将视频帧序列编码为时空离散 token；LAM 从相邻帧对中推断离散 latent action code（无需任何动作标注）；动力学模型以 latent action 为条件，用 MaskGIT 自回归预测下一帧 token 序列。</figcaption>
</figure>

Genie 在 3 万小时的平台游戏视频上训练（无动作标注），11B 参数，论文以 $\Delta_t\text{PSNR}$（推理时 PSNR 相对于 teacher forcing 基线的下降量）衡量生成质量衰减速度，作为 latent action 对齐程度的代理指标。Genie 的意义在于把"动作标注"这个瓶颈绕开了：互联网上有海量视频，但几乎没有配套的机器人动作标签。Genie 2 进一步扩展到 3D 场景，能在给定单张图像后生成完整的可交互 3D 世界。Bi et al. 于 2025 年发布的 [Motus](https://arxiv.org/abs/2512.13030)（A Unified Latent Action World Model）在具身操作任务上验证了类似思路，通过统一的 latent action 表征从异构视频数据中提取动作知识，再用少量有标注数据对齐到真实控制，实现跨具身迁移。

**学习范式**：介于观察型和交互型之间。训练只用视频（观察型），但推理时支持动作条件生成（交互型）。这个思路直接启发了后面要讲的 WAM 系列。

**局限**：latent action 是自动归纳的，不与真实物理动作对齐，无法直接用于机器人控制。从 latent action 到真实 policy 仍需额外的对齐步骤。


## 接下来：另外两个架构族

Genie 的 latent action 技巧引出了下一页要从两个不同方向回答的问题：LoopWM 追问动力学模型的深度能否与参数量解耦，WAM 追问世界模型和策略模型是否真的需要是两个分开的模块。两者都直接建立在这里介绍的思路之上，值得先在此停顿一下再继续。
