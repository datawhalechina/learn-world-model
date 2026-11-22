---
title: "两两对读与原论文"
description: "展开两篇论文，比较未来预测在哪一步影响动作。"
lecture: 6
---

# 两两对读与原论文

展开两篇论文，比较未来预测在哪一步影响动作。

## 两两对读

<details>
<summary>Fast-WAM · Motus</summary>

两篇都能在部署时直接输出动作。Fast-WAM 用掩码阻断动作对未来视频的依赖，Motus 靠分开设置视频和动作的噪声进度，让同一模型切换直出和联合模式。

</details>

<details>
<summary>Fast-WAM · LingBot-VA</summary>

Fast-WAM 压缩当前观测到动作的计算时间。LingBot-VA 则在动作执行期间提前计算下一段，收益取决于模型能否用最新反馈修正过时的预测。

</details>

<details>
<summary>DreamZero · Fast-WAM</summary>

同样把视频预测作为训练目标，Fast-WAM 在部署时能删掉未来视频 token，因为动作只读首帧特征。DreamZero 的动作和未来画面在同一段内联合去噪，每一轮都更新两者。

</details>

<details>
<summary>Cosmos Policy · Fast-WAM</summary>

Fast-WAM 在产生一段动作之前省下视频计算。Cosmos 的规划收益来自比较多种候选的后果。一次预测是否影响动作，要看策略是否还拥有其他候选可选。

</details>

<details>
<summary>LingBot-VA · Motus</summary>

Motus 的逆动力学模式以给定的视觉序列反推动作。LingBot-VA 把下一段视觉变化也交给模型预测，并在前后两段之间重新引入真实反馈，形成闭环。

</details>

<details>
<summary>DreamZero · Motus</summary>

面对没有动作标签的视频，Motus 从光流提取潜在动作来训练动作专家。DreamZero 只对来自其他机器人或人类的额外录像计算视觉损失，执行动作的监督仍来自目标机器人。

</details>

<details>
<summary>Cosmos Policy · Motus</summary>

两者都能用条件生成回答“给定动作会看到什么”。Cosmos 在规划时还把预测的终点送入价值函数，让同一起点的多段动作接受同一套排序。

</details>

<details>
<summary>DreamZero · LingBot-VA</summary>

两者都能用实拍更新历史。LingBot-VA 先预测下段画面，再从视觉变化反推动作。DreamZero 在同一轮中同时更新画面与动作。

</details>

<details>
<summary>Cosmos Policy · LingBot-VA</summary>

LingBot-VA 从最新反馈估计正在执行的动作将到哪里，以便提前计算下一段。Cosmos 在执行前比较同一起点的多个可能终点。前者对齐时间，后者对候选排序。

</details>

<details>
<summary>Cosmos Policy · DreamZero</summary>

DreamZero 把预测视频和动作作为同一次联合采样的结果。Cosmos 规划模式先提出动作，再对各自后果打分。两者的区别在于一次联合生成就能交付动作，还是要在多个候选之间作选择。

</details>

## 原论文

- [Fast-WAM: Do World Action Models Need Test-time Future Imagination?](https://arxiv.org/abs/2603.16666)
- [Motus: A Unified Latent Action World Model](https://arxiv.org/abs/2512.13030)
- [Causal World Modeling for Robot Control](https://arxiv.org/abs/2601.21998)
- [World Action Models are Zero-shot Policies](https://arxiv.org/abs/2602.15922)
- [Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning](https://arxiv.org/abs/2601.16163)

文章的编排参考 [Nemo’s Diffusion RL guide](https://knightnemo.github.io/blog/posts/diffusion-rl-zh/).
