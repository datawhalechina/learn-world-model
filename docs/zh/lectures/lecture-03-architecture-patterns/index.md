---
title: 从潜在动力学到规划与架构选择
description: 先用 CEM-MPC、潜在 Actor-Critic 与 TD-MPC 补完整智能体闭环，再比较替代动力学骨干，并按需阅读前沿系统。
lecture: 3
difficulty: 中高
---

# 从潜在动力学到规划与架构选择

你在 P02 中实现的 RSSM 已经能预测潜在未来，却还不能告诉智能体最终应执行哪个动作。本讲先补完整这个闭环。只有能够追踪预测如何影响动作之后，我们才比较替代模型架构。

- **Part A，核心规划闭环**：CEM-MPC 搜索、Dreamer 潜在 Actor-Critic 与 TD-MPC。读完后完成 P03。
- **Part B，核心骨干选择**：以 RSSM 为基线，再判断何时值得换用 Transformer 或扩散骨干。读完后完成 P04。
- **Part C，可选前沿综述**：JEPA、RWM、空间 3D/4D 世界模型、Genie、LoopWM、WAM、系统接入模式与 VLA 机制、以及 LS-Imagine 案例。这些页面用于扩展研究判断，不是 P03 或 P04 的前置要求。

如果目标是搭建可运行系统，请依次读完 Part A 与 Part B，把 Part C 当作选读。如果目标是建立文献地图，则在完成核心路线后继续阅读 Part C。
