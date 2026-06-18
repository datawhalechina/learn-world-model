---
title: 项目
---

# 项目

五个动手项目从零开始构建一条完整的世界模型流水线。每个项目都是一个自包含的 Jupyter notebook：基于合成数据训练，无需任何外部下载，可在 CPU、GPU 或 TPU 上运行，并保存一个供下一个项目加载的权重文件。请按顺序完成：P01 的编码器成为 P02 的观测编码器，P02 的动力学模型成为 P03 的骨干网络和 P04 的基线，P03 与 P04 训练出的两套系统在 P05 中进行对比。

这些 markdown 页面只保留叙述文字和代码。所有 output、图表、表格和其他产物都请到对应的 `.ipynb` 文件中查看。

notebook 与这些页面放在一起。在 Jupyter 或 Colab 中打开任意一个，从头到尾运行，训练好的权重就会落到工作目录中，供下游项目读取。当上游权重缺失时，每个 notebook 都会回退到随机初始化，因此单个项目即使独立运行也不会报错，可作为冒烟测试；但只有在真实权重就位后，跨项目的对比结果才具有意义。

## 项目流程

| # | 项目 | 前置 | 产出 | 交付物 |
|---|------|------|------|--------|
| P01 | [训练 VAE 编码器](./p01_vae_encoder) | L02 Part A | `vae_encoder.pt` | 64×64 帧上的 CNN VAE；ELBO 损失曲线；展示解耦维度的潜在遍历 |
| P02 | [构建 RSSM 动力学模型](./p02_rssm_dynamics) | P01、L02 Part B | `rssm.pt` | GRU、MDN-RNN 与 RSSM 对比；rollout 图；1 步到 5 步预测误差曲线 |
| P03 | [训练 Dreamer 智能体](./p03_dreamer_agent) | P02、L03 Part B | `dreamer.pt` | 编码器 + RSSM + 潜在 Actor-Critic 训练循环；奖励曲线；FID 与奖励相关性自评 |
| P04 | [替换动力学骨干网络](./p04_transformer_backbone) | P02、L03 Part A | `transformer_wm.pt` | 用 STORM 风格的类别 VAE 加因果 Transformer 替换 RSSM；架构对比报告 |
| P05 | [世界模型评估仪表盘](./p05_evaluation_dashboard) | P03、P04、L04 | -- | 加载两套训练好的模型并排打分：PSNR、奖励相关性、token 损失与潜在漂移 |

## 权重文件如何串联

各项目共享一组在流水线中向前传递的权重文件。P01 训练 VAE 并写出 `vae_encoder.pt`。P02 加载该编码器，训练动力学模型，写出 `rssm.pt`。此后路径分叉：P03 把编码器和 RSSM 组合成 Dreamer 智能体，保存为 `dreamer.pt`；P04 则复用 RSSM 作为基线，训练 Transformer 骨干，保存为 `transformer_wm.pt`。P05 加载 `dreamer.pt` 与 `transformer_wm.pt` 完成最终评估，闭合整条流水线。

```mermaid
graph TD
    P01[P01 VAE 编码器] -->|vae_encoder.pt| P02[P02 RSSM 动力学]
    P01 -->|vae_encoder.pt| P03[P03 Dreamer 智能体]
    P02 -->|rssm.pt| P03
    P02 -->|rssm.pt| P04[P04 Transformer 骨干]
    P03 -->|dreamer.pt| P05[P05 评估仪表盘]
    P04 -->|transformer_wm.pt| P05
```
