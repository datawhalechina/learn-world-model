---
title: 项目
---

# 项目

六个动手项目从零开始构建一条完整的世界模型流水线。请按顺序完成：P01 的编码器成为 P02 的观测编码器，P02 的动力学模型成为 P03 的骨干网络和 P04 的基线，P03 与 P04 训练出的两套系统在 P05 中进行对比，P06 再对这两套系统的因果保真度进行探查。每个项目都是 notebook 优先的教程，可在 CPU、GPU 或 TPU 上运行，只使用合成数据，并把权重文件继续传给下一步。

## 硬件要求

本章节的每个 notebook 都是在 Google Colab 上使用单块 T4 GPU（16 GB）开发并运行的。任何显存和算力相当或更高的加速卡，无论是同代或更新的 Nvidia GPU、AMD GPU 还是 TPU，都能无需任何改动地运行全部六个项目。一块中端消费级 GPU 就足够，所有项目都不需要多卡训练。

如果你手头没有合适的 GPU 机器，以下云平台都可以满足需求：

| 平台 | 硬件 | 适用场景 | 链接 |
|---|---|---|---|
| Google Colab | T4、L4、A100 | 本课程的参考环境；免费版可用于冒烟测试，Pro 版提供稳定的 T4/L4 算力 | [colab.research.google.com/signup](https://colab.research.google.com/signup) |
| Kaggle Notebooks | 双 T4、P100 | 每周 30 小时免费 GPU 时长，无需订阅 | [kaggle.com/docs/notebooks](https://www.kaggle.com/docs/notebooks) |
| AMD Radeon Cloud | Radeon | 免费 150 小时 GPU 时长，一键启动本课程 notebook | [developer.amd.com.cn](https://developer.amd.com.cn/login?source=eHjyREaw5) |
| Lambda Cloud | A10、A100、H100 | 按小时计费的 Nvidia 按需实例，无需长期承诺 | [lambda.ai/service/gpu-cloud](https://lambda.ai/service/gpu-cloud) |
| RunPod | 型号覆盖广，社区版与安全云两档 | 短期训练任务性价比最高的按需与抢占式定价 | [runpod.io](https://www.runpod.io/) |
| Google Cloud TPU | TPU v4/v5e | 专门验证 TPU 代码路径 | [cloud.google.com/tpu](https://cloud.google.com/tpu) |

以上平台均已验证可以无需改动直接运行这些 notebook。代码只使用标准的 PyTorch 算子，不含任何 CUDA 专用调用，因此在 AMD 硬件的 ROCm 环境下也能无需修改直接运行。

这些 markdown 页面只保留叙述文字和代码，output、图表、表格等产物都在对应的 `.ipynb` 文件中查看。

打开任意 notebook 并从头到尾运行即可。如果上游权重缺失，notebook 会回退到随机初始化，仍可作为冒烟测试，但只有在真实权重就位后，跨项目的对比结果才具有意义。

## 项目流程

| # | 项目 | 前置 | 产出 | 交付物 |
|---|------|------|------|--------|
| P01 | [训练 VAE 编码器](./p01_vae_encoder) | L02 Part A | `vae_encoder.pt` | 64×64 帧上的 CNN VAE；ELBO 损失曲线；展示解耦维度的潜在遍历 |
| P02 | [构建 RSSM 动力学模型](./p02_rssm_dynamics) | P01、L02 Part B | `rssm.pt` | GRU、MDN-RNN 与 RSSM 对比；rollout 图；1 步到 5 步预测误差曲线 |
| P03 | [训练 Dreamer 智能体](./p03_dreamer_agent) | P02、L03 Part A | `dreamer.pt` | 编码器 + RSSM + 潜在 Actor-Critic 训练循环；奖励曲线；FID 与奖励相关性自评 |
| P04 | [替换动力学骨干网络](./p04_transformer_backbone) | P03、L03 Part B | `transformer_wm.pt` | 用 STORM 风格的类别 VAE 加因果 Transformer 替换 RSSM；架构对比报告 |
| P05 | [世界模型评估仪表盘](./p05_evaluation_dashboard) | P03、P04、L04 | -- | 加载两套训练好的模型并排打分：PSNR、奖励相关性、token 损失与潜在漂移 |
| P06 | [反事实的动作条件世界模型](./p06_counterfactual_world_model) | P03、P04 | `causal_wm.pt` | 因果之梯分析：干预与反事实推演、逆动力学正则化的世界模型，以及动作影响度指标 |

## 权重文件如何串联

各项目共享一组在流水线中向前传递的权重文件。P01 训练 VAE 并写出 `vae_encoder.pt`；P02 加载该编码器，训练动力学模型，写出 `rssm.pt`。此后路径分叉：P03 把编码器和 RSSM 组合成 Dreamer 智能体，保存为 `dreamer.pt`；P04 则复用 RSSM 作为基线，训练 Transformer 骨干，保存为 `transformer_wm.pt`。P05 加载 `dreamer.pt` 与 `transformer_wm.pt` 完成准确度评估，P06 再加载同样的两套权重探查因果保真度，并训练一个自有的动作正则化模型，保存为 `causal_wm.pt`。

```mermaid
graph TD
    P01[P01 VAE 编码器] -->|vae_encoder.pt| P02[P02 RSSM 动力学]
    P01 -->|vae_encoder.pt| P03[P03 Dreamer 智能体]
    P02 -->|rssm.pt| P03
    P02 -->|rssm.pt| P04[P04 Transformer 骨干]
    P03 -->|dreamer.pt| P05[P05 评估仪表盘]
    P04 -->|transformer_wm.pt| P05
    P03 -->|dreamer.pt| P06[P06 反事实世界模型]
    P04 -->|transformer_wm.pt| P06
```
