# 课程大纲：第 0–19 章 + 5 个 Programming Assignments

## 先修与资源

零基础导论不设先修，初中生可以从概念、纸上活动和小游戏开始。进入第 3 章后的编程主线，需要逐步学习 Python；线性代数、概率和 PyTorch 由配套“够用数学”单元随用随学。强化学习不是先修。

建议资源：PA0 只需 CPU；PA1 建议 4 GB 显存；PA2–PA3 建议 12–24 GB 单卡。所有作业必须提供 tiny 配置，保证没有 GPU 也能跑通正确性测试。

## 第零部分：先学会看地图（无需数学）

### 第 0 章　机器人到底需要几个“大脑”？

- 从“把红色积木放进盒子”认识 perception、reasoning、action 与 prediction
- VLM、VLA、policy、world model、planner、simulator、WAM
- 直接行动、先想再做、混合系统
- 练习：看到系统输入输出就判断它属于什么

正文：[第 0 章](chapters/00-robot-brain-map.md)

### 第 1 章　七个方向不是七个名字

- Latent MBRL、互动视频、JEPA、3D/4D、驾驶、机器人 VLA/WAM、数字 Agent WM
- 用“动作输入、预测对象、服务对象、评价方法、数据来源”五问分类
- 练习：给 Dreamer、Genie、V-JEPA、GAIA、OpenVLA 等放置坐标

正文：[第 1 章](chapters/01-choose-a-world-model-direction.md)

### 第 2 章　世界不只有机器人：数字 Agent 世界

- Web、终端、OS、Android、数据库和 MCP 也是环境
- 区分 executable environment generation 与 predictive language world model
- 区分 code-as-WM、LLM-as-WM 与 hybrid
- 训练数据生成、推理时 lookahead、action verification 和运行时安全

正文：[第 2 章](chapters/02-digital-agent-worlds.md)

### 桥梁单元　够用的 Python、概率和向量

- 状态、列表、函数、随机数和画图
- 概率分布、条件概率、期望
- 向量只是“一排记录特征的数”；矩阵只是批量变换
- 用 7 格世界边做边学，不先上完整数学课

## 第一部分：世界、模型与决策

### 第 3 章　用代码造第一个世界模型

- 从 agent–environment loop 定义 state、observation、action、history、policy、model
- 区分 renderer、simulator、planner；区分 predictor、simulator、evolver
- 第一个反例：视频越清晰，规划不一定越好
- 实验：30 分钟内在 LineWorld 上比较无模型策略与精确模型规划

### 第 4 章　白盒世界：MDP、POMDP 与贝尔曼方程

- 转移核、奖励、终止、折扣和 return
- 部分可观测下为什么需要 memory / belief state
- 手算 value iteration 与 model predictive control
- 实现：不依赖 RL 库的表格环境和 planner

### 第 5 章　从经验中学习动力学

- maximum likelihood、count model、supervised dynamics
- aleatoric 与 epistemic uncertainty
- train/validation 切分为什么必须按轨迹考虑
- 实验：数据覆盖率、平滑和规划性能的关系

### 第 6 章　模型会撒谎：误差、分布偏移与利用漏洞

- one-step error 与 compounding error
- uncertainty-aware planning、ensemble、short rollout
- model exploitation 和 offline data coverage
- 评估：calibration、multi-step rollout、policy ranking agreement

### PA0　会做梦的 7 格世界

实现计数式随机动力学、value iteration 和闭环评估。详见 [PA0](../pa/pa0/README.md)。

## 第二部分：从像素学会“状态”

### 第 7 章　像素不是状态：表征学习

- sufficient state、Markov property、invariance
- autoencoder、VAE、β-VAE
- reconstruction quality 与 control sufficiency 的冲突
- 实验：同样重建误差，不同 latent 对规划的影响

### 第 8 章　把世界切成 token

- VQ-VAE / VQGAN 的 codebook、commitment loss、dead codes
- video token 的空间与时间压缩率
- 手算 codebook update；实现最小 vector quantizer

### 第 9 章　记忆与随机性：RSSM

- deterministic hidden state + stochastic state
- prior、posterior、KL balancing、free nats
- observation / reward / continue heads
- teacher forcing 与 open-loop imagination

### 第 10 章　在梦里规划

- random shooting、CEM、MPC
- latent rollout、reward prediction、terminal prediction
- 与真实环境闭环比较
- 实验：horizon 越长为什么可能越差

### PA1　PixelWorld：从像素恢复可规划状态

交付：数据生成器、VAE 或 VQ tokenizer、latent dynamics、CEM planner。核心指标同时报告 reconstruction、multi-step prediction、return 和 wall-clock。

## 第三部分：从 Dreamer 到现代生成世界

### 第 11 章　Dreamer-lite：在 imagination 中训练策略

- actor、critic、λ-return、stop-gradient
- dynamics learning 与 behavior learning 的更新边界
- 实现：小型 continuous-control 或 visual-control agent

### 第 12 章　MuZero 视角：只预测决策所需信息

- representation、dynamics、prediction 三个网络
- value/reward/policy targets 与 search
- 和 observation reconstruction 路线的对照

### 第 13 章　Transformer 世界模型

- causal token dynamics、action tokenization、scheduled sampling
- KV cache 与实时交互
- IRIS / Δ-IRIS 一类模型的最小复现思路
- 实验：tokenizer bottleneck 与 temporal model bottleneck 谁更关键

### 第 14 章　Diffusion 与 flow 世界模型

- 为什么未来是多模态分布
- diffusion / flow matching 用于 video dynamics
- action conditioning、classifier-free guidance、temporal consistency
- 与 autoregressive model 在质量、延迟、可交互性上的比较

### PA2　Dreamer-lite

从 RSSM 开始完成 imagination actor-critic。禁止调用现成 Dreamer 实现。设 hidden test 检查 KL、λ-return、stop-gradient 和 rollout shape。

### PA3　两种可交互像素世界

在相同数据、token budget 和训练时长下，对比 causal Transformer 与 diffusion/flow dynamics；必须进行闭环 agent evaluation，而非只交视频样例。

## 第四部分：World Foundation Models

### 第 15 章　JEPA：预测特征，不预测每个像素

- joint-embedding prediction、target encoder、collapse prevention
- actionless pretraining → action-conditioned adaptation
- V-JEPA 2 式 planning 接口
- 对照：generative loss、contrastive loss、feature prediction loss

### 第 16 章　空间智能：从 2D 视频到 3D/4D 状态

- camera geometry、depth、point cloud、occupancy、scene flow
- NeRF、3D Gaussian、mesh 的教学级直觉
- object permanence、多视角一致性、动态场景
- renderer 和 simulator 的结构分界

### 第 17 章　数据、系统与扩展律

- 数据混合：video、action-labeled video、simulation、robot trajectories
- video tokenizer 吞吐、序列长度、显存与 FLOPs 估算
- distributed training 只讲世界模型特有瓶颈
- 规模实验：固定 compute 下分配 tokenizer / dynamics / data budget

### 第 18 章　VLA、World Model 与机器人行动

- VLM → VLA：从语义 token 到离散/连续 robot action
- behavior cloning、Diffusion Policy、action chunking、flow matching
- RT-2、OpenVLA、Octo、π0、GR00T、Gemini Robotics 的接口差异
- action-first、model-first、world-action model 三种系统设计
- robot data、Open X-Embodiment、仿真数据、互联网视频与 embodiment gap
- 实验：2D language pick-and-place 中比较 VLA 直接行动与 world-model lookahead

### 第 19 章　评估、安全与走向真实世界

- open-loop：perception、FVD/LPIPS 类生成指标、rollout error
- closed-loop：counterfactual fidelity、return、policy ranking、optimization lift
- uncertainty、OOD、模型被 planner 利用、因果混淆
- 驾驶、机器人、游戏三种部署边界和安全要求
- 研究报告怎么避免 cherry-pick 好看轨迹

## 第五部分：毕业设计

### PA4　三选一 capstone

#### A. Interactive Game World

在 MiniGrid / Atari 子集 / 开源游戏轨迹上训练 action-conditioned world model；agent 必须在生成环境和真实环境中都评估。

#### B. Driving World Model

使用公开驾驶数据做多视角 future prediction、BEV/occupancy forecasting 或小型闭环 CARLA 项目；明确传感器、动作和安全指标。

#### C. Robot World Model

使用公开 manipulation 数据或仿真器，比较 model-based planning 与 behavior cloning / VLA-style policy；必须报告 OOD 物体或布局泛化。

交付统一包含：proposal、数据卡、可复现实验、负结果、模型卡、5 页报告、10 分钟 demo。最终分数以问题定义、实验可信度和闭环效用为主，不以视频华丽程度为主。

## 12 周建议节奏

| 周 | 内容 | 里程碑 |
|---|---|---|
| 1 | 第 0–3 章 | 方向判断，PA0 out |
| 2 | 第 4–6 章 | PA0 due |
| 3 | 第 7–8 章 | PA1 out |
| 4 | 第 9–10 章 | PA1 checkpoint |
| 5 | 第 11 章 | PA1 due，PA2 out |
| 6 | 第 12 章 | PA2 checkpoint |
| 7 | 第 13 章 | PA2 due，PA3 out |
| 8 | 第 14–15 章 | PA3 checkpoint |
| 9 | 第 16 章 | PA3 due，PA4 proposal |
| 10 | 第 17–18 章 | capstone baseline |
| 11 | 第 19 章 | capstone evaluation |
| 12 | 展示与复盘 | PA4 due |

## 评分建议

- PA0 10%，PA1 15%，PA2 20%，PA3 20%，PA4 30%，阅读/复现实验记录 5%。
- 每个 PA 中：正确性 40%，实验设计 25%，闭环评价 20%，代码可读性与复现 15%。
- leaderboard 只作为反馈，不把昂贵算力直接换成成绩。
