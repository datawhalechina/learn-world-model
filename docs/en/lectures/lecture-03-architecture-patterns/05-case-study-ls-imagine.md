# ICLR 2025 Oral | Playing MineCraft with Pure Vision on a Single 3090: LS-Imagine for Reinforcement Learning over Long Short-Term Imagination in Open Worlds

LS-Imagine plays Minecraft through pure visual observation, learning RL control policies by mimicking human players without cheats or privileged information.

Training visual reinforcement learning agents in high-dimensional open worlds presents numerous challenges. While model-based reinforcement learning (MBRL) methods improve sample efficiency by learning interactive world models, these agents often suffer from "myopia" because they are typically trained only on short imagined experience fragments. We argue that the primary challenge in open-world decision-making lies in **how to improve exploration efficiency in vast state spaces, particularly for tasks requiring long-term reward consideration**. Therefore, we propose a novel reinforcement learning method: **LS-Imagine**, which constructs a **Long Short-Term World Model** that simulates goal-driven jump-style state transitions and computes corresponding **Affordance Maps** by zooming into specific regions of single images, enabling the agent to expand its imagination horizon within limited state transition steps and explore behaviors that may yield favorable long-term rewards.

> Paper Title: Open-World Reinforcement Learning over Long Short-Term Imagination
> Authors: Jiajian Li\*, Qi Wang\*, Yunbo Wang (corresponding author), Xin Jin, Yang Li, Wenjun Zeng, Xiaokang Yang (\* equal contribution)
> Project Page: https://qiwang067.github.io/ls-imagine
> Paper Link: https://openreview.net/pdf?id=vzItLaEoDa
> Code Link: https://github.com/qiwang067/LS-Imagine

## 1. Introduction

In the context of reinforcement learning, decision-making in open worlds exhibits the following characteristics:

1. **Vast State Space**: The agent operates in an interactive environment with an enormous state space.
2. **Highly Flexible Policy**: The learned policy possesses high flexibility, enabling the agent to interact with various objects in the environment.
3. **Environmental Perception Uncertainty**: The agent cannot fully observe the internal state and physical dynamics of the external world, meaning its perception of the environment (e.g., raw images) typically carries significant uncertainty.

For example, **Minecraft** is a typical open-world game that satisfies the above properties.

Based on recent advances in visual control, the goal of open-world decision-making is to train agents to approach human-level intelligence using only high-dimensional visual observations. However, this also brings numerous challenges. For instance, in Minecraft tasks:
- High-level API-based methods (such as Voyager) perform high-level control through environment-specific APIs, which do not conform to standard visual control settings, limiting generalization ability and applicability.
- Model-free reinforcement learning methods (such as DECKARD) lack understanding of the underlying mechanisms of the environment, relying primarily on costly trial-and-error exploration, resulting in low sample efficiency and poor exploration effectiveness.
- Model-based reinforcement learning methods (such as DreamerV3), while improving sample efficiency, exhibit "myopia" problems due to optimizing policies only on short-term experience, making effective long-term exploration difficult.

To improve the efficiency of behavior learning in model-based reinforcement learning, we propose a novel method: LS-Imagine. The core of this method lies in **enabling the world model to efficiently simulate the long-term impact of specific behaviors without repeatedly performing step-by-step predictions**.

<figure style="text-align: center;">
  <img src="https://notes.sjtu.edu.cn/uploads/upload_e6f68aa4966902c40be894bca5146eae.gif" alt="Intro" width="800">
  <figcaption style="font-size: 14px; color: gray;">Figure 1: Overall framework of LS-Imagine</figcaption>
</figure>

As shown in *Figure 1*, the core of LS-Imagine lies in training a **Long Short-Term World Model** that integrates task-specific guidance during representation learning. After training, the world model can perform both **immediate state transitions** and **jump-style state transitions**, while generating corresponding intrinsic rewards, thereby optimizing the policy in a **joint space of short-term and long-term imagination**. Jump-style state transitions enable the agent to bypass intermediate states and directly simulate task-relevant future states $s_{t+H}$ in a single imagination step, encouraging the agent to explore behaviors that may yield favorable long-term rewards.

However, this approach raises a classic "chicken-and-egg" problem:
> **Without real data showing that the agent has already achieved the goal, how can we effectively train the model to simulate jump-style transitions from the current state to future states highly correlated with the goal?**

To address this problem, we continuously perform **zoom-in** operations on specific regions of observation images to simulate the continuous video frames the agent would observe while approaching that region, and perform **correlation assessment** between these video frames and the task's text description, thereby generating **Affordance Maps** to highlight potentially key regions in the observation related to the task. Based on this, we collect image observation pairs from adjacent time steps as well as image pairs spanning longer time intervals through interaction with the environment as a dataset, and train specific branches of the world model to enable it to perform **immediate state transitions** and **jump-style state transitions**. After the world model is trained, we generate a series of **imagined latent state sequences** based on the world model to optimize the agent's policy. During decision-making, jump-style state transitions can be leveraged to directly estimate long-term rewards, thereby enhancing the agent's decision-making capability.

## 2. Main Innovations and Contributions

We propose a novel model-based reinforcement learning method capable of simultaneously performing immediate state transitions and jump-style state transitions, applying them to behavior learning to improve the agent's exploration efficiency in open worlds.

LS-Imagine brings the following four specific contributions:

> 1. A world model architecture combining long-term and short-term components.
> 2. A method for generating affordance maps by simulating exploration processes through image zooming.
> 3. A novel intrinsic reward mechanism based on affordance maps.
> 4. An improved behavior learning method that incorporates long-term value estimation and operates on mixed long-short-term imagination sequences.

## 3. Method

LS-Imagine includes the following key algorithmic steps:

### 1. Affordance Map Computation

As shown in *Figure 2*, to generate affordance maps, we **simulate and evaluate the agent's exploration process** **without relying on real successful trajectories**.

<figure style="text-align: center;">
  <img src="https://notes.sjtu.edu.cn/uploads/upload_a091520394b81365d574618755a4dfd5.gif" alt="Affordance" width="800">
  <figcaption style="font-size: 14px; color: gray;">Figure 2: Affordance map computation process</figcaption>
</figure>

Specifically, for a single-frame observation image, we use a sliding bounding box to scan the entire observation image from left to right and top to bottom. For each position of the sliding bounding box, we crop 16 images starting from the original image, narrowing the field of view to focus on the region where the bounding box is located, and resize them back to the original image size, obtaining 16 consecutive frames to simulate the visual changes as the agent moves toward the region indicated by the bounding box.

Subsequently, we use the pre-trained MineCLIP model to evaluate the correlation between the simulated exploration video and the task text description, using this as the potential exploration value of that region. After the sliding bounding box scans the entire image, we fuse the correlation values from all bounding box positions to generate a complete affordance map, providing guidance for the agent's exploration.

### 2. Fast Affordance Map Generation

The affordance map computation process in step 1 above involves extensive window traversal and computation using a pre-trained video-text alignment model for each window position. This method is computationally intensive and time-consuming, making it difficult to apply to real-time tasks. To address this, we designed a multimodal U-Net architecture based on Swin-Unet, and trained this multimodal U-Net architecture using the virtual exploration-based affordance map computation method described above to annotate data as supervision signals, enabling it to efficiently generate affordance maps at each time step using visual observations and language instructions, as shown in *Figure 3*.

<figure style="text-align: center;">
  <img src="https://notes.sjtu.edu.cn/uploads/upload_640edca6ff9490db215e621936cda834.png" alt="UNet" width="600">
  <figcaption style="font-size: 14px; color: gray;">Figure 3: Efficient affordance map generation using multimodal U-Net</figcaption>
</figure>

### 3. Computing Intrinsic Rewards from Affordance Maps and Assessing the Necessity of Jump-Style State Transitions

As shown in *Figure 4*, to leverage the task-relevant prior knowledge provided by affordance maps, we compute the mean of element-wise multiplication between the affordance map and a 2D Gaussian matrix of the same size, using it as the affordance-driven intrinsic reward. This reward encourages the agent to continuously approach the target and align it in the center of the view.

<figure style="text-align: center;">
  <img src="https://notes.sjtu.edu.cn/uploads/upload_683addb53cad561141585afd7c259701.png" alt="Intrinsic" width="360">
  <figcaption style="font-size: 14px; color: gray;">Figure 4: Affordance-driven intrinsic reward computation method</figcaption>
</figure>

Furthermore, to assess the necessity of jump-style transitions during imagination, we introduce a jumping flag. As shown in *Figure 5*, when a distant task-relevant target appears in the agent's observation, it manifests as highly concentrated high-value regions on the affordance map, which also causes the kurtosis of the affordance map to increase significantly. In such cases, the agent should adopt jump-style state transitions (also called long-term transitions) to efficiently reach the target region.

<figure style="text-align: center;">
  <img src="https://notes.sjtu.edu.cn/uploads/upload_8c0cb4bbb7171d1315b62f4c8f7f1e3a.png" alt="Jumping flag based on affordance map kurtosis" width="480">
  <figcaption style="font-size: 14px; color: gray;">Figure 5: Assessment of jump-style state transition necessity</figcaption>
</figure>

### 4. Long Short-Term World Model

In LS-Imagine, the world model needs to simultaneously support immediate state transitions (short-term state transitions) and jump-style state transitions (long-term state transitions). Therefore, as shown in *Figure 6 (a)*, we designed short-term and long-term branches in the state transition model. The short-term state transition model combines the current state and action to perform single-step immediate state transitions to predict the next adjacent time step's state. The long-term transition model simulates goal-oriented jump-style state transitions, guiding the agent to rapidly imagine exploration toward the goal. The agent can decide which type of transition to adopt based on the current state and predict the next state through the selected transition branch.

<figure style="text-align: center;">
  <img src="https://notes.sjtu.edu.cn/uploads/upload_7aba677832aaeecff96e1a91a0f9932b.png" alt="Long short-term world model architecture and behavior learning" width="800">
  <figcaption style="font-size: 14px; color: gray;">Figure 6: Long short-term world model architecture and behavior learning based on long short-term imagination</figcaption>
</figure>

Unlike traditional world model architectures, we specifically designed a Jump predictor to determine which type of transition should be performed based on the current state. Additionally, for jump-style state transitions, we designed an Interval predictor to estimate the number of environment time steps $\hat {\Delta}_t^\prime$ between states before and after the jump, as well as the cumulative discounted reward $\hat G_t^\prime$ during that period, which will be used to estimate long-term rewards in subsequent behavior learning. Furthermore, we also input the affordance map $\mathcal{M}_t$ to the encoder, which can provide goal-based prior guidance for the agent to enhance the effectiveness of the decision-making process.

Based on this architecture, the agent interacts with the environment and collects new data, obtaining sample pairs from adjacent time steps corresponding to short-term state transitions, and modeling sample pairs spanning longer time intervals corresponding to long-term state transitions based on affordance maps. We use this data to update the replay buffer and sample from it to train the long short-term world model.

### 5. Behavior Learning on Long Short-Term Imagination Sequences

As shown in *Figure 6 (b)*, LS-Imagine employs an **actor-critic algorithm** to learn behavior through latent state sequences predicted by the world model. The actor's objective is to optimize the policy to maximize the discounted cumulative reward $R_t$, while the critic's role is to estimate the discounted cumulative reward for each state based on the current policy.

<figure style="text-align: center;">
  <img src="https://notes.sjtu.edu.cn/uploads/upload_f97361fef42f533d4d71c44ce522febb.png" alt="Long short-term imagination sequence rollout" width="800">
  <figcaption style="font-size: 14px; color: gray;">Figure 7: Dynamically selecting long-term or short-term transition models to predict long short-term imagination sequences</figcaption>
</figure>

As shown in *Figure 7*, starting from the initial state encoded from sampled observations and affordance maps, we dynamically select the long-term or short-term state transition model based on the jumping flag $\hat{j}_t$ predicted by the jump predictor to predict subsequent states. In a long short-term imagination sequence with **imagination horizon $L$**, we predict information such as the reward $\hat{r}_t$ corresponding to the state, the continue flag $\hat{c}_t$, the number of environment time steps $\hat {\Delta}_t$ between adjacent states, and the cumulative discounted reward $\hat G_t$ during that period through various predictors in the world model, and adopt an improved bootstrap $\lambda$-returns combining long-term and short-term imagination to compute the discounted cumulative reward for each state:

$$
R_{t}^{\lambda} \doteq \begin{cases}
\hat{c}_{t} \{\hat{G}_{t+1} + \gamma^{\hat{\Delta}_{t+1}} \left[ (1-\lambda) v_{\psi} (\hat{s}_{t+1}) + \lambda R_{t+1}^{\lambda} \right] \} & \text{if } t < L \\
v_{\psi} (\hat{s}_{L}) & \text{if } t = L
\end{cases},
$$

and employ the actor-critic algorithm for behavior learning.

## 4. Experimental Results

We conducted experiments in the Minecraft game environment to test the LS-Imagine agent. We set up 5 open-ended tasks as shown in *Table 1* for experimentation:

<center><figcaption style="font-size: 14px; color: gray;">Table 1: Minecraft task descriptions</figcaption></center>



<div style="display: flex; justify-content: center;">
    <table style="width: 60%; table-layout: fixed; border-collapse: collapse; text-align: center;">
      <colgroup>
        <col style="width: 40%;">
        <col style="width: 40%;">
        <col style="width: 20%;">
      </colgroup>
      <thead>
        <tr>
          <th style="border-bottom: 2px solid black; text-align: center;">Task</th>
          <th style="border-bottom: 2px solid black; text-align: center;">Language Description</th>
          <th style="border-bottom: 2px solid black; text-align: center;">Max Steps</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Collect logs in plains</td>
          <td>"Cut a tree."</td>
          <td>1000</td>
        </tr>
        <tr>
          <td>Collect water with bucket</td>
          <td>"Obtain water."</td>
          <td>1000</td>
        </tr>
        <tr>
          <td>Collect sand</td>
          <td>"Obtain sand."</td>
          <td>1000</td>
        </tr>
        <tr>
          <td>Shear sheep</td>
          <td>"Obtain wool."</td>
          <td>1000</td>
        </tr>
        <tr>
          <td>Mine iron ore</td>
          <td>"Mine iron ore."</td>
          <td>2000</td>
        </tr>
      </tbody>
    </table>
</div>

We compared LS-Imagine with various methods including VPT, STEVE-1, PTGM, Director, and DreamerV3. The evaluation metrics include **success rate in completing tasks within specified steps** and **average interaction steps required to complete tasks**. The numerical results are shown in *Table 2*.

<center><figcaption style="font-size: 14px; color: gray;">Table 2: Numerical results for success rate and interaction steps required to complete tasks</figcaption></center>
<div align="center">
<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="2">Collect logs in plains</th>
      <th colspan="2">Collect water with bucket</th>
      <th colspan="2">Collect sand</th>
      <th colspan="2">Shear sheep</th>
      <th colspan="2">Mine iron ore</th>
    </tr>
    <tr>
      <th>succ. (%)</th>
      <th>succ. step</th>
      <th>succ. (%)</th>
      <th>succ. step</th>
      <th>succ. (%)</th>
      <th>succ. step</th>
      <th>succ. (%)</th>
      <th>succ. step</th>
      <th>succ. (%)</th>
      <th>succ. step</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>VPT</td>
      <td>6.97</td>
      <td>963.32</td>
      <td>0.61</td>
      <td>987.65</td>
      <td>12.99</td>
      <td>880.54</td>
      <td>1.94</td>
      <td>987.49</td>
      <td>0.00</td>
      <td>—</td>
    </tr>
    <tr>
      <td>STEVE-1</td>
      <td>57.00</td>
      <td>752.47</td>
      <td>6.00</td>
      <td>989.07</td>
      <td>37.00</td>
      <td>770.40</td>
      <td>3.00</td>
      <td>992.36</td>
      <td>0.00</td>
      <td>—</td>
    </tr>
    <tr>
      <td>PTGM</td>
      <td>41.86</td>
      <td>811.19</td>
      <td>2.78</td>
      <td>977.78</td>
      <td>17.71</td>
      <td>833.64</td>
      <td>21.54</td>
      <td>887.03</td>
      <td>15.14</td>
      <td><b>1586.03</b></td>
    </tr>
    <tr>
      <td>Director</td>
      <td>8.67</td>
      <td>968.09</td>
      <td>20.90</td>
      <td>931.74</td>
      <td>36.36</td>
      <td>825.35</td>
      <td>1.27</td>
      <td>995.99</td>
      <td>7.82</td>
      <td>1906.31</td>
    </tr>
    <tr>
      <td>DreamerV3</td>
      <td>53.33</td>
      <td>711.22</td>
      <td>55.72</td>
      <td>628.79</td>
      <td>59.88</td>
      <td><b>548.76</b></td>
      <td>25.13</td>
      <td>841.14</td>
      <td>16.79</td>
      <td>1789.06</td>
    </tr>
    <tr>
      <td><b>LS-Imagine</b></td>
      <td><b>80.63</b></td>
      <td><b>503.35</b></td>
      <td><b>77.31</b></td>
      <td><b>502.61</b></td>
      <td><b>62.68</b></td>
      <td>601.18</td>
      <td><b>54.28</b></td>
      <td><b>633.78</b></td>
      <td><b>20.28</b></td>
      <td>1748.55</td>
    </tr>
  </tbody>
</table>
</div>

We found that **LS-Imagine** performs significantly better than comparison models, with its advantages being particularly pronounced **in task scenarios with sparsely distributed targets**.

Additionally, we present visualization results of reconstructed observation images and affordance maps based on long short-term imagination state sequences in *Figure 10*. The first row shows **latent states before and after jump-style state transitions, decoded back to pixel space** to intuitively present state changes. The second row visualizes **affordance maps reconstructed from latent states** to more clearly understand how affordance maps facilitate jump-style state transitions and whether they can provide effective goal-oriented guidance. The last row **overlays affordance maps on reconstructed observation images through transparent superposition** to more intuitively highlight the regions the agent focuses on.

<figure style="text-align: center;">
  <img src="https://notes.sjtu.edu.cn/uploads/upload_1fbb8ad05b0fc85b35061e41fb057206.gif" alt="Visualization of long short-term imagination sequences" width="800">
  <figcaption style="font-size: 14px; color: gray;">Figure 10: Visualization of long short-term imagination sequences</figcaption>
</figure>

These visualization results demonstrate that LS-Imagine's long short-term world model can **adaptively decide when to perform long-term imagination based on current visual observations**. Furthermore, the generated affordance maps can **effectively align with regions highly correlated with the final goal**, thereby facilitating more efficient policy exploration by the agent.

Moreover, given that our method relies on affordance maps to identify high-value exploration regions to achieve long-term state jumps, one might think that if the target is occluded or invisible, our method would fail. To demonstrate that **our affordance map generation method is not merely a target recognition algorithm and does not only highlight relevant regions when targets are visible**, we present examples of affordance maps generated when targets are occluded or invisible in *Figure 11*.

<figure style="text-align: center;">
  <img src="https://notes.sjtu.edu.cn/uploads/upload_2781ce68ace8424ad9350dad8c929a65.png" width="800">
  <figcaption style="font-size: 14px; color: gray;">Figure 11: Affordance maps when targets are occluded or invisible</figcaption>
</figure>

Thanks to the MineCLIP model's pre-training on a large number of expert demonstration videos, **our affordance map generation method can generate affordance maps that provide effective guidance for exploration even when targets are completely occluded or invisible**. For example, as shown in *Figure 11(a)*, in the **village-finding task**, although the village is not visible in the current observation, the affordance map can still provide clear exploration directions, suggesting the agent explore toward the forest on the right or the open area on the left hillside. Similarly, in the **mining task** shown in *Figure 11(b)*, although ores are typically located underground and occluded in the current observation, the affordance map can still guide the agent to dig into the mountain on the right or underground ahead. These examples fully demonstrate that **even when targets are occluded, affordance maps can still help agents explore effectively**.

## 5. Conclusion

Our work proposes a novel method, LS-Imagine, aimed at overcoming the challenges faced in training visual reinforcement learning agents in high-dimensional open worlds. By expanding the imagination horizon and leveraging a long short-term world model, LS-Imagine can efficiently perform policy exploration in vast state spaces. Additionally, introducing goal-based jump-style state transitions and affordance maps enables the agent to better understand long-term value, thereby enhancing its decision-making capability. Experimental results show that in the Minecraft environment, LS-Imagine achieves significant performance improvements compared to existing methods. This not only highlights LS-Imagine's potential in open-world reinforcement learning but also provides new inspiration for future research in this field.

The paper's code, checkpoints, and environment configuration documentation are all provided. We welcome GitHub stars and citations!

GitHub link: https://github.com/qiwang067/LS-Imagine

Citation:

```bibtex
@inproceedings{li2025open,
    title={Open-World Reinforcement Learning over Long Short-Term Imagination},
    author={Jiajian Li and Qi Wang and Yunbo Wang and Xin Jin and Yang Li and Wenjun Zeng and Xiaokang Yang},
    booktitle={ICLR},
    year={2025}
}
```
