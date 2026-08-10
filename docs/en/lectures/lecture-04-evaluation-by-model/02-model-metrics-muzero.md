---
title: "Value and Search Quality: MuZero"
description: Search quality and representation stability evaluation metrics and diagnostic rules for the implicit world model MuZero.
lecture: 4
---

# Value and Search Quality: MuZero as a Worked Example

## MuZero (Implicit World Model)

MuZero does not reconstruct pixels. Its world model is entirely hidden inside three networks: the representation function, the dynamics function, and the prediction function. Evaluating it requires approaching from two angles: **search quality** and **representation stability**.

### Value Accuracy

Before MCTS search, the network produces an initial value estimate `V₀` at the root node. After search completes, a refined estimate `V*` is obtained by weighting Q-values by visit counts. The mean squared error between the two measures how accurately the network can act "without searching."

$$\text{ValueAcc} = 1 - \frac{\text{MSE}(V_0, V^*)}{\text{Var}(V^*)}$$

Values closer to 1 are better. In a mature MuZero, `V₀` should closely track `V*`, with search serving as "verification" rather than "correction."

**Diagnostic rule**: If value accuracy remains below 0.6 for an extended period, the reward model (the `r̂` output of the dynamics function) needs retraining, or the replay buffer contains too many out-of-distribution samples from old policies. Try increasing the weight of **priority replay** (prioritized experience replay, which assigns sampling probability based on each transition's TD error: a larger error indicates the model predicts that sample poorly, so it is sampled more frequently for training) to make recent data sampled more often.

### MCTS Visit Entropy

In the search tree, the visit counts `n_i` of each child node form a distribution. Entropy is defined as:

$$H = -\sum_i \frac{n_i}{N} \log \frac{n_i}{N}$$

**High entropy** indicates the model is uncertain across many actions and searches broadly. **Low entropy** indicates strong confidence in a particular action.

**Important context dependence**: high entropy is not necessarily bad. In highly stochastic games (such as the opening of Go), many moves objectively have similar value, and high entropy correctly reflects the reality that "multiple actions are valuable." Low entropy only signals healthy confidence convergence late in the game when the outcome is clear. During early training or in highly stochastic positions, abnormally low entropy instead indicates the model has prematurely biased toward certain actions, a sign of insufficient coverage rather than convergence.

**Diagnostic rule**: If entropy is consistently very low and value accuracy is also low, the model is exhibiting pseudo-confidence, making judgments without genuine basis. This typically requires adding exploration noise (**Dirichlet noise**, random noise sampled from a Dirichlet distribution and added to the prior policy distribution at the root node, forcing MCTS to explore different actions early in search and preventing it from always searching only the branches the policy network considers optimal) or increasing replay buffer diversity.

### Representation Stability

This is a diagnostic metric specific to MuZero, used to check the robustness of the representation network.

$$\text{Stability} = \mathbb{E}_o\!\left[\cos\_\text{sim}(h(o),\, h(o + \varepsilon))\right]$$

> **📖 Cosine similarity**: measures the similarity in direction between two vectors, independent of vector magnitude: $\cos\_sim(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{|\mathbf{u}||\mathbf{v}|}$. The range is $[-1, 1]$: 1 means identical direction, 0 means orthogonal (unrelated), -1 means opposite direction. It is used here instead of Euclidean distance because the absolute magnitude of representation vectors is not important. What matters is whether they "point in the same direction" in high-dimensional space: same direction means the model assigns similar semantics to two similar inputs.

Given a position `o` with a slight random perturbation `ε` added (e.g., Gaussian noise applied to image observations with standard deviation around 1% of the pixel value range), the latent state output by the representation network should be very close to the unperturbed output, with a target cosine similarity `> 0.95`.

**Why this matters**: if representations are unstable, MCTS will make drastically different search decisions on physically nearly identical adjacent positions, causing the policy to vary sharply under slight perturbations. For real-world robots or game AI, this means the policy is extremely sensitive to sensor noise and cannot be trusted.

**Diagnostic rule**: If stability falls below 0.9, the training data for the representation network lacks diversity among similar positions, or the network capacity is too small, causing features to jump nonlinearly across similar inputs. Increasing network width or using a contrastive learning loss (such as **SimCLR**-style positive pairs, where SimCLR is a self-supervised contrastive learning framework: two random augmentations of the same image form a "positive pair," and the encoder is trained to bring positive-pair representations close together while pushing negative samples apart, learning robust visual representations) can improve this effectively.

<figure>
<img src="/muzero/muzero-training-plots.png" alt="MuZero training curves on Atari games" style="width:100%;display:block;margin:0 auto">
<figcaption>MuZero training curves on selected Atari games reported by Schrittwieser et al. (2020) (episode return vs. training steps). The three lines correspond to MuZero, AlphaZero (with rules), and R2D2 (model-free baseline). MuZero matches AlphaZero performance without being given game rules, validating the effectiveness of implicit world models.</figcaption>
</figure>
