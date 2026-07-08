---
title: "Debate Four: Where Does the Data Come From"
description: 'The era of "downloading humanity," the challenge of acquiring physical-world data, the data strategy at AMI Labs, and the ethical boundaries of data privacy.'
lecture: 5
---

# Debate Four: Where Does the Data Come From

## Debate Four: The Era of "Downloading Humanity" -- Where Does the Data Come From?

### Xie Saining's Assessment

Xie Saining offers a sweeping characterization of the current stage of AI development:

> "Before, it was the era of downloading the internet. Now, it is the era of downloading humanity."

The first phase, "downloading the internet," refers to the training data for LLMs: **Common Crawl** (a publicly available dataset of continuously crawled web pages, containing raw text from billions of pages and serving as the primary training corpus for GPT-series and LLaMA models), Wikipedia, GitHub, and books -- every textual trace humanity has left in digital space. The scale of this data is staggering, but it has a fundamental limitation: language only, no physics.

The second phase, "downloading humanity," refers to the data that world models need: first-person human operation videos, industrial sensor data, robot teleoperation data, and medical procedure footage. These data records capture how humans **act in the physical world**, not merely how humans describe action.

Xie Saining makes a striking order-of-magnitude comparison: a four-year-old child, over those four years of life, receives an amount of visual information -- measured in frames and pixels -- larger than all the text tokens used to train GPT-4 combined.

<figure>
<img src="/iris/iris-imaginations.png" alt="IRIS multi-step imagination rollouts in Kung Fu Master" style="width:100%;display:block;margin:0 auto">
<figcaption>Micheli et al. (2022) IRIS imagination rollout example: starting from the same initial frame, the Transformer world model autoregressively generates multiple candidate future trajectories (one per row) in Atari Kung Fu Master. These trajectories are generated entirely inside the model with no interaction with the real game environment, and the policy learns from these purely imagined sequences. The core of the data problem: generating such training data requires large numbers of initial states accumulated through real interactions, and in the physical world these initial states are extremely difficult to obtain at scale.</figcaption>
</figure>

Crucially, this data does not get uploaded to YouTube. It sits in hospital operating rooms, factory assembly lines, agricultural worksites, and home kitchens, scattered across countless private spaces, neither digitized nor annotated.

### AMI Labs' Data Strategy

Xie Saining describes his team's approach: a grassroots coalition. Find companies that hold real-world data -- industrial manufacturing, healthcare, agriculture -- and let them exchange data for model capabilities, co-building a world model together.

He draws an analogy to the financial industry: the success of Mastercard and Visa was not because they themselves held large amounts of money, but because they built a network that every bank was willing to join. The data strategy for world models may require a similar network effect. No single company can accumulate sufficient physical-world data on its own, but a coalition can.

### Deeper Challenges

Behind this vision lie several serious challenges that Xie Saining does not sidestep.

**Annotation costs are extremely high**: annotating a video of an industrial robot operation -- capturing the state of every joint, the properties of every object, and the intent behind every action -- is far more complex than labeling the category of a single image.

**Privacy and ownership**: medical images carry patient privacy, factory data carries trade secrets, and home video carries personal privacy. Enabling data to circulate requires resolving complex legal and ethical problems.

**Ethical boundaries**: the phrase "downloading humanity" is itself unsettling. Human behavioral data, decision patterns, and physical movements will all be used to train machines. Where is that boundary? Who draws it?

### For You to Consider

If world models ultimately require sensor data from everyone's daily life -- the AR glasses you wear, the cameras in your home, the operation logs from your work -- would you trade that privacy for a more capable AI assistant?

Going deeper: is that choice genuinely one you can "agree to" or "refuse"? Or will this become infrastructure like the smartphone, where not participating means being excluded?


## Further Reading

- [Sutton (2019): The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html): the core argument that search and learning outperform human-engineered knowledge
- LeCun, Y. *A Path Towards Autonomous Machine Intelligence* (see Lecture 1 further reading)
- Ha & Schmidhuber (2018): World Models (see Lecture 1 further reading)
- Saining Xie interview, *Business Interview* (Zhang Xiaojun, 2024). [YouTube](https://www.youtube.com/watch?v=rIwgZWzUKm8)
- LeCun, Y. (2026, May). Interview on LLM safety, VLA, and JEPA. [YouTube](https://www.youtube.com/watch?v=ngBraLDqzdI)
