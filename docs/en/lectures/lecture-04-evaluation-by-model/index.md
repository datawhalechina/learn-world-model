---
title: "Diagnosing World Models: From Components to Deployment"
description: A model-independent diagnostic framework for representations, dynamics, task signals, planning, rollouts, calibration, and deployment, followed by worked model examples.
lecture: 4
difficulty: Advanced
---

# L04 · Diagnosing World Models: From Components to Deployment

Evaluation should begin with the role a component plays in the agent loop, not with the name of the architecture. A low reconstruction loss cannot prove that actions change predicted futures correctly. A high episode return cannot reveal whether success depends on an exploitable model error. A visually convincing rollout cannot establish that a planner chooses safe actions.

**Core principle**: *first locate the failed interface, then choose a metric that can expose that failure.*

This lecture is organized into four layers:

- **Diagnostic framework**: representation, one-step dynamics, long-horizon rollout, task signal, planning, and deployment
- **Worked examples**: Dreamer, TD-MPC, MuZero, STORM, and Diamond show how the framework changes with the model interface
- **Cross-model failures**: horizon drift, physical inconsistency, action-conditioning failure, and model exploitation
- **Deployment evaluation**: calibration, latency, transfer, safety, and system-level reliability

Complete P03 and P04 before reading this lecture. Then use the framework while completing P05, rather than treating P05 as a prerequisite. Having two trained systems makes the diagnostic rules concrete.
