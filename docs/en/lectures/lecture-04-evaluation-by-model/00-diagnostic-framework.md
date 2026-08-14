---
title: A Model-Independent Diagnostic Framework
description: A staged protocol for locating failures in representations, dynamics, task signals, planning, long-horizon rollouts, and deployed control systems.
lecture: 4
---

# A Model-Independent Diagnostic Framework

A world model is a set of interfaces connecting observations, latent states, predicted futures, task signals, and actions. Evaluation becomes transferable when it asks which interface failed before asking which metric to report.

## Six Diagnostic Questions

| Layer | Question | Evidence to collect | Failure that return can hide |
| --- | --- | --- | --- |
| Representation | Does the latent state preserve distinctions needed by the task? | probe accuracy, latent variance and rank, perturbation tests | collapsed or nuisance-dominated features |
| One-step dynamics | Does the transition predict the immediate effect of an action? | held-out transition, reward, and termination error. Opposite-action tests | action-ignoring predictions |
| Long-horizon rollout | Do errors remain controlled when predictions become subsequent inputs? | error-versus-horizon curves, free-running rollouts, identity and geometry tracking | teacher-forcing gap and drift |
| Task signal | Are reward, value, and continuation estimates calibrated on real outcomes? | reward correlation, value error, calibration curves, out-of-distribution slices | visually accurate but decision-useless futures |
| Planner or policy | Does predicted value improve real action selection? | search improvement, plan efficiency, real-versus-imagined return gap, exploitation tests | optimization of model defects |
| Deployment loop | Do sensing, timing, control, uncertainty, and safety assumptions survive outside the dataset? | latency, intervention, transfer, uncertainty coverage, constraint violations | failures outside the learned model box |

No single metric covers all six layers. FID and PSNR say something about observation prediction but little about action causality. Episode return tests the complete loop but cannot localize a failure. A useful suite contains both local and end-to-end tests.

## Diagnose in Dependency Order

1. Verify representation diversity and task relevance.
2. Test one-step transitions on held-out data, including interventions that change only the action.
3. Run free rollouts and plot error against horizon.
4. Validate reward, value, and termination predictions against real outcomes.
5. Compare action ranking before and after planning, then verify selected actions in the real environment.
6. Measure calibration, latency, safety constraints, and human intervention under deployment shift.

Stop at the first failed layer. Downstream scores can still be recorded, but they should not determine a remedy until the upstream defect is controlled.

## Three Evaluation Regimes

- **In-distribution** trajectories verify basic implementation correctness.
- **Policy-shifted** trajectories, collected by a newer or more optimized policy, reveal model exploitation and replay staleness.
- **Mechanism-shifted** trajectories change object combinations, dynamics parameters, delays, or action magnitudes to test causal generalization.

Stratify every regime by rollout horizon and uncertainty. An average can hide that a model is reliable on common one-step transitions and unsafe on rare long-horizon cases.

## Minimum Evaluation Record

Save the model checkpoint, dataset or environment version, data-collection policy version, seeds, horizon, action distribution, metric implementation, and raw per-episode results. Without these fields, a score cannot be reproduced or attributed to a component change.

The remaining pages are worked examples. Dreamer emphasizes representation and task signal, TD-MPC latent consistency and planning efficiency, MuZero value and search, STORM free-running autoregressive rollout, and Diamond physical and action-conditioning consistency.
