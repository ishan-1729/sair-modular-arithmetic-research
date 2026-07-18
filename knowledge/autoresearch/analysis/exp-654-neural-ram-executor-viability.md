---
type: Experiment Analysis
title: EXP-654 learned neural-RAM executor viability result
description: A full matched rung rejects the current addressable neural-RAM and non-addressable latent formulations before Tier-7 scale-up.
status: completed
tags: [autoresearch, neural-execution, neural-ram, memory, negative-result]
timestamp: "2026-07-18T08:10:54+05:30"
---

# EXP-654 learned neural-RAM executor viability result

## Outcome

Reject the current learned neural-RAM formulation before scale-up. After the complete frozen 4,000-update curriculum, neither the 74,865-parameter addressable executor nor the 74,801-parameter non-addressable workspace produces an exact autonomous answer at trained width 16 or transfer widths 24 and 32. The addressable arm's final training batch is also zero exact at the final prefix, so this is an optimization/representation failure at the viability surface rather than merely a length-extrapolation miss.

| Arm | Parameters | Width 16 exact | Width 24 exact | Width 32 exact | Width 16 prefix transition |
|---|---:|---:|---:|---:|---:|
| decoded control | 41,242 | 85.33% | 0% | 0% | 93.67% |
| decoded wider | 76,244 | 94.33% | 0% | 0% | 97.35% |
| non-addressable workspace | 74,801 | 0% | 0% | 0% | 27.40% |
| neural RAM | 74,865 | 0% | 0% | 0% | 23.98% |

The wider decoded arm is an important control result: matching the latent models' capacity improves trained-width exactness by nine points over the small decoded model, but neither decoded arm transfers one exact trajectory even to width 24. Capacity helps fit the practiced width; it does not by itself establish a width-independent algorithm.

## Memory diagnosis

The neural-RAM tensors move, but the addressing policy remains close to uniform. At width 16, mean maximum write probability is 0.1336 versus uniform `0.125`, normalized write entropy is 0.9996, and read entropy is 0.9976. Register updates are active at 0.0173 RMS and trace RMS is 0.2411, yet resetting memory at each bit, permuting addressed content, or forcing uniform allocation changes no exact outcome because the base arm already scores zero. Trained and completely randomized models both score zero exact on the randomized-dependence check.

This evidence does not show that addressable neural computers are irrelevant in principle. It rejects this specific soft eight-slot, single-controller, jointly learned execution surface under the frozen matched curriculum. The memory never becomes selective enough to support a causal claim, and additional higher-width compute would be unsupported.

## Resource and provenance

The formal run completed all 16,000 arm updates and every preregistered evaluation/ablation in 5,220.24 wall seconds and 20,716.37 aggregate CPU seconds, with 1,183,764,480 peak RSS. This is below the measured-throughput projection of 6,632 wall / 26,503 CPU seconds and well inside the 10,279 / 40,284 soft budgets and 15,419 / 60,426 hard ceilings. No arm, update, width, case, or diagnostic was reduced. The direct four-case width-256 neural-RAM probe projects 20.92 wall seconds per 100 cases, so runtime passes even though exactness and causal-memory gates fail.

MLflow parent run `5fc6fa3b51c24ca4bc52363d73362bd2` and its four arm runs complement the [canonical run record](../../../experiments/runs/exp-654-neural-ram-executor-viability/run.json), [metrics](../../../evals/results/exp-654-neural-ram-executor-viability/metrics.json), and [full raw summary](../../../experiments/runs/exp-654-neural-ram-executor-viability/formal-summary.json).

## Next implication

Do not scale or lightly retune this memory design. The supported signal is that decoded recurrence remains much easier to optimize and benefits from additional state capacity at trained width, while the current latent variants do not learn exact execution. Return to the strong Candidate-005 lineage and test whether the proven operand-swap complementarity can be distilled into a single trajectory through an explicit symmetry-consistency objective. This changes the training signal without imposing another unproven executor representation and directly targets candidate accuracy and runtime.
