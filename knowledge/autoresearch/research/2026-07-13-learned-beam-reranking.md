---
type: Research Note
title: Learned Beam Search and Reranking for Tier-6 Rollouts
description: Playwright-verified primary-source evidence guiding the EXP-622 beam-coverage gate and any later learned reranker.
status: completed
tags: [autoresearch, online-research, beam-search, structured-prediction, reranking]
timestamp: "2026-07-13T08:20:00+05:30"
---

# Learned beam search and reranking for Tier-6 rollouts

## Research question

Does EXP-621's gap between selected accuracy and oracle beam containment justify a wider beam and a separately trained selector, and what evidence should be required before paying that cost?

## Browser protocol and observation

The sources below were opened read-only on 2026-07-13 using repository-local Playwright 1.61.1 and its managed regular Chromium build 1228 in headless mode. Every navigation used a fresh non-persistent context with service workers blocked and non-GET/HEAD requests aborted. The two direct arXiv abstract pages and the corrective arXiv search returned HTTP 200 and visibly rendered their expected titles and abstracts. An initially considered identifier, `1603.05725`, visibly resolved to an unrelated group-theory paper and was discarded; the arXiv search identified the correct paper as `1502.02206`. Each context and the browser closed in a `finally` block. No access controls were bypassed and no browser artifacts or external state were written.

## Primary-source claims and applicability

### Beam-aware global sequence scoring

Wiseman and Rush train global sequence scores with a beam-search-aware objective, explicitly aligning training with test-time beam use and avoiding biases induced by purely local scoring. They report gains across word ordering, parsing, and machine translation. Source: [Sequence-to-Sequence Learning as Beam-Search Optimization](https://arxiv.org/abs/1606.02960).

Applicability: EXP-621's cumulative independent-bit likelihood was trained as a local transition objective, not to distinguish complete rollout hypotheses encountered inside a beam. Its 82.0% selected accuracy despite 86.0% oracle containment is therefore consistent with a search/scoring mismatch. If a later selector is trained, its examples should be actual competing beam states or trajectories, and its loss should optimize final rollout ranking rather than repeat local bit likelihood.

Limitation: the paper concerns language-style sequence models, not modular state transitions, and does not imply that increasing beam width will improve Tier-6 coverage or fit the submission CPU envelope.

### Learning search decisions beyond a reference policy

Chang et al. introduce LOLS for structured prediction and show how learning-to-search can optimize deviations from a learned policy rather than merely imitate a possibly suboptimal reference. Source: [Learning to Search Better Than Your Teacher](https://arxiv.org/abs/1502.02206).

Applicability: a Tier-6 reranker need not copy the frozen model's likelihood order. Offline exact labels may supervise which generic learned-search branch ultimately reaches the correct final state, while submission inference remains target-free and trained-parameter dependent.

Limitation: offline access to exact labels is permitted, but it can create selection leakage if checked-public answers influence model choice. Reranker data must come from generated training/validation splits, with a preregistered untouched seed for selection. The paper does not validate the specific least-confident-bit branch operator.

## Local evidence and decision

EXP-621 already establishes that width-2 search has useful but insufficient coverage at width 128: greedy is 165/200, selected beam is 164/200, and the correct final state occurs anywhere in the beam for 172/200. The seven-case oracle gap is evaluation-only evidence and is not a legal inference selector.

Before training any reranker, EXP-622 will freeze the checkpoint and branch operator and change only beam width from two to four on a fresh generated seed. It advances only if width-128 oracle containment reaches at least 92%, final states are valid and deterministic, and measured throughput projects the integrated candidate at no more than 285 wall seconds. This is a coverage/runtime feasibility gate, not a candidate accuracy claim. If it passes, the next experiment may train a compact global reranker on generated beam alternatives with final-trajectory supervision and compare it against the frozen likelihood order. If it fails, close this search branch without spending a reranker-training budget.
