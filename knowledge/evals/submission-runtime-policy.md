---
type: Evaluation Policy
title: Submission inference runtime policy
description: Separates the official 300-second eligibility limit from historical internal headroom targets and evaluator implementation details.
status: active
tags: [evaluation, runtime, timeout, submission]
timestamp: "2026-08-12T17:50:00+05:30"
---

# Submission inference runtime policy

## Binding eligibility gate

The competition's written inference limit is **300 seconds for the complete test set**. Release qualification therefore requires a measured inference receipt below 300 seconds under the official four-CPU, 8-GiB sandbox envelope. Load, test generation, preprocessing-isolation checks, and determinism checks are recorded separately from inference.

## What 285 seconds means

The value **285 seconds is not an official rule**. It first entered this repository in EXP622 as an internal reliability target: 300 seconds minus a fixed 15-second integration margin. Historical experiments whose frozen preregistrations used 285 seconds remain valid historical decisions for those identities, but 285 must not be represented as a competition eligibility requirement or silently inherited by new release decisions.

## Evaluator implementation

The current evaluator checks its timeout between batches. Consequently, an oversized batch can begin before 300 seconds and finish afterward. A result that merely appears as completed is therefore insufficient runtime evidence. New release decisions require either an exact evaluator timing field or a retained official-container log showing complete inference below 300 seconds.

## Current runtime evidence

EXP706 retains a pinned official-container log in which all 1,100 cases complete at 281.6 inference seconds. Its separately checked public evaluation completes in 270.859 seconds. These satisfy the written runtime gate; the 18.4-second slower-run margin remains an operational reliability risk to monitor.

Runtime qualification is necessary but not sufficient for release. Current upstream commit `99cac6e` explicitly prohibits deterministic full-width operand pre-reduction, which Candidate005/006 use. EXP706 is therefore retained as exact runtime evidence while current-rule submission promotion remains blocked.
