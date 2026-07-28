---
type: mechanism-map
title: EXP681 AbbIE mechanism map
timestamp: 2026-07-30T06:00:00+05:30
status: frozen
experiment: exp-681-abbie-recurrence
source: arxiv-2507.08567
description: Page-anchored AbbIE head, body, tail and C/D recurrence mapping for Candidate005.
---

# EXP681 AbbIE mechanism map

## Observed source evidence

The local immutable PDF shows a head, a recurrent body applied `r` times, and a
tail in Figure 2 on page 3. Section 2.2 defines form C as direct iteration of
`B(h)` and form D as the global residual `B(h)+h` in Equations 3 and 4; Equation
5 expands accumulated body deltas. These observations do not establish SAIR
generalization.

## Bounded-faithful SAIR mapping

| AbbIE element | Candidate005 element | Frozen interpretation |
| --- | --- | --- |
| Head | input convolutional stem | maps six feature planes to hidden state |
| Body `B` | one complete alternating dilation schedule | shared learned recurrent transform |
| Tail | output norm and two readouts | maps hidden state to bit and quotient logits |
| Form C | replace state with `B(state)` | primary tied recurrence |
| Form D | add prior state after `B` | matched residual control |

The body retains ordinary feature injection inside every Candidate005 GRU
update. Recurrence is over hidden-state transformation, never over input chunks.

## Falsifiers

- Recurrence 1 for initialized form C must be bit-exact to Candidate005.
- Body weights must update under training.
- Recurrence 4 must differ from recurrence 2.
- Resetting or shuffling recurrent state must change outputs.
- Randomizing body weights must change outputs.
- Any W256 claim requires absolute exact-answer accuracy and gains over form D,
  untied, randomized-body, and frozen controls.

The experiment may be informative while parked: a resource or mechanism
rejection is evidence about this exact package, not evidence against AbbIE.
