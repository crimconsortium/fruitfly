# Results

A fruit fly brain foraging the criminology citation graph. Open access is a
corridor, a paywall is a wall, and it explores until it runs out of things it
can read.

## Headline

| Measure | Value |
|---|---|
| Seeds drawn (random, reproducible) | 100 |
| Seeds excluded as truncated | 24 |
| Median papers reachable from a seed | 0.0 |
| Median paywalls hit | 0.0 |
| Seeds that are themselves paywalled | 99% |
| Seeds stuck immediately | 99% |
| Papers seen in total | 9722 |

Open access here means diamond, gold, green, or hybrid. Bronze and closed are
walls. That is stricter than OpenAlex's own `is_oa`, which counts bronze.

## Did the brain matter?

| Measure | Value |
|---|---|
| Real connectome channel selectivity | 41% |
| Degree-preserving shuffled control | 67% |
| Chance | 12% |
| Mean firing rate achieved | 9.927999999999999 Hz |

**The real wiring does NOT beat its own shuffle. The fly's choices are effectively noise and the connectome is decoration. This is the result; we publish it rather than tuning until it disappears.**

## The rendered run

- Trace: `data/traces/W4390146717.json`
- Seed paper: Race, class, and criminal adjudication: Is the US criminal justice system as biased as is often assumed? A meta-analytic review
- Journal: Aggression and Violent Behavior (2023)
- Seed access: closed
- Papers reached: 1
- Paywalls hit: 0
- Moves: 0
- Ended because: seed_paywalled

Chosen as the run closest to the median reachable-paper count, not the most
dramatic one.

## Distribution across all seeds

| Statistic | Papers reached | Paywalls hit |
|---|---|---|
| min | 1 | 0 |
| median | 1 | 0 |
| mean | 1.1 | 1.9 |
| max | 6 | 148 |

## Provenance

- Connectome: male-cns:v1.0, 20000 neurons and 1016133 signed edges after pruning. CC-BY.
- Journals: Web of Science Criminology & Penology, adopted as-is. See corpus/PROVENANCE.md.
- Metadata: OpenAlex, CC0.

Regenerate everything with the `go` workflow.
