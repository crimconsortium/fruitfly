# Results

A fruit fly brain foraging the criminology citation graph. Open access is a
corridor, a paywall is a wall, and it explores until it runs out of things it
can read.

## Can the fly even start?

Computed over **every** seed. A seed's access status is known whether or not its
crawl finished, so nothing is excluded here.

| Measure | Value |
|---|---|
| Seeds drawn | 100 |
| Blocked at the seed | 75 |
| Open at the seed | 25 |
| Share blocked | 75% |

Seed status counts: `{'closed': 73, 'green': 14, 'hybrid': 11, 'bronze': 2}`

## How far does it get when it can start?

| Measure | Value |
|---|---|
| Crawls completed | 80 |
| Crawls censored at the 800-paper cap | 20 |
| Share of open seeds censored | 80% |
| Median papers read beyond the seed (completed only) | 0 |
| Max papers read beyond the seed (completed only) | 133 |
| Median paywalls hit (completed only) | 0 |

20 crawls reached the cap of 800 papers and were stopped. Their reachable totals are lower bounds and are excluded from crawl-size averages only.

## The whole field, for context

Across 30,654 articles in 51 journals:

| Status | Articles |
|---|---|
| closed | 19,826 |
| hybrid | 5,193 |
| green | 3,780 |
| bronze | 1,263 |
| diamond | 562 |
| gold | 30 |

Open under our rule: **31%**. Including bronze: 35%.

## Did the brain matter?

| Measure | Value |
|---|---|
| Real connectome selectivity | 36% |
| Shuffled controls, mean of 5 | 89% |
| Shuffled controls, sd | 10% |
| Chance | 12% |
| Mean firing rate achieved | 11.45 Hz |

Replicates: `[0.9531, 0.8594, 0.875, 1.0, 0.75]`

**The shuffles BEAT the real wiring by more than two standard deviations. That is not a finding about brains, it is a warning about our metric: the selectivity measure is picking up something the shuffle supplies more of than reality does. Reported, not tuned away.**

## The rendered run

- Trace: `data/traces/W2977893088.json`
- Seed paper: Predictors and Contexts of Unsolved and Solved Sexual Offenses
- Journal: Crime & Delinquency (2019)
- Seed access: green
- Papers read: 6
- Paywalls hit: 99
- Moves: 6
- Ended because: no_passable_neighbours

Chosen from completed, non-censored crawls that actually moved, as the run
closest to the median reachable count. Not the most dramatic one.

## Provenance

- Connectome: male-cns:v1.0, 20000 neurons and 1017037 signed edges after pruning. CC-BY.
- Journals: Web of Science Criminology & Penology, adopted as-is. See corpus/PROVENANCE.md.
- Metadata: OpenAlex, CC0.

## Corrections

An earlier run of this pipeline reported that 99% of seeds were paywalled. That
was wrong. 25 of 100 seeds were open, but 24 of those hit the crawl cap and were
then excluded from the headline, leaving 75 blocked and 1 open: 75/76 = 98.7%.
Hitting a compute cap says nothing about whether a paper was readable. Seed-level
statistics now use all seeds, and censored crawls are counted rather than deleted.
tests/test_stats.py pins this.

Regenerate everything with the `go` workflow.
