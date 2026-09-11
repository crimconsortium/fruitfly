# fruitfly

A simulated fruit fly brain forages the criminology citation graph. Papers you can read
are corridors. Papers you cannot read are walls.

A [CrimRxiv](https://www.crimrxiv.com) / [CrimConsortium](https://crimconsortium.com) project.

## What this is

In September 2026, HHMI Janelia, Google Research, and collaborators published MaleCNS
v1.0: a complete wiring diagram of an adult male *Drosophila* central nervous system,
about 166,700 neurons, released CC-BY. Within days people were wiring simulated copies
of it into video games.

This is that meme pointed at something that matters. We drop the fly on a randomly
chosen criminology paper and let it walk outward through reference lists. An edge is
passable only if the paper at the other end is open access. Then we count how far it
gets.

The fly is a reactive random walker whose turns are driven by spikes propagating through
the real connectome. It is not smart and it is not learning. That is the point: even a
bug can explore an open literature, and even a bug is stopped by a paywall.

## We did not choose the journals

The corpus is the Web of Science research area **Criminology & Penology**, adopted
as-is. We made no inclusion or exclusion decisions. *European Journal of Criminology*
and *Crime Science* are not in the category; we did not add them. See
`corpus/PROVENANCE.md`.

The authoritative category list is downloadable only to Web of Science subscribers. The
conventional definition of criminology is itself paywalled.

## Open access policy

| Status | This project | Why |
|---|---|---|
| `diamond` | open | fully OA journal, no APCs |
| `gold` | open | OA journal indexed in DOAJ |
| `green` | open | free copy in an OA repository |
| `hybrid` | open | free under an open license in a toll journal |
| `bronze` | **wall** | free at the publisher's discretion, no open license |
| `closed` | **wall** | not free to read |

Stricter than OpenAlex's own `is_oa`, which counts bronze. Because that choice makes
paywalls look worse, the bronze-inclusive number is reported alongside it.

Two biases, both pushing the same way: excluding bronze understates reachability, and
`oa_status` only reports `green` when the best OA location is a repository, so we also
check each work's `locations` for a repository copy.

## What we have found so far

From the diagnostic run over 30,654 articles in 51 resolved journals, 2015-2026:
closed 64.7%, hybrid 16.9%, green 12.3%, bronze 4.1%, diamond 1.8%, gold 0.1%. Open
under our rule: **31.2%**. Gold accounts for 30 articles in the entire field, roughly a
nineteenth of diamond.

## Corrections

An earlier run reported that **99% of seed papers were paywalled. That was wrong; the
figure is 75%.** 25 of 100 seeds were correctly recorded as open, but 24 of those hit
the crawl cap, were marked truncated, and were then excluded from the headline. That
left 75 blocked and 1 open seed: 75/76 = 98.7%.

Hitting a compute cap makes a crawl's reachable total uncertain. It does not make the
seed paper's access status uncertain. Seed-level statistics now use every seed, censored
crawls are counted rather than deleted, and `tests/test_stats.py` fails if that
regresses. The single shuffled control has also been replaced with replicates, because
"our control beat reality" was a one-sample artifact, not a result.

## Layers

- **data** - cached OpenAlex citation subgraph and pruned MaleCNS connectome, fetched by
  GitHub Actions and committed as versioned artifacts. Nothing is fetched at render time.
- **engine** - a pure function of `(citation_graph, connectome, seed_paper, rng_seed)`
  emitting a `trace/v1` file. No drawing, no network, no globals.
- **renderer** - reads a trace, emits the video and the stats.

A later browser version replaces only the engine.

## Run it

Actions -> **go** -> Run workflow. That runs the tests and the whole pipeline, commits
the video and `RESULTS.md`, and opens an issue if anything fails. Optionally set a repo
variable `OPENALEX_MAILTO` to use OpenAlex's polite pool.

`diagnose` answers the narrower question of whether OA detection is working.

## Licenses and attribution

- Code: MIT. Our data and figures: CC-BY 4.0.
- Connectome: MaleCNS v1.0, HHMI Janelia / University of Cambridge / MRC LMB / Google
  Research, CC-BY.
- Bibliographic metadata: OpenAlex, CC0.
- Journal category: Web of Science Criminology & Penology, Clarivate. Names only.

## Status

Pipeline runs end to end. First run's headline was wrong and has been corrected; awaiting
a clean re-run before anything is published. Open items: 2 of 53 journal names did not
resolve, and the OpenAlex-topic robustness corpus is not built yet.
