# fruitfly

A simulated fruit fly brain forages the criminology citation graph. Papers you can read are corridors. Papers you cannot read are walls.

A [CrimRxiv](https://www.crimrxiv.com) / [CrimConsortium](https://crimconsortium.com) project.

## What this is

In September 2026, HHMI Janelia, Google Research, and collaborators published MaleCNS v1.0: a complete wiring diagram of an adult male *Drosophila* central nervous system, about 166,700 neurons, released CC-BY. Within days people were wiring simulated copies of it into video games.

This is that meme pointed at something that matters. We give the fly a real problem: start at a randomly chosen criminology paper and explore outward through its citations and references. An edge is passable only if the paper at the other end is open access. Then we count how far it gets.

The fly is a random walker whose turns are driven by the connectome, not by a script. It is not smart. That is the point: even a bug can explore an open literature, and even a bug is stopped by a paywall.

## We did not choose the journals

The corpus is the Web of Science research area **Criminology & Penology**, adopted as-is. We made no inclusion or exclusion decisions. If you think a journal is missing, that is Clarivate's boundary, not ours.

Notably absent from the category: *European Journal of Criminology* and *Crime Science*. We did not add them.

The list lives in `corpus/wos_criminology_penology.txt`, verbatim, including malformed entries. See `corpus/PROVENANCE.md` for where it came from and when.

The authoritative category list is downloadable only to Web of Science subscribers. The conventional definition of criminology is itself paywalled.

**Robustness check:** we re-run everything against OpenAlex's own topic classifier, which assigns every work a `primary_topic` from a hierarchy of 4 domains, 26 fields, 252 subfields, and roughly 4,500 topics with no human curation. If the two corpora disagree, both results get published.

## Open access policy

OpenAlex assigns every work an `oa_status` of `diamond`, `gold`, `green`, `hybrid`, `bronze`, or `closed`.

| Status | This project | Why |
|---|---|---|
| `diamond` | open | fully OA journal, no APCs |
| `gold` | open | OA journal indexed in DOAJ |
| `green` | open | free copy in an OA repository |
| `hybrid` | open | free under an open license in a toll journal |
| `bronze` | **wall** | free to read at publisher discretion, no open license, no guarantee it stays free |
| `closed` | **wall** | not free to read |

This is stricter than OpenAlex's own `is_oa`, which counts bronze as open. Because the stricter rule makes paywalls look worse, we also report the bronze-inclusive number as a sensitivity check. Both appear in the results.

Two known biases, in opposite directions:

- Excluding bronze understates reachability.
- `oa_status` assigns `green` only when the *best* OA location is a repository, so repository copies of papers labeled gold or hybrid are invisible in the status field. We check each work's `locations` for a repository host rather than trusting `oa_status` alone. This also understates reachability.

## Layers

- **data** — cached OpenAlex citation subgraph and pruned MaleCNS connectome, fetched by GitHub Actions and committed as versioned artifacts. Nothing is fetched at render time.
- **engine** — a pure function of `(citation_graph, connectome, seed_paper, rng_seed)` that emits a `trace/v1` file. No drawing, no network, no globals. Same inputs, same trace, always.
- **renderer** — reads a trace, emits the video and the stats table.

A later browser version replaces only the engine. The trace schema is versioned so the renderer keeps working.

Seeds are drawn at random from the corpus. We do not seed from CrimRxiv: branding belongs in the frame, not in the sampling.

## Licenses and attribution

- Code: MIT.
- Our data and figures: CC-BY 4.0.
- Connectome: MaleCNS v1.0, HHMI Janelia / University of Cambridge / MRC LMB / Google Research, CC-BY.
- Bibliographic metadata: OpenAlex, CC0.
- Journal category: Web of Science Criminology & Penology, Clarivate. Names only, used to identify journals.

## Status

Data layer in progress. Engine and renderer next.
