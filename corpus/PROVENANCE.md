# Corpus provenance

## What

`wos_criminology_penology.txt` contains 53 journal names constituting the Web of Science
research area **Criminology & Penology**.

## Where it came from

Retrieved 2026-09-11 from FIU Discovery's public listing of the Web of Science research
area Criminology & Penology: https://discovery.fiu.edu/display/wosc-criminology-penology

We used a public institutional mirror because Clarivate's authoritative category lists are
downloadable only to Web of Science subscribers
(https://mjl.clarivate.com/collection-list-downloads).

## Caveats we are not hiding

1. **It is a mirror, not the source of record.** A subscriber snapshot may differ. If you
   have Web of Science access, replace this file with the official export and re-run; the
   pipeline does not care where the names came from.
2. **It is a snapshot.** Category membership changes between JCR releases. This is the
   2026-09-11 state.
3. **One entry is malformed.** `CRIME AND JUSTICE: A REVIEW OF RESEARCH, VOL 20` is a
   mirror artifact of the annual series *Crime and Justice*. We left it in. The resolver
   flags it rather than us quietly fixing it.
4. **Names, not ISSNs.** The mirror gives display names only. `src/fetch_sources.py`
   resolves each to a canonical OpenAlex source and ISSN-L, and flags every uncertain
   match for human review in `data/sources.csv`.
5. **Absences are Clarivate's, not ours.** *European Journal of Criminology* and
   *Crime Science* are not in this category. We did not add them.

## Why adopt someone else's boundary

Any list we assembled ourselves would be a defensible target: every inclusion and exclusion
would be a choice we made, and a choice we made is a choice we could have made to favor the
result. Adopting the field's conventional, externally maintained category means the only
question left is whether the measurement is right.
