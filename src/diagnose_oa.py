"""Settle one question: is the open-access detection working, or is criminology 99% walled?

The first full run reported 99% of seed papers as paywalled. That is either a strong
finding or a bug, and RESULTS.md cannot tell the difference. This script prints the
evidence that can.

Three checks:

  1. GROUND TRUTH. One group_by call over the whole corpus gives the real distribution
     of oa_status across every article in the 53 Web of Science journals, 2015-2026.
     This does not depend on any of our crawling code. If the field is genuinely ~99%
     non-open under our rule, this call says so.

  2. THE SEEDS. Re-draw the identical sample (same OpenAlex seed) and count raw
     oa_status values, how many records are missing the open_access object entirely,
     and how many have an OA repository location that oa_status does not reflect.

  3. THE BATCH FILTER. The first run used filter=openalex_id:..., which is not a real
     OpenAlex filter key -- the correct key is ids.openalex, short form openalex. This
     queries the same 20 IDs three ways and reports the HTTP status and how many of the
     20 came back. If the old form returns 0 matches, that alone explains why every
     referenced work was recorded as a wall.

Output: data/diagnostics.json, plus a readable report on stderr and in the run summary.
"""
from __future__ import annotations

import collections
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

API = "https://api.openalex.org"
MAILTO = os.environ.get("OPENALEX_MAILTO", "")
ROOT = Path(__file__).resolve().parent.parent
SESSION = requests.Session()
SELECT = "id,display_name,publication_year,primary_location,open_access,locations,referenced_works"


def get(path: str, raise_for_status: bool = True, **params):
    if MAILTO:
        params["mailto"] = MAILTO
    for attempt in range(5):
        r = SESSION.get(f"{API}/{path}", params=params, timeout=90)
        if r.status_code == 200:
            return r.json(), 200
        if r.status_code in (429, 500, 502, 503):
            time.sleep(2 ** attempt)
            continue
        if raise_for_status:
            r.raise_for_status()
        return None, r.status_code
    raise RuntimeError(f"failed: {path} {params}")


def short(x):
    return (x or "").rsplit("/", 1)[-1]


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yml").read_text())
    y0, y1 = cfg["policy"]["years"]
    open_statuses = set(cfg["policy"]["open_statuses"])
    ccfg = cfg["citations"]

    sources = pd.read_csv(ROOT / "data" / "sources.csv")
    ids = [short(s) for s in sources["openalex_id"].dropna() if str(s).strip()]
    src_filter = f"primary_location.source.id:{'|'.join(ids)}"
    out = {"n_journals": len(ids)}
    log = []

    data, _ = get("works", filter=f"{src_filter},publication_year:{y0}-{y1},type:article",
                  group_by="open_access.oa_status")
    corpus = {g["key"]: g["count"] for g in data.get("group_by", [])}
    total = sum(corpus.values())
    corpus_open = sum(v for k, v in corpus.items() if k in open_statuses)
    out["corpus_oa_status_counts"] = corpus
    out["corpus_total"] = total
    out["corpus_open_share"] = round(corpus_open / total, 4) if total else None
    out["corpus_open_share_with_bronze"] = (
        round((corpus_open + corpus.get("bronze", 0)) / total, 4) if total else None
    )
    log.append("1. GROUND TRUTH across the whole corpus")
    log.append(f"   {total:,} articles in {len(ids)} journals, {y0}-{y1}")
    for k, v in sorted(corpus.items(), key=lambda kv: -kv[1]):
        log.append(f"   {k:9} {v:8,}  {v / total:6.1%}")
    log.append(f"   open under our rule:   {out['corpus_open_share']}")
    log.append(f"   open including bronze: {out['corpus_open_share_with_bronze']}")

    data, _ = get("works", filter=f"{src_filter},publication_year:{y0}-{y1},type:article",
                  sample=str(ccfg["n_seeds"]), seed=str(ccfg["rng_seed"]),
                  per_page=str(ccfg["n_seeds"]), select=SELECT)
    seeds = data.get("results", [])
    statuses = collections.Counter()
    missing_oa_object = 0
    repo_but_not_open = 0
    for w in seeds:
        oa = w.get("open_access")
        if not oa:
            missing_oa_object += 1
            statuses["MISSING_open_access"] += 1
            continue
        st = oa.get("oa_status") or "MISSING_oa_status"
        statuses[st] += 1
        if st not in open_statuses:
            for loc in w.get("locations") or []:
                if (loc.get("source") or {}).get("type") == "repository" and loc.get("is_oa"):
                    repo_but_not_open += 1
                    break
    out["n_seeds_returned"] = len(seeds)
    out["seed_oa_status_counts"] = dict(statuses)
    out["seeds_missing_open_access_object"] = missing_oa_object
    out["seeds_rescued_by_repository_location"] = repo_but_not_open
    log.append("")
    log.append("2. THE SEEDS actually drawn")
    log.append(f"   {len(seeds)} returned")
    for k, v in statuses.most_common():
        log.append(f"   {k:22} {v:4}")
    log.append(f"   missing open_access object: {missing_oa_object}")
    log.append(f"   not open by status but have an OA repository copy: {repo_but_not_open}")

    probe = []
    for w in seeds:
        probe.extend(short(r) for r in (w.get("referenced_works") or []))
        if len(probe) >= 20:
            break
    probe = probe[:20]
    out["probe_ids"] = probe
    log.append("")
    log.append("3. BATCH FILTER, 20 known reference IDs")
    for key in ("openalex_id", "openalex", "ids.openalex"):
        if not probe:
            log.append("   no reference IDs available to probe")
            break
        data, status = get("works", raise_for_status=False,
                           filter=f"{key}:{'|'.join(probe)}",
                           select=SELECT, per_page=str(len(probe)))
        results = (data or {}).get("results", [])
        matched = len({short(r["id"]) for r in results} & set(probe))
        out[f"batch_filter_{key.replace('.', '_')}"] = {
            "http_status": status, "returned": len(results), "matched_requested": matched,
        }
        log.append(f"   {key:14} HTTP {status}  returned {len(results):3}  requested-and-matched {matched:3}")

    (ROOT / "data" / "diagnostics.json").write_text(json.dumps(out, indent=2))
    report = "\n".join(log)
    print(report, file=sys.stderr)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as fh:
            fh.write("## OA diagnostic\n\n```\n" + report + "\n```\n")


if __name__ == "__main__":
    main()
