"""Crawl the OA-gated citation closure around random criminology seed papers.

For each seed we walk reference lists outward. A paper is *passable* if we could
actually read it under this project's OA policy. A paper confirmed not passable is a
wall. A paper we could not look up is UNRESOLVED and is never counted as a wall.

TWO VARIABLES, ONE OF WHICH CAN BE CENSORED
-------------------------------------------
Run #1 reported "99% of seeds paywalled". The truth was 75%. 25 of 100 seeds were
correctly recorded as open, but 24 of those hit the crawl cap, were marked truncated,
and were then excluded from the headline: 75 blocked + 1 open = 75/76 = 98.68%.

Hitting a compute cap makes a crawl's reachable TOTAL uncertain. It does not make the
seed paper's ACCESS STATUS uncertain. So seed-level statistics use ALL seeds, and
crawl-size statistics are reported for completed crawls with censored ones counted.

SURVIVING THE NETWORK
---------------------
Run #2 died after 44 seeds because one batch request exhausted its retries, and all 44
seeds of work were discarded. Three changes:

  * a batch that fails permanently marks those IDs unresolved and the crawl continues.
    The run aborts only if the share of failed batches exceeds max_failed_batch_share.
  * every completed seed is checkpointed to seeds.partial.json, and a re-run resumes
    from it unless --fresh is passed.
  * one process-wide work cache, so a reference shared by twenty seeds is fetched once.

Definitions:
  reachable             = passable papers entered, INCLUDING the seed itself
  reachable_beyond_seed = reachable - 1
  censored              = hit max_nodes_per_seed; reachable is a lower bound

Input:  data/sources.csv
Output: data/citations/{papers,edges}.parquet, seeds.json, manifest.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import pandas as pd
import yaml

from openalex import OpenAlexError, get, warn_if_no_mailto

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "citations"
CHECKPOINT = OUT / "seeds.partial.json"

SELECT = ",".join([
    "id", "display_name", "publication_year", "primary_location",
    "open_access", "locations", "referenced_works", "cited_by_count",
])

WORK_CACHE: dict[str, dict] = {}
API_STATS = collections.Counter()


def short(work_id: str) -> str:
    return (work_id or "").rsplit("/", 1)[-1]


def classify(work: dict, open_statuses: set, credit_repos: bool) -> tuple[str, bool]:
    oa = work.get("open_access") or {}
    status = oa.get("oa_status") or "unknown"
    passable = status in open_statuses
    if not passable and credit_repos:
        for loc in work.get("locations") or []:
            src = loc.get("source") or {}
            if src.get("type") == "repository" and loc.get("is_oa"):
                return status, True
    return status, passable


def fetch_works(ids: list[str], batch_size: int, api_cfg: dict) -> dict:
    """Batch-fetch by OpenAlex ID, using the shared cache. A batch that fails
    permanently is recorded and skipped; its IDs end up unresolved."""
    found = {i: WORK_CACHE[i] for i in ids if i in WORK_CACHE}
    missing = [i for i in ids if i not in WORK_CACHE]
    API_STATS["cache_hits"] += len(found)
    for i in range(0, len(missing), batch_size):
        chunk = missing[i:i + batch_size]
        API_STATS["batches_attempted"] += 1
        try:
            data = get("works",
                       max_attempts=api_cfg["max_attempts"],
                       max_backoff=api_cfg["max_backoff_seconds"],
                       filter=f"openalex:{'|'.join(chunk)}",
                       select=SELECT, per_page=str(len(chunk)))
        except OpenAlexError as exc:
            API_STATS["batches_failed"] += 1
            print(f"  BATCH FAILED ({len(chunk)} ids), continuing: {exc}")
            continue
        for w in data.get("results", []):
            key = short(w["id"])
            WORK_CACHE[key] = w
            found[key] = w
    return found


def summarize(seed_records: list[dict], status_counts, policy: dict,
              ccfg: dict, n_papers: int, api_stats: dict | None = None) -> dict:
    """Pure function. Seed stats over ALL seeds; crawl stats split by censoring.

    This is the function the 99% bug lived in. tests/test_stats.py pins it.
    """
    df = pd.DataFrame(seed_records)
    n = len(df)
    sens = set(policy["sensitivity_open_statuses"])
    censored = df["censored"] if "censored" in df else df["truncated"]
    complete = df[~censored]
    blocked = df[~df["seed_passable"]]
    open_seeds = df[df["seed_passable"]]
    counts = dict(status_counts)
    n_resolved = sum(v for k, v in counts.items() if k != "unresolved")

    def stat(frame, col, fn):
        return float(getattr(frame[col], fn)()) if len(frame) else None

    return {
        "definitions": {
            "reachable": "passable papers entered, including the seed itself",
            "reachable_beyond_seed": "reachable minus one",
            "walls": "papers confirmed not passable",
            "unresolved": "papers we could not look up; never counted as walls",
            "censored": f"crawl hit the {ccfg['max_nodes_per_seed']}-paper cap; "
                        "reachable is a lower bound, seed status is still known",
        },
        "seed_access": {
            "denominator": "all seeds, including censored crawls",
            "n_seeds": n,
            "n_blocked_at_seed": int(len(blocked)),
            "n_open_at_seed": int(len(open_seeds)),
            "share_blocked_at_seed": round(len(blocked) / n, 4) if n else None,
            "seed_status_counts": df["seed_oa_status"].value_counts().to_dict(),
        },
        "crawl_size": {
            "cap": ccfg["max_nodes_per_seed"],
            "n_complete": int(len(complete)),
            "n_censored": int(censored.sum()),
            "share_of_open_seeds_censored": (
                round(float(censored[df["seed_passable"]].mean()), 4) if len(open_seeds) else None
            ),
            "complete_crawls_only": {
                "median_reachable_beyond_seed": stat(complete, "reachable_beyond_seed", "median"),
                "mean_reachable_beyond_seed": stat(complete, "reachable_beyond_seed", "mean"),
                "max_reachable_beyond_seed": stat(complete, "reachable_beyond_seed", "max"),
                "median_walls": stat(complete, "walls", "median"),
                "max_walls": stat(complete, "walls", "max"),
            },
            "censored_crawls_note": (
                f"{int(censored.sum())} crawls reached the cap of "
                f"{ccfg['max_nodes_per_seed']} papers and were stopped. Their reachable "
                "totals are lower bounds and are excluded from crawl-size averages only."
            ),
        },
        "papers": {
            "n_papers": n_papers,
            "oa_status_counts": counts,
            "unresolved_share": (
                round(counts.get("unresolved", 0) / n_papers, 4) if n_papers else None
            ),
            "share_open_of_resolved": (
                round(sum(v for k, v in counts.items()
                          if k in set(policy["open_statuses"])) / n_resolved, 4)
                if n_resolved else None
            ),
            "share_open_of_resolved_with_bronze": (
                round(sum(v for k, v in counts.items() if k in sens) / n_resolved, 4)
                if n_resolved else None
            ),
        },
        "api": api_stats or {},
        "policy": policy,
        "citations_config": ccfg,
        "attribution": "Bibliographic metadata from OpenAlex (CC0).",
    }


def crawl_seed(seed: dict, cfg: dict, ccfg: dict, api_cfg: dict) -> tuple[dict, dict, list]:
    open_statuses = set(cfg["policy"]["open_statuses"])
    credit_repos = bool(cfg["policy"].get("credit_repository_locations"))
    sid = short(seed["id"])
    WORK_CACHE[sid] = seed
    status, passable = classify(seed, open_statuses, credit_repos)
    local = {sid: {"depth": 0, "status": status, "passable": passable, "resolved": True}}
    edges: list[tuple[str, str, str]] = []
    frontier = [sid] if passable else []
    censored = False
    walls = 0
    unresolved = 0

    while frontier:
        if len(local) >= ccfg["max_nodes_per_seed"]:
            censored = True
            break
        current, frontier = frontier, []
        refs_needed = []
        for node in current:
            for ref in (WORK_CACHE.get(node, {}).get("referenced_works") or []):
                r = short(ref)
                edges.append((sid, node, r))
                if r not in local:
                    refs_needed.append(r)
        if not refs_needed:
            break
        room = max(ccfg["max_nodes_per_seed"] - len(local), 0)
        refs_needed = list(dict.fromkeys(refs_needed))[:room]
        if not refs_needed:
            censored = True
            break
        fetched = fetch_works(refs_needed, ccfg["batch_size"], api_cfg)
        depth = min((local[c]["depth"] for c in current), default=0) + 1
        for rid in refs_needed:
            w = fetched.get(rid)
            if w is None:
                local[rid] = {"depth": depth, "status": "unresolved",
                              "passable": False, "resolved": False}
                unresolved += 1
                continue
            st, ps = classify(w, open_statuses, credit_repos)
            local[rid] = {"depth": depth, "status": st, "passable": ps, "resolved": True}
            if ps and depth < ccfg["max_depth"]:
                frontier.append(rid)
            elif not ps:
                walls += 1

    reachable = sum(1 for m in local.values() if m["passable"])
    record = {
        "seed": sid, "title": seed.get("display_name"),
        "journal": ((seed.get("primary_location") or {}).get("source") or {}).get("display_name"),
        "year": seed.get("publication_year"),
        "seed_oa_status": status, "seed_passable": passable,
        "world_size": len(local), "reachable": reachable,
        "reachable_beyond_seed": max(reachable - 1, 0),
        "walls": walls, "unresolved": unresolved,
        "censored": censored, "truncated": censored,
    }
    return record, local, edges


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any checkpoint and start over")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yml").read_text())
    ccfg = cfg["citations"]
    api_cfg = cfg.get("api", {"max_attempts": 8, "max_backoff_seconds": 120,
                              "max_failed_batch_share": 0.1})
    y0, y1 = cfg["policy"]["years"]
    OUT.mkdir(parents=True, exist_ok=True)
    warn_if_no_mailto()

    sources = pd.read_csv(ROOT / "data" / "sources.csv")
    source_ids = [short(s) for s in sources["openalex_id"].dropna().tolist() if str(s).strip()]
    if not source_ids:
        raise SystemExit("No resolved sources. Run src/fetch_sources.py first.")
    print(f"{len(source_ids)} resolved journals")

    sample = get("works",
                 max_attempts=api_cfg["max_attempts"],
                 max_backoff=api_cfg["max_backoff_seconds"],
                 filter=(f"primary_location.source.id:{'|'.join(source_ids)},"
                         f"publication_year:{y0}-{y1},type:article"),
                 sample=str(ccfg["n_seeds"]), seed=str(ccfg["rng_seed"]),
                 per_page=str(ccfg["n_seeds"]), select=SELECT)
    seeds = sample.get("results", [])
    print(f"sampled {len(seeds)} seeds with OpenAlex seed={ccfg['rng_seed']}")

    seed_records: list[dict] = []
    if CHECKPOINT.exists() and not args.fresh:
        seed_records = json.loads(CHECKPOINT.read_text())
        print(f"resuming from checkpoint: {len(seed_records)} seeds already done")
    done = {r["seed"] for r in seed_records}

    papers: dict[str, dict] = {}
    edges: list[tuple[str, str, str]] = []
    status_counts = collections.Counter()

    for n, seed in enumerate(seeds, 1):
        sid = short(seed["id"])
        if sid in done:
            continue
        record, local, seed_edges = crawl_seed(seed, cfg, ccfg, api_cfg)
        edges.extend(seed_edges)
        for pid, meta in local.items():
            status_counts[meta["status"]] += 1
            w = WORK_CACHE.get(pid, {})
            src = (w.get("primary_location") or {}).get("source") or {}
            papers.setdefault(pid, {
                "id": pid, "title": w.get("display_name"),
                "year": w.get("publication_year"), "journal": src.get("display_name"),
                "oa_status": meta["status"], "passable": meta["passable"],
                "resolved": meta["resolved"], "cited_by_count": w.get("cited_by_count"),
            })
        seed_records.append(record)
        CHECKPOINT.write_text(json.dumps(seed_records, indent=2))

        print(f"[{n:3d}/{len(seeds)}] {sid} {record['seed_oa_status']:10} "
              f"world={record['world_size']:5d} reachable={record['reachable']:5d} "
              f"walls={record['walls']:4d} unresolved={record['unresolved']:4d}"
              + (" CENSORED" if record["censored"] else ""))

        attempted = API_STATS["batches_attempted"]
        failed = API_STATS["batches_failed"]
        if attempted >= 20 and failed / attempted > api_cfg["max_failed_batch_share"]:
            raise SystemExit(
                f"ABORT: {failed}/{attempted} batches failed permanently, above the "
                f"{api_cfg['max_failed_batch_share']:.0%} threshold. The checkpoint at "
                f"{CHECKPOINT} holds {len(seed_records)} completed seeds; re-run to resume."
            )

    pd.DataFrame(papers.values()).to_parquet(OUT / "papers.parquet", index=False)
    pd.DataFrame(edges, columns=["seed", "src", "dst"]).drop_duplicates().to_parquet(
        OUT / "edges.parquet", index=False)
    (OUT / "seeds.json").write_text(json.dumps(seed_records, indent=2))

    api_stats = dict(API_STATS)
    api_stats["cache_size"] = len(WORK_CACHE)
    manifest = summarize(seed_records, status_counts, cfg["policy"], ccfg,
                         len(papers), api_stats)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    CHECKPOINT.unlink(missing_ok=True)

    sa = manifest["seed_access"]
    cs = manifest["crawl_size"]
    print("\n--- seed access (ALL seeds) ---")
    print(f"  blocked at seed: {sa['n_blocked_at_seed']}/{sa['n_seeds']} "
          f"= {sa['share_blocked_at_seed']}")
    print(f"  status counts: {sa['seed_status_counts']}")
    print("--- crawl size ---")
    print(f"  complete: {cs['n_complete']}, censored at cap {cs['cap']}: {cs['n_censored']}")
    print(f"  {cs['complete_crawls_only']}")
    print(f"--- api: {api_stats}")
    print(f"\n{len(papers):,} papers, {len(edges):,} edges -> data/citations/")


if __name__ == "__main__":
    main()
