"""Crawl the OA-gated citation closure around random criminology seed papers.

For each seed we walk reference lists outward. A paper is *passable* if we could
actually read it under this project's OA policy; the fly may continue through it.
A paper that is not passable is a wall: we record it, we do not expand it.

The closure we compute is the fly's entire world. 'Run until stuck' is not a
parameter, it is what happens when the frontier runs out of passable papers.
max_nodes_per_seed and max_depth exist only to stop a runaway open component, and
any seed that hits them is marked truncated and excluded from headline statistics.

Input:  data/sources.csv (from src/fetch_sources.py)
Output: data/citations/{papers,edges}.parquet, seeds.json, manifest.json
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

API = "https://api.openalex.org"
MAILTO = os.environ.get("OPENALEX_MAILTO", "")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "citations"
SESSION = requests.Session()

SELECT = ",".join([
    "id", "display_name", "publication_year", "primary_location",
    "open_access", "locations", "referenced_works", "cited_by_count",
])


def get(path: str, **params) -> dict:
    if MAILTO:
        params["mailto"] = MAILTO
    for attempt in range(6):
        r = SESSION.get(f"{API}/{path}", params=params, timeout=90)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503):
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
    raise RuntimeError(f"OpenAlex request failed: {path} {params}")


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


def fetch_works(ids: list[str], batch_size: int) -> dict:
    found = {}
    for i in range(0, len(ids), batch_size):
        chunk = [short(x) for x in ids[i:i + batch_size]]
        data = get(
            "works",
            filter=f"openalex_id:{'|'.join(chunk)}",
            select=SELECT,
            per_page=str(len(chunk)),
        )
        for w in data.get("results", []):
            found[short(w["id"])] = w
    return found


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yml").read_text())
    ccfg = cfg["citations"]
    open_statuses = set(cfg["policy"]["open_statuses"])
    credit_repos = bool(cfg["policy"].get("credit_repository_locations"))
    y0, y1 = cfg["policy"]["years"]
    OUT.mkdir(parents=True, exist_ok=True)

    sources = pd.read_csv(ROOT / "data" / "sources.csv")
    source_ids = [
        short(s) for s in sources["openalex_id"].dropna().tolist() if str(s).strip()
    ]
    if not source_ids:
        raise SystemExit("No resolved sources. Run src/fetch_sources.py first.")
    print(f"{len(source_ids)} resolved journals")

    sample = get(
        "works",
        filter=(
            f"primary_location.source.id:{'|'.join(source_ids)},"
            f"publication_year:{y0}-{y1},type:article"
        ),
        sample=str(ccfg["n_seeds"]),
        seed=str(ccfg["rng_seed"]),
        per_page=str(ccfg["n_seeds"]),
        select=SELECT,
    )
    seeds = sample.get("results", [])
    print(f"sampled {len(seeds)} seeds with OpenAlex seed={ccfg['rng_seed']}")

    papers: dict[str, dict] = {}
    edges: list[tuple[str, str, str]] = []
    seed_records = []

    for n, seed in enumerate(seeds, 1):
        sid = short(seed["id"])
        cache = {sid: seed}
        status, passable = classify(seed, open_statuses, credit_repos)
        local = {sid: {"depth": 0, "status": status, "passable": passable}}
        frontier = [sid] if passable else []
        truncated = False
        walls = 0

        while frontier:
            if len(local) >= ccfg["max_nodes_per_seed"]:
                truncated = True
                break
            current, frontier = frontier, []
            refs_needed = []
            for node in current:
                for ref in cache[node].get("referenced_works") or []:
                    r = short(ref)
                    edges.append((sid, node, r))
                    if r not in local:
                        refs_needed.append(r)
            if not refs_needed:
                break
            refs_needed = list(dict.fromkeys(refs_needed))[: ccfg["max_nodes_per_seed"]]
            fetched = fetch_works(refs_needed, ccfg["batch_size"])
            for rid in refs_needed:
                w = fetched.get(rid)
                if w is None:
                    local[rid] = {"depth": None, "status": "unknown", "passable": False}
                    walls += 1
                    continue
                cache[rid] = w
                st, ps = classify(w, open_statuses, credit_repos)
                depth = min(
                    (local[c]["depth"] for c in current if local[c]["depth"] is not None),
                    default=0,
                ) + 1
                local[rid] = {"depth": depth, "status": st, "passable": ps}
                if ps and depth < ccfg["max_depth"]:
                    frontier.append(rid)
                elif not ps:
                    walls += 1

        for pid, meta in local.items():
            w = cache.get(pid, {})
            src = (w.get("primary_location") or {}).get("source") or {}
            papers.setdefault(pid, {
                "id": pid,
                "title": w.get("display_name"),
                "year": w.get("publication_year"),
                "journal": src.get("display_name"),
                "oa_status": meta["status"],
                "passable": meta["passable"],
                "cited_by_count": w.get("cited_by_count"),
            })

        reachable = sum(1 for m in local.values() if m["passable"])
        seed_records.append({
            "seed": sid,
            "title": seed.get("display_name"),
            "journal": ((seed.get("primary_location") or {}).get("source") or {}).get("display_name"),
            "year": seed.get("publication_year"),
            "seed_oa_status": status,
            "seed_passable": passable,
            "world_size": len(local),
            "reachable": reachable,
            "walls": walls,
            "truncated": truncated,
        })
        print(
            f"[{n:3d}/{len(seeds)}] {sid} {status:8} world={len(local):4d} "
            f"reachable={reachable:4d} walls={walls:4d}"
            + (" TRUNCATED" if truncated else "")
        )

    pd.DataFrame(papers.values()).to_parquet(OUT / "papers.parquet", index=False)
    pd.DataFrame(edges, columns=["seed", "src", "dst"]).drop_duplicates().to_parquet(
        OUT / "edges.parquet", index=False
    )
    (OUT / "seeds.json").write_text(json.dumps(seed_records, indent=2))

    df = pd.DataFrame(seed_records)
    clean = df[~df["truncated"]]
    manifest = {
        "policy": cfg["policy"],
        "citations_config": ccfg,
        "n_seeds": len(df),
        "n_truncated": int(df["truncated"].sum()),
        "n_papers": len(papers),
        "headline": {
            "median_reachable": float(clean["reachable"].median()) if len(clean) else None,
            "mean_reachable": float(clean["reachable"].mean()) if len(clean) else None,
            "median_walls": float(clean["walls"].median()) if len(clean) else None,
            "share_seeds_stuck_immediately": float((clean["reachable"] <= 1).mean()) if len(clean) else None,
            "share_seeds_paywalled": float((~clean["seed_passable"]).mean()) if len(clean) else None,
        },
        "attribution": "Bibliographic metadata from OpenAlex (CC0).",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    print("\n--- headline (truncated seeds excluded) ---")
    for k, v in manifest["headline"].items():
        print(f"  {k}: {v}")
    print(f"\n{len(papers):,} papers, {len(edges):,} edges -> data/citations/")


if __name__ == "__main__":
    main()
