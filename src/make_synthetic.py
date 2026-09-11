"""Generate structurally realistic but entirely fake data, for testing only.

This exists so the engine and renderer can be built and debugged before the real
workflows finish. Everything it writes goes to data_synthetic/ and is NEVER used for
any published number.

  python src/make_synthetic.py
  # then point the engine at it by copying data_synthetic/* over data/*  (locally!)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data_synthetic"
STATUSES = ["diamond", "gold", "green", "hybrid", "bronze", "closed"]
WEIGHTS = [0.02, 0.10, 0.15, 0.08, 0.05, 0.60]
OPEN = {"diamond", "gold", "green", "hybrid"}


def main() -> None:
    rng = np.random.default_rng(7)
    (OUT / "connectome").mkdir(parents=True, exist_ok=True)
    (OUT / "citations").mkdir(parents=True, exist_ok=True)

    n = 4000
    bodies = np.arange(100000, 100000 + n)
    is_readout = np.zeros(n, dtype=bool)
    is_readout[rng.choice(n, 240, replace=False)] = True
    neurons = pd.DataFrame({
        "body": bodies,
        "type": [f"DN{i:03d}" if r else f"CB{i:04d}" for i, r in enumerate(is_readout)],
        "class": ["descending" if r else "central" for r in is_readout],
        "side": rng.choice(["L", "R"], n),
        "is_readout": is_readout,
        "nt": rng.choice(["acetylcholine", "gaba", "glutamate"], n, p=[0.6, 0.25, 0.15]),
    })

    m = 120000
    pre = rng.choice(bodies, m)
    post = rng.choice(bodies, m)
    keep = pre != post
    nt_map = dict(zip(neurons["body"], neurons["nt"]))
    edges = pd.DataFrame({
        "pre": pre[keep],
        "post": post[keep],
        "weight": rng.integers(5, 80, keep.sum()),
    })
    edges["sign"] = [1 if nt_map[b] == "acetylcholine" else -1 for b in edges["pre"]]
    neurons.to_parquet(OUT / "connectome" / "neurons.parquet", index=False)
    edges.to_parquet(OUT / "connectome" / "edges.parquet", index=False)

    n_seeds, papers, cedges, seed_recs = 12, {}, [], []
    counter = 0
    for s in range(n_seeds):
        seed_id = f"WS{s:04d}"
        frontier = [seed_id]
        papers[seed_id] = rng.choice(STATUSES, p=WEIGHTS)
        local = {seed_id}
        for _depth in range(4):
            nxt = []
            for node in frontier:
                if papers[node] not in OPEN:
                    continue
                for _ in range(rng.integers(3, 12)):
                    counter += 1
                    ref = f"WR{counter:05d}"
                    papers[ref] = rng.choice(STATUSES, p=WEIGHTS)
                    cedges.append((seed_id, node, ref))
                    local.add(ref)
                    nxt.append(ref)
            frontier = nxt
        reach = sum(1 for p_ in local if papers[p_] in OPEN)
        seed_recs.append({
            "seed": seed_id, "title": f"Synthetic paper {s}", "journal": "FAKE JOURNAL",
            "year": int(rng.integers(2015, 2027)), "seed_oa_status": papers[seed_id],
            "seed_passable": papers[seed_id] in OPEN, "world_size": len(local),
            "reachable": reach, "walls": len(local) - reach, "truncated": False,
        })

    pd.DataFrame([
        {
            "id": k, "title": f"Synthetic {k}", "year": int(rng.integers(1990, 2027)),
            "journal": "FAKE JOURNAL", "oa_status": v, "passable": v in OPEN,
            "cited_by_count": int(rng.integers(0, 400)),
        }
        for k, v in papers.items()
    ]).to_parquet(OUT / "citations" / "papers.parquet", index=False)
    pd.DataFrame(cedges, columns=["seed", "src", "dst"]).to_parquet(
        OUT / "citations" / "edges.parquet", index=False
    )
    (OUT / "citations" / "seeds.json").write_text(json.dumps(seed_recs, indent=2))
    print(f"synthetic: {n:,} neurons, {len(edges):,} edges, "
          f"{len(papers):,} papers, {n_seeds} seeds -> data_synthetic/")


if __name__ == "__main__":
    main()
