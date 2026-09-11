"""The fly. A pure function from (connectome, citation graph, seed, rng) to a trace.

No network. No drawing. No globals. Same inputs always produce the same trace.

How a decision is made, and what is and is not our choice:

  1. The fly stands on a paper. Its unvisited neighbours become candidates, capped at
     n_channels. Non-passable neighbours are walls: recorded as bumps, never candidates.
  2. Each candidate drives one band of input neurons at a rate set by its citation
     count. Walls drive their own band too, so they perturb the dynamics without being
     selectable. This encoding is OUR choice and it is declared here.
  3. Spikes propagate through the real, signed MaleCNS subgraph for window_ms.
  4. Descending neurons are partitioned into n_channels groups. The candidate whose
     group spiked most wins. Ties go to the RNG.
  5. No plasticity. Nothing about the fly changes between steps.

The fly is not smart and is not learning. What is real is the wiring it runs on and the
walls it cannot pass.

Usage:
  python src/engine.py --seed W2141234567
  python src/engine.py --all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONN = ROOT / "data" / "connectome"
CITE = ROOT / "data" / "citations"
TRACES = ROOT / "data" / "traces"

DEFAULTS = dict(
    dt_ms=1.0, window_ms=50, tau_v_ms=20.0, tau_syn_ms=5.0,
    v_rest=-52.0, v_threshold=-45.0, v_reset=-52.0,
    gain=0.02, n_input_neurons=800, n_channels=8, base_hz=60.0,
    wall_hz=90.0, max_steps=5000, learning=False,
)

TERRAIN = {
    "diamond": "corridor", "gold": "corridor", "green": "corridor",
    "hybrid": "gate", "bronze": "trapdoor", "closed": "wall", "unknown": "wall",
}


class Brain:
    """Leaky integrate-and-fire over the pruned, signed connectome."""

    def __init__(self, neurons: pd.DataFrame, edges: pd.DataFrame, p: dict):
        self.p = p
        bodies = neurons["body"].to_numpy()
        self.index = {b: i for i, b in enumerate(bodies)}
        self.n = len(bodies)

        pre = edges["pre"].map(self.index).to_numpy()
        post = edges["post"].map(self.index).to_numpy()
        ok = ~(pd.isna(pre) | pd.isna(post))
        w = edges["weight"].to_numpy()[ok] * edges["sign"].to_numpy()[ok] * p["gain"]
        self.W = sp.csr_matrix(
            (w, (pre[ok].astype(int), post[ok].astype(int))), shape=(self.n, self.n)
        )

        out_strength = np.asarray(abs(self.W).sum(axis=1)).ravel()
        readout_mask = neurons["is_readout"].fillna(False).to_numpy().astype(bool)
        self.readout_idx = np.flatnonzero(readout_mask)
        if self.readout_idx.size == 0:
            raise SystemExit("No readout neurons in the connectome artifacts.")

        cand = np.flatnonzero(~readout_mask)
        order = cand[np.argsort(-out_strength[cand], kind="stable")]
        self.input_idx = order[: p["n_input_neurons"]]

        k = p["n_channels"]
        self.input_bands = np.array_split(self.input_idx, k)
        self.readout_groups = np.array_split(self.readout_idx, k)

    def run(self, drives: np.ndarray, rng: np.random.Generator):
        """drives: per-channel Hz. Returns (spikes per readout group, total spikes)."""
        p = self.p
        steps = int(p["window_ms"] / p["dt_ms"])
        decay_syn = float(np.exp(-p["dt_ms"] / p["tau_syn_ms"]))
        alpha = p["dt_ms"] / p["tau_v_ms"]

        v = np.full(self.n, p["v_rest"], dtype=np.float32)
        current = np.zeros(self.n, dtype=np.float32)
        group_spikes = np.zeros(len(self.readout_groups), dtype=np.int64)
        total = 0

        prob = np.zeros(self.n, dtype=np.float32)
        for band, hz in zip(self.input_bands, drives):
            if band.size:
                prob[band] = hz * p["dt_ms"] / 1000.0

        for _ in range(steps):
            current *= decay_syn
            current += (rng.random(self.n) < prob).astype(np.float32) * 2.0
            v += alpha * (p["v_rest"] - v) + current
            fired = v >= p["v_threshold"]
            if fired.any():
                v[fired] = p["v_reset"]
                current += np.asarray(
                    self.W[np.flatnonzero(fired)].sum(axis=0), dtype=np.float32
                ).ravel()
                total += int(fired.sum())
                for g, idx in enumerate(self.readout_groups):
                    group_spikes[g] += int(fired[idx].sum())
        return group_spikes, total


def load_graphs():
    neurons = pd.read_parquet(CONN / "neurons.parquet")
    edges = pd.read_parquet(CONN / "edges.parquet")
    papers = pd.read_parquet(CITE / "papers.parquet").set_index("id")
    cedges = pd.read_parquet(CITE / "edges.parquet")
    seeds = json.loads((CITE / "seeds.json").read_text())
    return neurons, edges, papers, cedges, seeds


def run_seed(brain, papers, cedges, seed_rec, policy, p, rng_seed):
    sid = seed_rec["seed"]
    rng = np.random.default_rng(rng_seed)
    sub = cedges[cedges["seed"] == sid]
    adj: dict[str, list[str]] = {}
    for s, d in zip(sub["src"], sub["dst"]):
        adj.setdefault(s, []).append(d)

    def meta(pid):
        if pid in papers.index:
            row = papers.loc[pid]
            return str(row.get("oa_status") or "unknown"), bool(row.get("passable"))
        return "unknown", False

    steps, visited, walls_hit = [], [sid], 0
    seed_status, seed_passable = meta(sid)
    node_ids = {sid}
    current = sid
    stuck_reason = "no_passable_neighbours"

    if not seed_passable:
        stuck_reason = "seed_paywalled"
    else:
        for t in range(p["max_steps"]):
            neighbours = adj.get(current, [])
            cands, bumps = [], []
            for nb in neighbours:
                st, ps = meta(nb)
                node_ids.add(nb)
                if ps and nb not in visited:
                    cands.append((nb, st))
                elif not ps:
                    bumps.append(nb)
            walls_hit += len(bumps)

            backtracked = False
            if not cands:
                for prev in reversed(visited[:-1]):
                    alt = [
                        (nb, meta(nb)[0]) for nb in adj.get(prev, [])
                        if meta(nb)[1] and nb not in visited
                    ]
                    if alt:
                        current, cands, backtracked = prev, alt, True
                        break
            if not cands:
                steps.append({
                    "t": t, "at": current, "to": None, "candidates": [],
                    "bumps": bumps, "total_spikes": 0, "backtracked": False,
                })
                break

            cands = sorted(cands, key=lambda c: c[0])[: p["n_channels"]]
            drives = np.zeros(p["n_channels"])
            for i, (nb, _) in enumerate(cands):
                cited = 0
                if nb in papers.index:
                    cited = float(papers.loc[nb].get("cited_by_count") or 0)
                drives[i] = p["base_hz"] * (1.0 + np.log1p(cited) / 10.0)
            if bumps:
                drives[len(cands) % p["n_channels"]] += p["wall_hz"]

            group_spikes, total = brain.run(drives, rng)
            scores = group_spikes[: len(cands)].astype(float)
            scores += rng.random(len(cands)) * 1e-6
            winner = int(np.argmax(scores))

            steps.append({
                "t": t,
                "at": current,
                "to": cands[winner][0],
                "candidates": [
                    {
                        "id": nb, "channel": i, "passable": True, "oa_status": st,
                        "drive_hz": round(float(drives[i]), 2),
                        "readout_spikes": int(group_spikes[i]),
                    }
                    for i, (nb, st) in enumerate(cands)
                ],
                "bumps": bumps,
                "total_spikes": total,
                "backtracked": backtracked,
            })
            current = cands[winner][0]
            visited.append(current)
        else:
            stuck_reason = "hop_ceiling"

    nodes = []
    for pid in sorted(node_ids):
        st, ps = meta(pid)
        row = papers.loc[pid] if pid in papers.index else {}
        nodes.append({
            "id": pid,
            "title": (row.get("title") if hasattr(row, "get") else None),
            "journal": (row.get("journal") if hasattr(row, "get") else None),
            "year": (int(row["year"]) if hasattr(row, "get") and pd.notna(row.get("year")) else None),
            "oa_status": st,
            "passable": ps,
            "terrain": TERRAIN.get(st, "wall"),
        })

    return {
        "schema": "trace/v1",
        "engine": {**p, "rng_seed": rng_seed, "neurons": brain.n, "edges": int(brain.W.nnz)},
        "policy": {"open_statuses": policy["open_statuses"]},
        "seed": {
            "id": sid, "title": seed_rec.get("title"), "journal": seed_rec.get("journal"),
            "year": seed_rec.get("year"), "oa_status": seed_status, "passable": seed_passable,
        },
        "nodes": nodes,
        "steps": steps,
        "result": {
            "visited": len(visited),
            "reachable": len(set(visited)),
            "walls_hit": walls_hit,
            "n_steps": len(steps),
            "stuck_at": current,
            "stuck_reason": stuck_reason,
            "truncated": bool(seed_rec.get("truncated")),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", help="OpenAlex short id of one seed paper")
    ap.add_argument("--all", action="store_true", help="run every seed")
    ap.add_argument("--rng-seed", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yml").read_text())
    p = dict(DEFAULTS)
    rng_seed = args.rng_seed if args.rng_seed is not None else cfg["citations"]["rng_seed"]

    neurons, edges, papers, cedges, seeds = load_graphs()
    brain = Brain(neurons, edges, p)
    print(f"brain: {brain.n:,} neurons, {brain.W.nnz:,} edges, "
          f"{brain.readout_idx.size:,} readout, {brain.input_idx.size:,} inputs")

    TRACES.mkdir(parents=True, exist_ok=True)
    targets = seeds if args.all else [s for s in seeds if s["seed"] == args.seed]
    if not targets:
        raise SystemExit("No matching seed. Pass --seed <id> or --all.")

    summary = []
    for rec in targets:
        trace = run_seed(brain, papers, cedges, rec, cfg["policy"], p, rng_seed)
        (TRACES / f"{rec['seed']}.json").write_text(json.dumps(trace))
        r = trace["result"]
        summary.append({"seed": rec["seed"], **r})
        print(f"{rec['seed']}: visited={r['visited']:4d} walls={r['walls_hit']:4d} "
              f"steps={r['n_steps']:4d} stuck={r['stuck_reason']}")

    pd.DataFrame(summary).to_csv(TRACES / "summary.csv", index=False)
    print(f"\n{len(summary)} traces -> data/traces/")


if __name__ == "__main__":
    main()
