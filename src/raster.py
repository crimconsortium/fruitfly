"""Record the real spike raster for the decisions the video shows.

The trace stores what the fly decided. It does not store which neuron fired when,
because keeping that for 100 seeds would be gigabytes. This script re-runs one seed
and keeps the raster for a handful of decisions only.

Determinism makes that safe. `engine.run_seed` is a pure function of
(connectome, citation graph, seed, rng_seed), and recording consumes no randomness,
so pass one and pass two produce identical traces. Pass one learns how many decisions
there are; pass two records the chosen ones.

Output (committed, small):
  data/raster.npz
    body           int64[n]      simulated neuron ids, in brain index order
    is_readout     bool[n]
    is_input       bool[n]
    input_channel  int16[n]      -1 unless the neuron is in an input band
    readout_channel int16[n]     -1 unless the neuron is in a readout group
    steps          int32[k]      the decision indices recorded
    ms_<t>         int16[s]      millisecond of each spike in decision t
    idx_<t>        int32[s]      brain index of each spike in decision t

Usage:
  python src/raster.py --trace data/traces/W123.json --head 12 --tail 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import engine

ROOT = Path(__file__).resolve().parent.parent


def channel_map(groups, n: int) -> np.ndarray:
    out = np.full(n, -1, dtype=np.int16)
    for c, idx in enumerate(groups):
        if len(idx):
            out[np.asarray(idx, dtype=int)] = c
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="path to the trace to record")
    ap.add_argument("--head", type=int, default=12, help="record the first N decisions")
    ap.add_argument("--tail", type=int, default=1, help="also record the last N decisions")
    ap.add_argument("--out", default="data/raster.npz")
    args = ap.parse_args()

    trace = json.loads(Path(args.trace).read_text())
    sid = trace["seed"]["id"]

    cfg = yaml.safe_load((ROOT / "config.yml").read_text())
    p = dict(engine.DEFAULTS)
    rng_seed = trace["engine"].get("rng_seed", cfg["citations"]["rng_seed"])

    neurons, edges, papers, cedges, seeds = engine.load_graphs()
    brain = engine.Brain(neurons, edges, p)
    calib_summary = None
    if engine.CALIB.exists():
        calib = json.loads(engine.CALIB.read_text())
        brain.apply_calibration(calib)
        calib_summary = {k: calib.get(k) for k in engine.CALIB_KEYS}

    rec = next((s for s in seeds if s["seed"] == sid), None)
    if rec is None:
        raise SystemExit(f"Seed {sid} is not in data/citations/seeds.json.")

    # Which decisions to record. Only decisions that actually ran the brain have a
    # raster: the terminal step has no candidates and never fires.
    ran = [s["t"] for s in trace["steps"] if s.get("candidates")]
    if not ran:
        raise SystemExit(f"Trace {args.trace} has no decisions that ran the brain.")
    chosen = sorted(set(ran[: max(args.head, 0)]) | set(ran[len(ran) - max(args.tail, 0):]))
    print(f"recording {len(chosen)} of {len(ran)} decisions: {chosen}")

    rasters: dict[int, list[np.ndarray]] = {}
    replay = engine.run_seed(
        brain, papers, cedges, rec, cfg["policy"], p, rng_seed, calib_summary,
        record_steps=set(chosen), rasters=rasters,
    )
    if replay["result"] != trace["result"]:
        raise SystemExit(
            "FATAL: the replay diverged from the committed trace. Recording must not "
            "change the simulation.\n"
            f"  committed: {trace['result']}\n  replay:    {replay['result']}"
        )

    payload = {
        "body": brain.bodies.astype("int64"),
        "is_readout": np.isin(np.arange(brain.n), brain.readout_idx),
        "is_input": np.isin(np.arange(brain.n), brain.input_idx),
        "input_channel": channel_map(brain.input_bands, brain.n),
        "readout_channel": channel_map(brain.readout_groups, brain.n),
        "steps": np.array(chosen, dtype=np.int32),
    }
    total = 0
    for t in chosen:
        window = rasters.get(t, [])
        ms = np.concatenate([np.full(len(f), i, dtype=np.int16) for i, f in enumerate(window)]) \
            if window else np.zeros(0, dtype=np.int16)
        idx = np.concatenate(window).astype(np.int32) if window else np.zeros(0, dtype=np.int32)
        payload[f"ms_{t}"] = ms
        payload[f"idx_{t}"] = idx
        total += len(idx)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    print(f"{total:,} spikes across {len(chosen)} decisions -> {args.out} "
          f"({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
