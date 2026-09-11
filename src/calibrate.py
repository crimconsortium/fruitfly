"""Calibrate the brain, then check whether the brain matters.

Three things happen here, in order, and all three get written down:

  1. GAIN. With an arbitrary gain the network saturates: at gain=0.02 on synthetic
     data it fired at 481 Hz mean rate, which is nonsense and washes out every input
     difference. We binary-search gain to hit a target mean firing rate.

  2. READOUT GROUPS. Partitioning descending neurons into arbitrary contiguous groups
     produced 12% channel selectivity against a 12.5% chance baseline -- i.e. nothing,
     with one group winning every trial regardless of input. Instead we measure which
     channel each descending neuron actually prefers, z-score across channels, and take
     balanced groups of the most selective neurons per channel.

  3. THE CONTROL. Step 2 can manufacture selectivity out of noise by fitting it. So we
     rebuild everything on a degree-preserving edge shuffle of the same connectome and
     re-measure. On synthetic data the real graph scored 50% and its shuffle scored 52%:
     identical, meaning the wiring contributed nothing and the number came from the
     calibration procedure itself.

     If the real MaleCNS subgraph does not beat its own shuffle, that is the finding.
     We publish it. We do not tune until it goes away.

Output: data/calibration.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from engine import Brain, DEFAULTS

ROOT = Path(__file__).resolve().parent.parent
CONN = ROOT / "data" / "connectome"
OUT = ROOT / "data" / "calibration.json"

TARGET_HZ = 10.0
PROBE_HI = 200.0
PROBE_LO = 10.0
REPS = 4
TRIALS = 64
MARGIN = 0.05


def mean_rate(brain: Brain, gain: float, rng) -> float:
    brain.set_gain(gain)
    _, total = brain.run_detailed(np.full(brain.p["n_channels"], 60.0), rng)
    return total / brain.n / (brain.p["window_ms"] / 1000.0)


def tune_gain(brain: Brain, rng) -> tuple[float, float]:
    lo, hi = 1e-5, 1.0
    best = (lo, mean_rate(brain, lo, rng))
    for _ in range(18):
        mid = (lo * hi) ** 0.5
        rate = mean_rate(brain, mid, rng)
        if abs(rate - TARGET_HZ) < abs(best[1] - TARGET_HZ):
            best = (mid, rate)
        if rate > TARGET_HZ:
            hi = mid
        else:
            lo = mid
    return best


def response_matrix(brain: Brain, rng) -> np.ndarray:
    k = brain.p["n_channels"]
    resp = np.zeros((k, brain.n))
    for ch in range(k):
        for _ in range(REPS):
            d = np.full(k, PROBE_LO)
            d[ch] = PROBE_HI
            per_neuron, _ = brain.run_detailed(d, rng)
            resp[ch] += per_neuron
    return resp


def balanced_groups(brain: Brain, resp: np.ndarray) -> list[np.ndarray]:
    k = brain.p["n_channels"]
    R = resp[:, brain.readout_idx]
    z = (R - R.mean(axis=0)) / (R.std(axis=0) + 1e-9)
    per = max(1, len(brain.readout_idx) // k)
    taken, groups = set(), []
    for c in range(k):
        order = np.argsort(-z[c])
        picks = [int(i) for i in order if int(i) not in taken][:per]
        taken.update(picks)
        groups.append(brain.readout_idx[np.array(picks, dtype=int)])
    return groups


def selectivity(brain: Brain, groups, rng) -> float:
    k = brain.p["n_channels"]
    hits = 0
    for t in range(TRIALS):
        ch = t % k
        d = np.full(k, PROBE_LO)
        d[ch] = PROBE_HI
        per_neuron, _ = brain.run_detailed(d, rng)
        scores = [per_neuron[g].mean() if len(g) else -1 for g in groups]
        hits += int(np.argmax(scores)) == ch
    return hits / TRIALS


def shuffle_edges(edges: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Degree-preserving: permuting the post column preserves every in- and out-degree."""
    out = edges.copy()
    out["post"] = np.random.default_rng(seed).permutation(out["post"].to_numpy())
    return out[out["pre"] != out["post"]].reset_index(drop=True)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yml").read_text())
    p = dict(DEFAULTS)
    neurons = pd.read_parquet(CONN / "neurons.parquet")
    edges = pd.read_parquet(CONN / "edges.parquet")

    rng = np.random.default_rng(cfg["citations"]["rng_seed"])
    brain = Brain(neurons, edges, p)
    gain, rate = tune_gain(brain, rng)
    brain.set_gain(gain)
    print(f"gain={gain:.3e} -> {rate:.1f} Hz mean firing rate (target {TARGET_HZ})")

    groups = balanced_groups(brain, response_matrix(brain, rng))
    real = selectivity(brain, groups, rng)
    chance = 1.0 / p["n_channels"]
    print(f"real connectome selectivity:  {real:.1%}  (chance {chance:.1%})")

    sh = shuffle_edges(edges, seed=int(cfg["citations"]["rng_seed"]) + 1)
    brain_sh = Brain(neurons, sh, p)
    gain_sh, rate_sh = tune_gain(brain_sh, rng)
    brain_sh.set_gain(gain_sh)
    groups_sh = balanced_groups(brain_sh, response_matrix(brain_sh, rng))
    shuffled = selectivity(brain_sh, groups_sh, rng)
    print(f"shuffled control selectivity: {shuffled:.1%}")

    wiring_matters = real > shuffled + MARGIN
    verdict = (
        "The real wiring beats its own degree-preserving shuffle."
        if wiring_matters else
        "The real wiring does NOT beat its own shuffle. The fly's choices are "
        "effectively noise and the connectome is decoration. This is the result; "
        "we publish it rather than tuning until it disappears."
    )
    print(f"\nVERDICT: {verdict}")

    OUT.write_text(json.dumps({
        "gain": gain,
        "achieved_mean_hz": rate,
        "target_mean_hz": TARGET_HZ,
        "n_channels": p["n_channels"],
        "chance_level": chance,
        "selectivity_real": real,
        "selectivity_shuffled": shuffled,
        "shuffled_gain": gain_sh,
        "shuffled_achieved_mean_hz": rate_sh,
        "margin": MARGIN,
        "wiring_matters": bool(wiring_matters),
        "verdict": verdict,
        "trials": TRIALS,
        "probe_hz": {"high": PROBE_HI, "low": PROBE_LO},
        "readout_groups": [[int(b) for b in neurons["body"].to_numpy()[g]] for g in groups],
    }, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
