"""Calibrate the brain, then check whether the brain matters.

  1. GAIN. An arbitrary gain saturates the network (gain=0.02 gave 481 Hz on synthetic
     data, which washes out every input difference). We binary-search gain to hit a
     target mean firing rate.

  2. READOUT GROUPS. An arbitrary partition of descending neurons gave 12% channel
     selectivity against a 12.5% chance baseline, with one group winning every trial.
     Instead we measure which channel each descending neuron prefers, z-score across
     channels, and take balanced groups of the most selective neurons per channel.

  3. THE CONTROL, PLURAL. Step 2 can manufacture selectivity by fitting noise, so we
     rebuild everything on degree-preserving edge shuffles of the same connectome.
     A single shuffle is an anecdote: the first run scored real 41% vs shuffle 67%,
     which is not a null result, it is a control that beat reality -- the signature of
     a one-sample comparison on a metric with high variance. We now run several
     replicates and compare the real score against their mean and spread.

The verdict has three possible shapes and we publish whichever one comes out:
  real above the shuffle band  -> the wiring carries information
  real inside the band         -> null result, the connectome is decoration
  real below the band          -> the procedure is suspect, not reality

Output: data/calibration.json
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from engine import Brain, DEFAULTS

ROOT = Path(__file__).resolve().parent.parent
CONN = ROOT / "data" / "connectome"
OUT = ROOT / "data" / "calibration.json"

PROBE_HI = 200.0
PROBE_LO = 10.0
REPS = 4


def mean_rate(brain: Brain, gain: float, rng) -> float:
    brain.set_gain(gain)
    _, total = brain.run_detailed(np.full(brain.p["n_channels"], 60.0), rng)
    return total / brain.n / (brain.p["window_ms"] / 1000.0)


def tune_gain(brain: Brain, rng, target_hz: float):
    lo, hi = 1e-5, 1.0
    best = (lo, mean_rate(brain, lo, rng))
    for _ in range(18):
        mid = (lo * hi) ** 0.5
        rate = mean_rate(brain, mid, rng)
        if abs(rate - target_hz) < abs(best[1] - target_hz):
            best = (mid, rate)
        if rate > target_hz:
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


def selectivity(brain: Brain, groups, rng, trials: int) -> float:
    k = brain.p["n_channels"]
    hits = 0
    for t in range(trials):
        ch = t % k
        d = np.full(k, PROBE_LO)
        d[ch] = PROBE_HI
        per_neuron, _ = brain.run_detailed(d, rng)
        scores = [per_neuron[g].mean() if len(g) else -1 for g in groups]
        hits += int(np.argmax(scores)) == ch
    return hits / trials


def shuffle_edges(edges: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Degree-preserving: permuting the post column preserves every in- and out-degree."""
    out = edges.copy()
    out["post"] = np.random.default_rng(seed).permutation(out["post"].to_numpy())
    return out[out["pre"] != out["post"]].reset_index(drop=True)


def score_graph(neurons, edges, p, rng, target_hz, trials):
    brain = Brain(neurons, edges, p)
    gain, rate = tune_gain(brain, rng, target_hz)
    brain.set_gain(gain)
    groups = balanced_groups(brain, response_matrix(brain, rng))
    return brain, gain, rate, groups, selectivity(brain, groups, rng, trials)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yml").read_text())
    kcfg = cfg.get("calibration", {})
    target_hz = float(kcfg.get("target_mean_hz", 10.0))
    trials = int(kcfg.get("trials", 64))
    n_rep = int(kcfg.get("shuffle_replicates", 5))
    p = dict(DEFAULTS)

    neurons = pd.read_parquet(CONN / "neurons.parquet")
    edges = pd.read_parquet(CONN / "edges.parquet")
    base_seed = int(cfg["citations"]["rng_seed"])
    rng = np.random.default_rng(base_seed)

    brain, gain, rate, groups, real = score_graph(
        neurons, edges, p, rng, target_hz, trials)
    chance = 1.0 / p["n_channels"]
    print(f"gain={gain:.3e} -> {rate:.2f} Hz (target {target_hz})")
    print(f"real connectome selectivity: {real:.1%}  (chance {chance:.1%})")

    shuffles = []
    for i in range(n_rep):
        sh = shuffle_edges(edges, seed=base_seed + 101 + i)
        _, g_sh, r_sh, _, s_sh = score_graph(
            neurons, sh, p, rng, target_hz, trials)
        shuffles.append(s_sh)
        print(f"  shuffle {i + 1}/{n_rep}: {s_sh:.1%} (gain {g_sh:.2e}, {r_sh:.2f} Hz)")

    sh_mean = statistics.fmean(shuffles)
    sh_sd = statistics.stdev(shuffles) if len(shuffles) > 1 else 0.0
    band_hi = sh_mean + 2 * sh_sd
    band_lo = sh_mean - 2 * sh_sd

    if real > band_hi:
        verdict = ("The real wiring beats its degree-preserving shuffles by more than two "
                   "standard deviations. The connectome carries information here.")
        matters = True
    elif real < band_lo:
        verdict = ("The shuffles BEAT the real wiring by more than two standard deviations. "
                   "That is not a finding about brains, it is a warning about our metric: "
                   "the selectivity measure is picking up something the shuffle supplies "
                   "more of than reality does. Reported, not tuned away.")
        matters = False
    else:
        verdict = ("The real wiring is indistinguishable from its own degree-preserving "
                   "shuffles. Null result: the fly's route is noise. The paywalls it cannot "
                   "pass are still entirely real.")
        matters = False

    print(f"\nshuffles: mean {sh_mean:.1%}, sd {sh_sd:.1%}, n={n_rep}")
    print(f"VERDICT: {verdict}")

    OUT.write_text(json.dumps({
        "gain": gain,
        "achieved_mean_hz": round(rate, 2),
        "target_mean_hz": target_hz,
        "n_channels": p["n_channels"],
        "chance_level": round(chance, 4),
        "selectivity_real": round(real, 4),
        "selectivity_shuffled_replicates": [round(s, 4) for s in shuffles],
        "selectivity_shuffled_mean": round(sh_mean, 4),
        "selectivity_shuffled_sd": round(sh_sd, 4),
        "shuffle_band_2sd": [round(band_lo, 4), round(band_hi, 4)],
        "n_shuffle_replicates": n_rep,
        "trials_per_score": trials,
        "wiring_matters": bool(matters),
        "verdict": verdict,
        "probe_hz": {"high": PROBE_HI, "low": PROBE_LO},
        "readout_groups": [[int(b) for b in neurons["body"].to_numpy()[g]] for g in groups],
    }, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
