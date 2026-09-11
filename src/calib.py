"""Shared, defensive reading of data/calibration.json.

The calibration format changed once: a single `selectivity_shuffled` became
`selectivity_shuffled_mean` plus a list of replicates. That one rename crashed run #3
in the engine and run #4 in the renderer, both times because the field was indexed
directly instead of fetched with a default.

Everything that reads calibration goes through here. Nothing indexes calibration keys.
tests/test_calib.py covers the old format, the new format, and no calibration at all.
"""
from __future__ import annotations

# Fields copied into traces and onto the page. All optional.
KEYS = (
    "gain", "achieved_mean_hz", "chance_level", "selectivity_real",
    "selectivity_shuffled_mean", "selectivity_shuffled_sd",
    "selectivity_shuffled_replicates", "shuffle_band_2sd",
    "n_shuffle_replicates", "trials_per_score", "wiring_matters", "verdict",
)


def pct(x, nd: int = 0) -> str:
    return "n/a" if x is None else f"{x * 100:.{nd}f}%"


def shuffled_mean(calib: dict | None):
    """Mean shuffled selectivity, tolerating the older single-shuffle format."""
    if not calib:
        return None
    if calib.get("selectivity_shuffled_mean") is not None:
        return calib["selectivity_shuffled_mean"]
    return calib.get("selectivity_shuffled")


def replicates(calib: dict | None) -> list:
    if not calib:
        return []
    reps = calib.get("selectivity_shuffled_replicates")
    if reps:
        return list(reps)
    single = calib.get("selectivity_shuffled")
    return [single] if single is not None else []


def n_replicates(calib: dict | None) -> int:
    if not calib:
        return 0
    n = calib.get("n_shuffle_replicates")
    if n:
        return int(n)
    return len(replicates(calib))


def summary(calib: dict | None) -> dict:
    """The subset of calibration worth carrying around, with None for anything absent."""
    if not calib:
        return {}
    return {k: calib.get(k) for k in KEYS}


def summary_line(calib: dict | None) -> str:
    """One line fit for a video caption or a console log. Empty if uncalibrated."""
    if not calib:
        return ""
    real = calib.get("selectivity_real")
    shuf = shuffled_mean(calib)
    chance = calib.get("chance_level")
    n = n_replicates(calib)
    bits = [f"channel selectivity {pct(real)}"]
    if shuf is not None:
        label = f"shuffled-connectome control {pct(shuf)}"
        if n > 1:
            label += f" (mean of {n})"
        bits.append("vs " + label)
    if chance is not None:
        bits.append(f"chance {pct(chance)}")
    return ", ".join(bits)
