"""Pick the representative trace and write RESULTS.md.

Representative means the non-truncated run whose reachable-paper count is closest to
the median. Not the biggest, not the most dramatic: the typical one.

Stdout carries exactly one line, `trace=<path>`, so a workflow can append it to
$GITHUB_OUTPUT. Everything human-readable goes to stderr and to RESULTS.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load(path: Path, default=None):
    if path.exists():
        return json.loads(path.read_text())
    return default


def pct(x):
    return "n/a" if x is None else f"{x * 100:.0f}%"


def main() -> None:
    summary = pd.read_csv(ROOT / "data" / "traces" / "summary.csv")
    seeds = {s["seed"]: s for s in load(ROOT / "data" / "citations" / "seeds.json", [])}
    cite = load(ROOT / "data" / "citations" / "manifest.json", {}) or {}
    calib = load(ROOT / "data" / "calibration.json", {}) or {}
    conn = load(ROOT / "data" / "connectome" / "manifest.json", {}) or {}

    clean = summary[~summary["truncated"]] if "truncated" in summary else summary
    if clean.empty:
        clean = summary
    median = clean["reachable"].median()
    pick_idx = (clean["reachable"] - median).abs().idxmin()
    pick = clean.loc[pick_idx]
    trace = f"data/traces/{pick['seed']}.json"
    seed_meta = seeds.get(pick["seed"], {})

    head = cite.get("headline", {})
    lines = [
        "# Results",
        "",
        "A fruit fly brain foraging the criminology citation graph. Open access is a",
        "corridor, a paywall is a wall, and it explores until it runs out of things it",
        "can read.",
        "",
        "## Headline",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Seeds drawn (random, reproducible) | {cite.get('n_seeds', 'n/a')} |",
        f"| Seeds excluded as truncated | {cite.get('n_truncated', 'n/a')} |",
        f"| Median papers reachable from a seed | {head.get('median_reachable', 'n/a')} |",
        f"| Median paywalls hit | {head.get('median_walls', 'n/a')} |",
        f"| Seeds that are themselves paywalled | {pct(head.get('share_seeds_paywalled'))} |",
        f"| Seeds stuck immediately | {pct(head.get('share_seeds_stuck_immediately'))} |",
        f"| Papers seen in total | {cite.get('n_papers', 'n/a')} |",
        "",
        "Open access here means diamond, gold, green, or hybrid. Bronze and closed are",
        "walls. That is stricter than OpenAlex's own `is_oa`, which counts bronze.",
        "",
        "## Did the brain matter?",
        "",
    ]
    if calib:
        lines += [
            "| Measure | Value |",
            "|---|---|",
            f"| Real connectome channel selectivity | {pct(calib.get('selectivity_real'))} |",
            f"| Degree-preserving shuffled control | {pct(calib.get('selectivity_shuffled'))} |",
            f"| Chance | {pct(calib.get('chance_level'))} |",
            f"| Mean firing rate achieved | {calib.get('achieved_mean_hz', 'n/a')} Hz |",
            "",
            f"**{calib.get('verdict', '')}**",
            "",
        ]
    else:
        lines += ["No calibration found. The readout groups were arbitrary.", ""]

    lines += [
        "## The rendered run",
        "",
        f"- Trace: `{trace}`",
        f"- Seed paper: {seed_meta.get('title') or 'n/a'}",
        f"- Journal: {seed_meta.get('journal') or 'n/a'} ({seed_meta.get('year') or 'n/a'})",
        f"- Seed access: {seed_meta.get('seed_oa_status') or 'n/a'}",
        f"- Papers reached: {int(pick['reachable'])}",
        f"- Paywalls hit: {int(pick['walls_hit'])}",
        f"- Moves: {int(pick['n_steps'])}",
        f"- Ended because: {pick['stuck_reason']}",
        "",
        "Chosen as the run closest to the median reachable-paper count, not the most",
        "dramatic one.",
        "",
        "## Distribution across all seeds",
        "",
        "| Statistic | Papers reached | Paywalls hit |",
        "|---|---|---|",
        f"| min | {int(clean['reachable'].min())} | {int(clean['walls_hit'].min())} |",
        f"| median | {clean['reachable'].median():.0f} | {clean['walls_hit'].median():.0f} |",
        f"| mean | {clean['reachable'].mean():.1f} | {clean['walls_hit'].mean():.1f} |",
        f"| max | {int(clean['reachable'].max())} | {int(clean['walls_hit'].max())} |",
        "",
        "## Provenance",
        "",
        f"- Connectome: {conn.get('dataset', 'male-cns:v1.0')}, "
        f"{conn.get('counts', {}).get('neurons_kept', 'n/a')} neurons and "
        f"{conn.get('counts', {}).get('edges_kept', 'n/a')} signed edges after pruning. CC-BY.",
        "- Journals: Web of Science Criminology & Penology, adopted as-is. See corpus/PROVENANCE.md.",
        "- Metadata: OpenAlex, CC0.",
        "",
        "Regenerate everything with the `go` workflow.",
        "",
    ]

    (ROOT / "RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines), file=sys.stderr)
    print(f"trace={trace}")


if __name__ == "__main__":
    main()
