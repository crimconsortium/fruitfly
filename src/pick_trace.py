"""Pick the representative trace and write RESULTS.md.

Representative = among seeds whose crawl COMPLETED and that actually moved, the run
whose reachable-paper count is closest to the median. Censored crawls are excluded
from this choice because their totals are lower bounds, not from any seed-level number.

Stdout carries exactly one line, `trace=<path>`, for $GITHUB_OUTPUT.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def pct(x):
    return "n/a" if x is None else f"{x * 100:.0f}%"


def num(x, nd=1):
    return "n/a" if x is None else f"{x:.{nd}f}"


def main() -> None:
    summary = pd.read_csv(ROOT / "data" / "traces" / "summary.csv")
    seeds = {s["seed"]: s for s in load(ROOT / "data" / "citations" / "seeds.json", [])}
    cite = load(ROOT / "data" / "citations" / "manifest.json", {}) or {}
    calib = load(ROOT / "data" / "calibration.json", {}) or {}
    conn = load(ROOT / "data" / "connectome" / "manifest.json", {}) or {}
    diag = load(ROOT / "data" / "diagnostics.json", {}) or {}

    summary["censored"] = summary["seed"].map(
        lambda s: bool(seeds.get(s, {}).get("censored", False)))
    movers = summary[(~summary["censored"]) & (summary["n_steps"] > 0)]
    pool = movers if len(movers) else summary
    median = pool["reachable"].median()
    pick = pool.loc[(pool["reachable"] - median).abs().idxmin()]
    trace = f"data/traces/{pick['seed']}.json"
    meta = seeds.get(pick["seed"], {})

    sa = cite.get("seed_access", {})
    cs = cite.get("crawl_size", {})
    cc = cs.get("complete_crawls_only", {})

    lines = [
        "# Results",
        "",
        "A fruit fly brain foraging the criminology citation graph. Open access is a",
        "corridor, a paywall is a wall, and it explores until it runs out of things it",
        "can read.",
        "",
        "## Can the fly even start?",
        "",
        "Computed over **every** seed. A seed's access status is known whether or not its",
        "crawl finished, so nothing is excluded here.",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Seeds drawn | {sa.get('n_seeds', 'n/a')} |",
        f"| Blocked at the seed | {sa.get('n_blocked_at_seed', 'n/a')} |",
        f"| Open at the seed | {sa.get('n_open_at_seed', 'n/a')} |",
        f"| Share blocked | {pct(sa.get('share_blocked_at_seed'))} |",
        "",
        f"Seed status counts: `{sa.get('seed_status_counts', {})}`",
        "",
        "## How far does it get when it can start?",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Crawls completed | {cs.get('n_complete', 'n/a')} |",
        f"| Crawls censored at the {cs.get('cap', 'n/a')}-paper cap | {cs.get('n_censored', 'n/a')} |",
        f"| Share of open seeds censored | {pct(cs.get('share_of_open_seeds_censored'))} |",
        f"| Median papers beyond the seed (completed only) | {num(cc.get('median_reachable_beyond_seed'), 0)} |",
        f"| Max papers beyond the seed (completed only) | {num(cc.get('max_reachable_beyond_seed'), 0)} |",
        f"| Median paywalls hit (completed only) | {num(cc.get('median_walls'), 0)} |",
        "",
        f"{cs.get('censored_crawls_note', '')}",
        "",
    ]

    if diag.get("corpus_oa_status_counts"):
        lines += [
            "## The whole field, for context",
            "",
            f"Across {diag.get('corpus_total', 0):,} articles in {diag.get('n_journals', 0)} journals:",
            "",
            "| Status | Articles |",
            "|---|---|",
        ]
        for k, v in sorted(diag["corpus_oa_status_counts"].items(), key=lambda kv: -kv[1]):
            lines.append(f"| {k} | {v:,} |")
        lines += [
            "",
            f"Open under our rule: **{pct(diag.get('corpus_open_share'))}**. "
            f"Including bronze: {pct(diag.get('corpus_open_share_with_bronze'))}.",
            "",
        ]

    lines += ["## Did the brain matter?", ""]
    if calib:
        lines += [
            "| Measure | Value |",
            "|---|---|",
            f"| Real connectome selectivity | {pct(calib.get('selectivity_real'))} |",
            f"| Shuffled controls, mean of {calib.get('n_shuffle_replicates', 'n/a')} | {pct(calib.get('selectivity_shuffled_mean'))} |",
            f"| Shuffled controls, sd | {pct(calib.get('selectivity_shuffled_sd'))} |",
            f"| Chance | {pct(calib.get('chance_level'))} |",
            f"| Mean firing rate achieved | {num(calib.get('achieved_mean_hz'), 2)} Hz |",
            "",
            f"Replicates: `{calib.get('selectivity_shuffled_replicates', [])}`",
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
        f"- Seed paper: {meta.get('title') or 'n/a'}",
        f"- Journal: {meta.get('journal') or 'n/a'} ({meta.get('year') or 'n/a'})",
        f"- Seed access: {meta.get('seed_oa_status') or 'n/a'}",
        f"- Papers reached: {int(pick['reachable'])}",
        f"- Paywalls hit: {int(pick['walls_hit'])}",
        f"- Moves: {int(pick['n_steps'])}",
        f"- Ended because: {pick['stuck_reason']}",
        "",
        "Chosen from completed, non-censored crawls that actually moved, as the run",
        "closest to the median reachable count. Not the most dramatic one.",
        "",
        "## Provenance",
        "",
        f"- Connectome: {conn.get('dataset', 'male-cns:v1.0')}, "
        f"{conn.get('counts', {}).get('neurons_kept', 'n/a')} neurons and "
        f"{conn.get('counts', {}).get('edges_kept', 'n/a')} signed edges after pruning. CC-BY.",
        "- Journals: Web of Science Criminology & Penology, adopted as-is. See corpus/PROVENANCE.md.",
        "- Metadata: OpenAlex, CC0.",
        "",
        "## Corrections",
        "",
        "An earlier run of this pipeline reported that 99% of seeds were paywalled. That",
        "was wrong. 25 of 100 seeds were open, but 24 of those hit the crawl cap and were",
        "then excluded from the headline, leaving 75 blocked and 1 open: 75/76 = 98.7%.",
        "Hitting a compute cap says nothing about whether a paper was readable. Seed-level",
        "statistics now use all seeds, and censored crawls are counted rather than deleted.",
        "tests/test_stats.py pins this.",
        "",
        "Regenerate everything with the `go` workflow.",
        "",
    ]

    (ROOT / "RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines), file=sys.stderr)
    print(f"trace={trace}")


if __name__ == "__main__":
    main()
