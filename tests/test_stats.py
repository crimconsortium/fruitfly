"""Regression test for the 99% denominator bug.

The first full run had 75 seeds blocked at the seed, 1 open seed whose crawl finished,
and 24 open seeds whose crawls hit the cap. The headline then excluded censored crawls
before computing the share blocked, giving 75/76 = 98.68%, reported as 99%.

The correct answer is 75/100 = 75%. These tests pin that.
"""
import collections

from fetch_citations import summarize

POLICY = {
    "open_statuses": ["diamond", "gold", "green", "hybrid"],
    "closed_statuses": ["bronze", "closed"],
    "sensitivity_open_statuses": ["diamond", "gold", "green", "hybrid", "bronze"],
}
CCFG = {"max_nodes_per_seed": 1500, "n_seeds": 100}


def record(status, passable, reachable, censored, walls=0):
    return {
        "seed": f"W{abs(hash((status, reachable, censored))) % 10**8}",
        "title": "t", "journal": "j", "year": 2020,
        "seed_oa_status": status, "seed_passable": passable,
        "world_size": reachable, "reachable": reachable,
        "reachable_beyond_seed": max(reachable - 1, 0),
        "walls": walls, "unresolved": 0,
        "censored": censored, "truncated": censored,
    }


def first_run_shape():
    recs = [record("closed", False, 0, False) for _ in range(73)]
    recs += [record("bronze", False, 0, False) for _ in range(2)]
    recs += [record("green", True, 149, False, walls=148)]
    recs += [record("hybrid", True, 1500, True, walls=200) for _ in range(24)]
    return recs


def test_share_blocked_uses_all_seeds():
    s = summarize(first_run_shape(), collections.Counter(), POLICY, CCFG, 0)
    assert s["seed_access"]["n_seeds"] == 100
    assert s["seed_access"]["n_blocked_at_seed"] == 75
    assert s["seed_access"]["share_blocked_at_seed"] == 0.75


def test_censored_crawls_are_counted_not_deleted():
    s = summarize(first_run_shape(), collections.Counter(), POLICY, CCFG, 0)
    assert s["crawl_size"]["n_censored"] == 24
    assert s["crawl_size"]["n_complete"] == 76
    assert s["crawl_size"]["n_complete"] + s["crawl_size"]["n_censored"] == 100


def test_open_seeds_are_not_lost():
    s = summarize(first_run_shape(), collections.Counter(), POLICY, CCFG, 0)
    assert s["seed_access"]["n_open_at_seed"] == 25
    assert s["crawl_size"]["share_of_open_seeds_censored"] == 0.96


def test_crawl_size_stats_use_complete_crawls_only():
    s = summarize(first_run_shape(), collections.Counter(), POLICY, CCFG, 0)
    # 75 blocked seeds contribute 0, one open seed contributes 148.
    assert s["crawl_size"]["complete_crawls_only"]["max_reachable_beyond_seed"] == 148
    assert s["crawl_size"]["complete_crawls_only"]["median_reachable_beyond_seed"] == 0.0


def test_paper_level_shares_exclude_unresolved():
    counts = collections.Counter({"closed": 60, "green": 20, "hybrid": 10,
                                 "bronze": 10, "unresolved": 100})
    s = summarize(first_run_shape(), counts, POLICY, CCFG, 200)
    assert s["papers"]["share_open_of_resolved"] == 0.3
    assert s["papers"]["share_open_of_resolved_with_bronze"] == 0.4
    assert s["papers"]["unresolved_share"] == 0.5
