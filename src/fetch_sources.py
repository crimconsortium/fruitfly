"""Resolve the WoS Criminology & Penology journal names to OpenAlex sources.

Input:  corpus/wos_criminology_penology.txt (adopted as-is, never edited)
Output: data/sources.csv (for human review)

We do not choose journals. We only try to figure out which OpenAlex source each
name refers to, and we flag every case where we are not confident.
"""
from __future__ import annotations

import csv
import difflib
import os
import time
from pathlib import Path

import requests
import yaml

API = "https://api.openalex.org"
MAILTO = os.environ.get("OPENALEX_MAILTO", "")
ROOT = Path(__file__).resolve().parent.parent
STATUSES = ["diamond", "gold", "green", "hybrid", "bronze", "closed"]
SESSION = requests.Session()

# Regression assertions, not curation: these ISSNs are verified from published
# sources. If the resolver stops matching them, the resolver broke.
ISSN_ASSERTIONS = {
    "CRIMINOLOGY": "0011-1384",
    "JUSTICE QUARTERLY": "0741-8825",
    "JOURNAL OF CRIMINAL JUSTICE EDUCATION": "1051-1253",
    "TRAUMA VIOLENCE & ABUSE": "1524-8380",
}


def get(path: str, **params: str) -> dict:
    if MAILTO:
        params["mailto"] = MAILTO
    for attempt in range(5):
        r = SESSION.get(f"{API}/{path}", params=params, timeout=60)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503):
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
    raise RuntimeError(f"OpenAlex request failed: {path} {params}")


def _norm(s: str) -> str:
    s = s.lower().replace("&", "and")
    for sep in ("-an ", ": ", " - "):
        if sep in s:
            s = s.split(sep)[0]
    return s.replace("the ", "").strip()


def score(name: str, source: dict) -> float:
    titles = [source.get("display_name") or ""] + (source.get("alternate_titles") or [])
    return max(difflib.SequenceMatcher(None, _norm(name), _norm(t)).ratio() for t in titles)


def resolve(name: str):
    data = get("sources", search=_norm(name), per_page="25")
    cands = [s for s in data.get("results", []) if s.get("type") == "journal"]
    if not cands:
        return None, 0.0
    best = max(cands, key=lambda s: score(name, s))
    return best, score(name, best)


def oa_mix(source_id: str, y0: int, y1: int) -> dict:
    filt = f"primary_location.source.id:{source_id},publication_year:{y0}-{y1}"
    data = get("works", filter=filt, group_by="open_access.oa_status")
    counts = {s: 0 for s in STATUSES}
    for g in data.get("group_by", []):
        if g["key"] in counts:
            counts[g["key"]] = g["count"]
    return counts


def read_corpus(path: Path) -> list[str]:
    lines = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yml").read_text())
    y0, y1 = cfg["policy"]["years"]
    open_set = set(cfg["policy"]["open_statuses"])
    sens_set = set(cfg["policy"]["sensitivity_open_statuses"])
    min_score = cfg["resolver"]["min_match_score"]

    names = read_corpus(ROOT / cfg["corpus"]["primary"])
    outdir = ROOT / "data"
    outdir.mkdir(exist_ok=True)

    rows = []
    for name in names:
        src, sc = resolve(name)
        if src is None:
            rows.append({"wos_name": name, "review_flag": "NOT_FOUND"})
            print(f"NOT_FOUND      ----  {name}")
            continue
        counts = oa_mix(src["id"], y0, y1)
        total = sum(counts.values())
        opened = sum(v for k, v in counts.items() if k in open_set)
        opened_sens = sum(v for k, v in counts.items() if k in sens_set)
        expected = ISSN_ASSERTIONS.get(name)
        if expected and expected != src.get("issn_l"):
            flag = "ISSN_MISMATCH"
        elif sc < min_score:
            flag = "CHECK"
        else:
            flag = "OK"
        rows.append({
            "wos_name": name,
            "matched_name": src.get("display_name"),
            "match_score": round(sc, 3),
            "review_flag": flag,
            "openalex_id": src.get("id"),
            "issn_l": src.get("issn_l"),
            "asserted_issn_l": expected or "",
            "is_oa": src.get("is_oa"),
            "is_in_doaj": src.get("is_in_doaj"),
            "apc_usd": src.get("apc_usd") or "",
            "works_total": total,
            "open_share": round(opened / total, 4) if total else "",
            "open_share_with_bronze": round(opened_sens / total, 4) if total else "",
            **{f"n_{s}": counts[s] for s in STATUSES},
        })
        print(f"{flag:14} {sc:.2f}  {name} -> {src.get('display_name')}")

    fields = [
        "wos_name", "matched_name", "match_score", "review_flag", "openalex_id",
        "issn_l", "asserted_issn_l", "is_oa", "is_in_doaj", "apc_usd",
        "works_total", "open_share", "open_share_with_bronze",
    ] + [f"n_{s}" for s in STATUSES]
    with (outdir / "sources.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})

    needs_review = [r for r in rows if r.get("review_flag") != "OK"]
    print(f"\n{len(rows)} journals, {len(needs_review)} need review -> data/sources.csv")
    for r in needs_review:
        print(f"  {r['review_flag']}: {r['wos_name']} -> {r.get('matched_name')}")


if __name__ == "__main__":
    main()
