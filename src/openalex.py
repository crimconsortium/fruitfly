"""One HTTP client for OpenAlex, with honest logging and patient retries.

Run #2 died mid-crawl after 44 seeds: a batch exhausted six retries and the log never
said which status code caused it. Two things were wrong. The retry loop was silent, and
the caller treated one bad batch as fatal.

This module fixes the first: every retry prints the HTTP status, the Retry-After header
if the server sent one, and a snippet of the response body. Non-retryable statuses fail
immediately with the body attached instead of burning eight attempts on a 400.
"""
from __future__ import annotations

import os
import random
import sys
import time

import requests

API = "https://api.openalex.org"
MAILTO = os.environ.get("OPENALEX_MAILTO", "").strip()
SESSION = requests.Session()
RETRY_STATUSES = {408, 425, 429, 500, 502, 503, 504}


class OpenAlexError(RuntimeError):
    """A request that could not be completed, with the reason attached."""


def warn_if_no_mailto() -> None:
    if MAILTO:
        print(f"OpenAlex polite pool: {MAILTO}")
    else:
        print("WARNING: OPENALEX_MAILTO is not set, so requests go to the common pool "
              "and are rate-limited harder. Set a repository variable named "
              "OPENALEX_MAILTO to your email.", file=sys.stderr)


def get(path: str, *, max_attempts: int = 8, max_backoff: float = 120.0, **params) -> dict:
    if MAILTO:
        params["mailto"] = MAILTO
    last = "no attempt made"
    for attempt in range(1, max_attempts + 1):
        try:
            r = SESSION.get(f"{API}/{path}", params=params, timeout=120)
        except requests.RequestException as exc:
            last = f"network error: {exc}"
            r = None
        if r is not None:
            if r.status_code == 200:
                return r.json()
            body = r.text[:300].replace("\n", " ")
            last = f"HTTP {r.status_code}: {body}"
            if r.status_code not in RETRY_STATUSES:
                raise OpenAlexError(f"{last} | {path} {params}")
        retry_after = (r.headers.get("Retry-After") if r is not None else None)
        if retry_after and retry_after.strip().isdigit():
            wait = min(float(retry_after), max_backoff)
        else:
            wait = min(max_backoff, 2.0 ** attempt)
        wait *= 0.7 + 0.6 * random.random()
        print(f"    attempt {attempt}/{max_attempts}: {last}; sleeping {wait:.0f}s",
              file=sys.stderr)
        time.sleep(wait)
    raise OpenAlexError(f"gave up after {max_attempts} attempts. last: {last}")
