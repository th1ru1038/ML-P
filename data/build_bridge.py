"""
Build a mapping from Letterboxd slugs → MovieLens movieIds using TMDB as the bridge.

Run from the repo root:
    TMDB_API_KEY=your_key python data/build_bridge.py

Output: data/letterboxd_to_movielens.json  {slug: ml_movieId}
"""

import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths (all relative to repo root, not this file's location)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
LB_RATINGS_CSV = REPO_ROOT / "letterboxd_ratings.csv"
LINKS_CSV = REPO_ROOT / "data" / "ml-25m" / "links.csv"
OUTPUT_JSON = REPO_ROOT / "data" / "letterboxd_to_movielens.json"

TMDB_BASE = "https://api.themoviedb.org/3/search/movie"
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")


# ---------------------------------------------------------------------------
# Slug cleaning
# ---------------------------------------------------------------------------

def clean_slug(slug: str):
    """
    Extract the search title and optional year from a Letterboxd slug.

    "parasite-2019"  → ("parasite", 2019)
    "inception"      → ("inception", None)
    "1917"           → ("1917", None)       ← title IS the number, no year suffix
    "malcolm-x-1992" → ("malcolm x", 1992)
    """
    m = re.search(r"-(\d{4})$", slug)
    if m:
        year = int(m.group(1))
        base = slug[: m.start()]
    else:
        year = None
        base = slug
    title = base.replace("-", " ")
    return title, year


# ---------------------------------------------------------------------------
# TMDB search
# ---------------------------------------------------------------------------

def tmdb_search(title: str, year: int | None, api_key: str) -> int | None:
    """Return the TMDB movie id for the best match, or None on failure."""
    params = {"api_key": api_key, "query": title, "include_adult": "false"}
    if year:
        params["year"] = year

    try:
        resp = requests.get(TMDB_BASE, params=params, timeout=10)
        resp.raise_for_status()
        results = resp.json().get("results", [])
    except requests.RequestException as exc:
        print(f"    TMDB request error: {exc}")
        return None

    if results:
        return results[0]["id"]

    # If year-filtered search returned nothing, retry without the year filter.
    if year:
        params.pop("year")
        try:
            resp = requests.get(TMDB_BASE, params=params, timeout=10)
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if results:
                return results[0]["id"]
        except requests.RequestException:
            pass

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not TMDB_API_KEY:
        raise EnvironmentError(
            "TMDB_API_KEY environment variable is not set.\n"
            "Get a free key at https://www.themoviedb.org/settings/api\n"
            "Then run:  export TMDB_API_KEY=your_key"
        )

    # Load Letterboxd slugs (deduplicated, original order doesn't matter here)
    lb_df = pd.read_csv(LB_RATINGS_CSV)
    slugs = lb_df["movieId"].unique().tolist()
    print(f"Letterboxd slugs to bridge: {len(slugs)}")

    # Load links.csv and build tmdbId → ml_movieId lookup
    links = pd.read_csv(LINKS_CSV)
    links = links.dropna(subset=["tmdbId"])
    links["tmdbId"] = links["tmdbId"].astype(int)
    tmdb_to_ml = dict(zip(links["tmdbId"], links["movieId"]))
    print(f"MovieLens tmdbId entries: {len(tmdb_to_ml)}")

    bridge = {}
    failed = []

    for i, slug in enumerate(slugs, 1):
        title, year = clean_slug(slug)
        tmdb_id = tmdb_search(title, year, TMDB_API_KEY)

        if tmdb_id is None:
            print(f"  [{i:>3}/{len(slugs)}] TMDB miss      : {slug!r}  (searched: {title!r})")
            failed.append(slug)
            time.sleep(0.05)
            continue

        ml_id = tmdb_to_ml.get(tmdb_id)
        if ml_id is None:
            print(f"  [{i:>3}/{len(slugs)}] not in ML-25m  : {slug!r}  (tmdbId={tmdb_id}, title={title!r})")
            failed.append(slug)
            time.sleep(0.05)
            continue

        print(f"  [{i:>3}/{len(slugs)}] OK  {slug!r:45s} → tmdb={tmdb_id}  ml={ml_id}")
        bridge[slug] = int(ml_id)
        time.sleep(0.05)   # ~20 req/s, well inside TMDB's free-tier limit

    # Save
    with open(OUTPUT_JSON, "w") as fh:
        json.dump(bridge, fh, indent=2)

    print()
    print("=" * 60)
    print(f"Matched : {len(bridge)} / {len(slugs)}")
    print(f"Failed  : {len(failed)}")
    if failed:
        print("Failed slugs:")
        for s in failed:
            print(f"  {s}")
    print(f"Saved   : {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
