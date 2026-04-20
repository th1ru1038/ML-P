import json
from datetime import datetime, timezone

import pandas as pd

from recommender.data_processing import map_column


def _parse_timestamp(date_str: str) -> int:
    """Convert a YYYY-MM-DD diary date string to a UTC unix timestamp."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def fetch_user_diary(username: str) -> list[dict]:
    """
    Fetch all diary entries for one Letterboxd user.

    letterboxdpy returns:
        {"entries": {log_id (int): {"name": str, "slug": str,
                                    "release": int, "date": str,
                                    "rating": float|None, ...}}}

    We flatten that into a list of row dicts.
    """
    from letterboxdpy.user import User  # import here so the module is optional at import time

    user_obj = User(username)
    diary_data = user_obj.get_diary()

    rows = []
    for entry in diary_data.get("entries", {}).values():
        date_str = entry.get("date")
        slug = entry.get("slug")
        if not date_str or not slug:
            continue
        try:
            # date arrives as ISO 8601 e.g. "2026-03-02T00:00:00.000000Z"
            ts = _parse_timestamp(date_str[:10])
        except ValueError:
            continue
        actions = entry.get("actions") or {}
        rows.append(
            {
                "userId": username,
                "movieId": slug,
                "title": entry.get("name", slug),
                "release": entry.get("release"),
                "rating": actions.get("rating"),
                "timestamp": ts,
            }
        )
    return rows


def build_letterboxd_dataset(
    usernames: list,
    output_csv: str = "letterboxd_ratings.csv",
    mapping_json: str = "letterboxd_movie_mapping.json",
):
    """
    Scrape diary entries for all usernames and produce the files that the
    existing training pipeline expects.

    Returns
    -------
    df               DataFrame with columns [userId, movieId, movieId_mapped, timestamp]
                     — ready to pass straight into training.train().
    mapping          dict  slug -> int id (starting at 2; 0=PAD, 1=MASK are reserved)
    inverse_mapping  dict  int id -> slug
    slug_to_meta     dict  slug -> {"title": str, "release": int|None}
    """
    all_rows = []
    for username in usernames:
        print(f"Fetching diary for '{username}' …")
        try:
            rows = fetch_user_diary(username)
            print(f"  {len(rows)} diary entries")
            all_rows.extend(rows)
        except Exception as exc:
            print(f"  ERROR fetching '{username}': {exc}")

    if not all_rows:
        raise ValueError(
            "No diary entries were fetched. "
            "Check that the usernames are correct and letterboxdpy is installed."
        )

    df_full = pd.DataFrame(all_rows)

    # Build slug -> display metadata before we drop the extra columns.
    slug_to_meta = (
        df_full[["movieId", "title", "release"]]
        .drop_duplicates(subset="movieId")
        .set_index("movieId")[["title", "release"]]
        .to_dict(orient="index")
    )

    # The training pipeline only needs these three columns.
    df = df_full[["userId", "movieId", "timestamp"]].copy()
    df.sort_values(by="timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # map_column adds movieId_mapped and returns forward + inverse dicts.
    # Integers start at 2 (matching PAD=0, MASK=1 in data_processing.py).
    df, mapping, inverse_mapping = map_column(df, col_name="movieId")

    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} rows → {output_csv}")

    # Persist mappings so inference can reload them without re-scraping.
    # JSON requires string keys, so we stringify the integer keys.
    serialisable = {
        "mapping": {str(k): v for k, v in mapping.items()},
        "inverse_mapping": {str(v): k for k, v in mapping.items()},  # int_id -> slug
        "slug_to_meta": slug_to_meta,
    }
    with open(mapping_json, "w") as fh:
        json.dump(serialisable, fh, indent=2)
    print(f"Saved mappings → {mapping_json}")

    return df, mapping, inverse_mapping, slug_to_meta


def load_mappings(mapping_json: str = "letterboxd_movie_mapping.json"):
    """
    Reload the mappings saved by build_letterboxd_dataset.

    Returns
    -------
    mapping          slug  -> int id
    inverse_mapping  int id -> slug
    slug_to_meta     slug  -> {"title": str, "release": int|None}
    """
    with open(mapping_json) as fh:
        data = json.load(fh)

    mapping = {k: int(v) for k, v in data["mapping"].items()}
    inverse_mapping = {int(k): v for k, v in data["inverse_mapping"].items()}
    slug_to_meta = data["slug_to_meta"]
    return mapping, inverse_mapping, slug_to_meta
