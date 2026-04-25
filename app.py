"""
Flask API backed by the MovieLens-trained BERT4Rec model.

Start with:  python app.py
Frontend at: http://localhost:5000
API at:      POST http://localhost:5000/recommend  {"username": "someuser"}
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, request, send_from_directory

from recommender.data_processing import MASK, PAD
from recommender.letterboxd_data import fetch_user_diary
from recommender.models import Recommender

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = Path("recommender_models_ml/recommender.ckpt")
BRIDGE_JSON     = Path("data/letterboxd_to_movielens.json")
ML_MOVIES_CSV   = Path("data/ml-25m/movies.csv")

# Use ratings_small.csv when it exists (matches what run_movielens.py trains on),
# otherwise fall back to the full ratings.csv.
_RATINGS_CANDIDATES = [
    Path("data/ml-25m/ratings_small.csv"),
    Path("data/ml-25m/ratings.csv"),
]

HISTORY_SIZE = 120
TOP_N        = 30
MIN_MATCHES  = 5    # minimum bridge matches required to run inference

# ---------------------------------------------------------------------------
# Global state — populated once at startup, reused for every request
# ---------------------------------------------------------------------------
_model: Recommender | None = None
_ml_mapping: dict | None = None          # ml_movieId (int)  → mapped_int
_ml_inverse_mapping: dict | None = None  # mapped_int        → ml_movieId (int)
_title_lookup: dict | None = None        # ml_movieId (int)  → "Title (Year)"
_bridge: dict | None = None              # lb_slug (str)     → ml_movieId (int)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_ratings_csv() -> Path:
    for p in _RATINGS_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "No ratings CSV found in data/ml-25m/. "
        "Expected ratings_small.csv or ratings.csv."
    )


def _build_ml_mapping(ratings_csv: Path) -> tuple[dict, dict]:
    """
    Reconstruct the integer ID mapping that map_column() built during training.

    map_column (data_processing.py:17-18) does:
        values = sorted(list(df["movieId"].unique()))
        mapping = {k: i + 2 for i, k in enumerate(values)}

    We reproduce that exactly using only the movieId column so we don't have
    to load the full 25M-row file into memory.
    """
    print(f"  Building vocab mapping from {ratings_csv} …")
    df = pd.read_csv(ratings_csv, usecols=["movieId"])
    unique_ids = sorted(df["movieId"].unique().tolist())
    mapping = {k: i + 2 for i, k in enumerate(unique_ids)}
    inverse_mapping = {v: k for k, v in mapping.items()}
    print(f"  Vocab: {len(mapping)} unique ML movie IDs → integers [2, {len(mapping)+1}]")
    return mapping, inverse_mapping


def _load_checkpoint(path: Path, vocab_size: int) -> Recommender:
    model = Recommender(vocab_size=vocab_size, lr=1e-4, dropout=0.3)
    state = torch.load(str(path), map_location="cpu")
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


def _ensure_loaded() -> None:
    """Populate module-level globals. No-op after first successful call."""
    global _model, _ml_mapping, _ml_inverse_mapping, _title_lookup, _bridge

    if _model is not None:
        return

    if not CHECKPOINT_PATH.exists():
        raise RuntimeError("Model not trained yet. Run run_movielens.py first.")

    print("Loading MovieLens model and supporting data…")

    # 1. Bridge: Letterboxd slug → ML movieId
    with open(BRIDGE_JSON) as fh:
        raw = json.load(fh)
    _bridge = {slug: int(ml_id) for slug, ml_id in raw.items()}
    print(f"  Bridge: {len(_bridge)} Letterboxd slugs mapped to ML IDs")

    # 2. ML vocab mapping (must be identical to what training.py used)
    ratings_csv = _find_ratings_csv()
    _ml_mapping, _ml_inverse_mapping = _build_ml_mapping(ratings_csv)

    # 3. Title lookup: ML movieId → "Title (Year)"
    movies_df = pd.read_csv(ML_MOVIES_CSV)
    _title_lookup = dict(zip(movies_df["movieId"], movies_df["title"]))
    print(f"  Titles: {len(_title_lookup)} films loaded from movies.csv")

    # 4. Model
    vocab_size = len(_ml_mapping) + 2  # +2 for PAD=0 and MASK=1
    _model = _load_checkpoint(CHECKPOINT_PATH, vocab_size)
    print(f"  Model ready  vocab_size={vocab_size}  ckpt={CHECKPOINT_PATH}")


def _run_inference(rows: list[dict]) -> list[str] | None:
    """
    Translate a user's diary rows into a ranked list of film titles.

    Returns None when fewer than MIN_MATCHES films are in the bridge
    (caller should surface the "not enough matches" error).
    """
    rows_sorted = sorted(rows, key=lambda r: r["timestamp"])

    # Walk the watch history in order; translate slug → ML id → mapped int
    matched: list[int] = []
    for row in rows_sorted:
        ml_id = _bridge.get(row["movieId"])
        if ml_id is None:
            continue
        mapped_id = _ml_mapping.get(ml_id)
        if mapped_id is None:
            continue
        matched.append(mapped_id)

    if len(matched) < MIN_MATCHES:
        return None

    # Build input sequence: left-pad to (HISTORY_SIZE-1), then append MASK
    context   = matched[-(HISTORY_SIZE - 1):]
    pad_count = (HISTORY_SIZE - 1) - len(context)
    input_ids = [PAD] * pad_count + context + [MASK]

    src = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        logits = _model(src)   # (1, HISTORY_SIZE, vocab_size)

    # Read logits at the MASK position (last token)
    scores  = logits[0, -1].numpy()
    exclude = set(input_ids)
    ranked  = [i for i in np.argsort(scores)[::-1].tolist() if i not in exclude]

    results: list[str] = []
    for idx in ranked:
        if len(results) >= TOP_N:
            break
        ml_id = _ml_inverse_mapping.get(idx)
        if ml_id is None:
            continue
        title = _title_lookup.get(ml_id)
        if title is None:
            continue
        results.append(title)

    return results


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/recommend", methods=["POST"])
def recommend():
    body     = request.get_json(silent=True) or {}
    username = (body.get("username") or "").strip().lstrip("@")

    if not username:
        return jsonify({"error": "username is required"}), 400

    # Ensure model is loaded — fail fast with a clear message if not
    try:
        _ensure_loaded()
    except (RuntimeError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 503

    # Scrape the user's Letterboxd diary
    try:
        rows = fetch_user_diary(username)
    except Exception as exc:
        if "ResourceNotFoundError" in type(exc).__name__ or "not found" in str(exc).lower():
            return jsonify({
                "error": f"Letterboxd user '{username}' was not found. "
                         "Check the username and make sure the profile is public."
            }), 404
        return jsonify({"error": f"Failed to fetch diary: {exc}"}), 500

    if not rows:
        return jsonify({
            "error": f"No diary entries found for '{username}'. "
                     "The diary may be empty or set to private."
        }), 404

    recs = _run_inference(rows)

    if recs is None:
        return jsonify({
            "error": "Not enough films in your diary match our database. "
                     "Try adding more films to your Letterboxd."
        }), 422

    if not recs:
        return jsonify({"error": "Could not generate recommendations for this user."}), 422

    return jsonify({"username": username, "recommendations": recs})


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        _ensure_loaded()
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"\nWARNING: {exc}")
        print("The /recommend endpoint will return 503 until this is resolved.\n")

    app.run(debug=False, port=5000)
