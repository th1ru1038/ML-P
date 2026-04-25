"""
Flask API for the Letterboxd BERT4Rec recommender.

Start with:  python app.py
Frontend at: http://localhost:5000
API at:      POST http://localhost:5000/recommend
"""

from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory

from recommender.data_processing import MASK, PAD
from recommender.letterboxd_data import fetch_user_diary, load_mappings
from recommender.models import Recommender

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = Path("recommender_models/recommender.ckpt")
MAPPING_JSON = "letterboxd_movie_mapping.json"
DATA_CSV = "letterboxd_ratings.csv"
HISTORY_SIZE = 120
TOP_N = 20
TRAIN_EPOCHS = 100

# ---------------------------------------------------------------------------
# Global model state — loaded once at startup, reused for every request
# ---------------------------------------------------------------------------
_model: Recommender | None = None
_mapping: dict | None = None          # slug  -> int_id
_inverse_mapping: dict | None = None  # int_id -> slug
_slug_to_meta: dict | None = None     # slug  -> {"title": str, "release": int}
_load_error: str | None = None


def _load_model(checkpoint_path: str, vocab_size: int) -> Recommender:
    model = Recommender(vocab_size=vocab_size, lr=1e-4, dropout=0.3)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


def _ensure_loaded() -> None:
    """Load mapping + model into module-level globals. No-op if already loaded."""
    global _model, _mapping, _inverse_mapping, _slug_to_meta, _load_error

    if _model is not None:
        return

    mapping_path = Path(MAPPING_JSON)
    if not mapping_path.exists():
        _load_error = (
            f"{MAPPING_JSON} not found. "
            "Scrape some data first: python run_letterboxd.py"
        )
        raise RuntimeError(_load_error)

    _mapping, _inverse_mapping, _slug_to_meta = load_mappings(MAPPING_JSON)
    vocab_size = len(_mapping) + 2  # +2 for PAD=0 and MASK=1

    if not CHECKPOINT_PATH.exists():
        print(f"No checkpoint at {CHECKPOINT_PATH} — training now (this will take a while)…")
        from recommender.training import train
        result = train(
            data_csv_path=DATA_CSV,
            model_dir=str(CHECKPOINT_PATH.parent),
            epochs=TRAIN_EPOCHS,
        )
        ckpt = result["best_model_path"]
    else:
        ckpt = str(CHECKPOINT_PATH)

    _model = _load_model(ckpt, vocab_size)
    print(f"Model ready  checkpoint={ckpt}  vocab_size={vocab_size}")


def _run_inference(slugs: list[str]) -> list[str]:
    """
    Given a chronologically-ordered list of Letterboxd slugs, return up to
    TOP_N film titles that the model predicts the user would enjoy.
    """
    watched_ids = [_mapping[s] for s in slugs if s in _mapping]
    if not watched_ids:
        return []

    context = watched_ids[-(HISTORY_SIZE - 1):]
    pad_count = (HISTORY_SIZE - 1) - len(context)
    input_ids = [PAD] * pad_count + context + [MASK]

    src = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        logits = _model(src)  # (1, HISTORY_SIZE, vocab_size)

    scores = logits[0, -1].numpy()
    exclude = set(input_ids)
    ranked = [i for i in np.argsort(scores)[::-1].tolist() if i not in exclude]

    results = []
    for idx in ranked:
        if len(results) >= TOP_N:
            break
        slug = _inverse_mapping.get(idx)
        if slug is None:
            continue
        meta = _slug_to_meta.get(slug, {})
        title = meta.get("title", slug)
        year = meta.get("release", "")
        results.append(f"{title} ({year})" if year else title)

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
    body = request.get_json(silent=True) or {}
    username = (body.get("username") or "").strip()

    if not username:
        return jsonify({"error": "username is required"}), 400

    # Load model (no-op if already loaded)
    try:
        _ensure_loaded()
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503

    # Scrape the user's Letterboxd diary
    try:
        rows = fetch_user_diary(username)
    except Exception as exc:
        exc_name = type(exc).__name__
        if "ResourceNotFoundError" in exc_name or "not found" in str(exc).lower():
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

    slugs = [r["movieId"] for r in sorted(rows, key=lambda r: r["timestamp"])]
    recs = _run_inference(slugs)

    if not recs:
        return jsonify({
            "error": f"None of '{username}'s watched films are in the model's "
                     "vocabulary. Try retraining with more users' data."
        }), 422

    return jsonify({"username": username, "recommendations": recs})


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading model…")
    try:
        _ensure_loaded()
    except RuntimeError as exc:
        print(f"WARNING: {exc}")
        print("The /recommend endpoint will return 503 until the model is ready.")

    app.run(debug=False, port=5000)
