"""
Train BERT4Rec on MovieLens-25M and generate personalised recommendations
for a Letterboxd user using the slug→MovieLens bridge built by data/build_bridge.py.

Run from the repo root:
    python run_movielens.py
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
ML_RATINGS_CSV = "data/ml-25m/ratings_small.csv"
ML_MOVIES_CSV = "data/ml-25m/movies.csv"
BRIDGE_JSON = "data/letterboxd_to_movielens.json"
LB_RATINGS_CSV = "letterboxd_ratings.csv"

LB_USERNAME = "aayanr"       # Letterboxd username whose history drives inference

HISTORY_SIZE = 120
TOP_N = 20
EPOCHS = 10
BATCH_SIZE = 32
MODEL_DIR = "recommender_models_ml"
LOG_DIR = "recommender_logs_ml"
# ---------------------------------------------------------------------------


def run_training():
    """Train on MovieLens-25M. Returns the full output_json from train()."""
    from recommender.training import train

    print(f"MovieLens-25M has ~25M ratings across ~160K users.")
    print(f"Estimated time on Apple Silicon MPS: 30–60 min for {EPOCHS} epochs.")
    print(f"(num_workers=0 means single-threaded data loading — the main bottleneck.)\n")

    t0 = time.time()
    result = train(
        data_csv_path=ML_RATINGS_CSV,
        log_dir=LOG_DIR,
        model_dir=MODEL_DIR,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        history_size=HISTORY_SIZE,
    )
    elapsed = time.time() - t0
    print(f"\nTraining finished in {elapsed/60:.1f} min")
    return result


def load_model(checkpoint_path: str, vocab_size: int):
    from recommender.models import Recommender

    model = Recommender(vocab_size=vocab_size, lr=1e-4, dropout=0.3)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


def build_user_sequence(
    lb_username: str,
    bridge: dict,
    mapping: dict,
) -> list[int]:
    """
    Read the Letterboxd ratings CSV, translate each slug to a MovieLens
    mapped integer ID, and return the list in chronological order.

    Skips slugs that are absent from the bridge or from the ML mapping.
    """
    lb_df = pd.read_csv(LB_RATINGS_CSV)
    user_df = lb_df[lb_df["userId"] == lb_username].sort_values("timestamp")

    sequence = []
    skipped = []
    for slug in user_df["movieId"].tolist():
        ml_id = bridge.get(slug)
        if ml_id is None:
            skipped.append((slug, "not in bridge"))
            continue
        mapped_id = mapping.get(ml_id)
        if mapped_id is None:
            skipped.append((slug, f"ml_id={ml_id} not in mapping"))
            continue
        sequence.append(mapped_id)

    if skipped:
        print(f"\nSkipped {len(skipped)} films during sequence build:")
        for slug, reason in skipped:
            print(f"  {slug!r}  ({reason})")

    print(f"\nSequence length after bridge + mapping: {len(sequence)} films")
    return sequence


def predict(
    sequence: list[int],
    model,
    inverse_mapping: dict,
    movies_df: pd.DataFrame,
    history_size: int = HISTORY_SIZE,
    top_n: int = TOP_N,
) -> list[str]:
    from recommender.data_processing import PAD, MASK

    # Take last (history_size - 1) items, append MASK at position -1
    context = sequence[-(history_size - 1):]
    pad_count = (history_size - 1) - len(context)
    input_ids = [PAD] * pad_count + context + [MASK]

    src = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)  # (1, 120)

    with torch.no_grad():
        logits = model(src)  # (1, 120, vocab_size)

    scores = logits[0, -1].numpy()

    exclude = set(input_ids)
    ranked = np.argsort(scores)[::-1].tolist()
    ranked = [idx for idx in ranked if idx not in exclude]

    # Build a title lookup from movies.csv: ml_movieId → "Title (Year)"
    # movies.csv title field already includes the year, e.g. "Inception (2010)"
    title_lookup = dict(zip(movies_df["movieId"], movies_df["title"]))

    results = []
    for idx in ranked:
        if len(results) >= top_n:
            break
        ml_id = inverse_mapping.get(idx)
        if ml_id is None:
            continue
        title = title_lookup.get(ml_id)
        if title is None:
            continue
        results.append(title)

    return results


def main():
    # ------------------------------------------------------------------ #
    # 1. Train                                                            #
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("Step 1: Training BERT4Rec on MovieLens-25M")
    print("=" * 60)
    result = run_training()

    checkpoint_path = result["best_model_path"]
    mapping = result["mapping"]           # ml_movieId (int) → mapped int
    inverse_mapping = result["inverse_mapping"]  # mapped int → ml_movieId (int)
    vocab_size = len(mapping) + 2         # +2 for PAD=0 and MASK=1

    print(f"\nBest checkpoint : {checkpoint_path}")
    print(f"Vocab size      : {vocab_size}")

    # ------------------------------------------------------------------ #
    # 2. Load bridge and build my watch sequence                          #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("Step 2: Building personal watch sequence from Letterboxd history")
    print("=" * 60)

    bridge_path = Path(BRIDGE_JSON)
    if not bridge_path.exists():
        raise FileNotFoundError(
            f"{BRIDGE_JSON} not found.\n"
            "Run this first:  TMDB_API_KEY=your_key python data/build_bridge.py"
        )

    with open(bridge_path) as fh:
        # bridge keys are slugs (str), values are ml_movieIds (int)
        bridge = json.load(fh)

    # mapping keys from train() are the original MovieLens integer movieIds.
    # JSON round-trip makes them strings if they were stored — but here
    # train() returns them directly as Python dicts, so keys are already ints.
    sequence = build_user_sequence(LB_USERNAME, bridge, mapping)

    if not sequence:
        print("No films in sequence after bridging. Check that build_bridge.py ran successfully.")
        return

    # ------------------------------------------------------------------ #
    # 3. Inference                                                         #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print(f"Step 3: Generating top-{TOP_N} recommendations for '{LB_USERNAME}'")
    print("=" * 60)

    model = load_model(checkpoint_path, vocab_size=vocab_size)
    movies_df = pd.read_csv(ML_MOVIES_CSV)

    recs = predict(sequence, model, inverse_mapping, movies_df)

    if recs:
        print(f"\nTop {len(recs)} recommendations:\n")
        for i, title in enumerate(recs, 1):
            print(f"  {i:>2}. {title}")
    else:
        print("No recommendations generated.")


if __name__ == "__main__":
    main()
