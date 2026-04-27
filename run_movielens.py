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
ML_RATINGS_CSV = "data/ml-25m/ratings.csv"   # full dataset; change to ratings_small.csv for a quick test
ML_MOVIES_CSV  = "data/ml-25m/movies.csv"
BRIDGE_JSON    = "data/letterboxd_to_movielens.json"
LB_RATINGS_CSV = "letterboxd_ratings.csv"

LB_USERNAME  = "aayanr"   # Letterboxd username whose history drives inference
HISTORY_SIZE = 120
TOP_N        = 30         # candidates returned by predict(); main() prints first 20
EPOCHS       = 10
BATCH_SIZE   = 32
MODEL_DIR    = "recommender_models_ml"
LOG_DIR      = "recommender_logs_ml"
CHECKPOINT_PATH = f"{MODEL_DIR}/recommender.ckpt"
# ---------------------------------------------------------------------------


def run_training():
    """Train on MovieLens. Returns the full output_json from train()."""
    from recommender.training import train

    print(f"Training on {ML_RATINGS_CSV}")
    print(f"Estimated time on Apple Silicon MPS: 30–60 min for {EPOCHS} epochs.")
    print(f"(num_workers=0 → single-threaded data loading is the main bottleneck)\n")

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
    print(f"\nTraining finished in {elapsed / 60:.1f} min")
    return result


def build_mapping_from_csv(ratings_csv: str) -> tuple[dict, dict]:
    """
    Reconstruct the exact integer mapping that map_column() built during training.

    map_column (data_processing.py) does:
        values = sorted(list(df["movieId"].unique()))
        mapping = {k: i + 2 for i, k in enumerate(values)}

    We reproduce that here using only the movieId column so we avoid loading
    the full 25M-row file into memory when we only need the vocab.
    """
    print(f"Reconstructing vocab mapping from {ratings_csv} …")
    df = pd.read_csv(ratings_csv, usecols=["movieId"])
    unique_ids = sorted(df["movieId"].unique().tolist())
    mapping = {k: i + 2 for i, k in enumerate(unique_ids)}
    inverse_mapping = {v: k for k, v in mapping.items()}
    print(f"  {len(mapping)} unique ML movie IDs → integer range [2, {len(mapping) + 1}]")
    return mapping, inverse_mapping


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
    Translate a user's Letterboxd watch history into a chronologically ordered
    list of integer IDs that the model understands.

    Pipeline:
        Letterboxd slug
          → MovieLens movieId   (via data/letterboxd_to_movielens.json)
          → integer vocab ID    (via map_column mapping from ratings.csv)
    """
    lb_df   = pd.read_csv(LB_RATINGS_CSV)
    user_df = lb_df[lb_df["userId"] == lb_username].sort_values("timestamp")
    total   = len(user_df)

    sequence: list[int] = []
    skipped:  list[tuple[str, str]] = []

    for slug in user_df["movieId"].tolist():
        ml_id = bridge.get(slug)
        if ml_id is None:
            skipped.append((slug, "not in bridge"))
            continue
        mapped_id = mapping.get(ml_id)
        if mapped_id is None:
            skipped.append((slug, f"ml_id={ml_id} not in ML vocab"))
            continue
        sequence.append(mapped_id)

    matched = len(sequence)
    print(f"\nWatch history: {total} films total, {matched} matched, {total - matched} skipped")

    if skipped:
        print("Skipped films:")
        for slug, reason in skipped:
            print(f"  {slug!r}  ({reason})")

    return sequence


def predict(
    sequence: list[int],
    model,
    inverse_mapping: dict,
    movies_df: pd.DataFrame,
    history_size: int = HISTORY_SIZE,
    top_n: int = TOP_N,
) -> list[str]:
    """
    Run inference for a user sequence and return up to top_n film titles.

    Steps:
      1. Take last (history_size - 1) IDs chronologically
      2. Append MASK token (value=1)
      3. Left-pad to history_size with PAD (value=0)
      4. Pass through model; read logits at the MASK position (last token)
      5. Exclude PAD, MASK, and already-watched IDs
      6. Walk ranked predictions; convert integer ID → ML movieId → title
    """
    from recommender.data_processing import MASK, PAD

    # Build model input
    context   = sequence[-(history_size - 1):]
    pad_count = (history_size - 1) - len(context)
    input_ids = [PAD] * pad_count + context + [MASK]

    src = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)  # (1, 120)

    with torch.no_grad():
        logits = model(src)  # (1, 120, vocab_size)

    # Logits at the MASK position (last token in the sequence)
    scores = logits[0, -1].numpy()

    # Exclude PAD=0, MASK=1, and every watched integer ID
    exclude = set(input_ids)
    ranked  = [idx for idx in np.argsort(scores)[::-1].tolist() if idx not in exclude]

    # Inverse mapping: integer vocab ID → original ML movieId
    # Title lookup: ML movieId → "Title (Year)" from movies.csv
    title_lookup = dict(zip(movies_df["movieId"], movies_df["title"]))

    results: list[str] = []
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
    checkpoint = Path(CHECKPOINT_PATH)

    # ------------------------------------------------------------------ #
    # Step 1: Get mapping — from training if needed, from CSV if ckpt    #
    #         already exists.                                             #
    # ------------------------------------------------------------------ #
    if checkpoint.exists():
        print(f"Checkpoint found at {checkpoint} — skipping training.")
        checkpoint_path = str(checkpoint)
        mapping, inverse_mapping = build_mapping_from_csv(ML_RATINGS_CSV)
    else:
        print("=" * 60)
        print("Step 1: Training BERT4Rec on MovieLens")
        print("=" * 60)
        result = run_training()
        checkpoint_path = result["best_model_path"]
        # mapping keys are the original ML movieIds (ints); values are mapped ints
        mapping          = result["mapping"]
        inverse_mapping  = result["inverse_mapping"]

    vocab_size = len(mapping) + 2   # +2 for PAD=0 and MASK=1
    print(f"\nCheckpoint  : {checkpoint_path}")
    print(f"Vocab size  : {vocab_size}")

    # ------------------------------------------------------------------ #
    # Step 2: Build personal watch sequence                               #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("Step 2: Building personal watch sequence from Letterboxd history")
    print("=" * 60)

    bridge_path = Path(BRIDGE_JSON)
    if not bridge_path.exists():
        raise FileNotFoundError(
            f"{BRIDGE_JSON} not found.\n"
            "Run first:  TMDB_API_KEY=your_key python data/build_bridge.py"
        )

    with open(bridge_path) as fh:
        # JSON decodes integer values as Python ints — matches mapping key type
        bridge = json.load(fh)

    sequence = build_user_sequence(LB_USERNAME, bridge, mapping)

    if not sequence:
        print("\nNo films matched. Check that build_bridge.py ran successfully.")
        return

    # ------------------------------------------------------------------ #
    # Step 3: Inference                                                    #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print(f"Step 3: Generating recommendations for '{LB_USERNAME}'")
    print("=" * 60)

    model      = load_model(checkpoint_path, vocab_size=vocab_size)
    movies_df  = pd.read_csv(ML_MOVIES_CSV)

    recs = predict(sequence, model, inverse_mapping, movies_df, top_n=TOP_N)

    print(f"\nTop 20 recommendations for '{LB_USERNAME}':\n")
    for i, title in enumerate(recs[:20], 1):
        print(f"  {i:>2}. {title}")

    if not recs:
        print("  (none — try more training epochs or add more films to your diary)")


if __name__ == "__main__":
    main()
