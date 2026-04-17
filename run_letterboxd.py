"""
End-to-end Letterboxd → BERT4Rec pipeline.

Edit USERNAMES below, then run:
    python run_letterboxd.py
"""

import sys

import numpy as np
import torch

# ---------------------------------------------------------------------------
# CONFIG — edit these before running
# ---------------------------------------------------------------------------
USERNAMES = [
    "aayanr",       # first user will be used for inference demo
    # add more Letterboxd usernames here
]

HISTORY_SIZE = 120       # sequence length the model sees (matches training default)
TOP_N = 20               # how many recommendations to print
EPOCHS = 100             # training epochs (increase for better results)
BATCH_SIZE = 32
DATA_CSV = "letterboxd_ratings.csv"
MAPPING_JSON = "letterboxd_movie_mapping.json"
MODEL_DIR = "recommender_models"
LOG_DIR = "recommender_logs"
# ---------------------------------------------------------------------------


def collect_data():
    """Scrape Letterboxd diaries and write the CSV + mapping JSON."""
    from recommender.letterboxd_data import build_letterboxd_dataset

    df, mapping, inverse_mapping, slug_to_meta = build_letterboxd_dataset(
        usernames=USERNAMES,
        output_csv=DATA_CSV,
        mapping_json=MAPPING_JSON,
    )
    return df, mapping, inverse_mapping, slug_to_meta


def run_training(vocab_size: int):
    """Train the model and return the path to the best checkpoint."""
    from recommender.training import train

    result = train(
        data_csv_path=DATA_CSV,
        log_dir=LOG_DIR,
        model_dir=MODEL_DIR,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        history_size=HISTORY_SIZE,
    )
    return result["best_model_path"]


def load_model(checkpoint_path: str, vocab_size: int):
    """Load the trained Recommender from a Lightning checkpoint."""
    from recommender.models import Recommender

    model = Recommender(vocab_size=vocab_size, lr=1e-4, dropout=0.3)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


def predict_for_user(
    username: str,
    df,
    model,
    mapping: dict,
    inverse_mapping: dict,
    slug_to_meta: dict,
    history_size: int = HISTORY_SIZE,
    top_n: int = TOP_N,
):
    """
    Given a user's watch history, predict their next films.

    Strategy: use the last (history_size - 1) watched slugs as context,
    append a MASK token, run the model, return top_n predictions.
    """
    from recommender.data_processing import PAD, MASK

    user_df = df[df["userId"] == username].sort_values("timestamp")
    if user_df.empty:
        print(f"No data found for user '{username}'.")
        return []

    watched_slugs = user_df["movieId"].tolist()
    # map slugs to integer IDs; skip any that aren't in the vocabulary
    watched_ids = [mapping[s] for s in watched_slugs if s in mapping]

    if not watched_ids:
        print(f"None of '{username}'s films are in the vocabulary.")
        return []

    # Take the most recent (history_size - 1) films, leaving room for MASK
    context_ids = watched_ids[-(history_size - 1):]

    # Left-pad with PAD to reach history_size - 1, then append MASK
    pad_count = (history_size - 1) - len(context_ids)
    input_ids = [PAD] * pad_count + context_ids + [MASK]

    src = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)  # (1, history_size)

    with torch.no_grad():
        logits = model(src)  # (1, history_size, vocab_size)

    # Prediction at the MASK position (last token)
    scores = logits[0, -1].numpy()

    # Sort descending, exclude PAD/MASK and already-watched IDs
    exclude = set(input_ids)
    ranked = np.argsort(scores)[::-1].tolist()
    ranked = [idx for idx in ranked if idx not in exclude]

    results = []
    for idx in ranked:
        if len(results) >= top_n:
            break
        slug = inverse_mapping.get(idx)
        if slug is None:
            continue
        meta = slug_to_meta.get(slug, {})
        title = meta.get("title", slug)
        year = meta.get("release", "")
        label = f"{title} ({year})" if year else title
        results.append(label)

    return results


def main():
    if not USERNAMES:
        sys.exit("Add at least one Letterboxd username to USERNAMES at the top of this file.")

    # 1. Collect data
    print("=" * 60)
    print("Step 1: Collecting Letterboxd diary data")
    print("=" * 60)
    df, mapping, inverse_mapping, slug_to_meta = collect_data()

    vocab_size = len(mapping) + 2  # +2 for PAD=0 and MASK=1

    # 2. Train
    print("\n" + "=" * 60)
    print(f"Step 2: Training BERT4Rec  (vocab={vocab_size}, epochs={EPOCHS})")
    print("=" * 60)
    checkpoint_path = run_training(vocab_size=vocab_size)
    print(f"\nBest checkpoint: {checkpoint_path}")

    # 3. Inference on the first username
    test_user = USERNAMES[0]
    print("\n" + "=" * 60)
    print(f"Step 3: Recommendations for '{test_user}'")
    print("=" * 60)

    model = load_model(checkpoint_path, vocab_size=vocab_size)
    recs = predict_for_user(
        username=test_user,
        df=df,
        model=model,
        mapping=mapping,
        inverse_mapping=inverse_mapping,
        slug_to_meta=slug_to_meta,
    )

    if recs:
        print(f"\nTop {len(recs)} recommendations for '{test_user}':\n")
        for i, title in enumerate(recs, 1):
            print(f"  {i:>2}. {title}")
    else:
        print("No recommendations generated — the user may have too few diary entries.")


if __name__ == "__main__":
    main()
