# Team Tasks

---

## AAYAN — Inference Fix

**Status:** Partially complete — code done, data/checkpoint not shared

### What Aayan did

- Added `build_mapping_from_csv()` — reconstructs the vocab mapping from the ratings CSV without reloading the full dataset, so the checkpoint can be reused across runs without retraining
- Added `CHECKPOINT_PATH` config constant and updated `main()` to skip training when a checkpoint already exists
- Improved `build_user_sequence()` logging: now prints total films, matched count, and skipped count with reasons
- Cleaned up `predict()` with a step-by-step docstring and type annotations
- Bumped `TOP_N` from 20 → 30, `EPOCHS` from 10 → 25
- Added `num_sanity_val_steps=0` to the PyTorch Lightning trainer (suppresses the pre-flight validation pass)
- Added `flask>=2.0.0` to `requirements.txt`
- Successfully trained on his local machine using `ratings_small.csv` (10,000 users, MPS accelerator)

### What is still missing

The program **cannot run** on this machine because two large files were not shared:

1. **`data/ml-25m/ratings_small.csv`** — the training data Aayan used. Not in the repo (too large). Aayan needs to share this file, or you can download the full dataset:
   - Download `ml-25m.zip` from https://grouplens.org/datasets/movielens/
   - Unzip and place `ratings.csv` at `data/ml-25m/ratings.csv`
   - Update line 20 of `run_movielens.py`: change `ratings_small.csv` → `ratings.csv`

2. **`recommender_models_ml/recommender.ckpt`** — the trained model checkpoint. Not in the repo. Aayan needs to share this so training can be skipped.

### To verify once the checkpoint is available

1. Run `python3 run_movielens.py` — it should detect the checkpoint and skip training
2. Confirm `build_user_sequence()` prints a non-zero matched count
3. Confirm `Top 20 recommendations` prints human-readable titles

---

## BRINDA — Frontend Required

**Status:** Ready to start

### Context

The questionnaire layer is complete. `questionnaire.py` handles the full user-facing flow:

1. Runs a multi-turn Groq chat to understand the user's mood and preferences
2. Generates a `mood_summary` string
3. Collects the user's Letterboxd `username`
4. Calls `get_recommendations(username)` to fetch 5 candidate movie titles
5. Passes those titles + `mood_summary` to Groq, which returns the **top 3 picks with reasoning**

### What needs to be built

Build a simple **Flask** web frontend that:

- Accepts the output from `questionnaire.py` (see data format below)
- Displays the 3 recommendations as **cards**, one per movie
- Each card should show:
  - **Movie title**
  - Short reason why it was picked (from Groq's explanation)
  - A link to the movie's **TMDB page** (`https://www.themoviedb.org/search?query=<title>`)
- Keep the UI clean and simple — no frameworks required, plain HTML/CSS is fine

The Flask app does not need to embed the questionnaire chat itself for now — it can accept the final output as input (e.g. passed in as JSON or rendered server-side after `questionnaire.py` runs).

---

## Current Output Format from `questionnaire.py`

Both teammates should know exactly what data the pipeline produces.

### Terminal output (end of run)

```
==========================================================
  COLLECTED DATA
==========================================================
mood_summary : <1–2 sentence string describing the user's mood, preferred
               genre, tone, energy level, and emotional intensity>
username     : <Letterboxd username string>
==========================================================

==========================================================
  TOP 3 PICKS FOR TONIGHT
==========================================================
1. **<Movie Title>** — <1–2 sentence explanation of why this fits the mood>

2. **<Movie Title>** — <1–2 sentence explanation of why this fits the mood>

3. **<Movie Title>** — <1–2 sentence explanation of why this fits the mood>
==========================================================
```

### Concrete example

```
==========================================================
  COLLECTED DATA
==========================================================
mood_summary : The user is feeling low-energy and reflective tonight,
               looking for something emotionally resonant but not too
               heavy — a drama or quiet indie with a warm, hopeful tone.
username     : aayan_watches
==========================================================

==========================================================
  TOP 3 PICKS FOR TONIGHT
==========================================================
1. **Eternal Sunshine of the Spotless Mind** — Matches the reflective,
   emotionally rich mood perfectly; bittersweet but ultimately hopeful.

2. **La La Land** — A warm, visually beautiful film with just enough
   emotional weight to feel meaningful without being draining.

3. **The Grand Budapest Hotel** — Lighter in tone but still carefully
   crafted; good for a low-energy evening that still wants something
   with substance.
==========================================================
```

### Key variables (Python)

| Variable | Type | Description |
|---|---|---|
| `mood_summary` | `str` | 1–2 sentences summarising mood + preferences |
| `username` | `str` | Letterboxd username |
| `candidate_titles` | `list[str]` | 5 titles from `get_recommendations()` |
| `best_matches` | `str` | Groq's formatted top-3 response (plain text) |
