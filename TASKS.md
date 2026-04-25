# Team Tasks

---

## AAYAN — Inference Fix Required

**Status:** In Progress

### Goal

Get `run_movielens.py` working end-to-end: training on the MovieLens-25M dataset and producing real movie recommendations for a given Letterboxd username.

**Focus on `run_movielens.py` only — ignore `run_letterboxd.py`.**

### Background

`run_letterboxd.py` was the original approach but hit a dead end: the Letterboxd export contained only one user with 84 diary entries, giving BERT4Rec a `train_data` length of 1 — not enough to learn anything. `test_accuracy` was 0.0 and `predict_for_user()` returned nothing.

`run_movielens.py` is the correct path forward. It trains on the **MovieLens-25M dataset** (~25 million ratings across tens of thousands of users), which gives BERT4Rec the scale it needs to learn meaningful patterns. Training has been confirmed to work. The remaining gap is inference — producing actual recommendations for a user.

### Data location

```
/Users/thiruveleyudham/Documents/01_Purdue/Projects/Clubs/ML@P/Final Project/data/ml-25m
```

### What has already been done

- Full pipeline ran end-to-end without crashing
- TensorBoard installed and confirmed working
- Model trained for 100 epochs on MovieLens-25M
- Checkpoint saved successfully

### Immediate fix (unblocks everything)

`ratings.csv` is missing from `data/ml-25m/`. This is the primary blocker — the script crashes before doing anything.

**Step 1** — Download the file:
```
https://grouplens.org/datasets/movielens/
```
Download `ml-25m.zip` (~700 MB), unzip it, and place `ratings.csv` at:
```
data/ml-25m/ratings.csv
```

**Step 2** — Update line 20 in `run_movielens.py`:
```python
# before
ML_RATINGS_CSV = "data/ml-25m/ratings_small.csv"

# after
ML_RATINGS_CSV = "data/ml-25m/ratings.csv"
```

**Step 3** — Run the script:
```
python run_movielens.py
```

Training will take 30–60 min on Apple Silicon (MPS). A checkpoint will be saved to `recommender_models_ml/`.

### What needs to be done after training

Once training completes and the checkpoint exists:

1. Confirm `build_user_sequence()` returns a non-empty sequence — it reads `letterboxd_ratings.csv`, translates slugs via `data/letterboxd_to_movielens.json` (66 entries already mapped), and filters through the training vocabulary. If it returns 0 films, check the bridge file.
2. Verify `predict()` produces results — check that `inverse_mapping` round-trips correctly from internal vocab ID → MovieLens ID → title via `movies.csv`.
3. Confirm output titles are human-readable and surfaced in the terminal under `Top N recommendations`.

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
