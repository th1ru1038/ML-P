"""
Flask frontend for the Tonight's Feature questionnaire pipeline.

Routes:
    GET  /        → serves frontend/questionnaire.html
    POST /run     → runs questionnaire.py pipeline, returns JSON
    GET  /health  → {"status": "ok"}

Start with:
    GROQ_API_KEY=gsk_... python3 questionnaire_app.py
"""

import json
import os
import re
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
from flask import Flask, jsonify, send_from_directory
from groq import Groq

import questionnaire as q
from run_movielens import build_mapping_from_csv, load_model, predict

# ── Config ───────────────────────────────────────────────────────────────────
ML_RATINGS_CSV  = "data/ml-25m/ratings_small.csv"
ML_MOVIES_CSV   = "data/ml-25m/movies.csv"
BRIDGE_JSON     = "data/letterboxd_to_movielens.json"
CHECKPOINT_PATH = "recommender_models_ml/recommender.ckpt"
HISTORY_SIZE    = 120
TOP_N           = 5   # candidates passed to Groq for final 3 picks

FRONTEND_DIR = Path(__file__).parent / "frontend"

# ── Model state (loaded once at startup) ─────────────────────────────────────
_model           = None
_mapping         = None
_inverse_mapping = None
_bridge          = None
_movies_df       = None
_model_ready     = False


def _load_model_state() -> bool:
    """Load checkpoint, mapping, bridge, and movies CSV. Returns True on success."""
    global _model, _mapping, _inverse_mapping, _bridge, _movies_df, _model_ready

    ckpt = Path(CHECKPOINT_PATH)
    if not ckpt.exists():
        print(f"Warning: checkpoint not found at {CHECKPOINT_PATH} — will use fallback recommendations.")
        return False

    print("Loading ML model…")
    _mapping, _inverse_mapping = build_mapping_from_csv(ML_RATINGS_CSV)
    vocab_size = len(_mapping) + 2
    _model = load_model(str(ckpt), vocab_size)
    _movies_df = pd.read_csv(ML_MOVIES_CSV)

    bridge_path = Path(BRIDGE_JSON)
    if not bridge_path.exists():
        print(f"Warning: bridge file not found at {BRIDGE_JSON}.")
        return False
    with open(bridge_path) as fh:
        _bridge = json.load(fh)

    _model_ready = True
    print("Model ready.")
    return True


def _ml_candidates_for_user(username: str) -> list[str]:
    """
    Fetch the user's Letterboxd diary live, build their watch sequence,
    and run BERT4Rec inference. Returns up to TOP_N titles the user
    hasn't watched (predict() excludes the input sequence automatically).
    Falls back to the placeholder list if no films matched the bridge.
    """
    from recommender.letterboxd_data import fetch_user_diary

    rows = fetch_user_diary(username)
    rows.sort(key=lambda r: r["timestamp"])

    sequence: list[int] = []
    for row in rows:
        slug = row["movieId"]          # letterboxd_data.py stores slug here
        ml_id = _bridge.get(slug)
        if ml_id is None:
            continue
        mapped_id = _mapping.get(ml_id)
        if mapped_id is None:
            continue
        sequence.append(mapped_id)

    if not sequence:
        print(f"No bridge matches for '{username}', using fallback recommendations.")
        return q.get_recommendations(username)

    titles = predict(
        sequence, _model, _inverse_mapping, _movies_df,
        history_size=HISTORY_SIZE, top_n=TOP_N,
    )
    return titles if titles else q.get_recommendations(username)


# ── Title helpers ─────────────────────────────────────────────────────────────

_YEAR_SUFFIX = re.compile(r"\s*\(\d{4}\)\s*$")

# Matches a trailing ", Article" optionally followed by a year, e.g.
# "Matrix, The (1999)" or "Vie en Rose, La"
_TRAILING_ARTICLE = re.compile(
    r",\s+(The|A|An|Les|La|Le|Los|Das|Die|Der)(\s*\(\d{4}\))?\s*$"
)


def _strip_year(title: str) -> str:
    """'Inception (2010)' → 'Inception'"""
    return _YEAR_SUFFIX.sub("", title).strip()


def _format_title(title: str) -> str:
    """
    Convert MovieLens title format to natural display format.

    Moves a trailing article to the front and strips the year:
      'Matrix, The (1999)'                              → 'The Matrix'
      'Lord of the Rings: The Fellowship of the Ring, The (2001)'
                                                        → 'The Lord of the Rings: The Fellowship of the Ring'
      'Vie en Rose, La (2007)'                          → 'La Vie en Rose'
      'Inception (2010)'                                → 'Inception'
    """
    m = _TRAILING_ARTICLE.search(title)
    if m:
        article = m.group(1)
        base = title[: m.start()].strip()
        return f"{article} {base}"
    return _strip_year(title)


def _normalize(title: str) -> str:
    """Lowercase, strip year, collapse punctuation/whitespace for comparison."""
    t = _YEAR_SUFFIX.sub("", title.lower())
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _match_to_candidate(groq_title: str, candidates: list[str]) -> str:
    """
    Return the candidate whose normalised form best matches groq_title.
    Groq often abbreviates (e.g. 'Star Wars' instead of the full subtitle);
    this recovers the exact original string so TMDB search hits correctly.
    """
    g = _normalize(groq_title)
    g_words = set(g.split())

    best, best_score = None, -1
    for c in candidates:
        n = _normalize(c)
        # Substring containment (either direction) — handles abbreviations well
        if g in n or n in g:
            score = len(g) + len(n)          # prefer longer / more specific matches
            if score > best_score:
                best_score, best = score, c
            continue
        # Word-overlap fallback
        overlap = len(g_words & set(n.split()))
        if overlap > best_score:
            best_score, best = overlap, c

    return best if best else groq_title


# ── Groq call with exact-title prompt ────────────────────────────────────────

_EXACT_TITLE_PROMPT = (
    "A user wants a movie to watch tonight. Their mood and preferences:\n\n"
    "\"{mood_summary}\"\n\n"
    "From the numbered list below, choose the 3 films that best fit their mood. "
    "For each, write 1–2 sentences explaining why it suits them.\n\n"
    "IMPORTANT: Copy each film title EXACTLY as it appears in the list — "
    "do not shorten, rephrase, or omit any words.\n\n"
    "{titles}\n\n"
    "Format: a numbered list (1, 2, 3). Put the title in **bold**, "
    "then an em dash (—), then your explanation."
)


def _pick_best_matches_exact(client: Groq, mood_summary: str, titles: list[str]) -> str:
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(titles))
    prompt = _EXACT_TITLE_PROMPT.format(mood_summary=mood_summary, titles=numbered)
    response = client.chat.completions.create(
        model=q.MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


# ── Parsing ───────────────────────────────────────────────────────────────────

def _parse_best_matches(text: str) -> list[dict]:
    """
    Parse Groq's numbered list into individual recommendation dicts.

    Splits on the start of each numbered entry so it works regardless of
    whether Groq uses single newlines, double newlines, or no separator.
    """
    print(f"\n--- Groq raw output ---\n{text}\n-----------------------\n")

    # Split just before each "N. **" — handles \n, \n\n, or back-to-back entries
    items = [s.strip() for s in re.split(r"(?=\d+\.\s+\*\*)", text.strip()) if s.strip()]

    results = []
    for item in items:
        # Title: text between the first pair of ** markers
        title_m = re.search(r"\*\*(.+?)\*\*", item)
        if not title_m:
            continue
        title = title_m.group(1).strip()

        # Reason: everything after the first em dash (—) or plain hyphen ( - )
        if " — " in item:
            reason = item.split(" — ", 1)[1]
        elif " - " in item:
            reason = item.split(" - ", 1)[1]
        else:
            continue

        # Collapse all whitespace (newlines, extra spaces) into single spaces
        reason = " ".join(reason.split())

        results.append({"title": title, "reason": reason})

    return results[:3]


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "questionnaire.html")


@app.route("/run", methods=["POST"])
def run():
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return jsonify({"error": "GROQ_API_KEY environment variable is not set"}), 500

    try:
        # Blocks here while the user answers questions in the terminal
        mood_summary, username = q.run_questionnaire()

        if _model_ready:
            candidate_titles = _ml_candidates_for_user(username)
        else:
            candidate_titles = q.get_recommendations(username)

        client = Groq(api_key=api_key)
        # Use the exact-title prompt instead of questionnaire.py's default
        best_matches_text = _pick_best_matches_exact(client, mood_summary, candidate_titles)

        parsed = _parse_best_matches(best_matches_text)

        recommendations = []
        for rec in parsed:
            # Recover the exact original title even if Groq abbreviated it
            exact = _match_to_candidate(rec["title"], candidate_titles)
            display_title = _format_title(exact)
            recommendations.append({
                "title": display_title,
                "reason": rec["reason"],
                "tmdb_url": (
                    "https://www.themoviedb.org/search?query="
                    + quote_plus(display_title)
                ),
            })

        return jsonify({
            "mood_summary": mood_summary,
            "username": username,
            "recommendations": recommendations,
        })

    except Exception as e:
        return jsonify({"error": "something went wrong", "detail": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    _load_model_state()
    app.run(debug=False, port=5001)
