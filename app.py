import re
from dataclasses import dataclass
from urllib.parse import quote_plus

from flask import Flask, jsonify, render_template, request


app = Flask(__name__)


@dataclass
class Recommendation:
    title: str
    reason: str

    @property
    def tmdb_url(self) -> str:
        return f"https://www.themoviedb.org/search?query={quote_plus(self.title)}"


LINE_PATTERN = re.compile(
    r"^\s*\d+[\).\s-]*\*{0,2}(?P<title>.+?)\*{0,2}\s*(?:—|-|:)\s*(?P<reason>.+?)\s*$"
)


def parse_best_matches(best_matches: str) -> list[Recommendation]:
    recs: list[Recommendation] = []
    for raw_line in best_matches.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = LINE_PATTERN.match(line)
        if not match:
            continue
        recs.append(
            Recommendation(
                title=match.group("title").strip(),
                reason=match.group("reason").strip(),
            )
        )
    return recs[:3]


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/recommendations", methods=["POST"])
def recommendations():
    payload = request.get_json(silent=True)
    if payload is None:
        payload = request.form.to_dict()
    payload = payload or {}

    mood_summary = str(payload.get("mood_summary", "")).strip()
    username = str(payload.get("username", "")).strip()
    best_matches = str(payload.get("best_matches", "")).strip()

    recommendations = parse_best_matches(best_matches)
    if not recommendations:
        return (
            jsonify(
                {
                    "error": "No recommendations found. Provide `best_matches` as a numbered top-3 list."
                }
            ),
            400,
        )

    return render_template(
        "results.html",
        mood_summary=mood_summary,
        username=username,
        recommendations=recommendations,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)
