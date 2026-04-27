"""
questionnaire.py

Conducts a short Groq-powered chat to learn the user's mood and movie
preferences, then collects their Letterboxd username.

Outputs:
    mood_summary  (str)  — 1-2 sentence description of mood + preferences
    username      (str)  — Letterboxd username

Usage:
    python questionnaire.py
"""

import os

from groq import Groq

# ── Set GROQ_API_KEY in your environment (e.g. export GROQ_API_KEY=gsk_...) ──
API_KEY = os.getenv("GROQ_API_KEY", "")

MODEL = "llama-3.3-70b-versatile"

SYSTEM_INSTRUCTION = """
You are a warm, casual movie concierge helping someone pick a film for tonight.
Your job: ask the user 3-4 short questions — one at a time — to understand
their current mood and what kind of movie they're in the mood for.

Natural topics to weave in (don't be mechanical about it):
- How they're feeling / their energy level right now
- Whether they want something light or more intense / emotional
- Genre or vibe they're gravitating toward (comedy, thriller, drama, etc.)
- Any preference on pacing, era, or tone

Keep it friendly and conversational. Don't number the questions.
Don't ask all topics if the conversation naturally covers them early.

Once you've gathered enough to make a good recommendation and have asked
all your questions, wrap up naturally — then add the exact token [DONE]
at the very end of your message (on its own, after a newline). The user
won't notice it.
"""

SUMMARY_PROMPT = (
    "Based on our conversation, write a 1–2 sentence summary of this person's "
    "current mood and the type of movie they want tonight. "
    "Be specific: mention genre, tone, emotional intensity, and energy level "
    "if any of those came up."
)

RECOMMENDATION_PROMPT_TEMPLATE = (
    "A user is looking for a movie to watch tonight. Here is a summary of their "
    "current mood and preferences:\n\n\"{mood_summary}\"\n\n"
    "From the following list of 5 candidate movies, pick the 3 that best match "
    "their mood and explain briefly (1-2 sentences each) why each is a good fit:\n\n"
    "{titles}\n\n"
    "Format your response as a numbered list (1, 2, 3) with the movie title in bold "
    "followed by your explanation."
)


def run_questionnaire() -> tuple[str, str]:
    client = Groq(api_key=API_KEY)

    # Groq is stateless — conversation history is a plain list we manage ourselves
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]

    def send(user_text: str) -> str:
        messages.append({"role": "user", "content": user_text})
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )
        reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
        return reply

    # Seed the conversation so the model speaks first
    response_text = send("Hey, help me pick a movie for tonight.")
    _print_assistant(response_text)

    # Conversation loop — runs until the model signals [DONE]
    while "[DONE]" not in response_text:
        user_input = _prompt_user()
        response_text = send(user_input)
        _print_assistant(response_text)

    # Ask the model to distill the conversation into a mood summary
    mood_summary = send(SUMMARY_PROMPT).strip()

    # Collect Letterboxd username — no model call needed
    print("Assistant: Last thing — what's your Letterboxd username so I can "
          "see what you've already watched?\n")
    username = _prompt_user()

    return mood_summary, username


# ── Recommendation helpers ────────────────────────────────────────────────────

def get_recommendations(username: str) -> list[str]:
    """Placeholder — returns a hardcoded list of 5 candidate movie titles.

    In a real implementation this would fetch the user's Letterboxd watchlist
    or use a recommendation engine keyed on `username`.
    """
    return [
        "Eternal Sunshine of the Spotless Mind",
        "The Grand Budapest Hotel",
        "Parasite",
        "La La Land",
        "Hereditary",
    ]


def pick_best_matches(client: Groq, mood_summary: str, titles: list[str]) -> str:
    """Ask Groq to choose the 3 best-fitting titles for the user's mood."""
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(titles))
    prompt = RECOMMENDATION_PROMPT_TEMPLATE.format(
        mood_summary=mood_summary,
        titles=numbered,
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_assistant(text: str) -> None:
    clean = text.replace("[DONE]", "").strip()
    print(f"\nAssistant: {clean}\n")


def _prompt_user() -> str:
    while True:
        val = input("You: ").strip()
        if val:
            return val


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    if not API_KEY:
        raise ValueError("Set your Groq API key in the API_KEY constant at the top of this file.")

    print("=" * 58)
    print("  Movie Night Questionnaire (powered by Groq)")
    print("=" * 58)

    mood_summary, username = run_questionnaire()

    print("\n" + "=" * 58)
    print("  COLLECTED DATA")
    print("=" * 58)
    print(f"mood_summary : {mood_summary}")
    print(f"username     : {username}")
    print("=" * 58)

    candidate_titles = get_recommendations(username)
    client = Groq(api_key=API_KEY)
    best_matches = pick_best_matches(client, mood_summary, candidate_titles)

    print("\n" + "=" * 58)
    print("  TOP 3 PICKS FOR TONIGHT")
    print("=" * 58)
    print(best_matches)
    print("=" * 58)


if __name__ == "__main__":
    main()
