import os
import urllib.request

_CACHE = os.path.join(os.path.dirname(__file__), ".cache")
_GUESSES_URL = (
    "https://raw.githubusercontent.com/Kinkelin/WordleCompetition"
    "/main/data/official/official_allowed_guesses.txt"
)
_ANSWERS_URL = (
    "https://raw.githubusercontent.com/Kinkelin/WordleCompetition"
    "/main/data/official/shuffled_real_wordles.txt"
)


def _fetch(url: str, fname: str) -> list[str]:
    path = os.path.join(_CACHE, fname)
    if not os.path.exists(path):
        os.makedirs(_CACHE, exist_ok=True)
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, path)
    with open(path) as f:
        return [
            w.strip().lower()
            for w in f
            if w.strip() and not w.startswith("#")
        ]


def _load():
    guesses = _fetch(_GUESSES_URL, "guesses.txt")
    try:
        answers = _fetch(_ANSWERS_URL, "answers.txt")
    except Exception:
        answers = guesses
    # de-dup while preserving order
    seen: set[str] = set()
    all_words: list[str] = []
    for w in answers + guesses:
        if w not in seen:
            seen.add(w)
            all_words.append(w)
    return answers, guesses, all_words


ANSWERS, GUESSES, ALL_WORDS = _load()
