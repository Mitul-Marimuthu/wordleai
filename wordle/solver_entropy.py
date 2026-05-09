"""
Information-entropy Wordle solver.

On the first call to best_guess() the solver evaluates every candidate against
every remaining possible answer and picks the word that maximises expected
information (Shannon entropy over pattern distributions).

Pattern matrix caching
----------------------
Building patterns for all 12 k × 2 315 pairs is slow (~30 s).  Pass
cache=True (default) to save the matrix to .cache/patterns.npy so subsequent
runs are instant.
"""

import os

import numpy as np

from game import MAX_GUESSES, WIN_PATTERN, evaluate_guess
from words import ALL_WORDS, ANSWERS, GUESSES

_CACHE_PATH = os.path.join(os.path.dirname(__file__), ".cache", "patterns.npy")
_WORD_PATH = os.path.join(os.path.dirname(__file__), ".cache", "pattern_words.txt")


# ---------------------------------------------------------------------------
# Pattern matrix
# ---------------------------------------------------------------------------

def _compute_pattern_matrix(row_words: list[str], col_words: list[str]) -> np.ndarray:
    """Return uint8 matrix shape (len(row_words), len(col_words))."""
    R, C = len(row_words), len(col_words)
    mat = np.empty((R, C), dtype=np.uint8)
    for i, guess in enumerate(row_words):
        for j, target in enumerate(col_words):
            p = evaluate_guess(guess, target)
            val = 0
            for ch in p:
                val = val * 3 + (0 if ch == "-" else 1 if ch == "x" else 2)
            mat[i, j] = val
    return mat


def _load_or_build_matrix(row_words: list[str], col_words: list[str]) -> np.ndarray:
    if os.path.exists(_CACHE_PATH) and os.path.exists(_WORD_PATH):
        with open(_WORD_PATH) as f:
            cached_rows = f.read().split()
        if cached_rows == row_words:
            return np.load(_CACHE_PATH)

    print(
        f"Building pattern matrix ({len(row_words)}×{len(col_words)}) "
        "– this takes ~30 s and is cached afterwards …"
    )
    mat = _compute_pattern_matrix(row_words, col_words)
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    np.save(_CACHE_PATH, mat)
    with open(_WORD_PATH, "w") as f:
        f.write("\n".join(row_words))
    return mat


# ---------------------------------------------------------------------------
# Entropy helpers (vectorised)
# ---------------------------------------------------------------------------

_WIN_INT = sum(2 * (3 ** i) for i in range(5))  # +++++ encodes to 242


def _row_entropy(pattern_row: np.ndarray, col_mask: np.ndarray) -> float:
    """Shannon entropy of pattern distribution over currently-possible columns."""
    patterns = pattern_row[col_mask]
    if patterns.size == 0:
        return 0.0
    counts = np.bincount(patterns.astype(np.int32), minlength=243)
    probs = counts[counts > 0] / patterns.size
    return float(-np.sum(probs * np.log2(probs)))


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def top_entropy_answers(k: int) -> list[str]:
    """Return the k ANSWERS words with the highest first-guess entropy.

    These are good probe words and can serve as a reduced action+target space
    for RL training, making exploration tractable without action masking.
    """
    solver = EntropySolver()
    full_mask = np.ones(len(solver.answers), dtype=bool)
    scored = [
        (word, _row_entropy(solver._matrix[solver._gue_idx[word]], full_mask))
        for word in solver.answers
        if word in solver._gue_idx
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    words = [w for w, _ in scored[: min(k, len(scored))]]
    print(f"[top_k] top-{len(words)} answers by entropy: {words[:5]} …")
    return words


class EntropySolver:
    """
    Parameters
    ----------
    answers  : words that can be the secret answer
    guesses  : all valid guesses (answers ⊆ guesses)
    cache    : whether to persist the pattern matrix to disk
    """

    def __init__(
        self,
        answers: list[str] | None = None,
        guesses: list[str] | None = None,
        cache: bool = True,
    ) -> None:
        self.answers: list[str] = list(answers or ANSWERS)
        self.guesses: list[str] = list(guesses or ALL_WORDS)

        # row = guess index, col = answer index
        self._ans_idx: dict[str, int] = {w: i for i, w in enumerate(self.answers)}
        self._gue_idx: dict[str, int] = {w: i for i, w in enumerate(self.guesses)}

        if cache:
            self._matrix = _load_or_build_matrix(self.guesses, self.answers)
        else:
            self._matrix = _compute_pattern_matrix(self.guesses, self.answers)

        self._possible_mask = np.ones(len(self.answers), dtype=bool)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._possible_mask[:] = True

    # ------------------------------------------------------------------
    @property
    def possible(self) -> list[str]:
        return [w for w, ok in zip(self.answers, self._possible_mask) if ok]

    @property
    def n_possible(self) -> int:
        return int(self._possible_mask.sum())

    # ------------------------------------------------------------------
    def best_guess(self) -> str:
        n = self.n_possible
        if n == 1:
            return self.possible[0]
        if n == 2:
            return self.possible[0]

        best_word = ""
        best_ent = -1.0

        for i, word in enumerate(self.guesses):
            ent = _row_entropy(self._matrix[i], self._possible_mask)
            if ent > best_ent:
                best_ent = ent
                best_word = word

        return best_word

    # ------------------------------------------------------------------
    def update(self, guess: str, pattern: str) -> None:
        """Narrow the possible set given a guess and its feedback pattern."""
        g_idx = self._gue_idx.get(guess)
        if g_idx is None:
            # word not in our guess list – fall back to slow path
            val = 0
            for ch in pattern:
                val = val * 3 + (0 if ch == "-" else 1 if ch == "x" else 2)
            keep = np.array(
                [evaluate_guess(guess, w) == pattern for w in self.answers],
                dtype=bool,
            )
            self._possible_mask &= keep
            return

        p_val = 0
        for ch in pattern:
            p_val = p_val * 3 + (0 if ch == "-" else 1 if ch == "x" else 2)

        self._possible_mask &= self._matrix[g_idx] == p_val

    # ------------------------------------------------------------------
    def solve(self, target: str, first_guess: str | None = None) -> list[str]:
        """
        Simulate a full game.  Returns the list of guesses made.
        Raises ValueError if the target is not in self.answers.
        """
        self.reset()
        guesses_made: list[str] = []
        for _ in range(MAX_GUESSES):
            if not guesses_made and first_guess:
                guess = first_guess
            else:
                guess = self.best_guess()
            guesses_made.append(guess)
            pattern = evaluate_guess(guess, target)
            if pattern == WIN_PATTERN:
                return guesses_made
            self.update(guess, pattern)
        return guesses_made  # failed – list will have MAX_GUESSES entries
