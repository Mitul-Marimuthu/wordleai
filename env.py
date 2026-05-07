"""
WordleEnv – a Gymnasium environment for Wordle.

Observation (Box float32, shape=(208,)):
  Letter knowledge (dims 0–77): 26 letters × 3 one-hot
    [1,0,0] = unknown  |  [0,1,0] = absent (gray)  |  [0,0,1] = present in word
    "present" covers both yellow (position unknown) and green (confirmed).
  Position knowledge (dims 78–207): 5 positions × 26 letters
    0.0 = unknown, 0.5 = eliminated from this position, 1.0 = confirmed here (green)

Action:
  Discrete(len(action_words)) – index into the action word list.
  Defaults to ANSWERS so the space stays tractable for PPO.

Reward:
  +1   win  (+ shaped bonus: (MAX_GUESSES - n) / MAX_GUESSES if shaped_reward=True)
  -1   loss (exceeded MAX_GUESSES without solving)
  Per non-terminal step with dense_reward=True:
    + (n_words_eliminated / len(action_words))

Action masking (use_mask=True):
  Requires sb3-contrib MaskablePPO.  Returns a boolean array over action_words.
  Possible-word tracking is always active regardless of use_mask (used for dense reward).
  Pattern matrix is loaded unconditionally for fast O(1) mask updates.
"""

import os

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from game import MAX_GUESSES, WIN_PATTERN, evaluate_guess
from words import ANSWERS

# ─── Pattern-matrix lazy loader ──────────────────────────────────────────────

_PAT_MATRIX: np.ndarray | None = None
_WORD_TO_ROW: dict[str, int] | None = None


def _load_pattern_cache() -> tuple[np.ndarray | None, dict[str, int] | None]:
    global _PAT_MATRIX, _WORD_TO_ROW
    if _PAT_MATRIX is not None:
        return _PAT_MATRIX, _WORD_TO_ROW

    cache_dir = os.path.join(os.path.dirname(__file__), ".cache")
    mat_path  = os.path.join(cache_dir, "patterns.npy")
    wrd_path  = os.path.join(cache_dir, "pattern_words.txt")

    if os.path.exists(mat_path) and os.path.exists(wrd_path):
        _PAT_MATRIX = np.load(mat_path)
        with open(wrd_path) as f:
            words = f.read().split()
        _WORD_TO_ROW = {w: i for i, w in enumerate(words)}

    return _PAT_MATRIX, _WORD_TO_ROW


def _pattern_str_to_int(pattern: str) -> int:
    val = 0
    for ch in pattern:
        val = val * 3 + (0 if ch == "-" else 1 if ch == "x" else 2)
    return val


# ─── Environment ─────────────────────────────────────────────────────────────

OBS_SIZE = 208   # 26*3 + 5*26


class WordleEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        answers: list[str] | None = None,
        action_words: list[str] | None = None,
        target_vocab: list[str] | None = None,
        shaped_reward: bool = True,
        dense_reward: bool = True,
        use_mask: bool = False,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.answers: list[str] = answers if answers is not None else ANSWERS
        self.action_words: list[str] = (
            action_words if action_words is not None else self.answers
        )
        self.target_vocab: list[str] | None = target_vocab
        self.shaped_reward = shaped_reward
        self.dense_reward  = dense_reward
        self.use_mask      = use_mask
        self.render_mode   = render_mode

        self._word_to_idx: dict[str, int] = {
            w: i for i, w in enumerate(self.action_words)
        }

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(self.action_words))

        self._state: np.ndarray = np.zeros(OBS_SIZE, dtype=np.float32)
        self._target: str = ""
        self._n_guesses: int = 0
        self._done: bool = False
        self._game: list[tuple[str, str]] = []   # (guess, pattern) history this episode

        # Possible-word tracking — always active (dense reward + optional mask).
        self._possible: np.ndarray = np.ones(len(self.action_words), dtype=bool)

        # Load pattern cache for fast O(1) mask/dense-reward updates.
        mat, w2r = _load_pattern_cache()
        if mat is not None and w2r is not None:
            rows = []
            for w in self.action_words:
                r = w2r.get(w)
                if r is not None:
                    rows.append(mat[r])
                else:
                    rows.append(
                        np.array(
                            [_pattern_str_to_int(evaluate_guess(w, t))
                             for t in self.answers],
                            dtype=np.uint8,
                        )
                    )
            self._mask_matrix: np.ndarray | None = np.array(rows, dtype=np.uint8)
        else:
            self._mask_matrix = None

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        pool = self.target_vocab if self.target_vocab else self.answers
        self._target   = pool[self.np_random.integers(len(pool))]
        self._state    = np.zeros(OBS_SIZE, dtype=np.float32)
        # Mark all 26 letters as unknown: one-hot [1, 0, 0]
        for i in range(26):
            self._state[i * 3] = 1.0
        self._n_guesses = 0
        self._done      = False
        self._game      = []
        self._possible  = np.ones(len(self.action_words), dtype=bool)
        return self._state.copy(), {"target": self._target}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert not self._done, "call reset() before step()"

        guess   = self.action_words[int(action)]
        pattern = evaluate_guess(guess, self._target)
        self._n_guesses += 1
        self._game.append((guess, pattern))

        n_before = int(self._possible.sum())

        self._update_obs(guess, pattern)
        self._update_possible(guess, pattern)

        n_after = int(self._possible.sum())

        won  = pattern == WIN_PATTERN
        lost = (not won) and self._n_guesses >= MAX_GUESSES
        terminated = won or lost

        if won:
            reward = 1.0
            if self.shaped_reward:
                reward += (MAX_GUESSES - self._n_guesses) / MAX_GUESSES
        elif lost:
            reward = -1.0
        else:
            reward = 0.0

        if self.dense_reward and not terminated and n_before > 1:
            reward += (n_before - n_after) / len(self.action_words)

        self._done = terminated
        info = {
            "guess":    guess,
            "pattern":  pattern,
            "target":   self._target,
            "n_guesses": self._n_guesses,
            "won":      won,
        }
        return self._state.copy(), reward, terminated, False, info

    # ------------------------------------------------------------------
    # Action masking (called by MaskablePPO each step)
    # ------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        if not self._possible.any():
            return np.ones(len(self.action_words), dtype=bool)
        return self._possible.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_obs(self, guess: str, pattern: str) -> None:
        """Update the 208-dim letter-knowledge observation.

        Layout:
          [0:78]   26 letters × 3 one-hot: [unknown, absent, present-in-word]
          [78:208] 5 positions × 26 letters: 0=unknown, 0.5=eliminated, 1.0=confirmed
        """
        for col, (ch, fb) in enumerate(zip(guess, pattern)):
            li    = ord(ch) - ord("a")       # letter index 0–25
            lbase = li * 3                   # start of one-hot in letter section
            pbase = 78 + col * 26 + li       # position×letter cell

            # ── position knowledge ──────────────────────────────────
            if fb == "+":
                self._state[pbase] = 1.0     # confirmed at this position
            elif self._state[pbase] < 1.0:
                self._state[pbase] = 0.5     # yellow/gray → eliminated here

            # ── letter knowledge ────────────────────────────────────
            if fb in ("+", "x"):
                # confirmed present somewhere (green or yellow)
                self._state[lbase + 0] = 0.0
                self._state[lbase + 1] = 0.0
                self._state[lbase + 2] = 1.0
            elif fb == "-":
                # absent — only update if we haven't already confirmed it's present
                if self._state[lbase + 2] == 0.0:
                    self._state[lbase + 0] = 0.0
                    self._state[lbase + 1] = 1.0

    def _update_possible(self, guess: str, pattern: str) -> None:
        p_int = _pattern_str_to_int(pattern)

        if self._mask_matrix is not None:
            g_idx = self._word_to_idx.get(guess)
            if g_idx is not None:
                self._possible &= (self._mask_matrix[g_idx] == p_int)
                return

        # Slow fallback (no cache)
        for i, w in enumerate(self.action_words):
            if self._possible[i] and evaluate_guess(guess, w) != pattern:
                self._possible[i] = False

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> str | None:
        sym = {"+": "🟩", "x": "🟨", "-": "⬜"}
        lines = [
            f"{g.upper()}  {''.join(sym[c] for c in p)}"
            for g, p in self._game
        ]
        out = "\n".join(lines) or "(no guesses yet)"
        if self.render_mode == "human":
            print(out)
            return None
        return out
