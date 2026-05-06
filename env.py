"""
WordleEnv – a Gymnasium environment for Wordle.

Observation (Box float32, shape=(60,)):
  Rows 0-5 are guess slots.  Each slot contributes 10 values:
    indices  0-4  : letter index ∈ {0…25}, normalised to (idx+1)/27  (0 = empty)
    indices  5-9  : feedback    ∈ {0=empty,1=absent,2=yellow,3=green}, /3

Action:
  Discrete(len(action_words)) – index into the action word list.
  Defaults to ANSWERS so the space stays tractable for PPO.

Reward:
  +1   win
  -1   loss (exceeded MAX_GUESSES without solving)
   0   otherwise

  Optionally shaped: on win, reward += (MAX_GUESSES - guesses_taken) / MAX_GUESSES
  so fewer guesses yield higher return.

Action masking (use_mask=True):
  Requires sb3-contrib MaskablePPO.  After each guess the env exposes
  action_masks() returning a boolean array over action_words.  Only words
  still consistent with all observed feedback are True.

  Mask updates use the pre-computed pattern matrix from solver_entropy.py
  (.cache/patterns.npy) for O(1) per-step cost.  If the cache is absent
  it falls back to evaluate_guess on the fly.
"""

import os

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from game import MAX_GUESSES, WIN_PATTERN, evaluate_guess
from words import ANSWERS

# ─── Pattern-matrix lazy loader ──────────────────────────────────────────────
# Shared across all env instances; loaded once from disk.

_PAT_MATRIX: np.ndarray | None = None  # shape (n_guesses_words, n_answer_words)
_WORD_TO_ROW: dict[str, int] | None = None  # guess word → row index in matrix


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

class WordleEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        answers: list[str] | None = None,
        action_words: list[str] | None = None,
        target_vocab: list[str] | None = None,
        shaped_reward: bool = True,
        use_mask: bool = False,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.answers: list[str] = answers if answers is not None else ANSWERS
        self.action_words: list[str] = (
            action_words if action_words is not None else self.answers
        )
        # Curriculum hook: targets sampled from this list when set.
        self.target_vocab: list[str] | None = target_vocab
        self.shaped_reward = shaped_reward
        self.use_mask = use_mask
        self.render_mode = render_mode

        self._word_to_idx: dict[str, int] = {
            w: i for i, w in enumerate(self.action_words)
        }

        # obs: 6 slots × 10 values = 60 floats in [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(60,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(self.action_words))

        self._state: np.ndarray = np.zeros(60, dtype=np.float32)
        self._target: str = ""
        self._n_guesses: int = 0
        self._done: bool = False

        # Masking support ──────────────────────────────────────────────────
        # _possible[i] = True iff action_words[i] is still a valid answer
        # given feedback seen so far.  Stays None if use_mask=False.
        self._possible: np.ndarray | None = None

        if use_mask:
            mat, w2r = _load_pattern_cache()
            # Pre-extract the rows we care about (one per action_word)
            if mat is not None and w2r is not None:
                rows = []
                for w in self.action_words:
                    r = w2r.get(w)
                    if r is not None:
                        rows.append(mat[r])   # shape (n_answers,)
                    else:
                        # word not in cache – build its row lazily
                        rows.append(
                            np.array(
                                [_pattern_str_to_int(evaluate_guess(w, t))
                                 for t in self.answers],
                                dtype=np.uint8,
                            )
                        )
                # shape (n_action_words, n_answers)
                self._mask_matrix: np.ndarray | None = np.array(rows, dtype=np.uint8)
                # mapping: answer word → column index in _mask_matrix
                self._ans_to_col: dict[str, int] = {
                    w: i for i, w in enumerate(self.answers)
                }
            else:
                # No cache – will fall back to evaluate_guess at runtime
                self._mask_matrix = None
                self._ans_to_col = {}

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        pool = self.target_vocab if self.target_vocab else self.answers
        self._target = pool[self.np_random.integers(len(pool))]
        self._state = np.zeros(60, dtype=np.float32)
        self._n_guesses = 0
        self._done = False
        if self.use_mask:
            self._possible = np.ones(len(self.action_words), dtype=bool)
        return self._state.copy(), {"target": self._target}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert not self._done, "call reset() before step()"

        guess   = self.action_words[int(action)]
        pattern = evaluate_guess(guess, self._target)
        self._n_guesses += 1

        # write this row into state
        slot = (self._n_guesses - 1) * 10
        feedback_map = {"-": 1, "x": 2, "+": 3}
        for k, (ch, fb) in enumerate(zip(guess, pattern)):
            self._state[slot + k]     = (ord(ch) - ord("a") + 1) / 27.0
            self._state[slot + 5 + k] = feedback_map[fb] / 3.0

        # update mask
        if self.use_mask and self._possible is not None:
            self._update_mask(guess, pattern)

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
        if self._possible is None:
            return np.ones(len(self.action_words), dtype=bool)
        # Safety: never return an all-False mask (would crash MaskablePPO)
        if not self._possible.any():
            return np.ones(len(self.action_words), dtype=bool)
        return self._possible.copy()

    def _update_mask(self, guess: str, pattern: str) -> None:
        p_int = _pattern_str_to_int(pattern)

        if self._mask_matrix is not None:
            # Fast path: O(n) numpy operation using pre-computed rows
            g_idx = self._word_to_idx.get(guess)
            if g_idx is not None:
                # For each action word i, check pattern(guess, action_word_i) == p_int
                self._possible &= (self._mask_matrix[g_idx] == p_int)
                return

        # Slow path fallback: O(n) evaluate_guess calls
        for i, w in enumerate(self.action_words):
            if self._possible[i]:
                if evaluate_guess(guess, w) != pattern:
                    self._possible[i] = False

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> str | None:
        lines: list[str] = []
        for s in range(self._n_guesses):
            slot = s * 10
            letters = "".join(
                chr(int(round(self._state[slot + k] * 27)) + ord("a") - 1)
                if self._state[slot + k] > 0 else "."
                for k in range(5)
            )
            fb_vals = [int(round(self._state[slot + 5 + k] * 3)) for k in range(5)]
            fb_syms = [("⬜", "🟨", "🟩")[v] if v else "⬜" for v in fb_vals]
            lines.append(f"{letters.upper()}  {''.join(fb_syms)}")
        out = "\n".join(lines) or "(no guesses yet)"
        if self.render_mode == "human":
            print(out)
            return None
        return out
