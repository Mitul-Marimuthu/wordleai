"""
QuordleEnv — Gymnasium environment for Quordle.

Quordle: solve 4 Wordle boards simultaneously with a shared guess pool.
Every guess is applied to all 4 boards at once.  Win by solving all 4
within 9 guesses; lose if any board is unsolved after guess 9.

Observation (Box float32, shape=(836,)):
  Dims   0–831 : 4 × 208-dim letter-knowledge encoding (one per board).
                 Board b occupies dims [b*208 : (b+1)*208].
                 Encoding identical to WordleEnv:
                   [0:78]   26 letters × 3 one-hot (unknown / absent / present)
                   [78:208] 5 positions × 26 letters (0=unknown, 0.5=elim, 1.0=green)
  Dims 832–835 : solved flags — 1.0 if board b is solved, else 0.0.

Action:
  Discrete(len(action_words)) — index into the shared word list.

Reward:
  +1.0  per board solved on this step
  +(MAX_GUESSES - n_guesses) / MAX_GUESSES  shaped bonus per board win
  -1.0  if game ends with any board unsolved
  Dense (non-terminal, per step):
    + (words_eliminated_across_unsolved_boards) / (n_unsolved * len(action_words))

Action masking (use_mask=True):
  Union of possible-word sets across all unsolved boards.
  A word is kept if it could still be the answer on at least one unsolved board.
"""

import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ─── Shared game logic from the Wordle module ────────────────────────────────

_WORDLE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "wordle"))
if _WORDLE_DIR not in sys.path:
    sys.path.insert(0, _WORDLE_DIR)

from game import evaluate_guess, WIN_PATTERN  # noqa: E402
from words import ANSWERS                     # noqa: E402

# ─── Constants ───────────────────────────────────────────────────────────────

MAX_GUESSES = 9
N_BOARDS    = 4
BOARD_OBS   = 208
OBS_SIZE    = N_BOARDS * BOARD_OBS + N_BOARDS   # 836


def _pattern_str_to_int(pattern: str) -> int:
    val = 0
    for ch in pattern:
        val = val * 3 + (0 if ch == "-" else 1 if ch == "x" else 2)
    return val


# ─── Environment ─────────────────────────────────────────────────────────────

class QuordleEnv(gym.Env):
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
        self.answers      = answers      if answers      is not None else list(ANSWERS)
        self.action_words = action_words if action_words is not None else self.answers
        self.target_vocab = target_vocab   # shared mutable list; None = use self.answers
        self.shaped_reward = shaped_reward
        self.dense_reward  = dense_reward
        self.use_mask      = use_mask
        self.render_mode   = render_mode

        self._word_to_idx: dict[str, int] = {w: i for i, w in enumerate(self.action_words)}

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(self.action_words))

        # Per-board state (initialised properly in reset())
        self._targets:   list[str]                    = [""] * N_BOARDS
        self._states:    list[np.ndarray]             = [np.zeros(BOARD_OBS, dtype=np.float32)] * N_BOARDS
        self._solved:    list[bool]                   = [False] * N_BOARDS
        self._possible:  list[np.ndarray]             = [np.ones(len(self.action_words), dtype=bool)] * N_BOARDS
        self._games:     list[list[tuple[str, str]]]  = [[] for _ in range(N_BOARDS)]
        self._n_guesses: int  = 0
        self._done:      bool = False

        self._mask_matrix: np.ndarray | None = self._load_mask_matrix()

    # ── Pattern-cache loader ──────────────────────────────────────────────────

    def _load_mask_matrix(self) -> np.ndarray | None:
        cache_dir = os.path.join(_WORDLE_DIR, ".cache")
        mat_path  = os.path.join(cache_dir, "patterns.npy")
        wrd_path  = os.path.join(cache_dir, "pattern_words.txt")
        if not (os.path.exists(mat_path) and os.path.exists(wrd_path)):
            return None

        full_mat = np.load(mat_path)                      # (12972, 2315)
        with open(wrd_path) as f:
            row_words = f.read().split()
        w2r      = {w: i for i, w in enumerate(row_words)}
        full_col = {w: i for i, w in enumerate(ANSWERS)}

        col_indices = np.array(
            [full_col[w] for w in self.action_words if w in full_col],
            dtype=np.intp,
        )
        if len(col_indices) != len(self.action_words):
            return None

        rows = []
        for w in self.action_words:
            r = w2r.get(w)
            if r is not None:
                rows.append(full_mat[r][col_indices])
            else:
                rows.append(np.array(
                    [_pattern_str_to_int(evaluate_guess(w, t)) for t in self.action_words],
                    dtype=np.uint8,
                ))
        return np.array(rows, dtype=np.uint8)

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        # Sample 4 distinct targets from the current vocab (curriculum-aware)
        pool = list(self.target_vocab) if self.target_vocab is not None else self.answers
        n    = min(N_BOARDS, len(pool))
        idxs = self.np_random.choice(len(pool), size=n, replace=False)
        self._targets = [pool[i] for i in idxs]

        self._states = [np.zeros(BOARD_OBS, dtype=np.float32) for _ in range(N_BOARDS)]
        for state in self._states:
            for i in range(26):
                state[i * 3] = 1.0   # all letters unknown: one-hot [1, 0, 0]

        self._solved   = [False] * N_BOARDS
        self._possible = [np.ones(len(self.action_words), dtype=bool) for _ in range(N_BOARDS)]
        self._games    = [[] for _ in range(N_BOARDS)]
        self._n_guesses = 0
        self._done      = False

        return self._get_obs(), {"targets": list(self._targets)}

    def step(self, action: int):
        assert not self._done, "call reset() before step()"

        guess = self.action_words[int(action)]
        self._n_guesses += 1

        # Words possible before this guess (for dense reward)
        before = [int(self._possible[b].sum()) for b in range(N_BOARDS)]

        reward = 0.0
        patterns: list[str | None] = [None] * N_BOARDS
        for b in range(N_BOARDS):
            if self._solved[b]:
                continue
            pattern = evaluate_guess(guess, self._targets[b])
            patterns[b] = pattern
            self._games[b].append((guess, pattern))
            self._update_obs(b, guess, pattern)
            self._update_possible(b, guess, pattern)

            if pattern == WIN_PATTERN:
                self._solved[b] = True
                reward += 1.0
                if self.shaped_reward:
                    reward += (MAX_GUESSES - self._n_guesses) / MAX_GUESSES

        all_solved     = all(self._solved)
        out_of_guesses = self._n_guesses >= MAX_GUESSES
        terminated     = all_solved or out_of_guesses

        if out_of_guesses and not all_solved:
            reward -= 1.0

        if self.dense_reward and not terminated:
            after    = [int(self._possible[b].sum()) for b in range(N_BOARDS)]
            n_unsol  = sum(1 for s in self._solved if not s)
            if n_unsol > 0:
                eliminated = sum(
                    before[b] - after[b]
                    for b in range(N_BOARDS) if not self._solved[b]
                )
                reward += eliminated / (n_unsol * len(self.action_words))

        self._done = terminated
        info = {
            "guess":    guess,
            "patterns": patterns,          # list[str|None] — None for already-solved boards
            "n_guesses": self._n_guesses,
            "solved":   list(self._solved),
            "n_solved": sum(self._solved),
            "won":      all_solved,
        }
        return self._get_obs(), reward, terminated, False, info

    # ── Action masking ────────────────────────────────────────────────────────

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(len(self.action_words), dtype=bool)
        for b in range(N_BOARDS):
            if not self._solved[b]:
                mask |= self._possible[b]
        return mask if mask.any() else np.ones(len(self.action_words), dtype=bool)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        obs = np.empty(OBS_SIZE, dtype=np.float32)
        for b in range(N_BOARDS):
            obs[b * BOARD_OBS : (b + 1) * BOARD_OBS] = self._states[b]
        for b in range(N_BOARDS):
            obs[N_BOARDS * BOARD_OBS + b] = 1.0 if self._solved[b] else 0.0
        return obs

    def _update_obs(self, board: int, guess: str, pattern: str) -> None:
        state = self._states[board]
        for col, (ch, fb) in enumerate(zip(guess, pattern)):
            li    = ord(ch) - ord("a")
            lbase = li * 3
            pbase = 78 + col * 26 + li

            if fb == "+":
                state[pbase] = 1.0
            elif state[pbase] < 1.0:
                state[pbase] = 0.5

            if fb in ("+", "x"):
                state[lbase + 0] = 0.0
                state[lbase + 1] = 0.0
                state[lbase + 2] = 1.0
            elif fb == "-":
                if state[lbase + 2] == 0.0:
                    state[lbase + 0] = 0.0
                    state[lbase + 1] = 1.0

    def _update_possible(self, board: int, guess: str, pattern: str) -> None:
        p_int = _pattern_str_to_int(pattern)
        if self._mask_matrix is not None:
            g_idx = self._word_to_idx.get(guess)
            if g_idx is not None:
                self._possible[board] &= (self._mask_matrix[g_idx] == p_int)
                return
        for i, w in enumerate(self.action_words):
            if self._possible[board][i] and evaluate_guess(guess, w) != pattern:
                self._possible[board][i] = False

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render(self) -> str | None:
        sym   = {"+": "🟩", "x": "🟨", "-": "⬜"}
        col_w = 24

        # Build rows for each board independently
        board_rows: list[list[str]] = []
        for b in range(N_BOARDS):
            header = f"Board {b + 1} ({'✓' if self._solved[b] else f'? {self._targets[b]}'})"
            rows   = [header] + [
                f"{g.upper()}  {''.join(sym[c] for c in p)}"
                for g, p in self._games[b]
            ]
            board_rows.append(rows)

        max_rows = max(len(r) for r in board_rows)
        lines = []
        for row in range(max_rows):
            lines.append("  ".join(
                (board_rows[b][row] if row < len(board_rows[b]) else "").ljust(col_w)
                for b in range(N_BOARDS)
            ))

        out = "\n".join(lines)
        if self.render_mode == "human":
            print(out)
            return None
        return out


# ─── Smoke test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    env = QuordleEnv(render_mode="human")
    obs, info = env.reset(seed=42)
    print(f"Targets: {info['targets']}")
    print(f"Obs shape: {obs.shape}  (expected {OBS_SIZE})\n")

    for word in ["raise", "clout", "sniff", "derby", "lumpy",
                 "began", "trove", "phone", "windy"]:
        if word not in env._word_to_idx:
            print(f"  {word!r} not in vocab, skipping")
            continue
        action = env._word_to_idx[word]
        obs, reward, terminated, _, info = env.step(action)
        print(f"Guess: {word.upper()}  reward={reward:.3f}  "
              f"solved={info['solved']}  guesses={info['n_guesses']}")
        env.render()
        print()
        if terminated:
            print("Game over —", "WON" if info["won"] else "LOST")
            break
