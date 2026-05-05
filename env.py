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
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from game import MAX_GUESSES, WIN_PATTERN, evaluate_guess
from words import ANSWERS


class WordleEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        answers: list[str] | None = None,
        action_words: list[str] | None = None,
        shaped_reward: bool = True,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.answers: list[str] = answers if answers is not None else ANSWERS
        self.action_words: list[str] = (
            action_words if action_words is not None else self.answers
        )
        self.shaped_reward = shaped_reward
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

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._target = self.answers[self.np_random.integers(len(self.answers))]
        self._state = np.zeros(60, dtype=np.float32)
        self._n_guesses = 0
        self._done = False
        return self._state.copy(), {"target": self._target}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert not self._done, "call reset() before step()"

        guess = self.action_words[int(action)]
        pattern = evaluate_guess(guess, self._target)
        self._n_guesses += 1

        # write this row into state
        slot = (self._n_guesses - 1) * 10
        feedback_map = {"-": 1, "x": 2, "+": 3}
        for k, (ch, fb) in enumerate(zip(guess, pattern)):
            self._state[slot + k] = (ord(ch) - ord("a") + 1) / 27.0
            self._state[slot + 5 + k] = feedback_map[fb] / 3.0

        won = pattern == WIN_PATTERN
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
            "guess": guess,
            "pattern": pattern,
            "target": self._target,
            "n_guesses": self._n_guesses,
            "won": won,
        }
        return self._state.copy(), reward, terminated, False, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> str | None:
        _FB = {"-": "⬜", "x": "🟨", "+": "🟩"}
        lines: list[str] = []
        for s in range(self._n_guesses):
            slot = s * 10
            letters = "".join(
                chr(int(round(self._state[slot + k] * 27)) + ord("a") - 1)
                if self._state[slot + k] > 0
                else "."
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
