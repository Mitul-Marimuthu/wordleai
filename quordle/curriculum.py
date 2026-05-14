"""
quordle/train_rl.py — curriculum callback and training helpers for Quordle.

Curriculum design
-----------------
Target vocab expands in stages: 15 → 50 → 150 → 500 → 2315 words.
Action space stays fixed at Discrete(2315) throughout so the policy network
never needs to be rebuilt.  Only the *target pool* changes via a shared
mutable list — all envs see the update on their next reset().

Advance condition: win rate (all 4 boards solved) ≥ win_threshold on a
held-out eval env at the current stage size.
"""

import random

from stable_baselines3.common.callbacks import BaseCallback

from env import QuordleEnv, N_BOARDS

STAGES = [15, 50, 150, 500, 2_315]


def _active_stages(max_size: int) -> list[int]:
    """STAGES capped at max_size, always ending exactly at max_size."""
    stages = [s for s in STAGES if s < max_size]
    stages.append(max_size)
    return stages


class CurriculumCallback(BaseCallback):
    """
    Periodically evaluates the policy on the current target vocab.
    When all-4-boards win rate >= win_threshold, expands the shared
    target_vocab list to the next stage.
    """

    def __init__(
        self,
        target_vocab: list[str],
        all_answers: list[str],
        eval_env: QuordleEnv,
        check_freq: int = 30_000,
        win_threshold: float = 0.50,
        n_eval: int = 200,
        stages: list[int] | None = None,
        on_advance=None,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.target_vocab   = target_vocab
        self.all_answers    = all_answers
        self.eval_env       = eval_env
        self.check_freq     = check_freq
        self.win_threshold  = win_threshold
        self.n_eval         = n_eval
        self.stages         = stages if stages is not None else list(STAGES)
        self.on_advance     = on_advance   # optional callback(new_size) for UI updates
        self._stage         = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True
        if self._stage >= len(self.stages) - 1:
            return True

        wins = 0
        obs, _ = self.eval_env.reset()
        for _ in range(self.n_eval):
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True,
                                               action_masks=self.eval_env.action_masks())
                obs, _, term, trunc, info = self.eval_env.step(int(action))
                done = term or trunc
            if info["won"]:
                wins += 1
            obs, _ = self.eval_env.reset()

        win_rate  = wins / self.n_eval
        next_size = self.stages[self._stage + 1]

        if self.verbose >= 1:
            print(
                f"\n[Curriculum] stage={self._stage}  "
                f"vocab={len(self.target_vocab)}  "
                f"win_rate={win_rate:.1%}  "
                f"(need ≥{self.win_threshold:.0%} to advance → {next_size})"
            )

        if win_rate >= self.win_threshold:
            self._stage += 1
            new_words = random.sample(self.all_answers, self.stages[self._stage])
            self.target_vocab.clear()
            self.target_vocab.extend(new_words)
            if self.on_advance:
                self.on_advance(len(self.target_vocab))
            if self.verbose >= 1:
                print(f"[Curriculum] → stage {self._stage}: vocab={len(self.target_vocab)}")

        return True
