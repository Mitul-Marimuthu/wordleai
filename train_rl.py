"""
Train a PPO agent to play Wordle.

Usage
-----
  python train_rl.py                          # 1 M steps, standard
  python train_rl.py --masked                 # action masking (fastest)
  python train_rl.py --curriculum             # staged vocab expansion
  python train_rl.py --masked --curriculum    # both (recommended)
  python train_rl.py --timesteps 3000000 --masked --curriculum
  python train_rl.py --eval                   # evaluate a saved model

Action masking (--masked)
-------------------------
Uses sb3-contrib MaskablePPO.  After each guess the env returns a boolean
mask over the 2315-word action space: only words still consistent with all
observed feedback are allowed.  This collapses the effective action space
from 2315 → ~300 → ~30 → ~3 across a typical game, making credit assignment
dramatically easier and cutting required training steps by 10-100×.

Curriculum design
-----------------
Action space stays fixed at Discrete(2315) throughout training so the policy
network never needs to be rebuilt.  Curriculum changes only the *target* pool
via a shared mutable list.  Stages: 50 → 150 → 400 → 1 000 → 2 315 words.
"""

import argparse
import os
import random

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor

from env import WordleEnv
from words import ANSWERS


def _ppo_class(masked: bool):
    if masked:
        from sb3_contrib import MaskablePPO
        return MaskablePPO
    from stable_baselines3 import PPO
    return PPO


def _eval_callback_class(masked: bool):
    if masked:
        from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
        return MaskableEvalCallback
    from stable_baselines3.common.callbacks import EvalCallback
    return EvalCallback

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "ppo_wordle")
LOG_PATH = os.path.join(os.path.dirname(__file__), "logs")

# Vocab sizes at each curriculum stage
STAGES = [50, 150, 400, 1_000, 2_315]


# ---------------------------------------------------------------------------
# Curriculum callback
# ---------------------------------------------------------------------------

class CurriculumCallback(BaseCallback):
    """
    Periodically evaluates the policy on the current target vocab.
    When win-rate >= win_threshold, the target vocab is expanded in-place
    to the next stage.  All training and eval envs see the change immediately
    on their next reset() because they hold a reference to the same list.
    """

    def __init__(
        self,
        target_vocab: list[str],
        all_answers: list[str],
        eval_env: WordleEnv,
        check_freq: int = 20_000,
        win_threshold: float = 0.80,
        n_eval: int = 300,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.target_vocab = target_vocab   # shared mutable list
        self.all_answers = all_answers
        self.eval_env = eval_env
        self.check_freq = check_freq
        self.win_threshold = win_threshold
        self.n_eval = n_eval
        self._stage = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True
        if self._stage >= len(STAGES) - 1:
            return True

        wins = 0
        obs, _ = self.eval_env.reset()
        for _ in range(self.n_eval):
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = self.eval_env.step(int(action))
                done = term or trunc
            if info["won"]:
                wins += 1
            obs, _ = self.eval_env.reset()

        win_rate = wins / self.n_eval
        next_size = STAGES[self._stage + 1]
        if self.verbose >= 1:
            print(
                f"\n[Curriculum] stage={self._stage}  vocab={len(self.target_vocab)}"
                f"  win_rate={win_rate:.1%}"
                f"  (need ≥{self.win_threshold:.0%} to advance → {next_size})"
            )

        if win_rate >= self.win_threshold:
            self._stage += 1
            new_words = random.sample(self.all_answers, STAGES[self._stage])
            self.target_vocab.clear()
            self.target_vocab.extend(new_words)
            if self.verbose >= 1:
                print(f"[Curriculum] → stage {self._stage}: vocab={len(self.target_vocab)}")

        return True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _make_model(vec_env, log_path: str, masked: bool):
    PPO = _ppo_class(masked)
    policy = "MlpPolicy"
    return PPO(
        policy,
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log=log_path,
    )


def train(
    timesteps: int = 1_000_000,
    n_envs: int = 8,
    curriculum: bool = False,
    masked: bool = False,
    dense_reward: bool = True,
) -> object:
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    all_answers = list(ANSWERS)

    if curriculum:
        target_vocab: list[str] = random.sample(all_answers, STAGES[0])
        make_env  = lambda: WordleEnv(shaped_reward=True,  dense_reward=dense_reward, target_vocab=target_vocab, use_mask=masked)
        eval_env  = WordleEnv(shaped_reward=False, dense_reward=False, target_vocab=target_vocab, use_mask=masked)
    else:
        target_vocab = all_answers
        make_env  = lambda: WordleEnv(shaped_reward=True,  dense_reward=dense_reward, use_mask=masked)
        eval_env  = WordleEnv(shaped_reward=False, dense_reward=False, use_mask=masked)

    vec_env = VecMonitor(make_vec_env(make_env, n_envs=n_envs))
    model   = _make_model(vec_env, LOG_PATH, masked)

    callbacks: list = []
    if curriculum:
        callbacks.append(
            CurriculumCallback(
                target_vocab=target_vocab,
                all_answers=all_answers,
                eval_env=eval_env,
                check_freq=max(20_000 // n_envs, 1),
                win_threshold=0.80,
                n_eval=300,
            )
        )

    EvalCB = _eval_callback_class(masked)
    eval_vec = make_vec_env(lambda: WordleEnv(shaped_reward=False, use_mask=masked), n_envs=4)
    callbacks.append(
        EvalCB(
            eval_vec,
            best_model_save_path=os.path.dirname(MODEL_PATH),
            log_path=LOG_PATH,
            eval_freq=max(50_000 // n_envs, 1),
            n_eval_episodes=200,
            deterministic=True,
            verbose=1,
        )
    )

    mode = "+".join(filter(None, ["masked" if masked else "", "curriculum" if curriculum else "standard"]))
    print(f"Training PPO ({mode}) for {timesteps:,} timesteps …")
    if curriculum:
        print(f"  Stage 0: {STAGES[0]} target words → full {STAGES[-1]}")

    model.learn(
        total_timesteps=timesteps,
        callback=CallbackList(callbacks),
        progress_bar=True,
    )
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}.zip")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, n_games: int = 500, masked: bool = False) -> dict:
    env = WordleEnv(shaped_reward=False, use_mask=masked)
    wins = 0
    total_guesses = 0
    dist: dict[int, int] = {}

    obs, _ = env.reset()
    for _ in range(n_games):
        done = False
        while not done:
            kwargs = {"action_masks": env.action_masks()} if masked else {}
            action, _ = model.predict(obs, deterministic=True, **kwargs)
            obs, _, term, trunc, info = env.step(int(action))
            done = term or trunc
        n = info["n_guesses"]
        won = info["won"]
        if won:
            wins += 1
            total_guesses += n
            dist[n] = dist.get(n, 0) + 1
        else:
            dist[7] = dist.get(7, 0) + 1
        obs, _ = env.reset()

    win_rate = wins / n_games
    return {
        "win_rate": win_rate,
        "avg_guesses_on_win": total_guesses / max(wins, 1),
        "distribution": dist,
        "n_games": n_games,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps",       type=int, default=1_000_000)
    parser.add_argument("--n_envs",          type=int, default=8)
    parser.add_argument("--curriculum",      action="store_true")
    parser.add_argument("--masked",          action="store_true")
    parser.add_argument("--no_dense_reward", action="store_true",
                        help="Disable per-step dense reward (terminal rewards only)")
    parser.add_argument("--eval",            action="store_true")
    parser.add_argument("--n_eval",          type=int, default=500)
    args = parser.parse_args()

    if args.eval:
        path = MODEL_PATH + ".zip"
        if not os.path.exists(path):
            print(f"No model at {path}. Train first.")
            return
        PPOCls = _ppo_class(args.masked)
        model = PPOCls.load(MODEL_PATH, env=WordleEnv(use_mask=args.masked))
    else:
        model = train(
            timesteps=args.timesteps,
            n_envs=args.n_envs,
            curriculum=args.curriculum,
            masked=args.masked,
            dense_reward=not args.no_dense_reward,
        )

    print("\nEvaluating on full word bank …")
    stats = evaluate(model, n_games=args.n_eval, masked=args.masked)
    print(f"Win rate      : {stats['win_rate']:.1%}")
    print(f"Avg guesses   : {stats['avg_guesses_on_win']:.2f}  (wins only)")
    dist = stats["distribution"]
    print("Distribution  :", {k: dist[k] for k in sorted(dist)})


if __name__ == "__main__":
    main()
