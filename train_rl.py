"""
Train a PPO agent to play Wordle.

Usage
-----
  python train_rl.py                     # train with default settings
  python train_rl.py --timesteps 2000000
  python train_rl.py --eval              # evaluate a saved model
"""

import argparse
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor

from env import WordleEnv
from words import ANSWERS

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "ppo_wordle")
LOG_PATH = os.path.join(os.path.dirname(__file__), "logs")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(timesteps: int = 1_000_000, n_envs: int = 8) -> PPO:
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    vec_env = make_vec_env(
        lambda: WordleEnv(shaped_reward=True),
        n_envs=n_envs,
    )
    vec_env = VecMonitor(vec_env)

    eval_env = make_vec_env(lambda: WordleEnv(shaped_reward=False), n_envs=4)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.dirname(MODEL_PATH),
        log_path=LOG_PATH,
        eval_freq=max(10_000 // n_envs, 1),
        n_eval_episodes=200,
        deterministic=True,
        verbose=1,
    )

    model = PPO(
        "MlpPolicy",
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
        tensorboard_log=LOG_PATH,
    )

    print(f"Training PPO for {timesteps:,} timesteps …")
    model.learn(total_timesteps=timesteps, callback=eval_callback, progress_bar=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}.zip")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: PPO, n_games: int = 500) -> dict:
    env = WordleEnv(shaped_reward=False)
    wins = 0
    total_guesses = 0
    guess_dist: dict[int, int] = {}

    for _ in range(n_games):
        obs, _ = env.reset()
        done = False
        won = False
        n = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            n = info["n_guesses"]
            won = info["won"]
        if won:
            wins += 1
            total_guesses += n
            guess_dist[n] = guess_dist.get(n, 0) + 1
        else:
            guess_dist[7] = guess_dist.get(7, 0) + 1  # failure bucket

    win_rate = wins / n_games
    avg_guesses = total_guesses / max(wins, 1)
    return {
        "win_rate": win_rate,
        "avg_guesses_on_win": avg_guesses,
        "distribution": guess_dist,
        "n_games": n_games,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--eval", action="store_true", help="evaluate saved model only")
    parser.add_argument("--n_eval", type=int, default=500)
    args = parser.parse_args()

    if args.eval:
        path = MODEL_PATH + ".zip"
        if not os.path.exists(path):
            print(f"No model found at {path}. Train first.")
            return
        model = PPO.load(MODEL_PATH, env=WordleEnv())
    else:
        model = train(timesteps=args.timesteps, n_envs=args.n_envs)

    print("\nEvaluating …")
    stats = evaluate(model, n_games=args.n_eval)
    print(f"Win rate      : {stats['win_rate']:.1%}")
    print(f"Avg guesses   : {stats['avg_guesses_on_win']:.2f}  (wins only)")
    dist = stats["distribution"]
    print("Distribution  :", {k: dist[k] for k in sorted(dist)})


if __name__ == "__main__":
    main()
