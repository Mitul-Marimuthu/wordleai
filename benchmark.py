"""
Benchmark the entropy solver and (optionally) the RL agent against the same
word bank, printing a side-by-side comparison.

Usage
-----
  python benchmark.py                      # entropy solver only
  python benchmark.py --rl                 # also load the PPO model
  python benchmark.py --words 200          # sample 200 words instead of all
  python benchmark.py --first-guess crane  # fix opening guess for entropy solver
"""

import argparse
import random
import time

import numpy as np

from game import MAX_GUESSES, WIN_PATTERN, evaluate_guess
from solver_entropy import EntropySolver
from words import ANSWERS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(stats: dict) -> str:
    d = stats["distribution"]
    bar = "  ".join(f"{k}:{d.get(k,0):>4}" for k in range(1, 8))
    return (
        f"  Win rate  : {stats['win_rate']:.1%}\n"
        f"  Avg guess : {stats['avg_guesses']:.3f}\n"
        f"  Dist      : {bar}\n"
        f"  Time      : {stats['elapsed']:.1f}s"
    )


# ---------------------------------------------------------------------------
# Entropy benchmark
# ---------------------------------------------------------------------------

def bench_entropy(
    word_list: list[str],
    first_guess: str | None = None,
    cache: bool = True,
) -> dict:
    solver = EntropySolver(cache=cache)
    wins = 0
    total = 0
    dist: dict[int, int] = {}
    t0 = time.time()

    for target in word_list:
        guesses = solver.solve(target, first_guess=first_guess)
        n = len(guesses)
        won = evaluate_guess(guesses[-1], target) == WIN_PATTERN
        dist[n] = dist.get(n, 0) + 1
        total += n
        if won:
            wins += 1

    elapsed = time.time() - t0
    return {
        "win_rate": wins / len(word_list),
        "avg_guesses": total / len(word_list),
        "distribution": dist,
        "elapsed": elapsed,
    }


# ---------------------------------------------------------------------------
# RL benchmark
# ---------------------------------------------------------------------------

def bench_rl(word_list: list[str]) -> dict:
    from stable_baselines3 import PPO
    import os
    from env import WordleEnv

    model_path = os.path.join(os.path.dirname(__file__), "models", "ppo_wordle")
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(
            f"No trained model at {model_path}.zip. Run train_rl.py first."
        )

    model = PPO.load(model_path)
    env = WordleEnv(shaped_reward=False)
    wins = 0
    total = 0
    dist: dict[int, int] = {}
    t0 = time.time()

    for target in word_list:
        # Force the environment to use this specific target
        env._target = target
        obs = np.zeros(60, dtype=np.float32)
        env._state = obs.copy()
        env._n_guesses = 0
        env._done = False

        done = False
        won = False
        n = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            n = info["n_guesses"]
            won = info["won"]

        dist[n if won else 7] = dist.get(n if won else 7, 0) + 1
        total += n
        if won:
            wins += 1

    elapsed = time.time() - t0
    return {
        "win_rate": wins / len(word_list),
        "avg_guesses": total / len(word_list),
        "distribution": dist,
        "elapsed": elapsed,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--words", type=int, default=0,
        help="number of words to sample (0 = all answers)"
    )
    parser.add_argument("--rl", action="store_true", help="also benchmark RL agent")
    parser.add_argument("--first-guess", default=None, dest="first_guess")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    word_list = ANSWERS if not args.words else rng.sample(ANSWERS, min(args.words, len(ANSWERS)))

    print(f"Benchmarking on {len(word_list)} words …\n")

    print("── Entropy Solver ──────────────────────")
    e_stats = bench_entropy(word_list, first_guess=args.first_guess)
    print(_fmt(e_stats))

    if args.rl:
        print("\n── RL Agent (PPO) ──────────────────────")
        try:
            r_stats = bench_rl(word_list)
            print(_fmt(r_stats))
        except FileNotFoundError as exc:
            print(f"  {exc}")


if __name__ == "__main__":
    main()
