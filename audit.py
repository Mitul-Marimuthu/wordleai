"""
audit.py — failure-case audit for the trained Wordle RL agent.

Runs the agent on every answer word exactly once (deterministic), records
every loss, then clusters failures by the set of words still possible just
before the final (losing) guess.

Usage
-----
  python audit.py               # full 2315-word bank
  python audit.py --top_k 300  # must match training vocab
  python audit.py --n 500      # random sample of N words
"""

import argparse
import datetime
import os
import sys
from collections import defaultdict

import numpy as np
from sb3_contrib import MaskablePPO

from env import WordleEnv
from train_rl import MODEL_PATH
from words import ANSWERS


def run_audit(words: list[str]) -> list[dict]:
    env = WordleEnv(
        answers=words,
        action_words=words,
        shaped_reward=False,
        dense_reward=False,
        use_mask=True,
    )

    path = MODEL_PATH + ".zip"
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model at {path}. Train first.")
    model = MaskablePPO.load(MODEL_PATH, env=env)

    losses = []
    wins = 0

    for i, target in enumerate(words):
        obs, _ = env.reset()
        env._target = target   # force specific target; obs is always "all unknown"

        done = False
        guesses, patterns, step_possibles = [], [], []

        while not done:
            # capture candidates BEFORE this guess
            step_possibles.append([w for w, k in zip(words, env._possible) if k])
            mask = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            obs, _, term, trunc, info = env.step(int(action))
            guesses.append(info["guess"])
            patterns.append(info["pattern"])
            done = term or trunc

        if info["won"]:
            wins += 1
        else:
            losses.append({
                "target":         target,
                "guesses":        guesses,
                "patterns":       patterns,
                "final_possible": step_possibles[-1],   # candidates before losing guess
                "n_guesses":      info["n_guesses"],
            })

        if (i + 1) % 500 == 0:
            print(f"  {i + 1:,}/{len(words):,} …")

    print(f"\nWin rate : {wins}/{len(words)} = {wins / len(words):.1%}")
    print(f"Losses   : {len(losses)}")
    return losses

# dummy commnet
def cluster_report(losses: list[dict], words: list[str], top_n: int = 20) -> None:
    clusters: dict[frozenset, list[dict]] = defaultdict(list)
    for loss in losses:
        key = frozenset(loss["final_possible"])
        clusters[key].append(loss)

    ranked = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)

    W = 62
    print(f"\n{'─' * W}")
    print(f"{'FAILURE CLUSTER REPORT':^{W}}")
    print(f"{'─' * W}")
    print(f"Total losses : {len(losses)}   Distinct clusters : {len(clusters)}")
    print(f"{'─' * W}\n")

    sym = {"+": "🟩", "x": "🟨", "-": "⬜"}

    for rank, (word_set, records) in enumerate(ranked[:top_n], 1):
        candidates = sorted(word_set)
        targets    = sorted(set(r["target"] for r in records))

        print(f"Cluster #{rank}  "
              f"({len(records)} loss{'es' if len(records) != 1 else ''},  "
              f"{len(candidates)} candidates)")
        print(f"  Candidates : {' / '.join(candidates)}")
        print(f"  Lost on    : {' '.join(targets)}")

        ex = records[0]
        print(f"  Example    : target = {ex['target'].upper()}")
        for g, p in zip(ex["guesses"], ex["patterns"]):
            print(f"               {g.upper()}  {''.join(sym[c] for c in p)}")
        print()

    # ── summary stats ──────────────────────────────────────────────────────
    print(f"{'─' * W}")
    solo  = sum(1 for ws in clusters if len(ws) <= 2)
    multi = sum(len(rs) for ws, rs in clusters.items() if len(ws) >= 4)
    worst = sum(len(rs) for _, rs in ranked[:5])
    print(f"Clusters with ≤2 candidates (near-unavoidable): {solo}")
    print(f"Losses from ≥4-candidate clusters             : {multi}  "
          f"({multi / max(len(losses), 1):.0%} of all losses)")
    print(f"Losses from top-5 clusters                    : {worst}  "
          f"({worst / max(len(losses), 1):.0%} of all losses)")

    # ── guess-count distribution of losses ─────────────────────────────────
    dist: dict[int, int] = {}
    for loss in losses:
        n = loss["n_guesses"]
        dist[n] = dist.get(n, 0) + 1
    print(f"\nLoss distribution by guess count:")
    for k in sorted(dist):
        print(f"  {k} guesses : {dist[k]}")


class _Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, path: str) -> None:
        self._file = open(path, "w", encoding="utf-8")
        self._stdout = sys.stdout
        sys.stdout = self

    def write(self, data: str) -> None:
        self._stdout.write(data)
        self._file.write(data)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def close(self) -> None:
        sys.stdout = self._stdout
        self._file.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=0,
                        help="Must match training vocab (0 = full 2315)")
    parser.add_argument("--n",     type=int, default=0,
                        help="Random sample size (0 = run every word exactly once)")
    args = parser.parse_args()

    if args.top_k > 0:
        from solver_entropy import top_entropy_answers
        words = top_entropy_answers(args.top_k)
    else:
        words = list(ANSWERS)

    if args.n > 0:
        import random
        words = random.sample(words, min(args.n, len(words)))

    os.makedirs("logs", exist_ok=True)
    stamp   = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logpath = os.path.join("logs", f"audit_{stamp}.txt")
    tee = _Tee(logpath)
    print(f"Logging to {logpath}\n")

    try:
        print(f"Running failure audit on {len(words):,} words …\n")
        losses = run_audit(words)

        if not losses:
            print("No losses — 100% win rate!")
        else:
            cluster_report(losses, words)
    finally:
        tee.close()
        print(f"\nLog saved → {logpath}", file=sys.stdout)


if __name__ == "__main__":
    main()
