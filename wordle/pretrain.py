"""
pretrain.py — Behavioural-cloning warm-start for the Wordle RL agent.

The entropy solver plays N games.  Every (observation, solver_action) pair is
recorded as an expert demonstration, then used to train the PPO policy network
via supervised NLL loss.

Demo generation is parallelised across CPU cores (--workers, default: all
available up to 8).  On an M-chip Mac this brings the wall-clock time from
~28 min down to ~5 min for the default 50 k games.

Train / test split
------------------
A fraction of target words (--test_frac, default 0.15) is held out and never
used as targets during demo generation.  After training, the policy is evaluated
separately on train targets and test targets.  If test ≈ train win-rate the
model is genuinely generalising from letter-knowledge patterns, not memorising
specific words.

Usage
-----
  python pretrain.py                       # 50k games, full 2315 vocab, 8 workers
  python pretrain.py --top_k 300          # smaller vocab, faster iteration
  python pretrain.py --workers 4          # cap CPU usage
  # Fine-tune after:
  python visualizer.py --curriculum
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from stable_baselines3 import PPO

from env import WordleEnv
from train_rl import MODEL_PATH


# ─── Fast solver builder ──────────────────────────────────────────────────────

def _make_solver(action_vocab: list[str], target_vocab: list[str]):
    """
    Build an EntropySolver for (action_vocab × target_vocab).
    Subsets the existing cache matrix instead of recomputing 4 M+ evaluate_guess calls.
    """
    from words import ANSWERS
    from solver_entropy import EntropySolver

    cache_mat = os.path.join(os.path.dirname(__file__), ".cache", "patterns.npy")
    cache_wrd = os.path.join(os.path.dirname(__file__), ".cache", "pattern_words.txt")

    if os.path.exists(cache_mat) and os.path.exists(cache_wrd):
        with open(cache_wrd) as f:
            row_words = f.read().split()
        full_mat = np.load(cache_mat)              # (12972, 2315)
        row_idx  = {w: i for i, w in enumerate(row_words)}
        col_idx  = {w: i for i, w in enumerate(ANSWERS)}

        r = [row_idx[w] for w in action_vocab if w in row_idx]
        c = [col_idx[w] for w in target_vocab if w in col_idx]

        if len(r) == len(action_vocab) and len(c) == len(target_vocab):
            solver = object.__new__(EntropySolver)
            solver.answers        = list(target_vocab)
            solver.guesses        = list(action_vocab)
            solver._ans_idx       = {w: i for i, w in enumerate(target_vocab)}
            solver._gue_idx       = {w: i for i, w in enumerate(action_vocab)}
            solver._matrix        = full_mat[np.ix_(r, c)]
            solver._possible_mask = np.ones(len(target_vocab), dtype=bool)
            return solver

    from solver_entropy import EntropySolver
    return EntropySolver(answers=target_vocab, guesses=action_vocab, cache=False)


# ─── Worker (top-level so it's picklable for multiprocessing spawn) ───────────

def _generate_chunk(args: tuple) -> tuple[np.ndarray, np.ndarray, int]:
    """Generate a chunk of expert games in a worker process."""
    n_games, action_vocab, train_targets, seed = args

    from env import WordleEnv
    env    = WordleEnv(answers=train_targets, action_words=action_vocab)
    solver = _make_solver(action_vocab, train_targets)
    w2i    = {w: i for i, w in enumerate(action_vocab)}

    obs_list: list[np.ndarray] = []
    act_list: list[int]        = []
    wins = 0
    rng  = np.random.default_rng(seed)

    for _ in range(n_games):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        solver.reset()
        done = False
        while not done:
            guess  = solver.best_guess()
            action = w2i.get(guess)
            if action is None:
                action = next((w2i[w] for w in solver.possible if w in w2i), 0)
            obs_list.append(obs.copy())
            act_list.append(action)
            obs, _, term, trunc, info = env.step(action)
            solver.update(info["guess"], info["pattern"])
            done = term or trunc
        if info["won"]:
            wins += 1

    return (
        np.array(obs_list, dtype=np.float32),
        np.array(act_list, dtype=np.int64),
        wins,
    )


# ─── Parallel demo generation ─────────────────────────────────────────────────

def generate_demos(
    n_games: int,
    action_vocab: list[str],
    train_targets: list[str],
    workers: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    t0 = time.time()

    if workers == 1:
        obs_arr, act_arr, wins = _generate_chunk(
            (n_games, action_vocab, train_targets, 42)
        )
    else:
        chunk   = n_games // workers
        rem     = n_games % workers
        job_args = [
            (chunk + (1 if i < rem else 0), action_vocab, train_targets, 42 + i)
            for i in range(workers)
        ]
        obs_parts, act_parts, wins = [], [], 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_generate_chunk, a): i for i, a in enumerate(job_args)}
            done = 0
            for fut in as_completed(futs):
                o, a, w = fut.result()
                obs_parts.append(o)
                act_parts.append(a)
                wins += w
                done += job_args[futs[fut]][0]
                print(f"  {done:,}/{n_games:,} games done  "
                      f"({time.time() - t0:.0f}s elapsed)", flush=True)
        obs_arr = np.concatenate(obs_parts)
        act_arr = np.concatenate(act_parts)

    print(f"Dataset: {len(obs_arr):,} (obs, action) pairs  "
          f"expert win rate {wins / n_games:.1%}  "
          f"({time.time() - t0:.0f}s)\n")
    return obs_arr, act_arr


# ─── Behavioural cloning ──────────────────────────────────────────────────────

def bc_train(
    obs_arr: np.ndarray,
    act_arr: np.ndarray,
    model: PPO,
    epochs: int     = 30,
    batch_size: int = 512,
    lr: float       = 1e-3,
) -> None:
    policy    = model.policy
    device    = policy.device
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    obs_t = torch.tensor(obs_arr, device=device)
    act_t = torch.tensor(act_arr, device=device)
    n     = len(obs_t)

    print(f"BC training: {n:,} samples  {epochs} epochs  batch {batch_size}  lr {lr}")
    t0 = time.time()

    for epoch in range(epochs):
        perm       = torch.randperm(n, device=device)
        total_loss = 0.0
        total_acc  = 0
        n_batches  = 0

        for start in range(0, n, batch_size):
            idx   = perm[start : start + batch_size]
            obs_b = obs_t[idx]
            act_b = act_t[idx]

            _, log_prob, _ = policy.evaluate_actions(obs_b, act_b)
            loss = -log_prob.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                pred, _, _ = policy.forward(obs_b, deterministic=True)
                total_acc += (pred == act_b).sum().item()

            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        print(f"  epoch {epoch + 1:2d}/{epochs}  "
              f"loss {total_loss / n_batches:.4f}  "
              f"acc {total_acc / n:.1%}  "
              f"lr {scheduler.get_last_lr()[0]:.1e}  "
              f"({time.time() - t0:.0f}s)")


# ─── Policy evaluation ────────────────────────────────────────────────────────

def evaluate_policy(
    model: PPO,
    action_vocab: list[str],
    target_vocab: list[str],
    n_games: int = 500,
    label: str = "",
) -> dict:
    env  = WordleEnv(answers=target_vocab, action_words=action_vocab)
    wins = 0
    total_guesses = 0
    dist: dict[int, int] = {}
    obs, _ = env.reset()

    for _ in range(n_games):
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(int(action))
            done = term or trunc
        n = info["n_guesses"]
        key = n if info["won"] else 7
        dist[key] = dist.get(key, 0) + 1
        if info["won"]:
            wins          += 1
            total_guesses += n
        obs, _ = env.reset()

    stats = {
        "win_rate":    wins / n_games,
        "avg_guesses": total_guesses / max(wins, 1),
        "distribution": {k: dist[k] for k in sorted(dist)},
    }
    tag = f"[{label}] " if label else ""
    print(f"  {tag}win {stats['win_rate']:.1%}  "
          f"avg {stats['avg_guesses']:.2f} guesses  "
          f"dist {stats['distribution']}")
    return stats


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_games",    type=int,   default=50_000)
    parser.add_argument("--top_k",      type=int,   default=0,
                        help="Action vocab size — top-K entropy answers (0 = full 2315)")
    parser.add_argument("--test_frac",  type=float, default=0.15,
                        help="Fraction of targets held out for generalisation test")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=512)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--eval_games", type=int,   default=500)
    parser.add_argument("--workers",    type=int,   default=0,
                        help="Parallel workers for demo generation (0 = all CPUs up to 8)")
    parser.add_argument("--seed",       type=int,   default=0)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    rng = np.random.default_rng(args.seed)

    n_workers = args.workers or min(os.cpu_count() or 1, 8)

    # ── Build vocab ────────────────────────────────────────────────────
    if args.top_k > 0:
        from solver_entropy import top_entropy_answers
        action_vocab = top_entropy_answers(args.top_k)
    else:
        from words import ANSWERS
        action_vocab = list(ANSWERS)

    # ── Train / test split ─────────────────────────────────────────────
    shuffled      = list(rng.permutation(action_vocab))
    n_test        = max(1, int(len(shuffled) * args.test_frac))
    test_targets  = shuffled[:n_test]
    train_targets = shuffled[n_test:]
    print(f"Vocab {len(action_vocab)}  →  "
          f"train targets {len(train_targets)}  |  "
          f"test targets {len(test_targets)}  (held out)\n")

    # ── Generate expert demos (train targets only, parallel) ───────────
    print(f"Generating {args.n_games:,} expert games  "
          f"({n_workers} workers) …")
    obs_arr, act_arr = generate_demos(
        args.n_games, action_vocab, train_targets, workers=n_workers
    )

    # ── Build fresh PPO model ──────────────────────────────────────────
    env = WordleEnv(answers=action_vocab, action_words=action_vocab)
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=0,
    )

    # ── Pre-train ──────────────────────────────────────────────────────
    bc_train(obs_arr, act_arr, model,
             epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    # ── Evaluate: train vs held-out test ───────────────────────────────
    print(f"\nEvaluating on {args.eval_games} games each …")
    evaluate_policy(model, action_vocab, train_targets,
                    n_games=args.eval_games, label="train targets")
    evaluate_policy(model, action_vocab, test_targets,
                    n_games=args.eval_games, label="test targets (held out)")

    # ── Save ───────────────────────────────────────────────────────────
    model.save(MODEL_PATH)
    print(f"\nSaved → {MODEL_PATH}.zip")
    tune_flag = f"--top_k {args.top_k}" if args.top_k else ""
    print(f"Fine-tune:  python visualizer.py --curriculum {tune_flag}".strip())


if __name__ == "__main__":
    main()
