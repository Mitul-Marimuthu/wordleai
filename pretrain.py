"""
pretrain.py — Behavioural-cloning warm-start for the Wordle RL agent.

The entropy solver plays N games.  Every (observation, solver_action) pair is
recorded as an expert demonstration, then used to train the PPO policy network
via supervised NLL loss.

Train / test split
------------------
A fraction of target words (--test_frac, default 0.15) is held out and never
used as targets during demo generation.  After training, the policy is evaluated
separately on train targets and test targets.  If the test win-rate is close to
the train win-rate, the model is genuinely generalising from letter-knowledge
patterns rather than memorising specific words.

The 208-dim observation encodes what the agent *knows* (which letters are
absent / present / confirmed at each position) — not which specific word it is
looking for.  A policy that learns the strategy should transfer to unseen words
because the letter-knowledge state looks the same regardless of which target
is hidden.

Usage
-----
  # Full 2315-word game — meaningful generalisation test
  python pretrain.py --top_k 0 --n_games 50000 --epochs 30

  # Faster experiment on a smaller vocab
  python pretrain.py --top_k 300 --n_games 30000

  # Fine-tune after pretraining
  python visualizer.py --curriculum
"""

import argparse
import os
import time

import numpy as np
import torch
from stable_baselines3 import PPO

from env import WordleEnv
from train_rl import MODEL_PATH


# ─── Fast solver builder (uses cached pattern matrix) ────────────────────────

def _make_solver(action_vocab: list[str], target_vocab: list[str]):
    """
    Build an EntropySolver for (action_vocab × target_vocab) efficiently.
    Subsets the existing cache rather than recomputing ~4 M evaluate_guess calls.
    Falls back to computing from scratch if the cache is absent.
    """
    import numpy as np
    from words import ANSWERS
    from solver_entropy import EntropySolver

    cache_mat = os.path.join(os.path.dirname(__file__), ".cache", "patterns.npy")
    cache_wrd = os.path.join(os.path.dirname(__file__), ".cache", "pattern_words.txt")

    if os.path.exists(cache_mat) and os.path.exists(cache_wrd):
        with open(cache_wrd) as f:
            row_words = f.read().split()
        full_mat   = np.load(cache_mat)              # (12972, 2315)
        row_idx    = {w: i for i, w in enumerate(row_words)}
        col_idx    = {w: i for i, w in enumerate(ANSWERS)}

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

    return EntropySolver(answers=target_vocab, guesses=action_vocab, cache=False)


# ─── Expert demonstration generation ─────────────────────────────────────────

def generate_demos(
    n_games: int,
    action_vocab: list[str],
    train_targets: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Play n_games where the entropy solver guesses from action_vocab and targets
    are drawn only from train_targets.  Returns (observations, expert_actions).
    """
    env    = WordleEnv(answers=train_targets, action_words=action_vocab)
    solver = _make_solver(action_vocab, train_targets)

    word_to_idx = {w: i for i, w in enumerate(action_vocab)}

    obs_list: list[np.ndarray] = []
    act_list: list[int]        = []
    wins = 0
    rng  = np.random.default_rng(42)
    t0   = time.time()

    for g in range(n_games):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        solver.reset()
        done = False

        while not done:
            guess  = solver.best_guess()
            action = word_to_idx.get(guess)
            if action is None:           # solver went outside action vocab (shouldn't happen)
                for w in solver.possible:
                    if w in word_to_idx:
                        action = word_to_idx[w]
                        break
                if action is None:
                    action = 0

            obs_list.append(obs.copy())
            act_list.append(action)

            obs, _, term, trunc, info = env.step(action)
            solver.update(info["guess"], info["pattern"])
            done = term or trunc

        if info["won"]:
            wins += 1

        if (g + 1) % 5_000 == 0:
            print(f"  {g + 1:,}/{n_games:,}  win {wins / (g + 1):.1%}  "
                  f"({time.time() - t0:.0f}s)")

    obs_arr = np.array(obs_list, dtype=np.float32)
    act_arr = np.array(act_list, dtype=np.int64)
    print(f"Dataset: {len(obs_arr):,} (obs, action) pairs  "
          f"expert win rate {wins / n_games:.1%}\n")
    return obs_arr, act_arr


# ─── Behavioural cloning ──────────────────────────────────────────────────────

def bc_train(
    obs_arr: np.ndarray,
    act_arr: np.ndarray,
    model: PPO,
    epochs: int     = 20,
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

    print(f"BC training: {n:,} samples  {epochs} epochs  "
          f"batch {batch_size}  lr {lr}")
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
                pred_actions, _, _ = policy.forward(obs_b, deterministic=True)
                total_acc += (pred_actions == act_b).sum().item()

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
    print(f"  {tag}win rate {stats['win_rate']:.1%}  "
          f"avg guesses {stats['avg_guesses']:.2f}  "
          f"dist {stats['distribution']}")
    return stats


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_games",    type=int,   default=50_000,
                        help="Expert games to generate (all from train targets)")
    parser.add_argument("--top_k",      type=int,   default=0,
                        help="Action vocab size — top-K entropy answers (0 = full 2315)")
    parser.add_argument("--test_frac",  type=float, default=0.15,
                        help="Fraction of targets held out for generalisation test")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=512)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--eval_games", type=int,   default=500)
    parser.add_argument("--seed",       type=int,   default=0)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ── Build action vocab ─────────────────────────────────────────────
    if args.top_k > 0:
        from solver_entropy import top_entropy_answers
        action_vocab = top_entropy_answers(args.top_k)
    else:
        from words import ANSWERS
        action_vocab = list(ANSWERS)

    # ── Train / test split on targets ──────────────────────────────────
    shuffled      = list(rng.permutation(action_vocab))
    n_test        = max(1, int(len(shuffled) * args.test_frac))
    test_targets  = shuffled[:n_test]
    train_targets = shuffled[n_test:]
    print(f"Vocab {len(action_vocab)}  →  "
          f"train targets {len(train_targets)}  |  "
          f"test targets {len(test_targets)}  (held out)\n")

    # ── Generate expert demos (train targets only) ─────────────────────
    print(f"Generating {args.n_games:,} expert games on train targets …")
    obs_arr, act_arr = generate_demos(args.n_games, action_vocab, train_targets)

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

    # ── Pre-train via behavioural cloning ──────────────────────────────
    bc_train(obs_arr, act_arr, model,
             epochs=args.epochs,
             batch_size=args.batch_size,
             lr=args.lr)

    # ── Evaluate: train targets vs held-out test targets ───────────────
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
