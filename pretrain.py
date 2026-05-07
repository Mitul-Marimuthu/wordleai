"""
pretrain.py — Behavioural-cloning warm-start for the Wordle RL agent.

The entropy solver plays N games.  Every (observation, solver_action) pair is
recorded as an expert demonstration, then used to train the PPO policy network
via supervised NLL loss (equivalent to cross-entropy over the action logits).

The resulting model is saved to models/ppo_wordle.zip and is directly
compatible with visualizer.py / train_rl.py for PPO fine-tuning.

Usage
-----
  python pretrain.py                          # 30 k games, top-150 vocab
  python pretrain.py --n_games 50000 --epochs 30
  python pretrain.py --top_k 0               # full 2315-word vocab
  # then fine-tune with the live visualiser:
  python visualizer.py --curriculum --top_k 150
"""

import argparse
import os
import time

import numpy as np
import torch
from stable_baselines3 import PPO

from env import WordleEnv
from train_rl import MODEL_PATH


# ─── Expert demonstration generation ─────────────────────────────────────────

def generate_demos(
    n_games: int,
    vocab: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Play n_games with the entropy solver on the given vocab.
    Returns (observations, expert_actions) as numpy arrays.
    """
    from solver_entropy import EntropySolver

    full_vocab = len(vocab) == 2315   # skip rebuilding matrix if using defaults
    env    = WordleEnv(answers=vocab, action_words=vocab)
    solver = EntropySolver(answers=vocab, guesses=vocab, cache=full_vocab)

    word_to_idx = {w: i for i, w in enumerate(vocab)}

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
            # Fallback: solver picked a word outside our action space
            if action is None:
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

            # NLL loss (behavioural cloning objective)
            _, log_prob, _ = policy.evaluate_actions(obs_b, act_b)
            loss = -log_prob.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            # Accuracy: compare deterministic actions to expert
            with torch.no_grad():
                pred_actions, _, _ = policy.forward(obs_b, deterministic=True)
                total_acc += (pred_actions == act_b).sum().item()

            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = total_loss / n_batches
        accuracy = total_acc / n
        print(f"  epoch {epoch + 1:2d}/{epochs}  "
              f"loss {avg_loss:.4f}  "
              f"acc {accuracy:.1%}  "
              f"lr {scheduler.get_last_lr()[0]:.1e}  "
              f"({time.time() - t0:.0f}s)")


# ─── Policy evaluation ────────────────────────────────────────────────────────

def evaluate_policy(model: PPO, vocab: list[str], n_games: int = 500) -> dict:
    env  = WordleEnv(answers=vocab, action_words=vocab)
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

    return {
        "win_rate":    wins / n_games,
        "avg_guesses": total_guesses / max(wins, 1),
        "distribution": {k: dist[k] for k in sorted(dist)},
    }


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_games",    type=int,   default=30_000,
                        help="Expert games to generate")
    parser.add_argument("--top_k",      type=int,   default=150,
                        help="Vocab size — top-K entropy answers (0 = full 2315)")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch_size", type=int,   default=512)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--eval_games", type=int,   default=500)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # ── Build vocab ────────────────────────────────────────────────────
    if args.top_k > 0:
        from solver_entropy import top_entropy_answers
        vocab = top_entropy_answers(args.top_k)
    else:
        from words import ANSWERS
        vocab = list(ANSWERS)

    # ── Generate expert demos ──────────────────────────────────────────
    print(f"Generating {args.n_games:,} expert games  "
          f"(vocab size {len(vocab)}) …")
    obs_arr, act_arr = generate_demos(args.n_games, vocab)

    # ── Build fresh PPO model (architecture only; weights random) ──────
    env = WordleEnv(answers=vocab, action_words=vocab)
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

    # ── Evaluate ───────────────────────────────────────────────────────
    print(f"\nEvaluating pre-trained policy on {args.eval_games} games …")
    stats = evaluate_policy(model, vocab, n_games=args.eval_games)
    print(f"Win rate    : {stats['win_rate']:.1%}")
    print(f"Avg guesses : {stats['avg_guesses']:.2f}  (wins only)")
    print(f"Distribution: {stats['distribution']}")

    # ── Save ───────────────────────────────────────────────────────────
    model.save(MODEL_PATH)
    print(f"\nSaved → {MODEL_PATH}.zip")
    print(f"Fine-tune:  python visualizer.py --curriculum --top_k {args.top_k}")


if __name__ == "__main__":
    main()
