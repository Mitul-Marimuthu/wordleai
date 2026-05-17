# Wordle AI

A full reinforcement learning pipeline for Wordle — and its harder cousin Quordle — built from scratch. The project progresses from a hand-crafted entropy solver through behavioural cloning, PPO fine-tuning, curriculum learning, and a live pygame visualiser.

---

## Results

| Agent | Win Rate | Avg Guesses |
|---|---|---|
| Entropy solver | **100%** | ~3.5 |
| Unmasked PPO (no BC) | ~50–60% | — |
| Unmasked PPO + BC pretrain | ~77–87% | — |
| **MaskablePPO + BC + curriculum** | **~95%** | — |

---

## Project structure

```
wordleai/
  wordle/          — complete Wordle pipeline
  quordle/         — Quordle (4 simultaneous boards)
  myenv/           — shared virtual environment
  requirements.txt
```

---

## Phase 1 — Core game engine (`wordle/`)

### `game.py`
Two-pass `evaluate_guess` algorithm:
- **Green pass**: mark letters in the correct position (`+`)
- **Yellow pass**: mark remaining letters present elsewhere (`x`)
- Absent letters marked `-`

### `words.py`
Downloads and caches two word lists:
- **ANSWERS** — 2,315 real Wordle answers
- **GUESSES** — 10,657 valid guess words (12,972 total)

---

## Phase 2 — Entropy solver (`wordle/solver_entropy.py`)

Builds and caches a **12,972 × 2,315 pattern matrix** (one row per guess word, one column per answer). On each turn, picks the guess that maximises Shannon entropy over the remaining possible answers — guaranteed to collapse the answer set as fast as possible.

- First guess is always **"soare"** (highest first-guess entropy)
- Achieves **100% win rate** at **~3.5 average guesses**
- Pattern matrix is pre-computed once and reused across all other modules

---

## Phase 3 — Gymnasium environment (`wordle/env.py`)

`WordleEnv` wraps the game as a standard `gym.Env`.

**Observation (208 dims, float32):**
```
Dims   0–77  : 26 letters × 3 one-hot  [unknown | absent | present]
Dims  78–207 : 5 positions × 26 letters  (0=unknown, 0.5=eliminated, 1.0=green)
```
This encodes *what the agent knows*, not raw board pixels — the model reasons about letter constraints rather than memorising board patterns.

**Action space:** `Discrete(2315)` — index into the answer word list.

**Reward:**
- `+1` on win (+ shaped bonus `(MAX_GUESSES − n) / MAX_GUESSES`)
- `−1` on loss
- Dense per-step reward: `words_eliminated / vocab_size` — gives gradient signal on every guess

**Action masking** (`use_mask=True`): uses `sb3-contrib MaskablePPO`. After each guess, only words still consistent with all feedback remain valid. The effective action space collapses from 2,315 → ~300 → ~30 → ~3 across a typical game.

---

## Phase 4 — PPO training (`wordle/train_rl.py`)

Standard PPO baseline with configurable flags:

```bash
python train_rl.py --masked --curriculum
```

**Curriculum learning:** target vocabulary expands in stages `[50, 150, 400, 1000, 2315]`. The agent starts on 50 easy words; once win rate ≥ 80% the pool expands. All envs share a mutable reference to the same target list so they see the expansion immediately on the next reset — no environment rebuilds needed.

**Finding:** unmasked PPO plateaus at **77–87%** regardless of training time. The 2,315-wide action space is too large for random exploration to discover good strategies. Masking solves this.

---

## Phase 5 — Behavioural cloning pretrain (`wordle/pretrain.py`)

The entropy solver plays **50,000 games** as an expert. Every `(208-dim obs, action)` pair is recorded as a demonstration, then used to train the PPO policy network via NLL loss (`-log_prob.mean()`), equivalent to cross-entropy on the action logits.

**Key design decisions:**
- **Train/test split** (`--test_frac 0.15`): 15% of target words are held out and never seen during demo generation. Evaluating on both sets confirms the policy is genuinely learning letter-elimination strategy, not memorising specific words.
- **Parallel demo generation** (`--workers 8`): uses `ProcessPoolExecutor` with `spawn` (required on macOS). Drops wall-clock time from ~28 min to ~5 min on 8 cores.
- **Cache subsetting**: subsets the existing 12,972 × 2,315 pattern matrix with `np.ix_()` instead of recomputing 4M+ `evaluate_guess` calls.
- **Cosine-annealed LR** + gradient norm clipping at 1.0.

```bash
python pretrain.py                # 50k games, full 2315 vocab
python pretrain.py --top_k 300   # smaller vocab for faster iteration
```

After BC, the policy achieves **~85% win rate** before any RL fine-tuning, and generalises to held-out words.

---

## Phase 6 — Live visualiser (`wordle/visualizer.py`)

pygame window showing the current episode in real time alongside a rolling history of episode lengths.

**Features:**
- Left panel: 6×5 Wordle board with green/yellow/grey tile colours
- Right panel: bar chart of recent episode lengths (green = win, red = fail), rolling average line
- Stats bar: FAST/SLOW indicator, episode count, rolling average, win rate
- Dynamic speed control: `[−]` / `[+]` buttons or `−` / `=` keys (0–5s per guess)
- Auto slow-mode: once rolling average drops below 6 guesses the board slows to 1s/guess so it's human-readable
- **Graceful shutdown**: closing the window signals the training thread via a `threading.Event`; the `finally` block saves the model before exit

**Fine-tuning a BC model:** when an existing `ppo_wordle.zip` is detected, the LR is automatically lowered to `3e-5` (vs default `3e-4`) and `ent_coef` to `0.003` to preserve BC-learned weights.

```bash
cd wordle
python visualizer.py --curriculum --masked
```

---

## Phase 7 — Failure audit (`wordle/audit.py`)

Runs the trained agent deterministically on every answer word and clusters losses by the set of words still possible just before the final guess.

```bash
cd wordle
python audit.py
```

**Output:** ranked failure clusters — groups of words the agent couldn't distinguish (e.g. batch / catch / match / patch / watch). Identifies whether failures are:
- **Structural** (≥4 candidates remaining — unavoidable without lookahead)
- **Near-unavoidable** (≤2 candidates — 50/50 guess regardless of strategy)

Results are logged to `logs/audit_YYYY-MM-DD_HH-MM-SS.txt`.

---

## Phase 8 — Quordle (`quordle/`)

Quordle: solve **4 Wordle boards simultaneously** with a **shared guess pool** of 9 guesses. Each guess is applied to all 4 boards at once. Win only when all 4 boards are solved.

### `quordle/env.py` — `QuordleEnv`

**Observation (836 dims):**
```
Dims   0–831 : 4 × 208-dim letter-knowledge (one per board)
Dims 832–835 : 4 solved flags (1.0 = board done)
```

**Action masking:** union of possible words across all *unsolved* boards. A word is kept in the mask if it could still be the answer on at least one unsolved board.

**Reward:** `+1` per board solved (+ shaped bonus), `−1` if game ends with any board unsolved, dense elimination signal normalised per unsolved board.

**Reuses the Wordle pattern cache** (`../wordle/.cache/patterns.npy`) — no recomputation.

### `quordle/visualizer.py`

2×2 board grid with the same tile colours and stats bar as the Wordle visualiser.

- Green border appears around each board as it is solved
- Bar chart tracks boards solved per episode (0–4)
- Speed adjuster and graceful shutdown carried over from Wordle

### `quordle/curriculum.py`

Stages: **15 → 50 → 150 → 500 → 2,315** words. Advances when all-4-boards win rate ≥ 40% at the current stage. Vocab size shown live in the stats bar.

```bash
cd quordle
python visualizer.py --curriculum
```

---

## Setup

```bash
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

The pattern matrix (~55 MB) is built automatically on first run and cached to `wordle/.cache/`.

---

## Recommended workflow

```bash
# 1. BC pretrain (~5 min, restores 85% baseline)
cd wordle
python pretrain.py

# 2. PPO fine-tune with live view
python visualizer.py --curriculum --masked

# 3. Audit failures once converged
python audit.py

# 4. Quordle
cd ../quordle
python visualizer.py --curriculum
```
