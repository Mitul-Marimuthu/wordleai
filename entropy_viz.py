"""
entropy_viz.py – watch the entropy solver play Wordle live.

Usage
-----
  python entropy_viz.py                       # entropy only, 1s/guess
  python entropy_viz.py --delay 0.5           # faster
  python entropy_viz.py --word crane          # fix the target word
  python entropy_viz.py --compare             # entropy vs RL side by side

Controls
--------
  SPACE / RIGHT  – skip to next game
  ESC / Q        – quit
"""

import argparse
import os
import queue
import random
import threading
import time
from collections import deque

import pygame

from env import WordleEnv
from game import MAX_GUESSES, WIN_PATTERN, evaluate_guess
from solver_entropy import EntropySolver
from words import ANSWERS

# ─── Shared ─────────────────────────────────────────────────────────────────

_entropy_q: "queue.Queue[dict]" = queue.Queue(maxsize=300)
_rl_q:      "queue.Queue[dict]" = queue.Queue(maxsize=300)
_skip = threading.Event()

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "ppo_wordle")

# ─── Solver threads ──────────────────────────────────────────────────────────

def _entropy_loop(delay: float, fixed_word: str | None) -> None:
    solver = EntropySolver(cache=True)
    rng    = random.Random()
    while True:
        target = fixed_word if fixed_word else rng.choice(ANSWERS)
        solver.reset()
        game: list[tuple[str, str]] = []
        _entropy_q.put({"type": "new_game"})

        for _ in range(MAX_GUESSES):
            if _skip.is_set():
                break
            guess = solver.best_guess()
            time.sleep(delay)
            if _skip.is_set():
                break
            pattern = evaluate_guess(guess, target)
            game.append((guess, pattern))
            solver.update(guess, pattern)
            _entropy_q.put({"type": "step", "game": list(game),
                            "remaining": solver.n_possible})
            if pattern == WIN_PATTERN:
                break

        won = bool(game) and game[-1][1] == WIN_PATTERN
        _entropy_q.put({"type": "end", "target": target,
                        "n": len(game), "won": won})
        _skip.clear()
        time.sleep(delay)


def _load_rl_model():
    """Try MaskablePPO first, fall back to PPO. Returns (model, masked) or None."""
    if not os.path.exists(MODEL_PATH + ".zip"):
        return None
    try:
        from sb3_contrib import MaskablePPO
        return MaskablePPO.load(MODEL_PATH), True
    except Exception:
        pass
    try:
        from stable_baselines3 import PPO
        return PPO.load(MODEL_PATH), False
    except Exception:
        return None


def _rl_loop(delay: float, fixed_word: str | None) -> None:
    result = _load_rl_model()
    if result is None:
        _rl_q.put({"type": "no_model"})
        return
    model, masked = result

    rng = random.Random()
    while True:
        target = fixed_word if fixed_word else rng.choice(ANSWERS)
        env = WordleEnv(shaped_reward=False, use_mask=masked)
        obs, _ = env.reset()
        # override the random target with the shared one
        env._target = target
        if masked and env._possible is not None:
            env._possible[:] = True

        game: list[tuple[str, str]] = []
        _rl_q.put({"type": "new_game"})

        done = False
        while not done:
            if _skip.is_set():
                break
            time.sleep(delay)
            if _skip.is_set():
                break
            kwargs = {"action_masks": env.action_masks()} if masked else {}
            action, _ = model.predict(obs, deterministic=True, **kwargs)
            obs, _, term, trunc, info = env.step(int(action))
            done = term or trunc
            game.append((info["guess"], info["pattern"]))
            _rl_q.put({"type": "step", "game": list(game), "remaining": None})

        won = bool(game) and game[-1][1] == WIN_PATTERN
        _rl_q.put({"type": "end", "target": target,
                   "n": len(game), "won": won})
        _skip.clear()
        time.sleep(delay)


# ─── Colours ─────────────────────────────────────────────────────────────────

C = {
    "bg":      (18,  18,  19),
    "panel":   (26,  26,  27),
    "divider": (50,  50,  52),
    "correct": (83,  141, 78),
    "present": (181, 159, 59),
    "absent":  (58,  58,  60),
    "empty":   (26,  26,  27),
    "border":  (70,  70,  72),
    "text":    (255, 255, 255),
    "dim":     (110, 110, 110),
    "win_bar": (83,  141, 78),
    "los_bar": (155, 50,  50),
    "avg_ln":  (255, 200, 60),
    "title":   (200, 200, 200),
    "rem":     (100, 180, 240),
    "rl_col":  (200, 140, 255),
}

def _fb_color(ch: str) -> tuple:
    return {"+" : C["correct"], "x": C["present"], "-": C["absent"]}.get(ch, C["empty"])


# ─── Panel geometry ───────────────────────────────────────────────────────────
#
# Single mode  (--compare not set):
#   W=940  H=660   left=board(420px)  right=graph(520px)
#
# Compare mode (--compare):
#   W=1300 H=700   two panels each 650px wide
#   Each panel: board on top (BOARD_AREA_H px), graph below
#

STATUS_H    = 60

# single mode
W_S, H_S   = 940,  660
DIVIDER_S  = 420
CELL_S     = 56
GAP_S      = 7

# compare mode
W_C, H_C       = 1300, 720
PANEL_W        = W_C // 2           # 650
BOARD_AREA_H   = 390                # each panel: board lives here
GRAPH_AREA_H   = H_C - BOARD_AREA_H - STATUS_H  # 270
CELL_C         = 42
GAP_C          = 5


# ─── Generic draw helpers ─────────────────────────────────────────────────────

def _board_dims(cell, gap):
    bw = 5 * cell + 4 * gap
    bh = MAX_GUESSES * cell + (MAX_GUESSES - 1) * gap
    return bw, bh


def draw_board(surf, font_lg, font_md,
               game, game_num, target_reveal, remaining, last_won,
               px, pw, cell, gap, area_h, label="", label_color=None):
    """Render a Wordle board inside the panel whose left edge is `px`."""
    pygame.draw.rect(surf, C["panel"], (px, 0, pw, area_h))

    bw, bh = _board_dims(cell, gap)
    bx = px + (pw - bw) // 2
    by = 90

    # label (e.g. "ENTROPY SOLVER" / "RL AGENT")
    lc = label_color or C["title"]
    lbl = font_md.render(label, True, lc)
    surf.blit(lbl, lbl.get_rect(centerx=px + pw // 2, top=8))

    # game number
    gn = font_md.render(f"Game  {game_num:,}", True, C["dim"])
    surf.blit(gn, gn.get_rect(centerx=px + pw // 2, top=30))

    # result badge
    if last_won is not None and target_reveal:
        badge = "WIN" if last_won else "FAIL"
        col   = C["correct"] if last_won else C["los_bar"]
        b = font_md.render(f"{badge}  —  {target_reveal.upper()}", True, col)
        surf.blit(b, b.get_rect(centerx=px + pw // 2, top=52))

    # remaining words (entropy only)
    if remaining is not None and game:
        rt = font_md.render(
            f"{remaining} word{'s' if remaining != 1 else ''} possible",
            True, C["rem"])
        surf.blit(rt, rt.get_rect(centerx=px + pw // 2, top=72))

    # 6 × 5 grid
    for row in range(MAX_GUESSES):
        for col in range(5):
            x    = bx + col * (cell + gap)
            y    = by + row * (cell + gap)
            rect = pygame.Rect(x, y, cell, cell)
            if row < len(game):
                guess, pattern = game[row]
                pygame.draw.rect(surf, _fb_color(pattern[col]), rect, border_radius=4)
                ltr = font_lg.render(guess[col].upper(), True, C["text"])
                surf.blit(ltr, ltr.get_rect(center=rect.center))
            else:
                pygame.draw.rect(surf, C["empty"],  rect, border_radius=4)
                pygame.draw.rect(surf, C["border"], rect, 2, border_radius=4)

    # colour key
    kx = bx
    ky = by + bh + 14
    for key_label, col in (("correct (+)", C["correct"]),
                            ("present (x)", C["present"]),
                            ("absent  (-)", C["absent"])):
        pygame.draw.rect(surf, col, (kx, ky, 10, 10), border_radius=2)
        kl = font_md.render(key_label, True, C["dim"])
        surf.blit(kl, (kx + 14, ky - 1))
        kx += kl.get_width() + 20


def draw_graph(surf, font_md, font_sm,
               hist, avg, game_num, wins,
               px, py, pw, ph, label_color=None):
    """Render the episode-length bar chart inside (px, py, pw, ph)."""
    GX = px + 20
    GY = py + 10
    GW = pw - 40
    GH = ph - 20

    pygame.draw.rect(surf, C["panel"],
                     (px, py, pw, ph), border_radius=6)

    wr = f"{wins / max(game_num, 1):.1%} win  •  avg {avg:.2f}"
    lc = label_color or C["title"]
    t = font_md.render(wr, True, lc)
    surf.blit(t, (GX, GY + 2))

    IY = GY + 26
    IH = GH - 26

    if not hist:
        w = font_md.render("Waiting …", True, C["dim"])
        surf.blit(w, w.get_rect(centerx=GX + GW // 2, centery=IY + IH // 2))
        return

    MAX_VAL = 7
    n       = len(hist)
    spacing = GW / n
    bar_w   = max(2, min(16, int(spacing) - 1))

    for i, ep in enumerate(hist):
        val = ep["n"] if ep["won"] else MAX_GUESSES + 1
        bh  = int((val / MAX_VAL) * IH)
        bxx = GX + int(i * spacing)
        byy = IY + IH - bh
        pygame.draw.rect(surf, C["win_bar"] if ep["won"] else C["los_bar"],
                         (bxx, byy, bar_w, bh))

    avg_y = IY + IH - int((avg / MAX_VAL) * IH)
    pygame.draw.line(surf, C["avg_ln"], (GX, avg_y), (GX + GW, avg_y), 2)
    al = font_sm.render(f"avg {avg:.2f}", True, C["avg_ln"])
    surf.blit(al, (GX + GW - al.get_width() - 4, avg_y - al.get_height() - 2))

    for v in range(1, 8):
        ty   = IY + IH - int((v / MAX_VAL) * IH)
        tick = font_sm.render(str(v) if v < 7 else "F", True, C["dim"])
        surf.blit(tick, (GX - 16, ty - tick.get_height() // 2))
        pygame.draw.line(surf, C["dim"], (GX - 4, ty), (GX, ty), 1)


def draw_statusbar(surf, font_md, w, h, delay, compare):
    sy = h - STATUS_H
    pygame.draw.rect(surf, C["panel"], (0, sy, w, STATUS_H))
    pygame.draw.line(surf, C["border"], (0, sy), (w, sy), 1)
    mode = "Entropy vs RL" if compare else "Entropy solver"
    items = [
        (f"{mode}  •  {delay:.1f}s / guess", C["title"]),
        ("SPACE  skip",                       C["dim"]),
        ("ESC  quit",                         C["dim"]),
    ]
    x = 20
    for txt, col in items:
        s = font_md.render(txt, True, col)
        surf.blit(s, (x, sy + (STATUS_H - s.get_height()) // 2))
        x += s.get_width() + 50


# ─── State helpers ────────────────────────────────────────────────────────────

def _blank_state() -> dict:
    return {
        "game": [], "target": None, "remaining": None,
        "last_won": None, "game_num": 0, "wins": 0,
        "avg": 0.0, "hist": deque(maxlen=150), "_total": 0,
        "no_model": False,
    }


def _apply_msg(state: dict, msg: dict) -> None:
    t = msg["type"]
    if t == "new_game":
        state["game"] = []; state["target"] = None
        state["remaining"] = None; state["last_won"] = None
        state["game_num"] += 1
    elif t == "step":
        state["game"]      = msg["game"]
        state["remaining"] = msg.get("remaining")
    elif t == "end":
        state["target"]   = msg["target"]
        state["last_won"] = msg["won"]
        n = msg["n"]
        if msg["won"]:
            state["wins"] += 1
        state["hist"].append({"n": n, "won": msg["won"]})
        state["_total"] += n if msg["won"] else MAX_GUESSES + 1
        state["avg"] = state["_total"] / len(state["hist"])
    elif t == "no_model":
        state["no_model"] = True


def _drain(q, state):
    try:
        while True:
            _apply_msg(state, q.get_nowait())
    except queue.Empty:
        pass


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay",   type=float, default=1.0)
    parser.add_argument("--word",    default=None)
    parser.add_argument("--compare", action="store_true",
                        help="Show entropy solver and RL agent side by side")
    args = parser.parse_args()

    compare = args.compare
    W = W_C if compare else W_S
    H = H_C if compare else H_S

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(
        "Entropy Solver vs RL Agent" if compare else "Entropy Solver — live"
    )
    clock = pygame.time.Clock()

    try:
        font_lg = pygame.font.SysFont("Helvetica Neue", 28 if compare else 30, bold=True)
        font_md = pygame.font.SysFont("Helvetica Neue", 16 if compare else 17)
        font_sm = pygame.font.SysFont("Helvetica Neue", 12 if compare else 13)
    except Exception:
        font_lg = pygame.font.Font(None, 32 if compare else 34)
        font_md = pygame.font.Font(None, 19 if compare else 20)
        font_sm = pygame.font.Font(None, 15 if compare else 16)

    entropy_state = _blank_state()
    rl_state      = _blank_state()

    # Launch threads
    threading.Thread(target=_entropy_loop, args=(args.delay, args.word),
                     daemon=True).start()
    if compare:
        threading.Thread(target=_rl_loop, args=(args.delay, args.word),
                         daemon=True).start()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                if event.key in (pygame.K_SPACE, pygame.K_RIGHT):
                    _skip.set()

        _drain(_entropy_q, entropy_state)
        if compare:
            _drain(_rl_q, rl_state)

        screen.fill(C["bg"])

        if compare:
            cell, gap = CELL_C, GAP_C

            # Left panel — entropy
            draw_board(screen, font_lg, font_md,
                       entropy_state["game"], entropy_state["game_num"],
                       entropy_state["target"], entropy_state["remaining"],
                       entropy_state["last_won"],
                       px=0, pw=PANEL_W, cell=cell, gap=gap,
                       area_h=BOARD_AREA_H,
                       label="ENTROPY  SOLVER", label_color=C["rem"])

            draw_graph(screen, font_md, font_sm,
                       list(entropy_state["hist"]), entropy_state["avg"],
                       entropy_state["game_num"], entropy_state["wins"],
                       px=0, py=BOARD_AREA_H, pw=PANEL_W, ph=GRAPH_AREA_H,
                       label_color=C["rem"])

            # Vertical divider
            pygame.draw.line(screen, C["divider"],
                             (PANEL_W, 0), (PANEL_W, H - STATUS_H), 1)

            # Right panel — RL agent
            if rl_state["no_model"]:
                msg = font_md.render(
                    "No trained model found.  Run train_rl.py first.", True, C["dim"])
                screen.blit(msg, msg.get_rect(
                    centerx=PANEL_W + PANEL_W // 2, centery=H // 2))
            else:
                draw_board(screen, font_lg, font_md,
                           rl_state["game"], rl_state["game_num"],
                           rl_state["target"], rl_state["remaining"],
                           rl_state["last_won"],
                           px=PANEL_W, pw=PANEL_W, cell=cell, gap=gap,
                           area_h=BOARD_AREA_H,
                           label="RL  AGENT  (PPO)", label_color=C["rl_col"])

                draw_graph(screen, font_md, font_sm,
                           list(rl_state["hist"]), rl_state["avg"],
                           rl_state["game_num"], rl_state["wins"],
                           px=PANEL_W, py=BOARD_AREA_H,
                           pw=PANEL_W, ph=GRAPH_AREA_H,
                           label_color=C["rl_col"])

        else:
            # ── Single mode (original layout) ─────────────────────────────
            cell, gap = CELL_S, GAP_S

            # Board fills left panel
            draw_board(screen, font_lg, font_md,
                       entropy_state["game"], entropy_state["game_num"],
                       entropy_state["target"], entropy_state["remaining"],
                       entropy_state["last_won"],
                       px=0, pw=DIVIDER_S, cell=cell, gap=gap,
                       area_h=H - STATUS_H,
                       label="ENTROPY  SOLVER", label_color=C["rem"])

            # Graph fills right panel
            draw_graph(screen, font_md, font_sm,
                       list(entropy_state["hist"]), entropy_state["avg"],
                       entropy_state["game_num"], entropy_state["wins"],
                       px=DIVIDER_S, py=0,
                       pw=W - DIVIDER_S, ph=H - STATUS_H,
                       label_color=C["rem"])

        draw_statusbar(screen, font_md, W, H, args.delay, compare)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
