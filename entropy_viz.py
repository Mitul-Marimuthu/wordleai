"""
entropy_viz.py – watch the entropy solver play Wordle live.

Usage
-----
  python entropy_viz.py                       # entropy only, 1s/guess
  python entropy_viz.py --delay 0.5           # faster
  python entropy_viz.py --word crane          # fix the target word
  python entropy_viz.py --compare             # entropy vs RL side by side
  python entropy_viz.py --graphs              # live statistics graphs (full speed)

Controls
--------
  SPACE / RIGHT  – skip to next game  (simulation modes)
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


# ─── Graph-mode state & helpers ───────────────────────────────────────────────

def _blank_graph_state() -> dict:
    return {
        "game_num": 0,
        "wins":     0,
        "rolling":  deque(maxlen=50),
        "wins_series": [],   # [(game_num, cumulative_wins)]
        "avg_series":  [],   # [(game_num, rolling_avg)]
        "no_model": False,
    }


def _apply_graph_msg(gst: dict, msg: dict) -> None:
    t = msg["type"]
    if t == "end":
        gst["game_num"] += 1
        won = msg["won"]
        n   = msg["n"]
        if won:
            gst["wins"] += 1
        gst["rolling"].append(n if won else MAX_GUESSES + 1)
        avg = sum(gst["rolling"]) / len(gst["rolling"])
        gst["wins_series"].append((gst["game_num"], gst["wins"]))
        gst["avg_series"].append((gst["game_num"], avg))
    elif t == "no_model":
        gst["no_model"] = True


def _drain_graph(q: queue.Queue, gst: dict) -> None:
    try:
        while True:
            _apply_graph_msg(gst, q.get_nowait())
    except queue.Empty:
        pass


def _subsample(pts: list, max_pts: int = 800) -> list:
    """Thin a point list to at most max_pts evenly-spaced entries."""
    n = len(pts)
    if n <= max_pts:
        return pts
    idxs = [int(i * (n - 1) / (max_pts - 1)) for i in range(max_pts)]
    return [pts[i] for i in idxs]


def draw_line_chart(
    surf,
    font_md,
    font_sm,
    title: str,
    y_min: float,
    y_max: float,
    y_ticks: list,
    series: list,          # [(label, color, [(x, y), ...])]
    gx: int, gy: int, gw: int, gh: int,
    x_max: int | None = None,
    ref_lines: list | None = None,  # [(y_val, color, label)]
) -> None:
    """Render a multi-series line chart inside (gx, gy, gw, gh)."""

    pygame.draw.rect(surf, C["panel"], (gx, gy, gw, gh), border_radius=6)

    # title
    ts = font_md.render(title, True, C["title"])
    surf.blit(ts, (gx + 10, gy + 8))

    # legend (top-right, right-to-left)
    lx = gx + gw - 12
    for lbl, col, _ in reversed(series):
        ls = font_sm.render(lbl, True, col)
        lx -= ls.get_width()
        surf.blit(ls, (lx, gy + 10))
        pygame.draw.line(surf, col, (lx - 22, gy + 17), (lx - 5, gy + 17), 2)
        lx -= 30

    # inner plot margins
    IX = gx + 58
    IY = gy + 34
    IW = gw - 70
    IH = gh - 52

    pygame.draw.rect(surf, (22, 22, 24), (IX, IY, IW, IH))

    # determine x range
    all_xs = [x for _, _, pts in series for x, _ in pts]
    if not all_xs and x_max is None:
        msg = font_md.render("Waiting for data …", True, C["dim"])
        surf.blit(msg, msg.get_rect(centerx=IX + IW // 2, centery=IY + IH // 2))
        pygame.draw.rect(surf, C["border"], (IX, IY, IW, IH), 1)
        return

    xmax = x_max if x_max is not None else (max(all_xs) if all_xs else 1)
    xmax = max(xmax, 1)

    def sx(x: float) -> int:
        return IX + int(x / xmax * IW)

    def sy(y: float) -> int:
        span = y_max - y_min
        if span == 0:
            return IY + IH // 2
        return IY + IH - int((y - y_min) / span * IH)

    # y grid + labels
    for tick in y_ticks:
        ty = sy(tick)
        if IY - 1 <= ty <= IY + IH + 1:
            pygame.draw.line(surf, (38, 38, 42), (IX, ty), (IX + IW, ty), 1)
            tl = font_sm.render(
                f"{tick:.0%}" if y_max <= 1.0 else str(tick),
                True, C["dim"],
            )
            surf.blit(tl, (IX - tl.get_width() - 5, ty - tl.get_height() // 2))

    # x grid + labels (5 ticks)
    n_xticks = 5
    x_step   = max(1, xmax // n_xticks)
    for xi in range(0, int(xmax) + 1, x_step):
        tx = sx(xi)
        pygame.draw.line(surf, (38, 38, 42), (tx, IY), (tx, IY + IH), 1)
        xl = font_sm.render(str(xi), True, C["dim"])
        surf.blit(xl, (tx - xl.get_width() // 2, IY + IH + 4))

    # x-axis label
    xl_lbl = font_sm.render("games played", True, C["dim"])
    surf.blit(xl_lbl, (IX + IW - xl_lbl.get_width(), IY + IH + 4))

    # reference lines (dashed-ish via short segments)
    for y_val, col, rlbl in (ref_lines or []):
        ry = sy(y_val)
        if IY <= ry <= IY + IH:
            for dx in range(0, IW, 12):
                pygame.draw.line(surf, col,
                                 (IX + dx, ry),
                                 (IX + min(dx + 7, IW), ry), 1)
            rls = font_sm.render(rlbl, True, col)
            surf.blit(rls, (IX + 6, ry - rls.get_height() - 2))

    # data series
    for lbl, col, pts in series:
        pts = _subsample(pts)
        if len(pts) < 2:
            continue
        screen_pts = [
            (max(IX, min(IX + IW, sx(x))),
             max(IY, min(IY + IH, sy(y))))
            for x, y in pts
        ]
        pygame.draw.lines(surf, col, False, screen_pts, 2)

    pygame.draw.rect(surf, C["border"], (IX, IY, IW, IH), 1)


def draw_graph_statusbar(surf, font_md, w, h,
                          e_games, e_wins, r_games, r_wins) -> None:
    sy = h - STATUS_H
    pygame.draw.rect(surf, C["panel"], (0, sy, w, STATUS_H))
    pygame.draw.line(surf, C["border"], (0, sy), (w, sy), 1)

    items = [
        ("GRAPHS  mode  —  full speed", C["title"]),
        (f"Entropy  {e_games:,} games  {e_wins/max(e_games,1):.1%} win",  C["rem"]),
        (f"RL Agent  {r_games:,} games  {r_wins/max(r_games,1):.1%} win", C["rl_col"]),
        ("ESC  quit", C["dim"]),
    ]
    x = 20
    for txt, col in items:
        s = font_md.render(txt, True, col)
        surf.blit(s, (x, sy + (STATUS_H - s.get_height()) // 2))
        x += s.get_width() + 40


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay",   type=float, default=1.0)
    parser.add_argument("--word",    default=None)
    parser.add_argument("--compare", action="store_true",
                        help="Entropy vs RL side-by-side boards + per-model bar charts")
    parser.add_argument("--graphs",  action="store_true",
                        help="Live stats graphs: wins over time + avg guesses (full speed, no boards)")
    args = parser.parse_args()

    graphs  = args.graphs
    compare = args.compare and not graphs   # --graphs takes priority

    W = 1060 if graphs else (W_C if compare else W_S)
    H = 700  if graphs else (H_C if compare else H_S)

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(
        "Wordle — Statistics Graphs" if graphs else
        "Entropy Solver vs RL Agent" if compare else
        "Entropy Solver — live"
    )
    clock = pygame.time.Clock()

    try:
        font_lg = pygame.font.SysFont("Helvetica Neue", 28 if compare else 30, bold=True)
        font_md = pygame.font.SysFont("Helvetica Neue", 16)
        font_sm = pygame.font.SysFont("Helvetica Neue", 12)
    except Exception:
        font_lg = pygame.font.Font(None, 32 if compare else 34)
        font_md = pygame.font.Font(None, 19)
        font_sm = pygame.font.Font(None, 15)

    # ── Launch solver threads ──────────────────────────────────────────────
    delay = 0.0 if graphs else args.delay
    threading.Thread(target=_entropy_loop, args=(delay, args.word),
                     daemon=True).start()
    if compare or graphs:
        threading.Thread(target=_rl_loop, args=(delay, args.word),
                         daemon=True).start()

    # ── Per-mode state ─────────────────────────────────────────────────────
    entropy_state = _blank_state()
    rl_state      = _blank_state()
    entropy_gst   = _blank_graph_state()
    rl_gst        = _blank_graph_state()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                if event.key in (pygame.K_SPACE, pygame.K_RIGHT) and not graphs:
                    _skip.set()

        screen.fill(C["bg"])

        # ── GRAPHS MODE ───────────────────────────────────────────────────
        if graphs:
            _drain_graph(_entropy_q, entropy_gst)
            _drain_graph(_rl_q, rl_gst)

            eg = entropy_gst
            rg = rl_gst

            # shared x_max so both lines are on the same scale
            x_max = max(eg["game_num"], rg["game_num"], 1)

            PAD  = 20
            MIDH = (H - STATUS_H) // 2

            # ── Graph 1: Games Won vs Games Played ────────────────────────
            y_max_wins = x_max
            n_ticks    = 6
            tick_step  = max(1, y_max_wins // n_ticks)
            y_ticks_w  = list(range(0, y_max_wins + tick_step, tick_step))

            draw_line_chart(
                screen, font_md, font_sm,
                title    = "Games Won vs Games Played",
                y_min    = 0,
                y_max    = max(y_max_wins, 1),
                y_ticks  = y_ticks_w,
                series   = [
                    ("Entropy Solver", C["rem"],    eg["wins_series"]),
                    ("RL Agent",       C["rl_col"], rg["wins_series"]),
                ],
                gx=PAD, gy=PAD,
                gw=W - 2*PAD, gh=MIDH - PAD,
                x_max=x_max,
                ref_lines=None,
            )

            # ── Graph 2: Rolling Average Guesses ──────────────────────────
            draw_line_chart(
                screen, font_md, font_sm,
                title    = "Rolling Average Guesses per Game  (window 50,  7 = fail)",
                y_min    = 1.0,
                y_max    = 7.0,
                y_ticks  = [1, 2, 3, 4, 5, 6, 7],
                series   = [
                    ("Entropy Solver", C["rem"],    eg["avg_series"]),
                    ("RL Agent",       C["rl_col"], rg["avg_series"]),
                ],
                gx=PAD, gy=MIDH + PAD,
                gw=W - 2*PAD, gh=MIDH - PAD,
                x_max=x_max,
                ref_lines=[(3.5, C["avg_ln"], "entropy benchmark ≈ 3.5")],
            )

            draw_graph_statusbar(screen, font_md, W, H,
                                 eg["game_num"], eg["wins"],
                                 rg["game_num"], rg["wins"])

        # ── COMPARE MODE ──────────────────────────────────────────────────
        elif compare:
            _drain(_entropy_q, entropy_state)
            _drain(_rl_q, rl_state)
            cell, gap = CELL_C, GAP_C

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

            pygame.draw.line(screen, C["divider"],
                             (PANEL_W, 0), (PANEL_W, H - STATUS_H), 1)

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
            draw_statusbar(screen, font_md, W, H, args.delay, compare)

        # ── SINGLE MODE ───────────────────────────────────────────────────
        else:
            _drain(_entropy_q, entropy_state)
            cell, gap = CELL_S, GAP_S
            draw_board(screen, font_lg, font_md,
                       entropy_state["game"], entropy_state["game_num"],
                       entropy_state["target"], entropy_state["remaining"],
                       entropy_state["last_won"],
                       px=0, pw=DIVIDER_S, cell=cell, gap=gap,
                       area_h=H - STATUS_H,
                       label="ENTROPY  SOLVER", label_color=C["rem"])
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
