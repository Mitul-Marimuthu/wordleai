"""
entropy_viz.py – watch the entropy solver play Wordle live.

Usage
-----
  python entropy_viz.py              # 1 second between guesses
  python entropy_viz.py --delay 0.5  # faster
  python entropy_viz.py --delay 2    # slower
  python entropy_viz.py --word crane # solve a specific word, then loop

Controls
--------
  SPACE / RIGHT  – skip to next game immediately
  ESC / Q        – quit
"""

import argparse
import queue
import random
import threading
import time
from collections import deque

import pygame

from game import MAX_GUESSES, WIN_PATTERN, evaluate_guess
from solver_entropy import EntropySolver
from words import ANSWERS

# ─── Shared ─────────────────────────────────────────────────────────────────

_q: "queue.Queue[dict]" = queue.Queue(maxsize=300)
_skip = threading.Event()   # set by UI to jump to next game

# ─── Solver thread ───────────────────────────────────────────────────────────

def _solve_loop(delay: float, fixed_word: str | None) -> None:
    solver = EntropySolver(cache=True)
    rng    = random.Random()

    while True:
        target = fixed_word if fixed_word else rng.choice(ANSWERS)
        solver.reset()
        game:  list[tuple[str, str]] = []

        _q.put({"type": "new_game", "target": "?????"})

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

            _q.put({
                "type":      "step",
                "game":      list(game),
                "remaining": solver.n_possible,
            })

            if pattern == WIN_PATTERN:
                break

        won = bool(game) and game[-1][1] == WIN_PATTERN
        _q.put({
            "type":   "end",
            "target": target,
            "n":      len(game),
            "won":    won,
        })

        _skip.clear()
        time.sleep(delay)          # brief pause between games


# ─── Layout / colours ────────────────────────────────────────────────────────

W, H     = 940, 660
DIVIDER  = 420

CELL = 56
GAP  = 7
BOARD_W = 5 * CELL + 4 * GAP
BOARD_H = MAX_GUESSES * CELL + (MAX_GUESSES - 1) * GAP
BOARD_X = (DIVIDER - BOARD_W) // 2
BOARD_Y = 100

C = {
    "bg":      (18,  18,  19),
    "panel":   (26,  26,  27),
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
}


def _fb_color(ch: str) -> tuple:
    return {"+" : C["correct"], "x": C["present"], "-": C["absent"]}.get(ch, C["empty"])


# ─── Draw helpers ────────────────────────────────────────────────────────────

def draw_board(surf, font_lg, font_md,
               game: list[tuple[str, str]],
               game_num: int,
               target_reveal: str | None,
               remaining: int | None,
               last_won: bool | None) -> None:

    pygame.draw.rect(surf, C["panel"], (0, 0, DIVIDER, H - 60))

    # header
    header = font_md.render(f"Game  {game_num:,}", True, C["title"])
    surf.blit(header, header.get_rect(centerx=DIVIDER // 2, top=12))

    # result badge
    if last_won is not None and target_reveal:
        badge = "WIN" if last_won else "FAIL"
        col   = C["correct"] if last_won else C["los_bar"]
        b = font_md.render(f"{badge}  —  {target_reveal.upper()}", True, col)
        surf.blit(b, b.get_rect(centerx=DIVIDER // 2, top=36))

    # possible-words counter
    if remaining is not None and game:
        rem_txt = font_md.render(f"{remaining} word{'s' if remaining != 1 else ''} still possible", True, C["rem"])
        surf.blit(rem_txt, rem_txt.get_rect(centerx=DIVIDER // 2, top=62))

    # 6 × 5 board
    for row in range(MAX_GUESSES):
        for col in range(5):
            x    = BOARD_X + col * (CELL + GAP)
            y    = BOARD_Y + row * (CELL + GAP)
            rect = pygame.Rect(x, y, CELL, CELL)
            if row < len(game):
                guess, pattern = game[row]
                pygame.draw.rect(surf, _fb_color(pattern[col]), rect, border_radius=4)
                ltr = font_lg.render(guess[col].upper(), True, C["text"])
                surf.blit(ltr, ltr.get_rect(center=rect.center))
            else:
                pygame.draw.rect(surf, C["empty"],  rect, border_radius=4)
                pygame.draw.rect(surf, C["border"], rect, 2, border_radius=4)

    # colour key
    kx  = BOARD_X
    ky  = BOARD_Y + BOARD_H + 18
    for label, col in (("correct (+)", C["correct"]),
                       ("present (x)", C["present"]),
                       ("absent  (-)", C["absent"])):
        pygame.draw.rect(surf, col, (kx, ky, 12, 12), border_radius=2)
        lbl = font_md.render(label, True, C["dim"])
        surf.blit(lbl, (kx + 16, ky - 1))
        kx += lbl.get_width() + 28


def draw_stats(surf, font_md, font_sm,
               hist: list[dict], avg: float,
               game_num: int, wins: int) -> None:

    GX = DIVIDER + 20
    GY = 20
    GW = W - DIVIDER - 40
    GH = H - 60 - GY - 20

    pygame.draw.rect(surf, C["panel"], (GX - 10, GY - 10, GW + 20, GH + 20), border_radius=8)

    # title + summary stats
    wr_str = f"{wins / max(game_num, 1):.1%} win rate  •  avg {avg:.2f} guesses"
    t = font_md.render(wr_str, True, C["title"])
    surf.blit(t, (GX, GY + 2))

    IY = GY + 28
    IH = GH - 28

    if not hist:
        w = font_md.render("Waiting for first game …", True, C["dim"])
        surf.blit(w, w.get_rect(centerx=GX + GW // 2, centery=IY + IH // 2))
        return

    MAX_VAL  = 7
    n        = len(hist)
    spacing  = GW / n
    bar_w    = max(2, min(18, int(spacing) - 1))

    for i, ep in enumerate(hist):
        val  = ep["n"] if ep["won"] else MAX_GUESSES + 1
        bh   = int((val / MAX_VAL) * IH)
        bx   = GX + int(i * spacing)
        by   = IY + IH - bh
        col  = C["win_bar"] if ep["won"] else C["los_bar"]
        pygame.draw.rect(surf, col, (bx, by, bar_w, bh))

    # average line
    avg_y = IY + IH - int((avg / MAX_VAL) * IH)
    pygame.draw.line(surf, C["avg_ln"], (GX, avg_y), (GX + GW, avg_y), 2)
    lbl = font_sm.render(f"avg {avg:.2f}", True, C["avg_ln"])
    surf.blit(lbl, (GX + GW - lbl.get_width() - 4, avg_y - lbl.get_height() - 2))

    # y-axis labels
    for v in range(1, 8):
        ty   = IY + IH - int((v / MAX_VAL) * IH)
        tick = font_sm.render(str(v) if v < 7 else "F", True, C["dim"])
        surf.blit(tick, (GX - 16, ty - tick.get_height() // 2))
        pygame.draw.line(surf, C["dim"], (GX - 4, ty), (GX, ty), 1)


def draw_statusbar(surf, font_md, delay: float, game_num: int) -> None:
    sy = H - 56
    pygame.draw.rect(surf, C["panel"], (0, sy, W, 56))
    pygame.draw.line(surf, C["border"], (0, sy), (W, sy), 1)

    items = [
        (f"Entropy solver  •  {delay:.1f}s / guess", C["title"]),
        (f"SPACE  skip game", C["dim"]),
        (f"ESC  quit", C["dim"]),
    ]
    x = 20
    for txt, col in items:
        s = font_md.render(txt, True, col)
        surf.blit(s, (x, sy + (56 - s.get_height()) // 2))
        x += s.get_width() + 50


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, default=1.0,
                        help="seconds between guesses (default 1.0)")
    parser.add_argument("--word",  default=None,
                        help="fix the target word (loops on same word)")
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Entropy Solver — live")
    clock  = pygame.time.Clock()

    try:
        font_lg = pygame.font.SysFont("Helvetica Neue", 30, bold=True)
        font_md = pygame.font.SysFont("Helvetica Neue", 17)
        font_sm = pygame.font.SysFont("Helvetica Neue", 13)
    except Exception:
        font_lg = pygame.font.Font(None, 34)
        font_md = pygame.font.Font(None, 20)
        font_sm = pygame.font.Font(None, 16)

    state = {
        "game":      [],
        "target":    None,
        "remaining": None,
        "last_won":  None,
        "game_num":  0,
        "wins":      0,
        "avg":       0.0,
        "hist":      deque(maxlen=150),
        "_total":    0,
    }

    t = threading.Thread(
        target=_solve_loop,
        args=(args.delay, args.word),
        daemon=True,
    )
    t.start()

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

        # drain queue
        try:
            while True:
                msg = _q.get_nowait()

                if msg["type"] == "new_game":
                    state["game"]      = []
                    state["target"]    = None
                    state["remaining"] = None
                    state["last_won"]  = None
                    state["game_num"] += 1

                elif msg["type"] == "step":
                    state["game"]      = msg["game"]
                    state["remaining"] = msg["remaining"]

                elif msg["type"] == "end":
                    state["target"]   = msg["target"]
                    state["last_won"] = msg["won"]
                    n = msg["n"]
                    if msg["won"]:
                        state["wins"] += 1
                    state["hist"].append({"n": n, "won": msg["won"]})
                    state["_total"] += n if msg["won"] else MAX_GUESSES + 1
                    state["avg"] = state["_total"] / len(state["hist"])

        except queue.Empty:
            pass

        screen.fill(C["bg"])
        draw_board(screen, font_lg, font_md,
                   state["game"],
                   state["game_num"],
                   state["target"],
                   state["remaining"],
                   state["last_won"])
        draw_stats(screen, font_md, font_sm,
                   list(state["hist"]),
                   state["avg"],
                   state["game_num"],
                   state["wins"])
        draw_statusbar(screen, font_md, args.delay, state["game_num"])

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
