"""
quordle/visualizer.py — pygame live view of Quordle RL training.

Layout
------
  Left  : 2×2 grid of Wordle boards (current episode)
  Right : bar chart — boards solved per episode (0–4, green=win)
          rolling average line overlaid

Timing
------
  FAST mode — no delay during early random play
  SLOW mode — 0.6 s/guess once rolling avg boards solved ≥ SLOW_THRESHOLD
              (agent starts reliably solving at least 2 boards)

Usage
-----
  cd quordle
  python visualizer.py
  python visualizer.py --timesteps 5000000
  python visualizer.py --timesteps 5000000 --n_steps 256
"""

import argparse
import os
import queue
import threading
import time
from collections import deque

import pygame
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from env import QuordleEnv, MAX_GUESSES, N_BOARDS

# ─── Shared state ─────────────────────────────────────────────────────────────

_q:    "queue.Queue[dict]" = queue.Queue(maxsize=800)
_slow  = threading.Event()
_stop  = threading.Event()

ROLLING_WIN    = 50
SLOW_THRESHOLD = 2.0    # avg boards solved ≥ this → engage slow mode
GUESS_DELAY    = 0.6

_delay: list[float] = [0.0]
DELAY_STEP = 0.25
DELAY_MAX  = 5.0

_btn: dict[str, pygame.Rect] = {}


def _change_delay(delta: float) -> None:
    _delay[0] = round(max(0.0, min(DELAY_MAX, _delay[0] + delta)), 2)


MODEL_DIR  = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "quordle_ppo")


# ─── Callback ─────────────────────────────────────────────────────────────────

class VizCallback(BaseCallback):
    def __init__(self) -> None:
        super().__init__(verbose=0)
        self._games: list[list[tuple[str, str]]] = [[] for _ in range(N_BOARDS)]
        self._rolling: deque[float] = deque(maxlen=ROLLING_WIN)
        self._hist:    deque[dict]  = deque(maxlen=200)
        self._n_ep = 0

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        done = bool(self.locals["dones"][0])

        if "guess" in info:
            guess   = info["guess"]
            solved  = info["solved"]
            # append guess to all boards that weren't yet solved last step
            for b in range(N_BOARDS):
                if len(self._games[b]) < info["n_guesses"]:
                    self._games[b].append((guess, "-----"))   # placeholder

        _put({"type": "step", "games": [list(g) for g in self._games],
              "solved": info.get("solved", [False]*N_BOARDS)})

        if done:
            n_solved = info.get("n_solved", 0)
            won      = info.get("won", False)
            self._n_ep += 1
            self._rolling.append(n_solved)
            avg = sum(self._rolling) / len(self._rolling)
            self._hist.append({
                "n_solved": n_solved,
                "n_guesses": info.get("n_guesses", MAX_GUESSES),
                "won": won,
            })
            _put({"type": "episode", "ep": self._n_ep,
                  "n_solved": n_solved, "won": won, "avg": avg,
                  "hist": list(self._hist)})

            if not _slow.is_set() and avg >= SLOW_THRESHOLD:
                _slow.set()
                _delay[0] = GUESS_DELAY
                print(f"[viz] Rolling avg {avg:.2f} boards ≥ {SLOW_THRESHOLD} → delay {GUESS_DELAY}s")

            self._games = [[] for _ in range(N_BOARDS)]

        if _delay[0] > 0:
            time.sleep(_delay[0])

        if _stop.is_set():
            return False

        return True


# ─── Training thread ──────────────────────────────────────────────────────────

def _train(timesteps: int, n_steps: int) -> None:
    env = QuordleEnv(shaped_reward=True, dense_reward=True, use_mask=True)

    if os.path.exists(MODEL_PATH + ".zip"):
        print("[viz] Resuming from existing model …")
        model = MaskablePPO.load(MODEL_PATH, env=env)
        model.set_env(env)
    else:
        model = MaskablePPO(
            "MlpPolicy", env,
            learning_rate=3e-4,
            n_steps=n_steps,
            batch_size=max(64, n_steps // 8),
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs=dict(net_arch=[512, 256]),
            verbose=0,
        )

    try:
        model.learn(total_timesteps=timesteps, callback=CallbackList([VizCallback()]))
    finally:
        os.makedirs(MODEL_DIR, exist_ok=True)
        model.save(MODEL_PATH)
        print(f"[viz] Model saved → {MODEL_PATH}.zip")
        _put({"type": "done"})


def _put(msg: dict) -> None:
    try:
        _q.put_nowait(msg)
    except queue.Full:
        pass


# ─── Layout constants ─────────────────────────────────────────────────────────

W, H     = 1080, 720
DIVIDER  = 440          # left panel (boards) | right panel (graph)
STATS_H  = 56

CELL     = 28
GAP      = 3
BOARD_W  = 5 * CELL + 4 * GAP   # 152
BOARD_H  = MAX_GUESSES * CELL + (MAX_GUESSES - 1) * GAP   # 276

# 2×2 grid origin inside the left panel
GRID_X   = (DIVIDER - 2 * BOARD_W - 20) // 2   # horizontal start
GRID_Y   = 50                                    # below title
GRID_GAP = 20                                    # gap between boards

# Top-left corners of each board: [0]=TL, [1]=TR, [2]=BL, [3]=BR
_BOARD_POS = [
    (GRID_X,              GRID_Y),
    (GRID_X + BOARD_W + GRID_GAP, GRID_Y),
    (GRID_X,              GRID_Y + BOARD_H + GRID_GAP),
    (GRID_X + BOARD_W + GRID_GAP, GRID_Y + BOARD_H + GRID_GAP),
]

C = {
    "bg":      (18,  18,  19),
    "panel":   (26,  26,  27),
    "correct": (83,  141, 78),
    "present": (181, 159, 59),
    "absent":  (58,  58,  60),
    "border":  (70,  70,  72),
    "text":    (255, 255, 255),
    "dim":     (110, 110, 110),
    "title":   (200, 200, 200),
    "avg":     (255, 200, 60),
    "win":     (83,  141, 78),
    "partial": (181, 159, 59),
    "loss":    (155, 50,  50),
    "slow_on": (83,  141, 78),
    "slow_of": (130, 130, 130),
}


# ─── Drawing ──────────────────────────────────────────────────────────────────

def _cell_color(fb: str) -> tuple:
    return {"+" : C["correct"], "x": C["present"], "-": C["absent"]}.get(fb, C["panel"])


def draw_boards(surf: pygame.Surface,
                font_lg: pygame.font.Font,
                font_md: pygame.font.Font,
                games: list[list[tuple[str, str]]],
                solved: list[bool],
                ep_num: int) -> None:

    pygame.draw.rect(surf, C["panel"], (0, 0, DIVIDER, H - STATS_H))

    ep_surf = font_md.render(f"Episode  {ep_num:,}", True, C["title"])
    surf.blit(ep_surf, ep_surf.get_rect(centerx=DIVIDER // 2, top=12))

    for b in range(N_BOARDS):
        bx, by = _BOARD_POS[b]

        # Board label
        label = font_md.render(
            f"{'✓' if solved[b] else f'B{b+1}'}",
            True, C["correct"] if solved[b] else C["dim"],
        )
        surf.blit(label, (bx, by - 18))

        # Grid cells
        for row in range(MAX_GUESSES):
            for col in range(5):
                x = bx + col * (CELL + GAP)
                y = by + row * (CELL + GAP)
                rect = pygame.Rect(x, y, CELL, CELL)

                if row < len(games[b]):
                    guess, pattern = games[b][row]
                    if len(pattern) == 5:
                        pygame.draw.rect(surf, _cell_color(pattern[col]), rect, border_radius=3)
                        ltr = font_lg.render(guess[col].upper(), True, C["text"])
                        surf.blit(ltr, ltr.get_rect(center=rect.center))
                    else:
                        pygame.draw.rect(surf, C["absent"], rect, border_radius=3)
                else:
                    pygame.draw.rect(surf, C["panel"], rect, border_radius=3)
                    pygame.draw.rect(surf, C["border"], rect, 1, border_radius=3)


def _bar_color(n_solved: int) -> tuple:
    if n_solved == N_BOARDS:
        return C["win"]
    if n_solved >= 2:
        return C["partial"]
    return C["loss"]


def draw_graph(surf: pygame.Surface,
               font_md: pygame.font.Font,
               font_sm: pygame.font.Font,
               hist: list[dict],
               avg: float) -> None:

    GX = DIVIDER + 20
    GY = 20
    GW = W - DIVIDER - 40
    GH = H - STATS_H - GY - 20

    pygame.draw.rect(surf, C["panel"], (GX - 10, GY - 10, GW + 20, GH + 20), border_radius=8)

    t = font_md.render("Boards solved per episode  (4 = win)", True, C["dim"])
    surf.blit(t, (GX, GY + 2))

    IY = GY + 28
    IH = GH - 28

    if not hist:
        w = font_md.render("Waiting for first episode …", True, C["dim"])
        surf.blit(w, w.get_rect(centerx=GX + GW // 2, centery=IY + IH // 2))
        return

    MAX_VAL = N_BOARDS
    n       = len(hist)
    bar_w   = max(2, min(18, (GW - 10) // n - 1))
    spacing = GW / n

    for i, ep in enumerate(hist):
        val = ep["n_solved"]
        bh  = int((val / MAX_VAL) * IH) if MAX_VAL > 0 else 0
        bx  = GX + int(i * spacing)
        by  = IY + IH - bh
        pygame.draw.rect(surf, _bar_color(val), (bx, by, max(bar_w, 2), max(bh, 1)))

    # rolling avg line
    avg_y = IY + IH - int((avg / MAX_VAL) * IH)
    pygame.draw.line(surf, C["avg"], (GX, avg_y), (GX + GW, avg_y), 2)
    avg_lbl = font_sm.render(f"avg {avg:.2f}", True, C["avg"])
    surf.blit(avg_lbl, (GX + GW - avg_lbl.get_width() - 4, avg_y - avg_lbl.get_height() - 2))

    # y-axis ticks (0-4)
    for v in range(MAX_VAL + 1):
        ty   = IY + IH - int((v / MAX_VAL) * IH)
        tick = font_sm.render(str(v), True, C["dim"])
        surf.blit(tick, (GX - 14, ty - tick.get_height() // 2))
        pygame.draw.line(surf, C["dim"], (GX - 4, ty), (GX, ty), 1)


def draw_stats(surf: pygame.Surface,
               font_md: pygame.font.Font,
               ep_num: int,
               avg: float,
               win_rate: float,
               done: bool) -> None:

    sy = H - STATS_H
    cy = sy + STATS_H // 2
    pygame.draw.rect(surf, C["panel"], (0, sy, W, STATS_H))
    pygame.draw.line(surf, C["border"], (0, sy), (W, sy), 1)

    d           = _delay[0]
    slow_active = d > 0
    wr_col      = C["win"] if win_rate >= 0.5 else C["avg"] if win_rate >= 0.2 else C["dim"]

    items = [
        (f"{'● SLOW' if slow_active else '● FAST'}  {d:.2f}s/guess",
         C["slow_on"] if slow_active else C["slow_of"]),
        (f"episode  {ep_num:,}", C["text"]),
        (f"avg boards  {avg:.2f} / {N_BOARDS}", C["avg"]),
        (f"win rate  {win_rate:.1%}", wr_col),
    ]
    if done:
        items.append(("training complete", C["win"]))

    x = 20
    for txt, col in items:
        s = font_md.render(txt, True, col)
        surf.blit(s, (x, sy + (STATS_H - s.get_height()) // 2))
        x += s.get_width() + 40

    # Speed adjuster [−] / [+]
    btn_w, btn_h = 26, 26
    btn_y = cy - btn_h // 2

    plus_rect = pygame.Rect(W - 20 - btn_w, btn_y, btn_w, btn_h)
    pygame.draw.rect(surf, C["border"], plus_rect, border_radius=4)
    surf.blit(font_md.render("+", True, C["text"]), font_md.render("+", True, C["text"]).get_rect(center=plus_rect.center))
    _btn["speed_up"] = plus_rect

    val_s = font_md.render(f"{d:.2f}s", True, C["avg"])
    val_x = plus_rect.left - val_s.get_width() - 8
    surf.blit(val_s, (val_x, cy - val_s.get_height() // 2))

    minus_rect = pygame.Rect(val_x - 8 - btn_w, btn_y, btn_w, btn_h)
    pygame.draw.rect(surf, C["border"], minus_rect, border_radius=4)
    surf.blit(font_md.render("−", True, C["text"]), font_md.render("−", True, C["text"]).get_rect(center=minus_rect.center))
    _btn["speed_down"] = minus_rect

    spd = font_md.render("speed", True, C["dim"])
    surf.blit(spd, (minus_rect.left - spd.get_width() - 10, cy - spd.get_height() // 2))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=3_000_000)
    parser.add_argument("--n_steps",   type=int, default=512)
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Quordle RL — live training")
    clock = pygame.time.Clock()

    try:
        font_lg = pygame.font.SysFont("Helvetica Neue", 22, bold=True)
        font_md = pygame.font.SysFont("Helvetica Neue", 17)
        font_sm = pygame.font.SysFont("Helvetica Neue", 13)
    except Exception:
        font_lg = pygame.font.Font(None, 26)
        font_md = pygame.font.Font(None, 20)
        font_sm = pygame.font.Font(None, 16)

    state = {
        "games":   [[] for _ in range(N_BOARDS)],
        "solved":  [False] * N_BOARDS,
        "hist":    [],
        "avg":     0.0,
        "ep":      0,
        "done":    False,
    }

    t = threading.Thread(
        target=_train,
        args=(args.timesteps, args.n_steps),
        daemon=False,
    )
    t.start()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                _stop.set()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    _stop.set()
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    _change_delay(DELAY_STEP)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    _change_delay(-DELAY_STEP)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if _btn.get("speed_up")   and _btn["speed_up"].collidepoint(event.pos):
                    _change_delay(DELAY_STEP)
                elif _btn.get("speed_down") and _btn["speed_down"].collidepoint(event.pos):
                    _change_delay(-DELAY_STEP)

        try:
            while True:
                msg = _q.get_nowait()
                if msg["type"] == "step":
                    state["games"]  = msg["games"]
                    state["solved"] = msg["solved"]
                elif msg["type"] == "episode":
                    state["avg"]  = msg["avg"]
                    state["ep"]   = msg["ep"]
                    state["hist"] = msg["hist"]
                elif msg["type"] == "done":
                    state["done"] = True
        except queue.Empty:
            pass

        hist     = state["hist"]
        win_rate = sum(1 for h in hist if h["won"]) / len(hist) if hist else 0.0

        screen.fill(C["bg"])
        draw_boards(screen, font_lg, font_md,
                    state["games"], state["solved"], state["ep"])
        draw_graph(screen, font_md, font_sm, hist, state["avg"])
        draw_stats(screen, font_md, state["ep"], state["avg"], win_rate, state["done"])

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

    if t.is_alive():
        print("Waiting for model to save …")
        t.join()


if __name__ == "__main__":
    main()
