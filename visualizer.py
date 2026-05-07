"""
visualizer.py – pygame live view of the Wordle RL agent training.

Layout
------
  Left  : Wordle board (current episode)
  Right : bar chart of recent episode lengths (green = win, red = fail)
          rolling-average line overlaid in yellow

Timing
------
  FAST mode  – no delay; graph fills up quickly (early training is random)
  SLOW mode  – 1 second added after every guess so you can read the board
               Triggers automatically once the rolling-average episode
               length (failures counted as 7) drops to ≤ 6, i.e. the
               agent starts solving at least some games.

Usage
-----
  source myenv/bin/activate
  python visualizer.py
  python visualizer.py --curriculum --timesteps 3000000
  python visualizer.py --timesteps 5000000 --n_steps 256
"""

import argparse
import os
import queue
import random
import threading
import time
from collections import deque

import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from env import WordleEnv
from game import MAX_GUESSES
from words import ANSWERS

# ─── Shared between training thread and pygame thread ───────────────────────

_q: "queue.Queue[dict]" = queue.Queue(maxsize=600)
_slow = threading.Event()   # set once rolling avg ≤ SLOW_THRESHOLD

ROLLING_WIN    = 50          # window for rolling average
SLOW_THRESHOLD = 6.0         # trigger slow mode when avg ≤ this
GUESS_DELAY    = 1.0         # default delay (s) when slow mode auto-triggers

# Mutable delay shared between threads — list so assignment is GIL-atomic.
# Training thread reads _delay[0]; pygame thread writes it via _change_delay().
_delay: list[float] = [0.0]

DELAY_STEP = 0.25
DELAY_MAX  = 5.0

# Button rects populated each frame by draw_stats(), read in the event loop.
_btn: dict[str, pygame.Rect] = {}


def _change_delay(delta: float) -> None:
    _delay[0] = round(max(0.0, min(DELAY_MAX, _delay[0] + delta)), 2)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "ppo_wordle")


# ─── Callback (runs inside training thread) ──────────────────────────────────

class VizCallback(BaseCallback):
    """
    Sends board state and episode stats to the render queue after every step.
    Sleeps GUESS_DELAY seconds per step once slow-mode is active.
    """

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self._game: list[tuple[str, str]] = []     # (guess, pattern) this episode
        self._rolling: deque[float] = deque(maxlen=ROLLING_WIN)
        self._hist: deque[dict]  = deque(maxlen=150)  # episode summaries
        self._n_ep = 0

    # ------------------------------------------------------------------
    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        done = bool(self.locals["dones"][0])

        if "guess" in info:
            self._game.append((info["guess"], info["pattern"]))

        _put({"type": "step", "game": list(self._game)})

        if done:
            n   = info.get("n_guesses", MAX_GUESSES)
            won = info.get("won", False)
            self._n_ep += 1
            cost = n if won else MAX_GUESSES + 1   # 7 = failure sentinel
            self._rolling.append(cost)
            avg = sum(self._rolling) / len(self._rolling)
            self._hist.append({"n": n, "won": won})

            _put({
                "type": "episode",
                "ep":   self._n_ep,
                "n":    n,
                "won":  won,
                "avg":  avg,
                "hist": list(self._hist),
            })

            # auto-engage delay once agent starts solving games
            if not _slow.is_set() and avg <= SLOW_THRESHOLD:
                _slow.set()
                _delay[0] = GUESS_DELAY
                print(f"[viz] Rolling avg {avg:.2f} ≤ {SLOW_THRESHOLD} → delay {GUESS_DELAY}s")

            self._game = []

        if _delay[0] > 0:
            time.sleep(_delay[0])

        return True


# ─── Training thread ─────────────────────────────────────────────────────────

def _train(timesteps: int, n_steps: int, curriculum: bool, masked: bool, top_k: int) -> None:
    from train_rl import _ppo_class, _active_stages
    PPO = _ppo_class(masked)

    if top_k > 0:
        from solver_entropy import top_entropy_answers
        all_ans = top_entropy_answers(top_k)
        env_kw  = dict(answers=all_ans, action_words=all_ans)
    else:
        all_ans = list(ANSWERS)
        env_kw  = {}

    stages = _active_stages(len(all_ans))

    if curriculum:
        from train_rl import CurriculumCallback
        tv = random.sample(all_ans, stages[0])
        env = WordleEnv(shaped_reward=True, target_vocab=tv, use_mask=masked, **env_kw)
        cur_cb = CurriculumCallback(
            target_vocab=tv,
            all_answers=all_ans,
            eval_env=WordleEnv(shaped_reward=False, target_vocab=tv, use_mask=masked, **env_kw),
            check_freq=20_000,
            win_threshold=0.80,
            n_eval=200,
            stages=stages,
            verbose=1,
        )
        cbs: list[BaseCallback] = [VizCallback(), cur_cb]
    else:
        env = WordleEnv(shaped_reward=True, use_mask=masked, **env_kw)
        cbs = [VizCallback()]

    if os.path.exists(MODEL_PATH + ".zip"):
        print("[viz] Resuming from existing model …")
        model = PPO.load(MODEL_PATH, env=env)
        model.set_env(env)
    else:
        model = PPO(
            "MlpPolicy", env,
            learning_rate=3e-4,
            n_steps=n_steps,
            batch_size=max(64, n_steps // 8),
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=0,
        )

    try:
        model.learn(total_timesteps=timesteps, callback=CallbackList(cbs))
    finally:
        os.makedirs(MODEL_DIR, exist_ok=True)
        model.save(MODEL_PATH)
        print(f"[viz] Model saved → {MODEL_PATH}.zip")
        _put({"type": "done"})


def _put(msg: dict) -> None:
    try:
        _q.put_nowait(msg)
    except queue.Full:
        pass   # renderer behind; drop non-critical frames


# ─── Colors / layout constants ───────────────────────────────────────────────

W, H = 940, 660

C = {
    "bg":       (18,  18,  19),
    "panel":    (26,  26,  27),
    "correct":  (83,  141, 78),    # green  +
    "present":  (181, 159, 59),    # yellow x
    "absent":   (58,  58,  60),    # gray   -
    "empty_c":  (26,  26,  27),
    "border":   (70,  70,  72),
    "text":     (255, 255, 255),
    "dim":      (110, 110, 110),
    "win_bar":  (83,  141, 78),
    "los_bar":  (155, 50,  50),
    "avg_line": (255, 200, 60),
    "title":    (200, 200, 200),
    "slow_on":  (83,  141, 78),
    "slow_off": (130, 130, 130),
}

CELL     = 56
GAP      = 7
DIVIDER  = 420   # x-coordinate separating board from graph panels

BOARD_W  = 5 * CELL + 4 * GAP          # 308 px
BOARD_H  = MAX_GUESSES * CELL + (MAX_GUESSES - 1) * GAP   # 371 px
BOARD_X  = (DIVIDER - BOARD_W) // 2    # centred in left panel
BOARD_Y  = 90                          # leave room for title


# ─── Drawing helpers ─────────────────────────────────────────────────────────

def _cell_color(fb: str) -> tuple:
    return {"+" : C["correct"], "x": C["present"], "-": C["absent"]}.get(fb, C["empty_c"])


def draw_board(surf: pygame.Surface,
               font_lg: pygame.font.Font,
               font_md: pygame.font.Font,
               game: list[tuple[str, str]],
               ep_num: int,
               last_won: bool | None) -> None:

    # panel background
    pygame.draw.rect(surf, C["panel"],
                     (0, 0, DIVIDER, H - 60), border_radius=0)

    # episode label
    ep_surf = font_md.render(f"Episode  {ep_num:,}", True, C["title"])
    surf.blit(ep_surf, ep_surf.get_rect(centerx=DIVIDER // 2, top=12))

    # result badge (previous episode)
    if last_won is not None:
        badge_txt = "WIN" if last_won else "FAIL"
        badge_col = C["correct"] if last_won else C["los_bar"]
        badge_surf = font_md.render(badge_txt, True, badge_col)
        surf.blit(badge_surf, badge_surf.get_rect(centerx=DIVIDER // 2, top=38))

    # 6×5 grid
    for row in range(MAX_GUESSES):
        for col in range(5):
            x = BOARD_X + col * (CELL + GAP)
            y = BOARD_Y + row * (CELL + GAP)
            rect = pygame.Rect(x, y, CELL, CELL)

            if row < len(game):
                guess, pattern = game[row]
                pygame.draw.rect(surf, _cell_color(pattern[col]), rect, border_radius=4)
                ltr = font_lg.render(guess[col].upper(), True, C["text"])
                surf.blit(ltr, ltr.get_rect(center=rect.center))
            else:
                pygame.draw.rect(surf, C["empty_c"], rect, border_radius=4)
                pygame.draw.rect(surf, C["border"],  rect, 2, border_radius=4)

    # colour key (horizontal row below the board)
    key_y = BOARD_Y + BOARD_H + 18
    kx = BOARD_X
    for label, col in (("correct (+)", C["correct"]),
                       ("present (x)", C["present"]),
                       ("absent  (-)", C["absent"])):
        pygame.draw.rect(surf, col, (kx, key_y, 12, 12), border_radius=2)
        lbl = font_md.render(label, True, C["dim"])
        surf.blit(lbl, (kx + 16, key_y - 1))
        kx += lbl.get_width() + 30


def draw_graph(surf: pygame.Surface,
               font_md: pygame.font.Font,
               font_sm: pygame.font.Font,
               hist: list[dict],
               avg: float) -> None:

    GX = DIVIDER + 20
    GY = 20
    GW = W - DIVIDER - 40
    GH = H - 60 - GY - 20    # leave stats bar + padding

    pygame.draw.rect(surf, C["panel"], (GX - 10, GY - 10, GW + 20, GH + 20), border_radius=8)

    # title
    t = font_md.render("Episode length  (7 = fail)", True, C["dim"])
    surf.blit(t, (GX, GY + 2))

    IY = GY + 28      # inner top
    IH = GH - 28      # inner height for bars

    if not hist:
        waiting = font_md.render("Waiting for first episode …", True, C["dim"])
        surf.blit(waiting, waiting.get_rect(centerx=GX + GW // 2, centery=IY + IH // 2))
        return

    MAX_VAL = 7
    n = len(hist)
    bar_w = max(2, min(18, (GW - 10) // n - 1))
    spacing = GW / n

    for i, ep in enumerate(hist):
        val  = ep["n"] if ep["won"] else MAX_GUESSES + 1
        bh   = int((val / MAX_VAL) * IH)
        bx   = GX + int(i * spacing)
        by   = IY + IH - bh
        col  = C["win_bar"] if ep["won"] else C["los_bar"]
        pygame.draw.rect(surf, col, (bx, by, max(bar_w, 2), bh))

    # rolling average line
    avg_y = IY + IH - int((avg / MAX_VAL) * IH)
    pygame.draw.line(surf, C["avg_line"], (GX, avg_y), (GX + GW, avg_y), 2)
    avg_lbl = font_sm.render(f"avg {avg:.2f}", True, C["avg_line"])
    surf.blit(avg_lbl, (GX + GW - avg_lbl.get_width() - 4, avg_y - avg_lbl.get_height() - 2))

    # y-axis tick labels (1–7)
    for v in range(1, 8):
        ty = IY + IH - int((v / MAX_VAL) * IH)
        tick = font_sm.render(str(v), True, C["dim"])
        surf.blit(tick, (GX - 14, ty - tick.get_height() // 2))
        pygame.draw.line(surf, C["dim"], (GX - 4, ty), (GX, ty), 1)


def draw_stats(surf: pygame.Surface,
               font_md: pygame.font.Font,
               ep_num: int,
               avg: float,
               win_rate: float,
               done: bool) -> None:

    sy  = H - 56
    cy  = sy + 28   # vertical centre of stats bar
    pygame.draw.rect(surf, C["panel"], (0, sy, W, 56))
    pygame.draw.line(surf, C["border"], (0, sy), (W, sy), 1)

    # ── left side: status labels ───────────────────────────────────────
    d = _delay[0]
    slow_active = d > 0
    wr_col = C["correct"] if win_rate >= 0.8 else C["avg_line"] if win_rate >= 0.5 else C["dim"]
    items = [
        (f"{'● SLOW' if slow_active else '● FAST'}  {d:.2f}s/guess",
         C["slow_on"] if slow_active else C["slow_off"]),
        (f"episode  {ep_num:,}", C["text"]),
        (f"rolling avg  {avg:.2f} / {SLOW_THRESHOLD:.0f}", C["avg_line"] if slow_active else C["dim"]),
        (f"win rate  {win_rate:.1%}", wr_col),
    ]
    if done:
        items.append(("training complete", C["correct"]))

    x = 20
    for txt, col in items:
        s = font_md.render(txt, True, col)
        surf.blit(s, (x, sy + (56 - s.get_height()) // 2))
        x += s.get_width() + 40

    # ── right side: speed adjuster  [−]  0.00s  [+] ───────────────────
    btn_w, btn_h = 26, 26
    btn_y = cy - btn_h // 2

    # [+]
    plus_rect = pygame.Rect(W - 20 - btn_w, btn_y, btn_w, btn_h)
    pygame.draw.rect(surf, C["border"], plus_rect, border_radius=4)
    ps = font_md.render("+", True, C["text"])
    surf.blit(ps, ps.get_rect(center=plus_rect.center))
    _btn["speed_up"] = plus_rect

    # delay value
    val_s = font_md.render(f"{d:.2f}s", True, C["avg_line"])
    val_x = plus_rect.left - val_s.get_width() - 8
    surf.blit(val_s, (val_x, cy - val_s.get_height() // 2))

    # [−]
    minus_rect = pygame.Rect(val_x - 8 - btn_w, btn_y, btn_w, btn_h)
    pygame.draw.rect(surf, C["border"], minus_rect, border_radius=4)
    ms = font_md.render("−", True, C["text"])   # − (minus sign)
    surf.blit(ms, ms.get_rect(center=minus_rect.center))
    _btn["speed_down"] = minus_rect

    # "speed" label
    spd = font_md.render("speed", True, C["dim"])
    surf.blit(spd, (minus_rect.left - spd.get_width() - 10, cy - spd.get_height() // 2))


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps",  type=int, default=2_000_000)
    parser.add_argument("--n_steps",    type=int, default=512,
                        help="PPO rollout length (smaller → more frequent policy updates)")
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--masked",     action="store_true",
                        help="Use action masking (MaskablePPO) — much faster convergence")
    parser.add_argument("--top_k",      type=int, default=0,
                        help="Restrict action+target space to top-K entropy answers (0=all 2315)")
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Wordle RL — live training")
    clock = pygame.time.Clock()

    try:
        font_lg = pygame.font.SysFont("Helvetica Neue", 30, bold=True)
        font_md = pygame.font.SysFont("Helvetica Neue", 17)
        font_sm = pygame.font.SysFont("Helvetica Neue", 13)
    except Exception:
        font_lg = pygame.font.Font(None, 34)
        font_md = pygame.font.Font(None, 20)
        font_sm = pygame.font.Font(None, 16)

    # Render state — updated from queue
    state = {
        "game":     [],
        "hist":     [],
        "avg":      7.0,
        "ep":       0,
        "last_won": None,
        "done":     False,
    }

    t = threading.Thread(
        target=_train,
        args=(args.timesteps, args.n_steps, args.curriculum, args.masked, args.top_k),
        daemon=True,
    )
    t.start()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                # +/= or numpad + → speed up (increase delay)
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    _change_delay(DELAY_STEP)
                # - or numpad - → speed down (decrease delay)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    _change_delay(-DELAY_STEP)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if _btn.get("speed_up") and _btn["speed_up"].collidepoint(event.pos):
                    _change_delay(DELAY_STEP)
                elif _btn.get("speed_down") and _btn["speed_down"].collidepoint(event.pos):
                    _change_delay(-DELAY_STEP)

        # drain the queue (non-blocking)
        try:
            while True:
                msg = _q.get_nowait()
                if msg["type"] == "step":
                    state["game"] = msg["game"]
                elif msg["type"] == "episode":
                    state["avg"]  = msg["avg"]
                    state["ep"]   = msg["ep"]
                    state["hist"] = msg["hist"]
                    state["last_won"] = msg["won"]
                elif msg["type"] == "done":
                    state["done"] = True
        except queue.Empty:
            pass

        hist = state["hist"]
        win_rate = sum(1 for h in hist if h["won"]) / len(hist) if hist else 0.0

        screen.fill(C["bg"])
        draw_board(screen, font_lg, font_md,
                   state["game"], state["ep"], state["last_won"])
        draw_graph(screen, font_md, font_sm, hist, state["avg"])
        draw_stats(screen, font_md,
                   state["ep"], state["avg"], win_rate, state["done"])

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
