"""
Core Wordle logic.

evaluate_guess(guess, target) -> str of length 5
  '+' correct position
  'x' wrong position
  '-' absent

Two-pass algorithm:
  Pass 1 – mark greens, count leftover target letters
  Pass 2 – yellows consume one occurrence from the remaining pool
"""

WIN_PATTERN = "+++++"
MAX_GUESSES = 6


def evaluate_guess(guess: str, target: str) -> str:
    result = ["-"] * 5
    remaining: dict[str, int] = {}

    # pass 1 – greens
    for i, (g, t) in enumerate(zip(guess, target)):
        if g == t:
            result[i] = "+"
        else:
            remaining[t] = remaining.get(t, 0) + 1

    # pass 2 – yellows
    for i, g in enumerate(guess):
        if result[i] != "+" and remaining.get(g, 0) > 0:
            result[i] = "x"
            remaining[g] -= 1

    return "".join(result)


def pattern_to_int(pattern: str) -> int:
    """Encode a pattern string to an integer in [0, 242]."""
    val = 0
    for ch in pattern:
        val = val * 3 + (0 if ch == "-" else 1 if ch == "x" else 2)
    return val


def int_to_pattern(val: int) -> str:
    syms = {0: "-", 1: "x", 2: "+"}
    chars = []
    for _ in range(5):
        chars.append(syms[val % 3])
        val //= 3
    return "".join(reversed(chars))
