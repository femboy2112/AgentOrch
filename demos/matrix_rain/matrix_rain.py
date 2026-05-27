#!/usr/bin/env python3
"""Polished Matrix-style digital rain terminal demo (stdlib only)."""

from __future__ import annotations

import argparse
import random
import shutil
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

ESC = "\x1b["

# Film-inspired set: mostly half-width katakana, plus Latin letters and numerals.
HALFWIDTH_KATAKANA = "ｦｧｨｩｪｫｬｭｮｯｰｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝﾞﾟ｢｣､･"
LATIN = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
DIGITS = "0123456789"
DEFAULT_GLYPHS = (HALFWIDTH_KATAKANA * 4) + LATIN + DIGITS

# Terminals do not provide true mirrored kana glyphs from the film's custom typeface,
# so mirror mode swaps where we can and otherwise keeps the original glyph.
MIRROR_TABLE = str.maketrans(
    {
        "｢": "｣",
        "｣": "｢",
        "(": ")",
        ")": "(",
        "[": "]",
        "]": "[",
        "<": ">",
        ">": "<",
        "/": "\\",
        "\\": "/",
    }
)

PALETTE_256 = [22, 22, 22, 28, 28, 34, 34, 40, 40, 46, 46, 82, 118, 154, 190]
HEAD_256 = 231
HEAD_TRUE = (232, 255, 240)
BRIGHT_TRUE = (0, 255, 65)
DARK_TRUE = (0, 24, 6)


@dataclass
class Column:
    x: int
    active: bool = False
    y: float = 0.0
    speed: float = 15.0
    trail: int = 18
    spawn_in: float = 0.0
    glyphs: Dict[int, str] = field(default_factory=dict)


class MatrixRain:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.use_truecolor = args.color == "truecolor"
        self.glyph_pool = self._build_glyph_pool(args.glyphs, mirror=not args.no_mirror)
        self.width = 0
        self.height = 0
        self.columns: List[Column] = []
        self.running = True
        self.needs_resize = True
        self.rng = random.Random()

    def _build_glyph_pool(self, glyphs: Optional[str], mirror: bool) -> str:
        pool = glyphs if glyphs else DEFAULT_GLYPHS
        pool = "".join(ch for ch in pool if not ch.isspace())
        if not pool:
            raise ValueError("glyph set cannot be empty")
        if mirror:
            return "".join(ch.translate(MIRROR_TABLE) for ch in pool)
        return pool

    def _terminal_size(self) -> Tuple[int, int]:
        size = shutil.get_terminal_size(fallback=(80, 24))
        return max(20, size.columns), max(8, size.lines)

    def _resize(self) -> None:
        self.width, self.height = self._terminal_size()
        self.columns = [self._new_column(i, start_random=True) for i in range(self.width)]
        self.needs_resize = False

    def _new_column(self, x: int, start_random: bool) -> Column:
        c = Column(x=x)
        self._reset_column(c, immediately=False)
        if start_random and self.rng.random() < min(0.95, self.args.density):
            self._spawn_column(c)
            c.y = self.rng.uniform(-float(self.height), float(self.height) * 0.6)
        return c

    def _spawn_delay(self) -> float:
        # Higher density -> shorter gaps between streams.
        base = self.rng.uniform(0.1, 2.6)
        return max(0.02, base / max(0.05, self.args.density))

    def _reset_column(self, c: Column, immediately: bool) -> None:
        c.active = False
        c.glyphs.clear()
        c.spawn_in = 0.0 if immediately else self._spawn_delay()

    def _spawn_column(self, c: Column) -> None:
        c.active = True
        c.y = self.rng.uniform(-self.height * 0.9, -1.0)
        c.speed = self.rng.uniform(10.0, 40.0) * self.args.speed
        min_trail = 7
        max_trail = max(12, int(self.height * 0.7))
        trail_bias = 0.22 + min(0.65, self.args.density * 0.25)
        c.trail = max(min_trail, int(self.rng.uniform(self.height * 0.18, self.height * trail_bias)))
        c.trail = min(c.trail, max_trail)

    def _update(self, dt: float) -> None:
        if self.needs_resize:
            self._resize()

        for c in self.columns:
            if c.active:
                c.y += c.speed * dt
                head_row = int(c.y)
                # Retain only near-trail glyphs.
                c.glyphs = {r: g for r, g in c.glyphs.items() if head_row - c.trail - 2 <= r <= head_row}
                if c.y - c.trail > self.height + self.rng.uniform(0, self.height * 0.25):
                    self._reset_column(c, immediately=False)
            else:
                c.spawn_in -= dt
                if c.spawn_in <= 0:
                    self._spawn_column(c)

    def _pick_glyph(self) -> str:
        return self.rng.choice(self.glyph_pool)

    def _tail_color_true(self, age: int, trail: int, glint: bool) -> Tuple[int, int, int]:
        if age <= 0:
            return HEAD_TRUE
        if age == 1:
            return BRIGHT_TRUE

        t = min(1.0, max(0.0, age / max(1, trail)))
        r = int(BRIGHT_TRUE[0] + (DARK_TRUE[0] - BRIGHT_TRUE[0]) * t)
        g = int(BRIGHT_TRUE[1] + (DARK_TRUE[1] - BRIGHT_TRUE[1]) * t)
        b = int(BRIGHT_TRUE[2] + (DARK_TRUE[2] - BRIGHT_TRUE[2]) * t)
        if glint:
            return (min(255, r + 24), min(255, g + 34), min(255, b + 24))
        return (r, g, b)

    def _tail_color_256(self, age: int, trail: int, glint: bool) -> int:
        if age <= 0:
            return HEAD_256
        idx = int((age / max(1, trail)) * (len(PALETTE_256) - 1))
        idx = max(0, min(len(PALETTE_256) - 1, idx))
        color = PALETTE_256[-1 - idx]
        if glint:
            return max(color, 154)
        return color

    def _render_frame(self) -> str:
        h = self.height
        w = self.width
        chars = [[" " for _ in range(w)] for _ in range(h)]
        score = [[0.0 for _ in range(w)] for _ in range(h)]
        colors_true: List[List[Tuple[int, int, int]]] = [[(0, 0, 0) for _ in range(w)] for _ in range(h)]
        colors_256 = [[22 for _ in range(w)] for _ in range(h)]

        mutation_rate = min(0.6, 0.07 * self.args.speed)

        for c in self.columns:
            if not c.active:
                continue

            head = int(c.y)
            start = max(0, head - c.trail)
            end = min(h - 1, head)
            if end < 0 or start >= h:
                continue

            for row in range(start, end + 1):
                age = head - row
                intensity = 1.2 if age == 0 else max(0.03, 1.0 - (age / max(1.0, c.trail)))
                if intensity < score[row][c.x]:
                    continue

                if age == 0:
                    ch = self._pick_glyph()
                else:
                    ch = c.glyphs.get(row)
                    if ch is None:
                        ch = self._pick_glyph()
                    elif 2 <= age <= max(3, c.trail - 2) and self.rng.random() < mutation_rate:
                        ch = self._pick_glyph()
                    c.glyphs[row] = ch

                glint = age > 1 and self.rng.random() < 0.012
                chars[row][c.x] = ch
                score[row][c.x] = intensity + (0.17 if glint else 0.0)
                if self.use_truecolor:
                    colors_true[row][c.x] = self._tail_color_true(age, c.trail, glint)
                else:
                    colors_256[row][c.x] = self._tail_color_256(age, c.trail, glint)

        out: List[str] = [f"{ESC}H"]
        current_color = None
        if self.use_truecolor:
            out.append(f"{ESC}48;2;0;0;0m")
            for y in range(h):
                for x in range(w):
                    ch = chars[y][x]
                    if ch == " ":
                        desired = (0, 0, 0)
                    else:
                        desired = colors_true[y][x]
                    if desired != current_color:
                        out.append(f"{ESC}38;2;{desired[0]};{desired[1]};{desired[2]}m")
                        current_color = desired
                    out.append(ch)
                if y != h - 1:
                    out.append("\n")
        else:
            out.append(f"{ESC}40m")
            for y in range(h):
                for x in range(w):
                    ch = chars[y][x]
                    desired = 22 if ch == " " else colors_256[y][x]
                    if desired != current_color:
                        out.append(f"{ESC}38;5;{desired}m")
                        current_color = desired
                    out.append(ch)
                if y != h - 1:
                    out.append("\n")

        out.append(f"{ESC}0m")
        return "".join(out)

    def run(self) -> None:
        start = time.monotonic()
        frame_time = 1.0 / self.args.fps
        last = start

        while self.running:
            now = time.monotonic()
            dt = max(0.0, min(0.1, now - last))
            last = now

            self._update(dt)
            sys.stdout.write(self._render_frame())
            sys.stdout.flush()

            if self.args.duration > 0 and now - start >= self.args.duration:
                break

            elapsed = time.monotonic() - now
            sleep_for = frame_time - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)


def positive_float(name: str, allow_zero: bool = False):
    def validator(value: str) -> float:
        try:
            v = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{name} must be a number") from exc
        if allow_zero:
            if v < 0:
                raise argparse.ArgumentTypeError(f"{name} must be >= 0")
        elif v <= 0:
            raise argparse.ArgumentTypeError(f"{name} must be > 0")
        return v

    return validator


def density_value(value: str) -> float:
    v = positive_float("density")(value)
    if not 0.05 <= v <= 3.0:
        raise argparse.ArgumentTypeError("density must be between 0.05 and 3.0")
    return v


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Movie-accurate Matrix digital rain (stdlib-only, ANSI terminal).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--fps", type=positive_float("fps"), default=30.0, help="target frames per second")
    p.add_argument("--speed", type=positive_float("speed"), default=1.0, help="global stream speed multiplier")
    p.add_argument(
        "--density",
        type=density_value,
        default=0.9,
        help="spawn density multiplier (higher = denser rain)",
    )
    p.add_argument(
        "--color",
        choices=("truecolor", "256"),
        default="truecolor",
        help="color mode for ANSI rendering",
    )
    p.add_argument("--no-mirror", action="store_true", help="disable mirrored glyph transform")
    p.add_argument(
        "--glyphs",
        type=str,
        default=None,
        help="override glyph pool (whitespace is ignored)",
    )
    p.add_argument(
        "--duration",
        type=positive_float("duration", allow_zero=True),
        default=0.0,
        help="seconds to run (0 = run until Ctrl-C)",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    try:
        args = parse_args(argv)
        rain = MatrixRain(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    def on_winch(_signum: int, _frame: object) -> None:
        rain.needs_resize = True

    def on_stop(_signum: int, _frame: object) -> None:
        rain.running = False

    signal.signal(signal.SIGWINCH, on_winch)
    signal.signal(signal.SIGINT, on_stop)

    try:
        sys.stdout.write(f"{ESC}?1049h{ESC}?25l{ESC}2J{ESC}H")
        sys.stdout.flush()
        rain.run()
    finally:
        sys.stdout.write(f"{ESC}0m{ESC}?25h{ESC}?1049l")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
