# Matrix Rain Demo

`matrix_rain.py` is a stdlib-only terminal program that renders a polished, film-inspired Matrix digital rain effect.

## Film Accuracy Notes

- Glyph mix follows reported Matrix code composition: mostly half-width katakana, mixed with Latin letters and numerals.
- Mirror mode is enabled by default to emulate the title-sequence mirrored look; `--no-mirror` disables it.
- Color styling uses a black background, a near-white head glyph, bright phosphor green (`#00FF41`-style), and darker green trail fade.

## Run

```bash
./demos/matrix_rain/matrix_rain.py
```

## CLI

```bash
./demos/matrix_rain/matrix_rain.py --help
```

Key options:

- `--fps` target frames per second
- `--speed` global stream-speed multiplier
- `--density` spawn density multiplier (`0.05` to `3.0`)
- `--color` `truecolor` or `256`
- `--no-mirror` disable mirrored transform
- `--glyphs` override glyph pool
- `--duration` seconds to run (`0` = until `Ctrl-C`)

## Example

```bash
./demos/matrix_rain/matrix_rain.py --fps 45 --speed 1.2 --density 1.1 --color truecolor
```
