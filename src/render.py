"""Render the video: a real fly brain on the left, one paper at a time on the right.

Left panel is the male central nervous system, MaleCNS v1.0, drawn from the annotated
soma positions. The lights are the spike raster recorded in `data/raster.npz` by
`src/raster.py`, replayed millisecond by millisecond. Nothing is a decorative particle.

Right panel is the paper the fly is on, one at a time, and every reference it senses on
that page: red for paywalled, green for open. Both counters accumulate the trace's real
numbers -- they are never interpolated toward the final total.

Each decision plays in three beats:
  SENSE   the references on the page appear, mostly walls
  FIRE    50 ms of brain activity in the 20,000-neuron subgraph
  CHOOSE  the winning descending group flashes green and the fly follows that citation

  python src/render.py --trace data/traces/W123.json

Outputs: site/fly.mp4, site/fly.gif, site/stats.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from brainview import (BrainView, INPUT_OPEN, INPUT_WALL, READOUT, WINNER,
                       role_colours)
from calib import summary_line

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"

W, H, FPS = 1280, 720, 30
BRAIN_W = 716
PANEL_X = BRAIN_W + 1
PANEL_W = W - PANEL_X
PAD = 28
COL = PANEL_W - PAD * 2          # usable text width on the right

BLACK = (7, 9, 14)
PANEL_BG = (13, 15, 21)
HAIR = (36, 41, 54)
WHITE = (240, 243, 248)
GREY = (132, 140, 156)
DIM = (92, 100, 116)
ORANGE = (246, 130, 18)
GREEN = (98, 226, 140)
RED = (240, 82, 96)
CYAN = (60, 214, 255)
AMBER = (255, 186, 72)

# Beats, in frames at 30 fps.
F_SENSE, F_FIRE, F_CHOOSE = 14, 30, 16
F_DECISION = F_SENSE + F_FIRE + F_CHOOSE
RASTER_MS = 50                    # the engine's integration window

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
_CACHE: dict[tuple[int, bool], ImageFont.FreeTypeFont] = {}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    key = (size, bold)
    if key not in _CACHE:
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        path = FONT_DIR / name
        _CACHE[key] = (ImageFont.truetype(str(path), size) if path.exists()
                       else ImageFont.load_default())
    return _CACHE[key]


def text_w(d: ImageDraw.ImageDraw, s: str, f) -> int:
    return int(d.textlength(s, font=f))


def wrap(d: ImageDraw.ImageDraw, s: str, f, width: int, max_lines: int) -> list[str]:
    """Wrap to width, then ellipsise the last line so it can never overflow."""
    words, lines, cur = s.split(), [], ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if text_w(d, trial, f) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
            if len(lines) == max_lines:
                break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    lines = lines[:max_lines]
    if not lines:
        return [""]
    # Anything left over, or a single word wider than the column, gets clipped hard.
    used = len(" ".join(lines).split())
    if used < len(words) or text_w(d, lines[-1], f) > width:
        tail = lines[-1]
        while tail and text_w(d, tail + "...", f) > width:
            tail = tail[:-1]
        lines[-1] = tail.rstrip(" ,;:") + "..."
    return lines


def fit_title(d: ImageDraw.ImageDraw, s: str, width: int, max_lines: int,
              sizes=(22, 20, 18, 16)):
    """Largest size at which the title fits without being clipped, else clipped small."""
    for size in sizes:
        f = font(size, True)
        lines = wrap(d, s, f, width, max_lines)
        if not lines[-1].endswith("..."):
            return f, lines
    f = font(sizes[-1], True)
    return f, wrap(d, s, f, width, max_lines)


def badge(d: ImageDraw.ImageDraw, x: int, y: int, label: str, fg, bg) -> int:
    f = font(13, True)
    tw = text_w(d, label, f)
    d.rounded_rectangle([x, y, x + tw + 18, y + 22], 4, fill=bg, outline=fg)
    d.text((x + 9, y + 4), label, font=f, fill=fg)
    return x + tw + 18


def scrim(img: Image.Image, box, alpha: int = 165, fade: str | None = None) -> None:
    """Darken a box. `fade` ramps the alpha in from 'top' or 'bottom' edge."""
    w, h = box[2] - box[0], box[3] - box[1]
    a = np.full((h, w), alpha, dtype=np.uint8)
    if fade == "top":
        a = (np.linspace(0, alpha, h)[:, None] * np.ones((1, w))).astype(np.uint8)
    elif fade == "bottom":
        a = (np.linspace(alpha, 0, h)[:, None] * np.ones((1, w))).astype(np.uint8)
    layer = Image.fromarray(np.dstack([
        np.full((h, w), 5, np.uint8), np.full((h, w), 7, np.uint8),
        np.full((h, w), 12, np.uint8), a]), "RGBA")
    img.paste(Image.alpha_composite(
        img.crop(box).convert("RGBA"), layer).convert("RGB"), (box[0], box[1]))


OA_STYLE = {
    "gold": ("OPEN ACCESS", GREEN),
    "green": ("OPEN ACCESS", GREEN),
    "diamond": ("OPEN ACCESS", GREEN),
    "hybrid": ("OPEN, PAID", AMBER),
    "bronze": ("FREE TO READ, NO LICENCE", AMBER),
    "closed": ("PAYWALLED", RED),
    "unknown": ("NO OPEN COPY FOUND", RED),
    "unresolved": ("NO OPEN COPY FOUND", RED),
}


def oa_badge(status: str):
    return OA_STYLE.get((status or "unknown").lower(), OA_STYLE["unknown"])


# --------------------------------------------------------------------------- panels


def brain_overlay(img: Image.Image, caption: str, ms: int | None, placed: int,
                  total: int) -> None:
    d = ImageDraw.Draw(img)
    f_lab, f_small = font(14, True), font(13)

    scrim(img, (0, 0, BRAIN_W, 58), 150)
    d.text((PAD, 14), "MALE CENTRAL NERVOUS SYSTEM", font=f_lab, fill=WHITE)
    d.text((PAD, 34), f"MaleCNS v1.0 - {total:,} annotated somas, "
                      f"{placed:,} of them simulated", font=f_small, fill=GREY)
    if ms is not None:
        stamp = f"t = {ms:2d} ms"
        d.text((BRAIN_W - PAD - text_w(d, stamp, f_lab), 14), stamp,
               font=f_lab, fill=ORANGE)

    scrim(img, (0, H - 74, BRAIN_W, H), 165)
    d.text((PAD, H - 64), caption, font=f_small, fill=GREY)
    keys = [(CYAN, "open reference"), (RED, "paywall drive"),
            (AMBER, "descending neurons"), (GREEN, "group that won")]
    x = PAD
    for colour, label in keys:
        d.ellipse([x, H - 36, x + 9, H - 27], fill=colour)
        d.text((x + 16, H - 39), label, font=f_small, fill=GREY)
        x += 16 + text_w(d, label, f_small) + 26


def draw_panel(d: ImageDraw.ImageDraw, *, step_no: int, step_total: int, phase: str,
               phase_colour, paper: dict, n_open: int, n_wall: int, reveal: float,
               note: str, note_colour, papers_read: int, walls_hit: int,
               winner_pulse: float) -> None:
    """The right-hand column. Every element sits at a fixed y so it never jumps."""
    d.rectangle([PANEL_X, 0, W, H], fill=PANEL_BG)
    d.line([PANEL_X, 0, PANEL_X, H], fill=HAIR)
    x = PANEL_X + PAD

    d.text((x, 22), f"DECISION {step_no} OF {step_total}", font=font(14, True), fill=DIM)
    d.text((x, 44), phase, font=font(21, True), fill=phase_colour)

    # -- the one paper -------------------------------------------------------
    d.line([x, 80, x + COL, 80], fill=HAIR)
    d.text((x, 92), "NOW READING", font=font(13, True), fill=ORANGE)

    # Fixed line pitch and a 4-line block, so nothing below can ever be pushed.
    f_title, lines = fit_title(d, paper.get("title") or "Untitled record", COL, 4,
                               sizes=(20, 18, 16, 15))
    for i, line in enumerate(lines):
        d.text((x, 112 + i * 25), line, font=f_title, fill=WHITE)

    venue = paper.get("journal") or "Venue not recorded"
    year = paper.get("year")
    meta = f"{venue}{f'  -  {year}' if year else ''}"
    for i, line in enumerate(wrap(d, meta, font(15), COL, 2)):
        d.text((x, 214 + i * 19), line, font=font(15), fill=GREY)
    label, colour = oa_badge(paper.get("oa_status"))
    badge(d, x, 256, label, colour, (colour[0] // 7, colour[1] // 7, colour[2] // 7))

    # -- references sensed on this page --------------------------------------
    d.line([x, 292, x + COL, 292], fill=HAIR)
    d.text((x, 302), "REFERENCES ON THIS PAGE", font=font(13, True), fill=ORANGE)
    total = n_open + n_wall
    d.text((x, 324), f"{total} sensed  -  {n_wall} paywalled  -  {n_open} open",
           font=font(15, True), fill=WHITE)

    cell, gap, cols, rows = 13, 4, 29, 4
    cap = cols * rows
    shown_open = min(n_open, cap)
    shown_wall = min(n_wall, max(cap - shown_open, 0))
    order = [GREEN] * shown_open + [RED] * shown_wall
    lit = int(round(len(order) * max(0.0, min(1.0, reveal))))
    gy = 348
    for i, colour in enumerate(order):
        cx = x + (i % cols) * (cell + gap)
        cy = gy + (i // cols) * (cell + gap)
        if i < lit:
            d.rectangle([cx, cy, cx + cell, cy + cell], fill=colour)
        else:
            d.rectangle([cx, cy, cx + cell, cy + cell], outline=(30, 34, 46))
    hidden = total - (shown_open + shown_wall)
    grid_bottom = gy + rows * (cell + gap)
    if hidden > 0:
        d.text((x, grid_bottom), f"+{hidden} more paywalled", font=font(13), fill=DIM)

    # -- fixed-height note strip --------------------------------------------
    ny = 442
    d.line([x, ny, x + COL, ny], fill=HAIR)
    for i, line in enumerate(wrap(d, note, font(15), COL, 2)):
        d.text((x, ny + 10 + i * 21), line, font=font(15), fill=note_colour)

    # -- counters ------------------------------------------------------------
    cy = 508
    d.line([x, cy, x + COL, cy], fill=HAIR)
    f_num, f_lab = font(42, True), font(13, True)
    d.text((x, cy + 20), f"{papers_read:,}", font=f_num, fill=GREEN)
    d.text((x, cy + 70), "PAPERS READ", font=f_lab, fill=GREY)
    x2 = x + COL // 2
    glow = int(220 + 35 * winner_pulse)
    d.text((x2, cy + 20), f"{walls_hit:,}", font=f_num, fill=(glow, 82, 96))
    d.text((x2, cy + 70), "PAYWALLS HIT", font=f_lab, fill=GREY)

    d.line([x, H - 62, x + COL, H - 62], fill=HAIR)
    d.text((x, H - 46), "fruitfly.crimconsortium.com", font=font(14, True), fill=DIM)


def card(lines, sub: str | None = None, bg=BLACK) -> Image.Image:
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    height = sum(f.size + 16 for _, f, _ in lines)
    y = (H - height) // 2 - 20
    for text, f, colour in lines:
        d.text(((W - text_w(d, text, f)) // 2, y), text, font=f, fill=colour)
        y += f.size + 16
    if sub:
        f = font(17)
        for i, line in enumerate(wrap(d, sub, f, int(W * 0.72), 3)):
            d.text(((W - text_w(d, line, f)) // 2, y + 24 + i * 26), line,
                   font=f, fill=GREY)
    return img


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--raster", default="data/raster.npz")
    ap.add_argument("--decisions", type=int, default=12,
                    help="how many opening decisions to play in full")
    args = ap.parse_args()

    trace = json.loads(Path(args.trace).read_text())
    SITE.mkdir(exist_ok=True)
    nodes = {n["id"]: n for n in trace["nodes"]}
    steps = trace["steps"]
    seed = trace["seed"]
    result = trace["result"]
    calib = trace.get("calibration") or {}
    n_ch = int(trace["engine"].get("channels", 8))

    r = np.load(ROOT / args.raster)
    have = {int(k.split("_")[1]) for k in r.files if k.startswith("idx_")}
    bv = BrainView(BRAIN_W, H, ROOT / "data" / "connectome" / "soma_xyz.parquet",
                   r["body"], pitch_deg=26.0, yaw_range=(-22.0, 22.0), fill=1.0)
    print(f"brain: {bv.n_points:,} somas, {bv.n_placed:,} of "
          f"{bv.n_sim:,} simulated neurons placed")

    # Real running totals, straight off the trace. Never interpolated.
    # `before` is the state on arriving at a decision, `after` once it is taken.
    read_before, wall_before, read_after, wall_after = [], [], [], []
    running_read, running_wall = 1, 0
    for s in steps:
        read_before.append(running_read)
        wall_before.append(running_wall)
        running_wall += len(s.get("bumps", []))
        if s.get("to"):
            running_read += 1
        read_after.append(running_read)
        wall_after.append(running_wall)
    assert read_after[-1] == result["reachable"], "paper count must match the trace"
    assert wall_after[-1] == result["walls_hit"], "wall count must match the trace"

    playable = [i for i in sorted(have) if i < len(steps)]
    head = [i for i in playable if i < len(steps) - 1][:args.decisions]
    last_fired = playable[-1] if playable else None
    if last_fired in head:
        last_fired = None

    frames: list[np.ndarray] = []
    yaw_period = FPS * 17.0

    def yaw_at(n: int) -> float:
        return 22.0 * float(np.sin(2 * np.pi * n / yaw_period))

    def paper_of(step) -> dict:
        return nodes.get(step["at"], seed)

    def compose(step_idx: int, step_no: int, phase: str, phase_colour, reveal: float,
                note: str, note_colour, ms: int | None, caption: str,
                read: int, walls: int, pulse: float = 0.0) -> np.ndarray:
        step = steps[step_idx]
        left = bv.render(yaw_at(len(frames)))
        brain_overlay(left, caption, ms, bv.n_placed, bv.n_points)
        img = Image.new("RGB", (W, H), BLACK)
        img.paste(left, (0, 0))
        draw_panel(ImageDraw.Draw(img), step_no=step_no, step_total=len(steps),
                   phase=phase, phase_colour=phase_colour, paper=paper_of(step),
                   n_open=len(step["candidates"]), n_wall=len(step.get("bumps", [])),
                   reveal=reveal, note=note, note_colour=note_colour,
                   papers_read=read, walls_hit=walls, winner_pulse=pulse)
        return np.asarray(img)

    def play(step_idx: int) -> None:
        step = steps[step_idx]
        step_no = step_idx + 1
        cands = step["candidates"]
        bumps = step.get("bumps", [])
        open_ch = {c["channel"] for c in cands}
        wall_ch = (len(cands) % n_ch) if bumps else None
        chosen = next((c for c in cands if c["id"] == step.get("to")), None)
        win_ch = chosen["channel"] if chosen else None
        colours, gains = role_colours(r["is_input"], r["input_channel"],
                                      r["is_readout"], r["readout_channel"],
                                      open_ch, wall_ch, win_ch)

        # SENSE -- the page's references come in. The brain is still.
        bv.heat[:] = 0.0
        bv.tint[:] = 0.0
        sense_note = (f"{len(bumps)} of these references are behind a paywall. "
                      f"Only the open ones can drive the brain.")
        for f in range(F_SENSE):
            frames.append(compose(
                step_idx, step_no, "SENSING REFERENCES", CYAN,
                (f + 1) / F_SENSE, sense_note, GREY, None,
                "The fly senses the references on this page. It is not firing yet.",
                read_before[step_idx], wall_before[step_idx]))

        # FIRE -- replay the recorded raster, real milliseconds.
        ms_arr, idx_arr = r[f"ms_{step_idx}"], r[f"idx_{step_idx}"]
        note = (f"{RASTER_MS} ms of activity in the 20,000-neuron subgraph. "
                f"{step['total_spikes']:,} spikes.")
        for f in range(F_FIRE):
            lo = int(round(RASTER_MS * f / F_FIRE))
            hi = int(round(RASTER_MS * (f + 1) / F_FIRE))
            bv.decay_heat()
            sel = idx_arr[(ms_arr >= lo) & (ms_arr < hi)]
            bv.fire(sel, colours[sel], gains[sel])
            frames.append(compose(
                step_idx, step_no, "BRAIN FIRING", ORANGE, 1.0, note, GREY, hi,
                "Every lit dot is one neuron that spiked, at its real soma position.",
                read_before[step_idx], wall_after[step_idx]))

        # CHOOSE -- the winning descending group, then the citation it follows.
        target = nodes.get(step.get("to"), {})
        if chosen:
            note = f"\u2192 next: {target.get('title') or step.get('to')}"
            note_colour = GREEN
        else:
            note = "No open reference left on this page. The walk stops here."
            note_colour = RED
        win_rows = np.flatnonzero(r["is_readout"] & (r["readout_channel"] == win_ch)) \
            if win_ch is not None else np.array([], dtype=int)
        for f in range(F_CHOOSE):
            bv.decay_heat()
            if len(win_rows) and f < 6:
                bv.fire(win_rows, colours[win_rows], gains[win_rows] * 1.4)
            frames.append(compose(
                step_idx, step_no, "FOLLOWING THE CITATION", GREEN, 1.0,
                note, note_colour, RASTER_MS,
                "Green is the descending group with the most spikes. That is the choice.",
                read_after[step_idx], wall_after[step_idx],
                pulse=float(np.sin(np.pi * f / F_CHOOSE))))

    # -- intro: the resting nervous system, full frame ------------------------
    WIDE_H = 440
    wide = BrainView(W, WIDE_H, ROOT / "data" / "connectome" / "soma_xyz.parquet",
                     r["body"], pitch_deg=26.0, yaw_range=(-22.0, 22.0), fill=0.92)

    def wide_frame(n: int, brightness: float) -> tuple[Image.Image, ImageDraw.ImageDraw]:
        img = Image.new("RGB", (W, H), BLACK)
        img.paste(wide.render(yaw_at(n), brightness=brightness), (0, 0))
        return img, ImageDraw.Draw(img)

    for f in range(int(2.8 * FPS)):
        img, d = wide_frame(f, 2.3)
        f_big, f_mid = font(44, True), font(19)
        for i, line in enumerate(["A FRUIT FLY BRAIN LOOKS FOR",
                                  "CRIMINOLOGY IT CAN READ"]):
            d.text(((W - text_w(d, line, f_big)) // 2, 468 + i * 54), line,
                   font=f_big, fill=WHITE if i == 0 else ORANGE)
        sub = ("MaleCNS v1.0 connectome (CC-BY) driving a walk through OpenAlex (CC0). "
               "Open access is a corridor. Paywalls are walls.")
        for i, line in enumerate(wrap(d, sub, f_mid, int(W * 0.66), 2)):
            d.text(((W - text_w(d, line, f_mid)) // 2, 596 + i * 26), line,
                   font=f_mid, fill=GREY)
        frames.append(np.asarray(img))

    for i in head:
        play(i)

    # -- the long middle, stated rather than faked ---------------------------
    if last_fired is not None:
        skipped = last_fired - head[-1] - 1
        jump = card([
            (f"{skipped} more decisions", font(46, True), WHITE),
            ("play out the same way", font(46, True), ORANGE),
        ], sub=(f"Between decision {head[-1] + 1} and decision {last_fired + 1} the fly "
                f"hits {wall_before[last_fired] - wall_after[head[-1]]:,} more "
                f"paywalls and reads "
                f"{read_before[last_fired] - read_after[head[-1]]:,} more papers."))
        frames += [np.asarray(jump)] * int(2.1 * FPS)
        play(last_fired)

    # -- the end of the road: nothing open left anywhere ----------------------
    stuck = len(steps) - 1
    if not steps[stuck]["candidates"]:
        for f in range(int(2.6 * FPS)):
            fade = max(0.14, 0.55 * (1.0 - f / (1.4 * FPS)))
            img, d = wide_frame(len(frames), fade)
            f_big, f_mid = font(44, True), font(18)
            d.text(((W - text_w(d, "NOWHERE LEFT TO GO", f_big)) // 2, 458),
                   "NOWHERE LEFT TO GO", font=f_big, fill=RED)
            f_sub = font(28, True)
            d.text(((W - text_w(d, "THE BRAIN GOES QUIET", f_sub)) // 2, 512),
                   "THE BRAIN GOES QUIET", font=f_sub, fill=WHITE)
            body = (f"Every open-access reference reachable from these "
                    f"{result['reachable']:,} papers has already been read. "
                    f"Everything else is a wall, so there is nothing left to fire on.")
            for i, line in enumerate(wrap(d, body, f_mid, int(W * 0.60), 3)):
                d.text(((W - text_w(d, line, f_mid)) // 2, 560 + i * 24), line,
                       font=f_mid, fill=GREY)
            frames.append(np.asarray(img))

    # -- outro ---------------------------------------------------------------
    line = summary_line(calib)
    end = card([
        (f"{result['reachable']:,} papers read", font(52, True), GREEN),
        (f"{result['walls_hit']:,} paywalls hit", font(52, True), RED),
        ("then it ran out of open access", font(24), GREY),
    ], sub=(line or "") + "   fruitfly.crimconsortium.com")
    frames += [np.asarray(end)] * int(3.4 * FPS)

    # A point cloud is high-entropy, so it needs a real rate-controlled encode
    # rather than imageio's default quality knob, or the file lands near 40 MB.
    mp4 = SITE / "fly.mp4"
    imageio.mimwrite(mp4, frames, fps=FPS, codec="libx264", macro_block_size=1,
                     pixelformat="yuv420p",
                     output_params=["-crf", "25", "-preset", "slow",
                                    "-movflags", "+faststart"])
    print(f"{mp4}: {mp4.stat().st_size / 1e6:.1f} MB, {len(frames)} frames, "
          f"{len(frames) / FPS:.1f}s")

    # The GIF is the social preview, so it is a short excerpt: the opening and the
    # first couple of decisions, at 10 fps.
    excerpt = frames[:int(7.2 * FPS)][::3]
    prepped = [Image.fromarray(f).resize((640, 360), Image.BOX)
               .filter(ImageFilter.GaussianBlur(0.5)) for f in excerpt]
    # One palette shared by every frame: the role colours survive and successive
    # frames compress against each other instead of flickering.
    stack = Image.new("RGB", (640, 360 * 6))
    for k, i in enumerate(np.linspace(0, len(prepped) - 1, 6).astype(int)):
        stack.paste(prepped[i], (0, k * 360))
    palette = stack.quantize(colors=96, dither=Image.NONE)
    small = [np.asarray(p.quantize(palette=palette, dither=Image.NONE).convert("RGB"))
             for p in prepped]
    gif = SITE / "fly.gif"
    imageio.mimwrite(gif, small, duration=1000 / (FPS / 3), loop=0)
    print(f"{gif}: {gif.stat().st_size / 1e6:.1f} MB, {len(small)} frames")

    (SITE / "stats.json").write_text(json.dumps({
        "seed": seed, "result": result, "calibration": calib,
        "calibration_line": line,
        "policy": trace["policy"], "engine": trace["engine"],
    }, indent=2))


if __name__ == "__main__":
    main()
