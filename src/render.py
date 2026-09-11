"""Render a trace/v1 file to an MP4 and a looping GIF.

Reads only the trace. Draws only what the trace contains: no invented spikes, no
smoothed-over gaps.

LAYOUT:
  - Left 65%: The Browser / Reading View. Shows the manuscript card the fly is
    inspecting: Title, Journal, Year, Access Status badge (OPEN vs PAYWALLED/BRONZE),
    plus live citations and references. The Fly is prominently visible (large animated
    sprite) sitting on / examining the paper. When bumping a paywall, an alert banner
    and visual recoil trigger.
  - Right 35%: Connectome Readout & Live Counters (Papers Read, Paywalls Hit, Firing Rate).

  python src/render.py --trace data/traces/W123.json

Outputs: site/fly.mp4, site/fly.gif, site/stats.json
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from calib import summary_line

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"

W, H, FPS = 1280, 720, 30
LEFT_W = int(W * 0.65)
PAD = 28

BLACK = (0, 0, 0)
ORANGE = (246, 130, 18)
WHITE = (245, 245, 245)
GREY = (140, 140, 140)
DARK_BG = (16, 16, 16)
CARD_BG = (24, 24, 24)
BORDER_COL = (45, 45, 45)
GREEN = (46, 204, 113)
RED = (231, 76, 60)

FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def font(size: int, bold: bool = False):
    p = FONT_PATHS[0] if bold else FONT_PATHS[1]
    if Path(p).exists():
        return ImageFont.truetype(p, size)
    for alt in FONT_PATHS:
        if Path(alt).exists():
            return ImageFont.truetype(alt, size)
    return ImageFont.load_default()


F_BIG = font(44, bold=True)
F_MID = font(26, bold=True)
F_BODY = font(18)
F_BODY_B = font(18, bold=True)
F_SMALL = font(15)
F_TINY = font(13)


def draw_fly_large(d: ImageDraw.ImageDraw, x: float, y: float, wings_up: bool, s: int = 8, alert: bool = False):
    """Draw a large, distinct fruit fly pixel character inspecting the page."""
    body = [
        (0, -3), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (0, 3),
        (-1, -2), (1, -2), (-1, -1), (1, -1), (-1, 0), (1, 0),
        (-1, 1), (1, 1), (-1, 2), (1, 2), (-2, 0), (2, 0), (-2, 1), (2, 1)
    ]
    b_col = (240, 240, 240) if not alert else (255, 100, 100)
    for bx, by in body:
        d.rectangle([x + bx * s, y + by * s, x + bx * s + s, y + by * s + s], fill=b_col)

    # Eyes: bright orange compound eyes
    for ex in (-2, 2):
        d.rectangle([x + ex * s, y - 4 * s, x + ex * s + s, y - 3 * s], fill=ORANGE)
        d.rectangle([x + ex * s, y - 3 * s, x + ex * s + s, y - 2 * s], fill=ORANGE)

    # Wings: fluttering translucent wings
    wy = -4 if wings_up else -1
    w_col = (180, 180, 190)
    for wx in (-5, -4, -3, 3, 4, 5):
        d.rectangle([x + wx * s, y + wy * s, x + wx * s + s, y + wy * s + s], fill=w_col)
        d.rectangle([x + wx * s, y + (wy + 1) * s, x + wx * s + s, y + (wy + 1) * s + s], fill=w_col)

    # Legs
    leg_col = (100, 100, 100)
    for lx, ly in [(-3, -2), (-4, 0), (-3, 3), (3, -2), (4, 0), (3, 3)]:
        d.rectangle([x + lx * s, y + ly * s, x + lx * s + s, y + ly * s + s], fill=leg_col)


def panel_frame(trace, current_node, is_bump, step, wings_up, fly_xy, papers_read, paywalls_hit):
    img = Image.new("RGB", (W, H), BLACK)
    d = ImageDraw.Draw(img)

    # Divider
    d.line([LEFT_W, 0, LEFT_W, H], fill=BORDER_COL, width=2)

    # Header
    d.text((PAD, PAD), "A FRUIT FLY LOOKS FOR CRIMINOLOGY IT CAN READ", font=F_MID, fill=ORANGE)
    d.text((PAD, PAD + 32), "Real MaleCNS v1.0 connectome inspecting OpenAlex literature", font=F_SMALL, fill=GREY)

    # Manuscript Browser Card (Left area)
    card_x1, card_y1 = PAD, PAD + 64
    card_x2, card_y2 = LEFT_W - PAD, H - PAD - 40
    d.rectangle([card_x1, card_y1, card_x2, card_y2], fill=CARD_BG, outline=BORDER_COL, width=2)

    # Header inside manuscript card
    d.rectangle([card_x1, card_y1, card_x2, card_y1 + 44], fill=(32, 32, 32))
    d.text((card_x1 + 16, card_y1 + 12), "CURRENT PAPER METADATA", font=F_TINY, fill=GREY)

    # Status Badge
    oa_status = (current_node.get("oa_status") or "unknown").upper()
    passable = current_node.get("passable", False)
    badge_col = GREEN if passable else RED
    badge_txt = f"[{oa_status} - READABLE]" if passable else f"[{oa_status} - PAYWALLED]"
    d.text((card_x2 - 190, card_y1 + 12), badge_txt, font=F_BODY_B, fill=badge_col)

    # Paper Title (wrapped)
    title_raw = current_node.get("title") or "Unknown Title"
    lines = textwrap.wrap(title_raw, width=44)
    ty = card_y1 + 64
    for line in lines[:3]:
        d.text((card_x1 + 20, ty), line, font=F_MID, fill=WHITE)
        ty += 32

    # Journal & Year
    j_name = current_node.get("journal") or "Unknown Journal"
    yr = current_node.get("year") or ""
    d.text((card_x1 + 20, ty + 10), f"Journal: {j_name} ({yr})", font=F_BODY_B, fill=ORANGE)

    # Inspection details / Abstract simulation
    d.line([card_x1 + 20, ty + 44, card_x2 - 20, ty + 44], fill=BORDER_COL, width=1)
    d.text((card_x1 + 20, ty + 56), "Citation Network Status:", font=F_SMALL, fill=GREY)

    cites = current_node.get("cited_by_count", 0)
    d.text((card_x1 + 20, ty + 80), f"- Total citations recorded: {cites}", font=F_BODY, fill=WHITE)

    if is_bump:
        # Visual alert when hitting a wall
        d.rectangle([card_x1 + 20, ty + 120, card_x2 - 20, ty + 170], fill=(60, 20, 20), outline=RED, width=2)
        d.text((card_x1 + 36, ty + 132), "ACCESS DENIED: Paywall encountered!", font=F_BODY_B, fill=RED)
    else:
        d.rectangle([card_x1 + 20, ty + 120, card_x2 - 20, ty + 170], fill=(20, 45, 25), outline=GREEN, width=1)
        d.text((card_x1 + 36, ty + 132), "FULL TEXT ACCESSED: Paper read successfully.", font=F_BODY_B, fill=GREEN)

    # Large Animated Fly inspecting the document
    draw_fly_large(d, fly_xy[0], fly_xy[1], wings_up, s=7, alert=is_bump)

    # Footer on left
    d.text((PAD, H - PAD - 20), "MaleCNS v1.0 Connectome (CC-BY) x OpenAlex (CC0)", font=F_TINY, fill=GREY)

    # Right Panel: HUD & Spikes
    x0 = LEFT_W + PAD
    d.text((x0, PAD), "DESCENDING-NEURON READOUT", font=F_TINY, fill=ORANGE)
    d.text((x0, PAD + 20), "mean spikes per channel", font=F_TINY, fill=GREY)

    cands = step.get("candidates", []) if step else []
    top = max([c.get("readout_spikes", 0) for c in cands], default=1.0) or 1.0
    bar_y = PAD + 56
    for i in range(8):
        val = next((c.get("readout_spikes", 0) for c in cands if c.get("channel") == i), 0.0)
        w_bar = int((val / top) * (W - x0 - PAD - 46))
        d.text((x0, bar_y), f"{i}", font=F_TINY, fill=GREY)
        d.rectangle([x0 + 18, bar_y, x0 + 18 + max(w_bar, 1), bar_y + 14],
                    fill=ORANGE if val == top and val > 0 else (90, 90, 90))
        bar_y += 24

    y2 = bar_y + 36
    d.text((x0, y2), f"papers read      {papers_read}", font=F_MID, fill=WHITE)
    d.text((x0, y2 + 38), f"paywalls hit     {paywalls_hit}", font=F_MID, fill=ORANGE)
    if step:
        d.text((x0, y2 + 76), f"spikes this step {step.get('total_spikes', 0):,}",
               font=F_SMALL, fill=GREY)

    d.text((x0, H - PAD - 42), "CRIMCONSORTIUM", font=F_SMALL, fill=ORANGE)
    d.text((x0, H - PAD - 18), "fruitfly.crimconsortium.com", font=F_TINY, fill=GREY)
    return img


def card(lines, sub=None):
    img = Image.new("RGB", (W, H), BLACK)
    d = ImageDraw.Draw(img)
    y = H // 2 - 26 * len(lines)
    for text, f, col in lines:
        w = d.textlength(text, font=f)
        d.text(((W - w) / 2, y), text, font=f, fill=col)
        y += f.size + 18
    if sub:
        w = d.textlength(sub, font=F_TINY)
        d.text(((W - w) / 2, H - PAD - 40), sub, font=F_TINY, fill=GREY)
    d.text((PAD, H - PAD - 20), "CRIMCONSORTIUM", font=F_SMALL, fill=ORANGE)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()

    trace = json.loads(Path(args.trace).read_text())
    SITE.mkdir(exist_ok=True)

    nodes_map = {n["id"]: n for n in trace["nodes"]}
    calib = trace.get("calibration") or {}
    seed = trace["seed"]

    title = card([
        ("A FRUIT FLY", F_BIG, WHITE),
        ("LOOKS FOR CRIMINOLOGY", F_BIG, WHITE),
        ("IT CAN READ", F_BIG, ORANGE),
    ], sub="MaleCNS v1.0 connectome (CC-BY) x OpenAlex (CC0). Open access = diamond, gold, green, hybrid.")

    steps = [s for s in trace["steps"] if s.get("to") or s.get("bumps")]
    total_frames = int(args.seconds * FPS)
    title_frames, card_frames = 2 * FPS, 4 * FPS
    walk_frames = max(total_frames - title_frames - card_frames, FPS)
    per_step = max(6, walk_frames // max(len(steps), 1))

    frames = [np.asarray(title)] * title_frames
    papers_read = 1
    paywalls_hit = 0

    fly_base_x = LEFT_W - 140
    fly_base_y = H // 2 + 60

    for s_idx, s in enumerate(steps):
        target_id = s.get("to") or s.get("at")
        curr_node = nodes_map.get(target_id, seed)
        bumps = s.get("bumps", [])
        paywalls_hit += len(bumps)
        if s.get("to") and curr_node.get("passable"):
            papers_read += 1

        for f in range(per_step):
            wings_up = (f // 3) % 2 == 0
            bob_x = fly_base_x + int(np.sin(f * 0.4) * 15)
            bob_y = fly_base_y + int(np.cos(f * 0.3) * 10)
            is_bump_frame = bool(bumps) and (f < per_step // 2)

            img = panel_frame(trace, curr_node, is_bump_frame, s, wings_up, (bob_x, bob_y),
                              papers_read, paywalls_hit)
            frames.append(np.asarray(img))

    r = trace["result"]
    sel = summary_line(calib)
    end = card([
        (f"{r['reachable']} papers read", F_BIG, WHITE),
        (f"{r['walls_hit']} paywalls hit", F_BIG, ORANGE),
        ("then it ran out of open access", F_MID, GREY),
    ], sub=sel or "fruitfly.crimconsortium.com")
    frames += [np.asarray(end)] * card_frames

    mp4 = SITE / "fly.mp4"
    imageio.mimwrite(mp4, frames, fps=FPS, quality=8, macro_block_size=1)
    print(f"{mp4}: {mp4.stat().st_size / 1e6:.1f} MB, {len(frames)} frames")

    small = [np.asarray(Image.fromarray(f).resize((640, 360), Image.NEAREST))
             for f in frames[::3]]
    gif = SITE / "fly.gif"
    imageio.mimwrite(gif, small, duration=1000 / (FPS / 3), loop=0)
    mb = gif.stat().st_size / 1e6
    print(f"{gif}: {mb:.1f} MB, {len(small)} frames" + ("  OVER 15 MB LIMIT" if mb > 15 else ""))

    (SITE / "stats.json").write_text(json.dumps({
        "seed": seed, "result": r, "calibration": calib,
        "calibration_line": sel,
        "policy": trace["policy"], "engine": trace["engine"],
    }, indent=2))


if __name__ == "__main__":
    main()
