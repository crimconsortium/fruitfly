"""Render an animated 'Single Paper Inspection' feed of the fly foraging criminology.

Radically simplified, clean visual model:
  - EXACTLY ONE paper card on screen at a time in the center.
  - No confusing rows of 4 doors or changing layouts.
  - The animated fruit fly crawls across the manuscript inspection card.
  - For each paper encountered:
      * If OPEN ACCESS (Diamond/Gold/Green/Hybrid):
          - Bright green header: [OPEN ACCESS - READABLE]
          - Full paper title, journal, and year clearly readable.
          - Fly crawls smoothly across the page, takes notes, and papers_read increments.
      * If PAYWALLED (Closed / Bronze / Unresolved):
          - Bright red header: [PAYWALLED - $39.95 / ACCESS DENIED]
          - Heavy red locked barrier stamp slams across the card.
          - Screen shake effect, fly bounces back / turns away, and paywalls_hit increments.
  - Right panel: Live motor channel firing (Channel 0 to 7), total spikes,
    and big retro arcade scorecards for PAPERS READ and PAYWALLS HIT.

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
LEFT_W = int(W * 0.68)
PAD = 28

BLACK = (12, 13, 16)
CARD_BG = (22, 24, 30)
CARD_BORDER = (45, 48, 58)
ORANGE = (246, 130, 18)
WHITE = (245, 245, 245)
GREY = (145, 150, 160)
LIGHT_GREY = (200, 205, 215)
GREEN = (46, 204, 113)
RED = (231, 76, 60)
RED_BG = (55, 22, 24)
GREEN_BG = (20, 50, 32)

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


F_BIG = font(40, bold=True)
F_TITLE = font(22, bold=True)
F_MID = font(18, bold=True)
F_BODY = font(16)
F_SMALL = font(13)
F_TINY = font(11)


def draw_fly(d: ImageDraw.ImageDraw, cx: float, cy: float, angle_deg: float, wings_up: bool, s: int = 7, alert: bool = False):
    """Draw an animated 2D fly sprite with compound eyes and fluttering wings."""
    rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)

    def rot(dx, dy):
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        return cx + rx * s, cy + ry * s

    body_pixels = [
        (0, -3), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (0, 3),
        (-1, -2), (1, -2), (-1, -1), (1, -1), (-1, 0), (1, 0),
        (-1, 1), (1, 1), (-1, 2), (1, 2)
    ]
    b_col = (245, 245, 245) if not alert else (255, 100, 100)
    for bx, by in body_pixels:
        px, py = rot(bx, by)
        d.rectangle([px - s / 2, py - s / 2, px + s / 2, py + s / 2], fill=b_col)

    # Compound eyes
    for ex in (-2, 2):
        px, py = rot(ex, -3)
        d.rectangle([px - s / 2, py - s / 2, px + s / 2, py + s / 2], fill=ORANGE)

    # Translucent wings
    wy = -4 if wings_up else -1
    w_col = (180, 190, 210)
    for wx in (-5, -4, -3, 3, 4, 5):
        px, py = rot(wx, wy)
        d.rectangle([px - s / 2, py - s / 2, px + s / 2, py + s / 2], fill=w_col)

    # Legs
    for lx, ly in [(-3, -2), (-4, 0), (-3, 2), (3, -2), (4, 0), (3, 2)]:
        px, py = rot(lx, ly)
        d.rectangle([px - s / 2, py - s / 2, px + s / 2, py + s / 2], fill=(100, 105, 115))


def render_single_paper_frame(paper, is_paywalled, encounter_num, total_encounters,
                              fly_state, step_data, papers_read, paywalls_hit, shake=(0, 0)):
    """Render ONE single clean paper card with the fly inspecting it."""
    img = Image.new("RGB", (W, H), BLACK)
    d = ImageDraw.Draw(img)

    sx, sy = shake
    cx1, cy1 = PAD + sx, PAD + 54 + sy
    cx2, cy2 = LEFT_W - PAD + sx, H - PAD - 20 + sy

    # Top Banner Header
    d.text((PAD, PAD + 10), "A FRUIT FLY LOOKS FOR CRIMINOLOGY IT CAN READ", font=F_TITLE, fill=ORANGE)
    d.text((PAD, PAD + 34), "MaleCNS v1.0 connectome (166k neurons) foraging research literature", font=F_TINY, fill=GREY)

    # Single Paper Card Container
    card_bg = RED_BG if is_paywalled else CARD_BG
    card_outline = RED if is_paywalled else (GREEN if paper.get("passable") else CARD_BORDER)
    d.rectangle([cx1, cy1, cx2, cy2], fill=card_bg, outline=card_outline, width=3)

    # Card Top Header Bar: Status & Counter
    header_h = 56
    bar_fill = (45, 15, 18) if is_paywalled else ((20, 50, 30) if paper.get("passable") else (30, 32, 40))
    d.rectangle([cx1, cy1, cx2, cy1 + header_h], fill=bar_fill, outline=card_outline, width=1)

    status_tag = (paper.get("oa_status") or ("CLOSED" if is_paywalled else "OPEN")).upper()
    if is_paywalled:
        badge_txt = f"PAYWALL  [{status_tag}] — ACCESS DENIED ($39.95)"
        badge_col = RED
    else:
        badge_txt = f"CORRIDOR  [{status_tag}] — FULL TEXT ACCESSIBLE"
        badge_col = GREEN

    d.text((cx1 + 18, cy1 + 16), badge_txt, font=F_MID, fill=badge_col)
    progress_txt = f"Encounter {encounter_num} of {total_encounters}"
    d.text((cx2 - d.textlength(progress_txt, font=F_SMALL) - 18, cy1 + 18), progress_txt, font=F_SMALL, fill=GREY)

    # Paper Title (Large, Clean Wrap)
    raw_title = paper.get("title") or "Unknown Research Manuscript"
    lines = textwrap.wrap(raw_title, width=42)
    ty = cy1 + 80
    for line in lines[:3]:
        d.text((cx1 + 24, ty), line, font=F_BIG, fill=WHITE)
        ty += 48

    # Journal & Publication Year
    j_name = paper.get("journal") or "Academic Journal"
    yr = paper.get("year") or ""
    j_str = f"Published in {j_name} ({yr})"
    d.text((cx1 + 24, ty + 12), j_str, font=F_MID, fill=ORANGE)

    # Metadata Divider
    d.line([cx1 + 24, ty + 50, cx2 - 24, ty + 50], fill=(60, 65, 78), width=1)

    # Simulated Abstract / Content Lines
    meta_y = ty + 68
    cites = paper.get("cited_by_count", 0)
    d.text((cx1 + 24, meta_y), f"Citation Impact: {cites:,} cited references in OpenAlex corpus", font=F_BODY, fill=LIGHT_GREY)

    # Large Access Stamp Box at bottom of card
    stamp_y1 = cy2 - 110
    stamp_y2 = cy2 - 24
    if is_paywalled:
        d.rectangle([cx1 + 24, stamp_y1, cx2 - 24, stamp_y2], fill=(60, 18, 20), outline=RED, width=2)
        d.text((cx1 + 44, stamp_y1 + 16), "WALL: Paper locked behind toll barrier.", font=F_MID, fill=RED)
        d.text((cx1 + 44, stamp_y1 + 44), "Fly connectome halted — cannot read references beyond paywall.", font=F_SMALL, fill=LIGHT_GREY)
    else:
        d.rectangle([cx1 + 24, stamp_y1, cx2 - 24, stamp_y2], fill=(18, 48, 26), outline=GREEN, width=2)
        d.text((cx1 + 44, stamp_y1 + 16), "CORRIDOR: Paper read successfully.", font=F_MID, fill=GREEN)
        d.text((cx1 + 44, stamp_y1 + 44), "Fly crawls through citations and proceeds into referenced literature.", font=F_SMALL, fill=LIGHT_GREY)

    # Animated Fly Sprite crawling on the paper
    fx, fy, angle, wings_up, is_bumping = fly_state
    draw_fly(d, fx + sx, fy + sy, angle, wings_up, s=7, alert=is_bumping)

    # Right Panel: Live Motor Channel Spikes & Scoreboard
    x0 = LEFT_W + PAD
    d.line([LEFT_W, 0, LEFT_W, H], fill=CARD_BORDER, width=2)

    d.text((x0, PAD + 10), "NEURON READOUT", font=F_MID, fill=ORANGE)
    d.text((x0, PAD + 32), "Descending motor channel firing", font=F_TINY, fill=GREY)

    cands = step_data.get("candidates", []) if step_data else []
    top = max([c.get("readout_spikes", 0) for c in cands], default=1.0) or 1.0
    bar_y = PAD + 60
    for i in range(8):
        val = next((c.get("readout_spikes", 0) for c in cands if c.get("channel") == i), 0.0)
        w_bar = int((val / top) * (W - x0 - PAD - 100))
        d.text((x0, bar_y), f"Channel {i}", font=F_TINY, fill=GREY)
        d.rectangle([x0 + 72, bar_y, x0 + 72 + max(w_bar, 2), bar_y + 13],
                    fill=ORANGE if val == top and val > 0 else (75, 78, 88))
        bar_y += 22

    # Arcade Scoreboard
    y2 = bar_y + 30
    box_w = W - x0 - PAD
    d.rectangle([x0, y2, x0 + box_w, y2 + 160], fill=(18, 20, 25), outline=CARD_BORDER, width=2)
    d.text((x0 + 16, y2 + 14), "PAPERS READ", font=F_SMALL, fill=GREY)
    d.text((x0 + 16, y2 + 32), f"{papers_read:03d}", font=F_BIG, fill=GREEN)

    d.text((x0 + 16, y2 + 86), "PAYWALLS HIT", font=F_SMALL, fill=GREY)
    d.text((x0 + 16, y2 + 104), f"{paywalls_hit:03d}", font=F_BIG, fill=RED)

    if step_data:
        d.text((x0, y2 + 180), f"Step spikes: {step_data.get('total_spikes', 0):,}", font=F_BODY, fill=WHITE)

    # Footer
    d.text((x0, H - PAD - 42), "CRIMCONSORTIUM", font=F_MID, fill=ORANGE)
    d.text((x0, H - PAD - 18), "fruitfly.crimconsortium.com", font=F_TINY, fill=GREY)

    return img


def title_card(lines, sub=None):
    img = Image.new("RGB", (W, H), BLACK)
    d = ImageDraw.Draw(img)
    y = H // 2 - 28 * len(lines)
    for text, f, col in lines:
        w = d.textlength(text, font=f)
        d.text(((W - w) / 2, y), text, font=f, fill=col)
        y += f.size + 18
    if sub:
        w = d.textlength(sub, font=F_SMALL)
        d.text(((W - w) / 2, H - PAD - 40), sub, font=F_SMALL, fill=GREY)
    d.text((PAD, H - PAD - 20), "CRIMCONSORTIUM", font=F_MID, fill=ORANGE)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--seconds", type=float, default=24.0)
    args = ap.parse_args()

    trace = json.loads(Path(args.trace).read_text())
    SITE.mkdir(exist_ok=True)

    nodes_map = {n["id"]: n for n in trace["nodes"]}
    calib = trace.get("calibration") or {}
    seed = trace["seed"]

    title = title_card([
        ("A FRUIT FLY", F_BIG, WHITE),
        ("LOOKS FOR CRIMINOLOGY", F_BIG, WHITE),
        ("IT CAN READ", F_BIG, ORANGE),
    ], sub="MaleCNS v1.0 Connectome (CC-BY) x OpenAlex (CC0). Open access = corridor, paywalls = walls.")

    # Flatten the run into an ordered list of ONE-PAPER-AT-A-TIME events:
    # Each event is either encountering a Paywalled paper (bump) or Reading an Open paper (to).
    events = []
    # Seed paper is event #1
    events.append({
        "paper": seed,
        "is_paywalled": not seed.get("passable", True),
        "step_data": trace["steps"][0] if trace["steps"] else None
    })

    for s in trace["steps"]:
        # Add paywall bumps encountered at this step
        for b_id in s.get("bumps", []):
            if b_id in nodes_map:
                events.append({
                    "paper": nodes_map[b_id],
                    "is_paywalled": True,
                    "step_data": s
                })
        # Add the open paper transitioned to
        t_id = s.get("to")
        if t_id and t_id in nodes_map and nodes_map[t_id].get("passable"):
            events.append({
                "paper": nodes_map[t_id],
                "is_paywalled": False,
                "step_data": s
            })

    # Pick 10-14 distinct events across the run
    if len(events) > 12:
        indices = np.linspace(0, len(events) - 1, 12, dtype=int)
        selected_events = [events[i] for i in indices]
    else:
        selected_events = events

    frames = [np.asarray(title)] * (2 * FPS)
    papers_read = 0
    paywalls_hit = 0

    total_encounters = len(selected_events)
    frames_per_event = int((args.seconds - 6.0) * FPS / max(total_encounters, 1))
    frames_per_event = max(frames_per_event, 30)

    # Center area of the card for the fly
    fly_cx = LEFT_W // 2
    fly_cy = H // 2 + 30

    for idx, ev in enumerate(selected_events, 1):
        paper = ev["paper"]
        is_paywalled = ev["is_paywalled"]
        step_data = ev["step_data"]

        if is_paywalled:
            paywalls_hit += 1
        else:
            papers_read += 1

        for f in range(frames_per_event):
            t = f / max(frames_per_event - 1, 1)
            wings_up = (f // 2) % 2 == 0

            if is_paywalled:
                # Paywall Encounter: Fly crawls up, hits barrier, recoils with screen shake
                if t < 0.4:
                    fx = fly_cx
                    fy = (fly_cy + 60) - (t / 0.4) * 60
                    ang = 0
                    is_bump = False
                    shake = (0, 0)
                else:
                    # Screen shake and recoil
                    fx = fly_cx + np.sin(f * 2.5) * 6
                    fy = fly_cy + (t - 0.4) * 40
                    ang = 180
                    is_bump = True
                    shake = (int(np.sin(f * 3) * 6), int(np.cos(f * 3) * 6))
            else:
                # Open Access: Fly calmly crawls across the page
                fx = (fly_cx - 80) + t * 160
                fy = fly_cy + np.sin(t * np.pi * 3) * 15
                ang = 90 + int(np.cos(t * np.pi * 3) * 20)
                is_bump = False
                shake = (0, 0)

            f_img = render_single_paper_frame(
                paper, is_paywalled, idx, total_encounters,
                (fx, fy, ang, wings_up, is_bump),
                step_data, papers_read, paywalls_hit, shake
            )
            frames.append(np.asarray(f_img))

    # End Summary Card
    r = trace["result"]
    sel = summary_line(calib)
    end = title_card([
        (f"{r['reachable']} papers read", F_BIG, WHITE),
        (f"{r['walls_hit']} paywalls hit", F_BIG, RED),
        ("then it ran out of open access", F_MID, GREY),
    ], sub=sel or "fruitfly.crimconsortium.com")
    frames += [np.asarray(end)] * (4 * FPS)

    mp4 = SITE / "fly.mp4"
    imageio.mimwrite(mp4, frames, fps=FPS, quality=8, macro_block_size=1)
    print(f"{mp4}: {mp4.stat().st_size / 1e6:.1f} MB, {len(frames)} frames")

    small = [np.asarray(Image.fromarray(f).resize((640, 360), Image.NEAREST))
             for f in frames[::3]]
    gif = SITE / "fly.gif"
    imageio.mimwrite(gif, small, duration=1000 / (FPS / 3), loop=0)
    print(f"{gif}: {gif.stat().st_size / 1e6:.1f} MB, {len(small)} frames")

    (SITE / "stats.json").write_text(json.dumps({
        "seed": seed, "result": r, "calibration": calib,
        "calibration_line": sel,
        "policy": trace["policy"], "engine": trace["engine"],
    }, indent=2))


if __name__ == "__main__":
    main()
