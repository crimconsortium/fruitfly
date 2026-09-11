"""Render an animated retro-arcade citation crawl video with clean paper-to-paper transitions.

Radical visual redesign:
  - Removed cluttered neuron readout entirely: full canvas dedicated to the paper journey.
  - Full width layout with no text overlapping.
  - Visually explicit paper-to-paper transition:
      1. Shows current paper card on left.
      2. Shows fly reading outgoing reference citations.
      3. When encountering paywalls: red locked barrier drops with screen shake and recoil.
      4. When following open access: fly crawls along a glowing citation arrow into the next card!
  - Running scoreboard in top right: PAPERS READ (green) and PAYWALLS HIT (red).

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
PAD = 32

BLACK = (10, 11, 14)
CARD_BG = (20, 22, 28)
CARD_BORDER = (45, 48, 58)
ORANGE = (246, 130, 18)
WHITE = (245, 245, 245)
GREY = (140, 145, 155)
LIGHT_GREY = (195, 200, 210)
GREEN = (46, 204, 113)
RED = (231, 76, 60)
RED_BG = (50, 20, 22)
GREEN_BG = (18, 45, 28)

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


F_BIG = font(34, bold=True)
F_TITLE = font(22, bold=True)
F_MID = font(18, bold=True)
F_BODY = font(15)
F_SMALL = font(13)
F_TINY = font(11)


def draw_fly(d: ImageDraw.ImageDraw, cx: float, cy: float, angle_deg: float, wings_up: bool, s: int = 7, alert: bool = False):
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
    b_col = (245, 245, 245) if not alert else (255, 90, 90)
    for bx, by in body_pixels:
        px, py = rot(bx, by)
        d.rectangle([px - s / 2, py - s / 2, px + s / 2, py + s / 2], fill=b_col)

    for ex in (-2, 2):
        px, py = rot(ex, -3)
        d.rectangle([px - s / 2, py - s / 2, px + s / 2, py + s / 2], fill=ORANGE)

    wy = -4 if wings_up else -1
    w_col = (180, 190, 210)
    for wx in (-5, -4, -3, 3, 4, 5):
        px, py = rot(wx, wy)
        d.rectangle([px - s / 2, py - s / 2, px + s / 2, py + s / 2], fill=w_col)

    for lx, ly in [(-3, -2), (-4, 0), (-3, 2), (3, -2), (4, 0), (3, 2)]:
        px, py = rot(lx, ly)
        d.rectangle([px - s / 2, py - s / 2, px + s / 2, py + s / 2], fill=(100, 105, 115))


def render_transition_frame(curr_paper, target_paper, action_type, fly_state, papers_read, paywalls_hit, shake=(0, 0)):
    """Render a clean wide canvas showing paper inspection and transition to citations."""
    img = Image.new("RGB", (W, H), BLACK)
    d = ImageDraw.Draw(img)

    sx, sy = shake
    cx1, cy1 = PAD + sx, PAD + 54 + sy
    cx2, cy2 = W - PAD - 260 + sx, H - PAD - 20 + sy

    # Top Header
    d.text((PAD, PAD + 10), "A FRUIT FLY LOOKS FOR CRIMINOLOGY IT CAN READ", font=F_TITLE, fill=ORANGE)
    d.text((PAD, PAD + 34), "MaleCNS v1.0 connectome navigating citation reference network", font=F_TINY, fill=GREY)

    # Scoreboard in Top Right
    sb_x1, sb_y1 = W - PAD - 230, PAD + 54
    sb_x2, sb_y2 = W - PAD, PAD + 230
    d.rectangle([sb_x1, sb_y1, sb_x2, sb_y2], fill=(18, 20, 26), outline=CARD_BORDER, width=2)
    d.text((sb_x1 + 16, sb_y1 + 14), "PAPERS READ", font=F_SMALL, fill=GREY)
    d.text((sb_x1 + 16, sb_y1 + 34), f"{papers_read:03d}", font=F_BIG, fill=GREEN)

    d.text((sb_x1 + 16, sb_y1 + 90), "PAYWALLS HIT", font=F_SMALL, fill=GREY)
    d.text((sb_x1 + 16, sb_y1 + 110), f"{paywalls_hit:04d}", font=F_BIG, fill=RED)

    # Branding in bottom right
    d.text((sb_x1, H - PAD - 42), "CRIMCONSORTIUM", font=F_MID, fill=ORANGE)
    d.text((sb_x1, H - PAD - 18), "fruitfly.crimconsortium.com", font=F_TINY, fill=GREY)

    # Main Paper Inspection Card
    is_paywalled = (action_type == "bump")
    card_bg = RED_BG if is_paywalled else CARD_BG
    card_outline = RED if is_paywalled else (GREEN if curr_paper.get("passable") else CARD_BORDER)
    d.rectangle([cx1, cy1, cx2, cy2], fill=card_bg, outline=card_outline, width=3)

    # Card Top Status Bar
    header_h = 52
    bar_fill = (50, 18, 20) if is_paywalled else ((20, 50, 30) if curr_paper.get("passable") else (30, 32, 40))
    d.rectangle([cx1, cy1, cx2, cy1 + header_h], fill=bar_fill, outline=card_outline, width=1)

    status_tag = (curr_paper.get("oa_status") or ("CLOSED" if is_paywalled else "OPEN")).upper()
    if is_paywalled:
        badge_txt = f"PAYWALL  [{status_tag}] — ACCESS DENIED ($39.95)"
        badge_col = RED
    else:
        badge_txt = f"CORRIDOR  [{status_tag}] — FULL TEXT ACCESSIBLE"
        badge_col = GREEN

    d.text((cx1 + 20, cy1 + 14), badge_txt, font=F_MID, fill=badge_col)

    # Paper Title (Clean Wrap, plenty of room now)
    raw_title = curr_paper.get("title") or "Unknown Manuscript"
    lines = textwrap.wrap(raw_title, width=54)
    ty = cy1 + 75
    for line in lines[:3]:
        d.text((cx1 + 24, ty), line, font=F_BIG, fill=WHITE)
        ty += 44

    # Journal & Publication Year
    j_name = curr_paper.get("journal") or "Criminology Journal"
    yr = curr_paper.get("year") or ""
    j_str = f"Journal: {j_name} ({yr})"
    d.text((cx1 + 24, ty + 10), j_str, font=F_MID, fill=ORANGE)

    # Divider
    d.line([cx1 + 24, ty + 46, cx2 - 24, ty + 46], fill=(55, 60, 72), width=1)

    # Citation Traversal Action Bar
    action_box_y1 = cy2 - 130
    action_box_y2 = cy2 - 20

    if is_paywalled:
        d.rectangle([cx1 + 20, action_box_y1, cx2 - 20, action_box_y2], fill=(65, 20, 22), outline=RED, width=2)
        d.text((cx1 + 36, action_box_y1 + 16), "WALL: Fly tried to read paywalled reference citation.", font=F_MID, fill=RED)
        d.text((cx1 + 36, action_box_y1 + 44), "Connectome blocked — fly recoils and turns back to explore open paths.", font=F_BODY, fill=LIGHT_GREY)
    else:
        d.rectangle([cx1 + 20, action_box_y1, cx2 - 20, action_box_y2], fill=(20, 52, 30), outline=GREEN, width=2)
        next_title = (target_paper.get("title") or "Next Reference") if target_paper else "Exploring references..."
        next_trunc = textwrap.shorten(next_title, width=55, placeholder="...")
        d.text((cx1 + 36, action_box_y1 + 16), "CORRIDOR: Paper read! Following citation link to next paper...", font=F_MID, fill=GREEN)
        d.text((cx1 + 36, action_box_y1 + 44), f"--> Crawling into: \"{next_trunc}\"", font=F_BODY, fill=ORANGE)

    # Fly Sprite
    fx, fy, angle, wings_up, is_bumping = fly_state
    draw_fly(d, fx + sx, fy + sy, angle, wings_up, s=7, alert=is_bumping)

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

    # Flatten run into distinct chronological steps
    events = []
    # Seed paper
    events.append({
        "type": "read",
        "curr": seed,
        "target": None,
    })

    for s in trace["steps"]:
        # Add bumps encountered
        for b_id in s.get("bumps", []):
            if b_id in nodes_map:
                events.append({
                    "type": "bump",
                    "curr": nodes_map[b_id],
                    "target": None,
                })
        # Add transition
        t_id = s.get("to")
        if t_id and t_id in nodes_map and nodes_map[t_id].get("passable"):
            events.append({
                "type": "read",
                "curr": nodes_map[t_id],
                "target": nodes_map.get(s.get("to")),
            })

    # Pick 10-14 representative chronological events
    if len(events) > 12:
        indices = np.linspace(0, len(events) - 1, 12, dtype=int)
        selected_events = [events[i] for i in indices]
    else:
        selected_events = events

    frames = [np.asarray(title)] * (2 * FPS)
    
    # Track true accumulated counts proportionally across the run
    final_read = trace["result"]["reachable"]
    final_walls = trace["result"]["walls_hit"]

    frames_per_event = int((args.seconds - 6.0) * FPS / max(len(selected_events), 1))
    frames_per_event = max(frames_per_event, 30)

    card_center_x = (PAD + (W - PAD - 260)) // 2
    card_center_y = H // 2 + 30

    for idx, ev in enumerate(selected_events, 1):
        action_type = ev["type"]
        curr_p = ev["curr"]
        target_p = ev["target"]

        # Scale running counters smoothly to reflect the true large-scale journey
        p_read = int(final_read * (idx / len(selected_events)))
        p_walls = int(final_walls * (idx / len(selected_events)))

        for f in range(frames_per_event):
            t = f / max(frames_per_event - 1, 1)
            wings_up = (f // 2) % 2 == 0

            if action_type == "bump":
                # Bump animation: Fly approaches wall, hits it with shake, recoils
                if t < 0.4:
                    fx = card_center_x
                    fy = (card_center_y + 40) - (t / 0.4) * 40
                    ang = 0
                    is_bump = False
                    shake = (0, 0)
                else:
                    fx = card_center_x + np.sin(f * 2.5) * 6
                    fy = card_center_y + (t - 0.4) * 35
                    ang = 180
                    is_bump = True
                    shake = (int(np.sin(f * 3) * 6), int(np.cos(f * 3) * 6))
            else:
                # Open Access: Fly crawls from left to right along the citation path
                fx = (card_center_x - 120) + t * 240
                fy = card_center_y + np.sin(t * np.pi * 2) * 20
                ang = 90 + int(np.cos(t * np.pi * 2) * 25)
                is_bump = False
                shake = (0, 0)

            f_img = render_transition_frame(
                curr_p, target_p, action_type,
                (fx, fy, ang, wings_up, is_bump),
                p_read, p_walls, shake
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
