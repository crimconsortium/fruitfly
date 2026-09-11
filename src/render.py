"""Render an animated retro-arcade 'Corridor vs Paywall' game view of the fly foraging papers.

Shows 10-15 key sequential decision rooms where the fly navigates doorways:
  - Open Access (Diamond / Gold / Green / Hybrid) -> Open green illuminated archway/corridor.
    Fly walks through into the new paper room.
  - Closed Access / Paywalls -> Heavy brick security gate with "$39.95 / ACCESS DENIED".
    Fly bumps into the wall with screen shake and recoils before picking a path.
  - Right Panel: Real-time descending-neuron spiking readout bars, spikes count,
    live "PAPERS READ" and "PAYWALLS HIT" arcade counters.

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
PAD = 24

BLACK = (10, 10, 12)
DARK_FLOOR = (22, 24, 28)
GRID_LINE = (35, 38, 45)
ORANGE = (246, 130, 18)
WHITE = (245, 245, 245)
GREY = (140, 140, 150)
BORDER_COL = (50, 52, 60)
GREEN = (46, 204, 113)
RED = (231, 76, 60)
DOOR_OPEN_BG = (25, 50, 35)
DOOR_WALL_BG = (55, 25, 25)

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


F_TITLE = font(22, bold=True)
F_BIG = font(42, bold=True)
F_MID = font(18, bold=True)
F_BODY = font(15)
F_SMALL = font(13)
F_TINY = font(11)


def draw_fly_sprite(d: ImageDraw.ImageDraw, cx: float, cy: float, angle_deg: float, wings_up: bool, s: int = 5, alert: bool = False):
    """Draw an animated 2D top-down fly sprite with orange compound eyes and fluttering wings."""
    rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)

    def rot(dx, dy):
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        return cx + rx * s, cy + ry * s

    # Fly body parts
    body_pixels = [
        (0, -3), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (0, 3),
        (-1, -2), (1, -2), (-1, -1), (1, -1), (-1, 0), (1, 0),
        (-1, 1), (1, 1), (-1, 2), (1, 2)
    ]
    b_col = (240, 240, 240) if not alert else (255, 90, 90)
    for bx, by in body_pixels:
        px, py = rot(bx, by)
        d.rectangle([px - s / 2, py - s / 2, px + s / 2, py + s / 2], fill=b_col)

    # Eyes: bright orange compound eyes
    for ex in (-2, 2):
        px, py = rot(ex, -3)
        d.rectangle([px - s / 2, py - s / 2, px + s / 2, py + s / 2], fill=ORANGE)

    # Wings: fluttering translucent wings
    wy = -4 if wings_up else -1
    w_col = (175, 185, 205)
    for wx in (-5, -4, -3, 3, 4, 5):
        px, py = rot(wx, wy)
        d.rectangle([px - s / 2, py - s / 2, px + s / 2, py + s / 2], fill=w_col)

    # Legs
    for lx, ly in [(-3, -2), (-4, 0), (-3, 2), (3, -2), (4, 0), (3, 2)]:
        px, py = rot(lx, ly)
        d.rectangle([px - s / 2, py - s / 2, px + s / 2, py + s / 2], fill=(90, 90, 100))


def draw_grid(d: ImageDraw.ImageDraw, x1, y1, x2, y2, cell=32):
    for x in range(x1, x2, cell):
        d.line([x, y1, x, y2], fill=GRID_LINE, width=1)
    for y in range(y1, y2, cell):
        d.line([x1, y, x2, y], fill=GRID_LINE, width=1)


def render_room_frame(curr_paper, doors, fly_state, step_data, papers_read, paywalls_hit, shake_offset=(0, 0)):
    """Render a single arcade hallway room showing doors ahead and the fly choosing."""
    img = Image.new("RGB", (W, H), BLACK)
    d = ImageDraw.Draw(img)

    sx, sy = shake_offset
    room_x1, room_y1 = PAD + sx, PAD + 54 + sy
    room_x2, room_y2 = LEFT_W - PAD + sx, H - PAD - 20 + sy

    # Floor background with grid
    d.rectangle([room_x1, room_y1, room_x2, room_y2], fill=DARK_FLOOR, outline=BORDER_COL, width=2)
    draw_grid(d, room_x1 + 2, room_y1 + 2, room_x2 - 2, room_y2 - 2, cell=28)

    # Room Header: Current Paper Info Banner
    d.rectangle([room_x1, room_y1, room_x2, room_y1 + 60], fill=(20, 22, 28), outline=BORDER_COL, width=1)
    curr_title = curr_paper.get("title") or "Unknown Seed Paper"
    short_title = textwrap.shorten(curr_title, width=65, placeholder="...")
    d.text((room_x1 + 14, room_y1 + 10), "CURRENT LOCATION (PAPER)", font=F_TINY, fill=GREY)
    d.text((room_x1 + 14, room_y1 + 26), short_title, font=F_MID, fill=WHITE)
    j_info = f"{(curr_paper.get('journal') or 'Journal')} ({curr_paper.get('year') or ''})"
    d.text((room_x2 - d.textlength(j_info, font=F_SMALL) - 14, room_y1 + 10), j_info, font=F_SMALL, fill=ORANGE)

    # Top Navigation Banner
    d.text((PAD, PAD + 10), "A FRUIT FLY LOOKS FOR CRIMINOLOGY IT CAN READ", font=F_TITLE, fill=ORANGE)
    d.text((PAD, PAD + 34), "MaleCNS v1.0 Connectome (166k neurons) navigating reference citations", font=F_TINY, fill=GREY)

    # Draw Doors / Outgoing Reference Paths at the Top of the Room
    n_doors = len(doors)
    if n_doors > 0:
        room_w = room_x2 - room_x1
        door_w = min(180, (room_w - 40) // n_doors - 14)
        total_doors_w = n_doors * door_w + (n_doors - 1) * 14
        start_dx = room_x1 + (room_w - total_doors_w) // 2

        for i, door in enumerate(doors):
            dx = start_dx + i * (door_w + 14)
            dy = room_y1 + 75
            dh = 95
            is_open = door.get("passable", False)
            bg_col = DOOR_OPEN_BG if is_open else DOOR_WALL_BG
            b_col = GREEN if is_open else RED

            # Doorway rectangle
            d.rectangle([dx, dy, dx + door_w, dy + dh], fill=bg_col, outline=b_col, width=2)

            # Archway header
            status_txt = (door.get("oa_status") or "CLOSED").upper()
            badge = f"CORRIDOR [{status_txt}]" if is_open else f"PAYWALL [{status_txt}]"
            d.rectangle([dx, dy, dx + door_w, dy + 22], fill=(15, 15, 18))
            d.text((dx + 6, dy + 5), badge, font=F_TINY, fill=b_col)

            # Paper title preview
            d_title = textwrap.shorten(door.get("title") or "Reference", width=22, placeholder="..")
            d.text((dx + 8, dy + 30), d_title, font=F_SMALL, fill=WHITE)

            # Sub-badge (Cost / Free)
            sub = "OPEN ACCESS" if is_open else "$39.95 LOCKED"
            d.text((dx + 8, dy + dh - 20), sub, font=F_TINY, fill=b_col)

            # Door index tag for neuron channel reference
            d.text((dx + door_w - 22, dy + 5), f"#{i}", font=F_TINY, fill=GREY)

    # Draw Animated Fly
    fx, fy, angle, wings_up, is_bumping = fly_state
    draw_fly_sprite(d, fx + sx, fy + sy, angle, wings_up, s=6, alert=is_bumping)

    # Bump alert overlay
    if is_bumping:
        d.rectangle([room_x1 + 30, room_y2 - 50, room_x2 - 30, room_y2 - 14], fill=(70, 20, 20), outline=RED, width=2)
        d.text((room_x1 + 45, room_y2 - 40), "BLOCKED BY PAYWALL! Fly bounces and recalculates trajectory...", font=F_BODY, fill=RED)

    # Right Panel: Arcade HUD, Spikes & Neuron Readout
    x0 = LEFT_W + PAD
    d.line([LEFT_W, 0, LEFT_W, H], fill=BORDER_COL, width=2)

    d.text((x0, PAD + 10), "NEURON READOUT", font=F_MID, fill=ORANGE)
    d.text((x0, PAD + 32), "Descending motor channel firing", font=F_TINY, fill=GREY)

    cands = step_data.get("candidates", []) if step_data else []
    top = max([c.get("readout_spikes", 0) for c in cands], default=1.0) or 1.0
    bar_y = PAD + 60
    for i in range(8):
        val = next((c.get("readout_spikes", 0) for c in cands if c.get("channel") == i), 0.0)
        w_bar = int((val / top) * (W - x0 - PAD - 46))
        d.text((x0, bar_y), f"Ch {i}", font=F_TINY, fill=GREY)
        d.rectangle([x0 + 36, bar_y, x0 + 36 + max(w_bar, 2), bar_y + 13],
                    fill=ORANGE if val == top and val > 0 else (75, 75, 85))
        bar_y += 22

    # Arcade Scoreboard Box
    y2 = bar_y + 30
    box_w = W - x0 - PAD
    d.rectangle([x0, y2, x0 + box_w, y2 + 160], fill=(18, 20, 25), outline=BORDER_COL, width=2)
    d.text((x0 + 14, y2 + 14), "PAPERS READ", font=F_SMALL, fill=GREY)
    d.text((x0 + 14, y2 + 32), f"{papers_read:03d}", font=F_BIG, fill=GREEN)

    d.text((x0 + 14, y2 + 86), "PAYWALLS HIT", font=F_SMALL, fill=GREY)
    d.text((x0 + 14, y2 + 104), f"{paywalls_hit:03d}", font=F_BIG, fill=RED)

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

    steps = [s for s in trace["steps"] if s.get("to") or s.get("bumps")]
    # Select 10 to 14 sequential steps to provide clear, digestible arcade pacing
    n_display_steps = min(len(steps), 12)
    selected_steps = steps[:n_display_steps] if n_display_steps else steps

    frames = [np.asarray(title)] * (2 * FPS)
    papers_read = 1
    paywalls_hit = 0

    room_x1, room_y1 = PAD, PAD + 54
    room_x2, room_y2 = LEFT_W - PAD, H - PAD - 20
    room_center_x = (room_x1 + room_x2) // 2
    room_bottom_y = room_y2 - 60

    curr_paper = seed

    frames_per_decision = int((args.seconds - 6.0) * FPS / max(len(selected_steps), 1))
    frames_per_decision = max(frames_per_decision, 28)

    for s in selected_steps:
        target_id = s.get("to")
        bump_ids = s.get("bumps", [])

        # Build door candidate list (chosen target + paywalled bumps)
        doors = []
        if target_id and target_id in nodes_map:
            doors.append(nodes_map[target_id])
        for bid in bump_ids[:3]:
            if bid in nodes_map and nodes_map[bid] not in doors:
                doors.append(nodes_map[bid])

        if not doors:
            continue

        n_doors = len(doors)
        room_w = room_x2 - room_x1
        door_w = min(180, (room_w - 40) // n_doors - 14)
        total_doors_w = n_doors * door_w + (n_doors - 1) * 14
        start_dx = room_x1 + (room_w - total_doors_w) // 2

        # 1. Bump animation phase (if paywall encountered)
        if bump_ids:
            paywalls_hit += len(bump_ids)
            # Fly walks toward wall, hits it, screen shakes, fly recoils
            bump_target_x = start_dx + (min(len(doors) - 1, 1)) * (door_w + 14) + door_w // 2
            bump_target_y = room_y1 + 130
            bump_frames = min(18, frames_per_decision // 2)

            for bf in range(bump_frames):
                t = bf / max(bump_frames - 1, 1)
                if t < 0.6:  # Walking to wall
                    fx = room_center_x + (bump_target_x - room_center_x) * (t / 0.6)
                    fy = room_bottom_y + (bump_target_y - room_bottom_y) * (t / 0.6)
                    ang = -20
                    is_bump = False
                    shake = (0, 0)
                else:  # Recoil and shake
                    fx = bump_target_x + np.sin(bf * 2.0) * 8
                    fy = bump_target_y + 18 + (t - 0.6) * 40
                    ang = 160
                    is_bump = True
                    shake = (int(np.sin(bf * 3) * 5), int(np.cos(bf * 3) * 5))

                wings = (bf // 2) % 2 == 0
                f_img = render_room_frame(curr_paper, doors, (fx, fy, ang, wings, is_bump),
                                          s, papers_read, paywalls_hit, shake)
                frames.append(np.asarray(f_img))

        # 2. Chosen Corridor Pass-through Phase
        if target_id and target_id in nodes_map:
            target_node = nodes_map[target_id]
            is_passable = target_node.get("passable", False)
            chosen_door_idx = 0
            for di, d_node in enumerate(doors):
                if d_node.get("id") == target_id:
                    chosen_door_idx = di
                    break

            target_door_x = start_dx + chosen_door_idx * (door_w + 14) + door_w // 2
            target_door_y = room_y1 + 80
            walk_frames = max(frames_per_decision - (len(bump_ids) * 12), 16)

            for wf in range(walk_frames):
                t = wf / max(walk_frames - 1, 1)
                fx = room_center_x + (target_door_x - room_center_x) * t
                fy = (room_bottom_y - 20) + (target_door_y - (room_bottom_y - 20)) * t
                ang = 0 if (target_door_x - room_center_x) == 0 else (
                    np.degrees(np.arctan2(target_door_y - room_bottom_y, target_door_x - room_center_x)) + 90
                )
                wings = (wf // 2) % 2 == 0
                f_img = render_room_frame(curr_paper, doors, (fx, fy, ang, wings, False),
                                          s, papers_read, paywalls_hit, (0, 0))
                frames.append(np.asarray(f_img))

            if is_passable:
                papers_read += 1
                curr_paper = target_node

    # End summary card
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
