"""Render a trace/v1 file to an MP4 and a looping GIF.

Reads only the trace. Draws only what the trace contains: no invented spikes, no
smoothed-over gaps. The right panel shows the actual per-channel descending-neuron
readout values recorded at each decision, and nothing else.

Layout is a deterministic spring layout seeded from the trace, so the same trace always
produces the same video.

  python src/render.py --trace data/traces/W123.json

Outputs: site/fly.mp4, site/fly.gif, site/stats.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"

W, H, FPS = 1280, 720, 30
LEFT_W = int(W * 0.70)
PAD = 28

BLACK = (0, 0, 0)
ORANGE = (246, 130, 18)
WHITE = (245, 245, 245)
GREY = (110, 110, 110)
DARK = (38, 38, 38)

TERRAIN_COLOR = {
    "corridor": WHITE,
    "gate": (200, 140, 60),
    "trapdoor": (120, 70, 20),
    "wall": DARK,
}

FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def font(size: int):
    for p in FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


F_BIG, F_MID, F_SMALL, F_TINY = font(54), font(30), font(20), font(15)


def layout(nodes, edges, seed: int):
    ids = [n["id"] for n in nodes]
    idx = {k: i for i, k in enumerate(ids)}
    n = len(ids)
    rng = np.random.default_rng(seed)
    pos = rng.random((n, 2)) * 2 - 1
    link = np.array([[idx[a], idx[b]] for a, b in edges if a in idx and b in idx])
    k = 1.0 / max(np.sqrt(n), 1)
    for step in range(220):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(delta, axis=-1) + 1e-6
        rep = (k ** 2 / dist ** 2)[..., None] * delta
        force = rep.sum(axis=1)
        if len(link):
            d = pos[link[:, 0]] - pos[link[:, 1]]
            dl = np.linalg.norm(d, axis=1, keepdims=True) + 1e-6
            att = d * (dl / k)
            np.add.at(force, link[:, 0], -att)
            np.add.at(force, link[:, 1], att)
        pos += force * (0.05 * (1 - step / 220))
        pos = np.clip(pos, -1.5, 1.5)
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    pos = (pos - lo) / span
    px = PAD + 44 + pos[:, 0] * (LEFT_W - 2 * PAD - 88)
    py = PAD + 74 + pos[:, 1] * (H - 2 * PAD - 148)
    return {ids[i]: (float(px[i]), float(py[i])) for i in range(n)}


def draw_fly(d: ImageDraw.ImageDraw, x: float, y: float, wings_up: bool, s: int = 5):
    body = [(0, -2), (0, -1), (0, 0), (0, 1), (-1, -1), (1, -1), (-1, 0), (1, 0)]
    for bx, by in body:
        d.rectangle([x + bx * s, y + by * s, x + bx * s + s, y + by * s + s], fill=(232, 232, 232))
    for ex in (-1, 1):
        d.rectangle([x + ex * s, y - 3 * s, x + ex * s + s, y - 2 * s], fill=ORANGE)
    wy = -2 if wings_up else 0
    for wx in (-3, -2, 2, 3):
        d.rectangle([x + wx * s, y + wy * s, x + wx * s + s, y + wy * s + s], fill=(160, 160, 160))


def panel_frame(trace, pos, visited, current_xy, bumps, step, wings_up, subtitle):
    img = Image.new("RGB", (W, H), BLACK)
    d = ImageDraw.Draw(img)
    nodes = {n["id"]: n for n in trace["nodes"]}

    d.line([LEFT_W, 0, LEFT_W, H], fill=DARK, width=2)
    d.text((PAD, PAD), "A FRUIT FLY LOOKS FOR CRIMINOLOGY IT CAN READ", font=F_SMALL, fill=ORANGE)
    d.text((PAD, PAD + 26), subtitle, font=F_TINY, fill=GREY)

    for a, b in trace["_edges"]:
        if a in pos and b in pos:
            passable = nodes.get(b, {}).get("passable")
            col = (60, 60, 60) if passable else (28, 28, 28)
            d.line([pos[a], pos[b]], fill=col, width=1)

    for nid, (x, y) in pos.items():
        node = nodes.get(nid, {})
        terrain = node.get("terrain", "wall")
        col = TERRAIN_COLOR.get(terrain, DARK)
        r = 6 if node.get("passable") else 5
        if nid in bumps:
            d.ellipse([x - 13, y - 13, x + 13, y + 13], outline=ORANGE, width=3)
        if terrain == "wall":
            d.rectangle([x - r, y - r, x + r, y + r], fill=col, outline=(70, 70, 70))
        elif terrain == "trapdoor":
            d.rectangle([x - r, y - r, x + r, y + r], fill=col, outline=ORANGE)
            d.line([x - r, y + r, x + r, y - r], fill=BLACK, width=1)
        else:
            d.ellipse([x - r, y - r, x + r, y + r], fill=col)
        if nid in visited:
            d.ellipse([x - r - 4, y - r - 4, x + r + 4, y + r + 4], outline=ORANGE, width=2)

    draw_fly(d, current_xy[0], current_xy[1], wings_up)

    x0 = LEFT_W + PAD
    d.text((x0, PAD), "DESCENDING-NEURON READOUT", font=F_TINY, fill=ORANGE)
    d.text((x0, PAD + 20), "mean spikes per channel", font=F_TINY, fill=GREY)
    cands = step.get("candidates", []) if step else []
    top = max([c["readout_spikes"] for c in cands], default=1.0) or 1.0
    bar_y = PAD + 56
    for i in range(8):
        val = next((c["readout_spikes"] for c in cands if c["channel"] == i), 0.0)
        w_bar = int((val / top) * (W - x0 - PAD - 46))
        d.text((x0, bar_y), f"{i}", font=F_TINY, fill=GREY)
        d.rectangle([x0 + 18, bar_y, x0 + 18 + max(w_bar, 1), bar_y + 14],
                    fill=ORANGE if val == top and val > 0 else (90, 90, 90))
        bar_y += 24

    y2 = bar_y + 26
    d.text((x0, y2), f"papers reached   {len(visited)}", font=F_SMALL, fill=WHITE)
    d.text((x0, y2 + 28), f"walls hit        {trace['_walls_so_far']}", font=F_SMALL, fill=WHITE)
    if step:
        d.text((x0, y2 + 56), f"spikes this step {step.get('total_spikes', 0):,}",
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

    edge_pairs = []
    for s in trace["steps"]:
        for c in s["candidates"]:
            edge_pairs.append((s["at"], c["id"]))
        for b in s["bumps"]:
            edge_pairs.append((s["at"], b))
    trace["_edges"] = list(dict.fromkeys(edge_pairs))
    pos = layout(trace["nodes"], trace["_edges"], trace["engine"]["rng_seed"])

    calib = trace.get("calibration") or {}
    seed = trace["seed"]
    title = card([
        ("A FRUIT FLY", F_BIG, WHITE),
        ("LOOKS FOR CRIMINOLOGY", F_BIG, WHITE),
        ("IT CAN READ", F_BIG, ORANGE),
    ], sub="MaleCNS v1.0 connectome (CC-BY) x OpenAlex (CC0). Open access = diamond, gold, green, hybrid.")

    steps = [s for s in trace["steps"] if s.get("to")]
    total_frames = int(args.seconds * FPS)
    title_frames, card_frames = 2 * FPS, 4 * FPS
    walk_frames = max(total_frames - title_frames - card_frames, FPS)
    per_step = max(3, walk_frames // max(len(steps), 1))

    frames = [np.asarray(title)] * title_frames
    visited, walls = {trace["seed"]["id"]}, 0
    subtitle = f"seed: {(seed.get('journal') or '?')[:44]} ({seed.get('oa_status')})"

    for s in steps:
        walls += len(s["bumps"])
        trace["_walls_so_far"] = walls
        a, b = pos.get(s["at"]), pos.get(s["to"])
        if a is None or b is None:
            continue
        for f in range(per_step):
            t = f / per_step
            xy = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            img = panel_frame(trace, pos, visited, xy, set(s["bumps"]) if f < per_step // 2 else set(),
                              s, (f // 3) % 2 == 0, subtitle)
            frames.append(np.asarray(img))
        visited.add(s["to"])

    r = trace["result"]
    sel = ""
    if calib:
        sel = (f"channel selectivity {calib['selectivity_real']:.0%} vs "
               f"shuffled-connectome control {calib['selectivity_shuffled']:.0%} "
               f"(chance {calib['chance_level']:.0%})")
    end = card([
        (f"{r['reachable']} papers reached", F_BIG, WHITE),
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
        "policy": trace["policy"], "engine": trace["engine"],
    }, indent=2))


if __name__ == "__main__":
    main()
