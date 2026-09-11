"""Render the cartoon cut: a fruit fly scrolling a reference list, door by door.

Same trace, same numbers as `src/render.py` -- this one just plays it as a joke.
Every title, journal, year and access status on screen is real, read out of the
trace. The counters are the trace's own running totals, never interpolated
toward the final figure: a page that slams 32 doors really has 32 walls in it.

The fly is an AI-generated sprite sheet, cut into six poses in `assets/`.

  PYTHONPATH=src python src/cartoon.py --trace data/traces/W123.json

Output: site/fly-cartoon.mp4, site/fly-cartoon.jpg (poster)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from render import fit_title, font, text_w, wrap

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
ASSETS = ROOT / "assets"

W, H, FPS = 1280, 720, 30

PAPER = (250, 246, 238)
CARD = (255, 253, 249)
INK = (38, 34, 32)
INK_SOFT = (122, 114, 108)
HAIR = (226, 218, 206)
RED = (214, 62, 74)
RED_BG = (253, 231, 231)
GREEN = (28, 150, 84)
GREEN_BG = (222, 246, 230)
ORANGE = (238, 118, 20)

# The list
ROW_X0, ROW_X1 = 198, 892
ROW_H, ROW_GAP = 76, 10
PITCH = ROW_H + ROW_GAP
VIEW_Y0, VIEW_Y1 = 196, 648
FOCUS_Y = 286                     # where the row the fly is standing on sits
LANE_CX = 98                      # centre of the fly's walking lane
SIDE_X = 916                      # right-hand column

POSES = ("read", "walkA", "walkB", "tap", "dazed", "fly")
LEAD = 6                          # rows played at full speed before the fast stretch

_FADES: tuple[Image.Image, Image.Image] | None = None


def _fades(n: int = 34) -> tuple[Image.Image, Image.Image]:
    """Paper-coloured gradients that hide the clipped top and bottom rows."""
    global _FADES
    if _FADES is None:
        ramp = np.linspace(255, 0, n).astype(np.uint8)[:, None].repeat(W, 1)
        base = np.dstack([np.full((n, W), c, np.uint8) for c in PAPER] + [ramp])
        _FADES = (Image.fromarray(base, "RGBA"),
                  Image.fromarray(base[::-1].copy(), "RGBA"))
    return _FADES


# ----------------------------------------------------------------- sprites
def load_sprites(height: int = 132) -> dict[str, Image.Image]:
    out = {}
    for name in POSES:
        im = Image.open(ASSETS / f"fly_{name}.png").convert("RGBA")
        if name == "dazed":            # drawn facing left; mirror so it reads as knocked back
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        w = max(1, round(im.width * height / im.height))
        out[name] = im.resize((w, height), Image.LANCZOS)
    return out


# ----------------------------------------------------------------- cartoon parts
def padlock(d: ImageDraw.ImageDraw, cx: float, cy: float, s: float,
            body=RED, shut: bool = True) -> None:
    """A chunky cartoon padlock, `s` = body half-width."""
    if s < 1.5:
        return
    lw = max(1, round(s * 0.28))
    top = cy - s * 0.35
    if shut:
        d.arc([cx - s * 0.62, top - s * 1.25, cx + s * 0.62, top + s * 0.25],
              180, 360, fill=INK, width=lw)
    else:
        d.arc([cx - s * 1.15, top - s * 1.3, cx + s * 0.1, top + s * 0.2],
              170, 330, fill=INK, width=lw)
    d.rounded_rectangle([cx - s, top, cx + s, cy + s * 1.15],
                        radius=s * 0.32, fill=body, outline=INK, width=lw)
    d.ellipse([cx - s * 0.24, cy + s * 0.05, cx + s * 0.24, cy + s * 0.53], fill=INK)


def door(d: ImageDraw.ImageDraw, cx: float, cy: float, s: float, ajar: float) -> None:
    """A little green door; `ajar` 0..1 swings it open."""
    h, w = s * 1.5, s
    d.rounded_rectangle([cx - w, cy - h, cx + w, cy + h], radius=s * 0.2,
                        fill=(196, 236, 208), outline=INK, width=max(1, round(s * 0.2)))
    lean = w * 2 * (1.0 - 0.75 * ajar)
    d.polygon([(cx - w, cy - h), (cx - w + lean, cy - h + h * 0.16 * ajar),
               (cx - w + lean, cy + h - h * 0.16 * ajar), (cx - w, cy + h)],
              fill=(120, 208, 150), outline=INK)
    d.ellipse([cx - w + lean - s * 0.34, cy - s * 0.12,
               cx - w + lean - s * 0.06, cy + s * 0.16], fill=INK)


def starburst(d: ImageDraw.ImageDraw, cx: float, cy: float, r: float, t: float) -> None:
    """Impact spikes, fading out over t = 0..1."""
    if t >= 1.0:
        return
    n, grow = 9, 0.6 + 0.9 * t
    for k in range(n):
        a = 2 * np.pi * k / n + t
        x0, y0 = cx + np.cos(a) * r * 0.55 * grow, cy + np.sin(a) * r * 0.55 * grow
        x1, y1 = cx + np.cos(a) * r * grow, cy + np.sin(a) * r * grow
        d.line([x0, y0, x1, y1], fill=ORANGE, width=max(1, int(4 * (1 - t))))


def stamp(img: Image.Image, cx: int, cy: int, label: str, colour, t: float,
          angle: float = -11.0) -> None:
    """A rubber stamp that thumps down: t = 0 huge and faint, t = 1 settled."""
    if t <= 0:
        return
    f = font(21, True)
    pad, scale = 12, 1.0 + 1.6 * (1.0 - t) ** 2
    tmp = Image.new("RGBA", (420, 90), (0, 0, 0, 0))
    td = ImageDraw.Draw(tmp)
    tw = text_w(td, label, f)
    box = [10, 10, 10 + tw + pad * 2, 54]
    td.rounded_rectangle(box, 6, outline=colour, width=4)
    td.text((10 + pad, 18), label, font=f, fill=colour)
    tmp = tmp.crop((6, 6, box[2] + 6, 60))
    tmp = tmp.rotate(angle, expand=True, resample=Image.BICUBIC)
    tw, th = int(tmp.width * scale), int(tmp.height * scale)
    tmp = tmp.resize((max(1, tw), max(1, th)), Image.LANCZOS)
    if t < 1.0:
        tmp.putalpha(tmp.getchannel("A").point(lambda v: int(v * min(1.0, 0.35 + t))))
    img.alpha_composite(tmp, (int(cx - tw / 2), int(cy - th / 2)))


# ----------------------------------------------------------------- page model
def clean(node: dict) -> dict:
    """Some OpenAlex records have no title or venue; never print a raw NaN."""
    out = dict(node)
    title = out.get("title")
    out["stub"] = not (isinstance(title, str) and title.strip())
    out["title"] = ("Paywalled reference - no public metadata" if out["stub"]
                    else title.strip())
    venue = out.get("journal")
    out["journal"] = venue.strip() if isinstance(venue, str) and venue.strip() else ""
    year = out.get("year")
    out["year"] = int(year) if isinstance(year, (int, float)) and year == year else None
    return out


def sub_line(row: dict) -> str:
    if row["stub"]:
        return "closed record, no metadata released"
    bits = [b for b in (row["journal"], row["year"]) if b]
    return " - ".join(str(b) for b in bits)


def build_pages(trace: dict, n_pages: int) -> list[dict]:
    """One page per decision: the paper, its walls, and the open reference it took."""
    nodes = {n["id"]: clean(n) for n in trace["nodes"]}
    pages = []
    read_n, wall_n = 1, 0
    for step in trace["steps"][:n_pages]:
        at = nodes[step["at"]]
        # Walls with real metadata lead the page and close it out; the records
        # OpenAlex gives no title for get played in the fast middle stretch.
        titled = [n for n in (nodes[b] for b in step["bumps"]) if not n["stub"]]
        stubs = [n for n in (nodes[b] for b in step["bumps"]) if n["stub"]]
        walls = titled[:LEAD] + stubs + titled[LEAD:]
        rows = [dict(w, closed=True) for w in walls]
        taken = step["to"]
        opens = [dict(nodes[c["id"]], closed=False)
                 for c in step["candidates"] if c["id"] == taken]
        opens += [dict(nodes[c["id"]], closed=False)
                  for c in step["candidates"] if c["id"] != taken]
        pages.append({
            "paper": at,
            "rows": rows + opens,          # walls first, then the doors that opened
            "n_walls": len(rows),
            "open_row": len(rows),         # index of the one it walked through
            "read_before": read_n,
            "wall_before": wall_n,
        })
        wall_n += len(rows)
        read_n += 1
    return pages


# ----------------------------------------------------------------- frame
def draw_frame(sp: dict[str, Image.Image], st: dict) -> Image.Image:
    img = Image.new("RGBA", (W, H), PAPER + (255,))
    d = ImageDraw.Draw(img)
    page = st["page"]

    # --- the paper the fly is standing on, as a card across the top
    d.rounded_rectangle([28, 24, 884, 172], 14, fill=CARD, outline=HAIR, width=2)
    d.text((48, 40), "THE PAGE IT IS ON", font=font(13, True), fill=ORANGE)
    f, lines = fit_title(d, page["paper"]["title"], 700, 2, sizes=(23, 21, 19, 17))
    for i, ln in enumerate(lines):
        d.text((48, 62 + i * 26), ln, font=f, fill=INK)
    d.text((48, 122), sub_line(page["paper"]), font=font(15), fill=INK_SOFT)
    d.text((48, 145), f"{len(page['rows'])} references on this page",
           font=font(14, True), fill=INK_SOFT)

    # --- the reference list, scrolling behind a soft mask
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ld = ImageDraw.Draw(lay)
    for i, row in enumerate(page["rows"]):
        y = VIEW_Y0 + i * PITCH - st["scroll"]
        if y < VIEW_Y0 - PITCH - 40 or y > VIEW_Y1 + 20:
            continue
        dx = st["shake"] if i == st["focus"] else 0.0
        closed = row["closed"]
        done = i in st["slammed"]
        if closed:
            bg, oc = (RED_BG, RED) if done else (CARD, HAIR)
        else:
            bg, oc = (GREEN_BG, GREEN)
        ld.rounded_rectangle([ROW_X0 + dx, y, ROW_X1 + dx, y + ROW_H], 10,
                             fill=bg, outline=oc, width=3 if (done or not closed) else 2)
        if closed:
            padlock(ld, ROW_X0 + 40 + dx, y + ROW_H / 2, 15,
                    body=RED if done else (232, 226, 216))
        else:
            door(ld, ROW_X0 + 40 + dx, y + ROW_H / 2, 15, st["open_t"])
        tf = font(17, True)
        title = wrap(ld, row["title"], tf, ROW_X1 - ROW_X0 - 200, 1)[0]
        ld.text((ROW_X0 + 76 + dx, y + 16), title, font=tf, fill=INK)
        ld.text((ROW_X0 + 76 + dx, y + 42), sub_line(row), font=font(13), fill=INK_SOFT)
        lab, col = ("PAYWALLED", RED) if closed else ("OPEN ACCESS", GREEN)
        lf = font(12, True)
        ld.text((ROW_X1 - 20 - text_w(ld, lab, lf) + dx, y + 30), lab, font=lf, fill=col)

    band = lay.crop((0, VIEW_Y0, W, VIEW_Y1))          # hard clip to the viewport
    if st["blur"] > 0:
        band = band.filter(ImageFilter.GaussianBlur(st["blur"]))
    img.alpha_composite(band, (0, VIEW_Y0))

    # soft top and bottom fades so rows do not get sliced off a hard edge
    top, bottom = _fades()
    img.alpha_composite(top, (0, VIEW_Y0))
    img.alpha_composite(bottom, (0, VIEW_Y1 - bottom.height))

    # --- the fly, standing beside the row it is reading
    row_y = VIEW_Y0 + st["focus"] * PITCH - st["scroll"]
    row_y = min(max(row_y, VIEW_Y0 + 12), VIEW_Y1 - ROW_H - 12)
    pose = sp[st["pose"]]
    fx = LANE_CX - pose.width // 2 + st["fly_dx"]
    fy = int(row_y + ROW_H - pose.height + 6 + st["fly_dy"])
    if st["pose"] == "fly":
        for k in range(3):                       # speed lines
            yy = fy + 20 + k * 22
            d.line([fx - 70 - k * 18, yy, fx - 12, yy], fill=(206, 198, 188), width=4)
    img.alpha_composite(pose, (int(fx), fy))

    # --- impact and stamp on the row being hit
    if st["hit_t"] < 1.0:
        starburst(d, ROW_X0 + 8, row_y + ROW_H / 2, 34, st["hit_t"])
    if st["stamp_t"] > 0:
        lift = 48 if st["stamp_col"] == GREEN else 0   # keep the title readable
        stamp(img, (ROW_X0 + ROW_X1) // 2 + 40, int(row_y + ROW_H // 2) - lift,
              st["stamp_label"], st["stamp_col"], st["stamp_t"])

    # --- right-hand tally
    d.line([SIDE_X - 18, 96, SIDE_X - 18, 648], fill=HAIR, width=2)
    d.text((SIDE_X, 40), "SCORE SO FAR", font=font(13, True), fill=ORANGE)
    d.text((SIDE_X, 72), f"{st['read_n']:,}", font=font(52, True), fill=GREEN)
    d.text((SIDE_X, 132), "PAPERS READ", font=font(14, True), fill=INK_SOFT)
    d.text((SIDE_X, 172), f"{st['wall_n']:,}", font=font(52, True), fill=RED)
    d.text((SIDE_X, 232), "DOORS SLAMMED", font=font(14, True), fill=INK_SOFT)

    d.text((SIDE_X, 288), "LOCKED ON THIS PAGE", font=font(13, True), fill=INK_SOFT)
    shown = min(st["page_slammed"], 32)
    for k in range(shown):
        padlock(d, SIDE_X + 13 + (k % 8) * 32, 330 + (k // 8) * 34, 11)
    if st["page_slammed"] > 32:
        d.text((SIDE_X, 330 + 4 * 34), f"+{st['page_slammed'] - 32} more",
               font=font(14, True), fill=RED)

    cf = font(15)
    for i, ln in enumerate(wrap(d, st["caption"], cf, W - SIDE_X - 44, 4)):
        d.text((SIDE_X, 508 + i * 22), ln, font=cf, fill=INK_SOFT)

    d.line([28, H - 58, W - 40, H - 58], fill=HAIR, width=2)
    d.text((28, H - 44), "Real titles, real access status, real running totals - "
                         "MaleCNS v1.0 x criminology citation graph",
           font=font(14), fill=INK_SOFT)
    d.text((W - 40 - text_w(d, "fruitfly.crimconsortium.com", font(14, True)), H - 44),
           "fruitfly.crimconsortium.com", font=font(14, True), fill=ORANGE)

    if st["flash"] > 0:
        img.alpha_composite(Image.new("RGBA", (W, H), (255, 255, 255,
                                                       int(90 * st["flash"]))))
    return img


def card(lines: list[tuple[str, int, tuple[int, int, int]]], sprite=None,
         sub: str | None = None) -> Image.Image:
    """A full-screen title card: (text, size, colour) lines, centred."""
    img = Image.new("RGBA", (W, H), PAPER + (255,))
    d = ImageDraw.Draw(img)
    total = sum(sz + 16 for _, sz, _ in lines)
    y = (H - total) // 2 - (60 if sprite else 0)
    for txt, sz, col in lines:
        f = font(sz, True)
        d.text(((W - text_w(d, txt, f)) // 2, y), txt, font=f, fill=col)
        y += sz + 16
    if sub:
        f = font(16)
        d.text(((W - text_w(d, sub, f)) // 2, y + 12), sub, font=f, fill=INK_SOFT)
    if sprite is not None:
        img.alpha_composite(sprite, ((W - sprite.width) // 2, y + 70))
    f = font(14, True)
    d.text(((W - text_w(d, "fruitfly.crimconsortium.com", f)) // 2, H - 60),
           "fruitfly.crimconsortium.com", font=f, fill=ORANGE)
    return img


# ----------------------------------------------------------------- timeline
def base_state(page: dict) -> dict:
    return {"page": page, "scroll": 0.0, "focus": 0, "pose": "read",
            "fly_dx": 0.0, "fly_dy": 0.0, "shake": 0.0, "slammed": set(),
            "open_t": 0.0, "hit_t": 1.0, "stamp_t": 0.0, "stamp_label": "LOCKED",
            "stamp_col": RED, "blur": 0.0, "flash": 0.0, "caption": "",
            "read_n": 1, "wall_n": 0, "page_slammed": 0}


def page_frames(sp, page: dict, slow: int, fast_step: int) -> list[Image.Image]:
    """Play one page: `slow` doors in full, the rest fast, then the open one."""
    out: list[Image.Image] = []
    st = base_state(page)
    st["read_n"], st["wall_n"] = page["read_before"], page["wall_before"]
    n_walls, open_i = page["n_walls"], page["open_row"]
    walk = ("walkA", "walkB")

    max_scroll = max(0.0, len(page["rows"]) * PITCH - (VIEW_Y1 - VIEW_Y0) + ROW_GAP)

    def focus_scroll(i: int) -> float:
        """Keep the list full: never scroll past the end just to reach a late row."""
        return min(max(0.0, i * PITCH - (FOCUS_Y - VIEW_Y0)), max_scroll)

    st["caption"] = (f"It senses {len(page['rows'])} references on this page. "
                     f"It has to try them one at a time.")

    for i in range(min(slow, n_walls)):
        target = focus_scroll(i)
        for f in range(6):                                    # walk over to the row
            st["scroll"] += (target - st["scroll"]) * 0.45
            st["focus"], st["pose"] = i, walk[f % 2]
            st["fly_dx"], st["fly_dy"] = 0.0, -3 * abs(np.sin(f * 1.3))
            st["hit_t"], st["stamp_t"], st["shake"] = 1.0, 0.0, 0.0
            out.append(draw_frame(sp, st))
        st["scroll"] = target
        for f in range(3):                                    # reach out and try it
            st["pose"], st["fly_dx"] = "tap", 6.0 + f * 3
            out.append(draw_frame(sp, st))
        st["slammed"].add(i)
        st["wall_n"] += 1
        st["page_slammed"] += 1
        for f in range(10):                                   # slam, bounce, stamp
            t = f / 9
            st["pose"] = "dazed"
            st["fly_dx"] = -18 * (1 - t) + 0.0
            st["fly_dy"] = -10 * np.sin(np.pi * t)
            st["shake"] = 9 * (1 - t) * np.sin(f * 2.6)
            st["hit_t"] = min(1.0, t * 1.6)
            st["stamp_t"] = min(1.0, 0.25 + t * 1.4)
            st["flash"] = max(0.0, 0.5 - t * 2)
            out.append(draw_frame(sp, st))
        st["flash"] = 0.0

    if n_walls > slow:                                        # the rest, at speed
        st["caption"] = (f"{n_walls - slow} more locked doors on this same page. "
                         f"Same result every time.")
        for i in range(slow, n_walls):
            st["focus"] = i
            st["slammed"].add(i)
            st["wall_n"] += 1
            st["page_slammed"] += 1
            for f in range(fast_step):
                target = focus_scroll(i)
                st["scroll"] += (target - st["scroll"]) * 0.7
                st["pose"] = "tap" if f == 0 else "dazed"
                st["fly_dx"] = 4.0 if f == 0 else -8.0
                st["fly_dy"] = 0.0
                st["shake"] = 7 * np.sin(f * 3.1)
                st["blur"] = 2.4
                st["hit_t"] = 0.4
                st["stamp_t"] = 0.0
                out.append(draw_frame(sp, st))
        st["blur"], st["hit_t"] = 0.0, 1.0

    # the open one
    st["caption"] = "One reference on this page is open access. That one lets it through."
    target = focus_scroll(open_i)
    for f in range(8):
        st["scroll"] += (target - st["scroll"]) * 0.4
        st["focus"], st["pose"] = open_i, walk[f % 2]
        st["fly_dx"], st["fly_dy"], st["shake"] = 0.0, -3 * abs(np.sin(f * 1.3)), 0.0
        out.append(draw_frame(sp, st))
    st["scroll"] = target
    for f in range(12):                                       # the door swings open
        t = f / 11
        st["open_t"] = t
        st["pose"] = "tap" if f < 4 else "read"
        st["fly_dx"] = 6.0 if f < 4 else 0.0
        st["stamp_t"] = min(1.0, max(0.0, (t - 0.3) * 2.2))
        st["stamp_label"], st["stamp_col"] = "THIS ONE OPENS", GREEN
        out.append(draw_frame(sp, st))
    st["read_n"] += 1
    for f in range(10):                                       # and it flies through
        t = f / 9
        st["pose"] = "fly"
        st["fly_dx"] = 60 * t * t + 20 * t
        st["fly_dy"] = -6 * t
        st["open_t"] = 1.0
        st["flash"] = max(0.0, 0.35 - t)
        out.append(draw_frame(sp, st))
    st["flash"] = 0.0
    return out


def montage_frames(sp, trace: dict, pages: list[dict], nodes: dict,
                   n_frames: int) -> list[Image.Image]:
    """The 130-odd decisions we do not have time to play, at speed."""
    steps = trace["steps"]
    first = len(pages)
    read0 = pages[-1]["read_before"] + 1
    wall0 = pages[-1]["wall_before"] + pages[-1]["n_walls"]
    out = []
    for f in range(n_frames):
        t = f / (n_frames - 1)
        k = first + int(t * (len(steps) - first - 1))
        step = steps[k]
        read_n = read0 + (k - first)
        wall_n = wall0 + sum(len(s["bumps"]) for s in steps[first:k])
        page = {"paper": nodes[step["at"]],
                "rows": [dict(w, closed=True) for w in
                         sorted((nodes[b] for b in step["bumps"]),
                                key=lambda n: n["stub"])][:14]
                        or [dict(nodes[step["at"]], closed=True)],
                "n_walls": len(step["bumps"]), "open_row": 99,
                "read_before": read_n, "wall_before": wall_n}
        st = base_state(page)
        st.update(read_n=read_n, wall_n=wall_n, focus=1,
                  slammed=set(range(len(page["rows"]))),
                  page_slammed=len(step["bumps"]),
                  scroll=(f * 63) % max(1, PITCH * max(1, len(page["rows"]) - 4)),
                  pose="dazed", blur=3.2, shake=6 * np.sin(f * 2.2),
                  caption=f"Decision {k + 1} of {len(steps)}. It keeps going.")
        out.append(draw_frame(sp, st))
    return out


def render(trace_path: Path, n_pages: int = 3) -> None:
    trace = json.loads(trace_path.read_text())
    nodes = {n["id"]: clean(n) for n in trace["nodes"]}
    result = trace["result"]
    sp = load_sprites()
    pages = build_pages(trace, n_pages)

    frames: list[Image.Image] = []
    fly_big = sp["fly"].resize((sp["fly"].width * 2, sp["fly"].height * 2),
                               Image.LANCZOS)
    frames += [card([("THIS IS THE SAME RUN,", 34, INK),
                     ("PLAYED AS A CARTOON", 44, ORANGE)],
                    sprite=fly_big,
                    sub="A fruit fly brain, dropped into the criminology citation "
                        "graph, trying every reference it can see.")] * 66
    for i, page in enumerate(pages):
        frames += page_frames(sp, page, slow=(5 if i == 0 else 3),
                              fast_step=(3 if i == 0 else 2))
    frames += montage_frames(sp, trace, pages, nodes, 150)

    walls, read = result["walls_hit"], result["reachable"]
    frames += [card([(f"{read} PAPERS READ", 46, GREEN),
                     (f"{walls:,} DOORS SLAMMED", 46, RED)],
                    sub="Then every open-access reference it could reach had already "
                        "been read, and it had nowhere left to go.")] * 78
    frames += [card([("OPEN ACCESS IS A CORRIDOR.", 36, INK),
                     ("A PAYWALL IS A WALL.", 44, ORANGE)],
                    sprite=sp["dazed"],
                    sub="fruitfly.crimconsortium.com - MaleCNS v1.0 connectome, "
                        "CC-BY, HHMI Janelia and Google Research")] * 84

    SITE.mkdir(exist_ok=True)
    mp4 = SITE / "fly-cartoon.mp4"
    with imageio.get_writer(mp4, fps=FPS, codec="libx264", quality=None,
                            output_params=["-crf", "24", "-preset", "slow",
                                           "-pix_fmt", "yuv420p",
                                           "-movflags", "+faststart"]) as wr:
        for im in frames:
            wr.append_data(np.asarray(im.convert("RGB")))
    frames[len(frames) // 3].convert("RGB").save(SITE / "fly-cartoon.jpg", quality=88)
    print(f"{mp4}: {mp4.stat().st_size / 1e6:.1f} MB, {len(frames)} frames, "
          f"{len(frames) / FPS:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--pages", type=int, default=3)
    a = ap.parse_args()
    render(Path(a.trace), a.pages)


if __name__ == "__main__":
    main()
