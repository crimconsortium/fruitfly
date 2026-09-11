"""Draw the fly's central nervous system, with the neurons that actually fired lit up.

Every dot is one real neuron at its real soma position from MaleCNS v1.0. Nothing here
is a decorative particle: the point cloud is `data/connectome/soma_xyz.parquet` and the
lights are the spike raster in `data/raster.npz`.

  - Dim dots: the 141,781 annotated somas of the whole male CNS, for context.
  - Brighter dots: the 20,000-neuron subgraph the fly actually runs on.
  - Lit dots: neurons that spiked, held by an exponential decay so a 50 ms cascade is
    legible at 30 frames per second.

Colour carries role, not mood:
  cyan   an input neuron carrying an open reference
  red    an input neuron carrying the wall drive (the paywalls pressing on the brain)
  amber  a descending neuron, the readout
  green  the descending group that won this decision
  white  everything else in the subgraph
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

CONTEXT = np.array([16, 24, 42], dtype=np.float32)
IDLE = np.array([44, 66, 112], dtype=np.float32)
SPIKE = np.array([104, 148, 214], dtype=np.float32)
INPUT_OPEN = np.array([60, 214, 255], dtype=np.float32)
INPUT_WALL = np.array([255, 78, 92], dtype=np.float32)
READOUT = np.array([255, 186, 72], dtype=np.float32)
WINNER = np.array([120, 255, 168], dtype=np.float32)


def _kernel(radius: int) -> list[tuple[int, int, float]]:
    """A small radial falloff, as (dx, dy, weight) triples."""
    out = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            r = (dx * dx + dy * dy) ** 0.5
            if r <= radius + 0.01:
                out.append((dx, dy, float(np.exp(-(r ** 2) / max(radius, 1) ** 2 * 1.6))))
    return out


class BrainView:
    """A camera on the CNS point cloud plus a decaying heat value per neuron."""

    def __init__(self, w: int, h: int, soma_path: Path, sim_bodies: np.ndarray,
                 pitch_deg: float = 18.0, decay: float = 0.80,
                 yaw_range: tuple[float, float] = (-30.0, 30.0), fill: float = 0.94):
        self.w, self.h = w, h
        self.decay = decay

        soma = pd.read_parquet(soma_path)
        body = soma["body"].to_numpy().astype("int64")
        pos = soma[["x", "y", "z"]].to_numpy().astype(np.float32)

        # Screen axes: x is left-right, y is dorsal-ventral, z runs brain to nerve cord.
        centre = np.array([pos[:, 0].mean(), pos[:, 1].mean(), pos[:, 2].mean()], np.float32)
        self.pos = pos - centre
        self.span = float(np.abs(self.pos).max())

        row_of = pd.Series(np.arange(len(body)), index=body)
        matched = row_of.reindex(sim_bodies)
        self.sim_row = matched.to_numpy(dtype=float)          # NaN where unplaced
        self.placed = ~np.isnan(self.sim_row)
        self.sim_row_int = np.where(self.placed, self.sim_row, 0).astype(int)

        self.is_sim = np.zeros(len(body), dtype=bool)
        self.is_sim[self.sim_row_int[self.placed]] = True

        self.n_points = len(body)
        self.n_sim = int(len(sim_bodies))
        self.n_placed = int(self.placed.sum())

        self.heat = np.zeros(self.n_sim, dtype=np.float32)
        self.tint = np.zeros((self.n_sim, 3), dtype=np.float32)

        pitch = np.radians(pitch_deg)
        self.rot_pitch = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)],
        ], dtype=np.float32)

        self.k_small = _kernel(1)
        self.k_glow = _kernel(3)

        # Fit the camera once, over the yaw range actually used, so the CNS fills the
        # frame at every angle instead of drifting in and out of it.
        self.fill = fill
        self.scale, self.offset = 1.0, np.zeros(2, np.float32)
        self._fit(yaw_range)

    def _fit(self, yaw_range) -> None:
        """Choose one scale and centre that keeps every yaw in frame."""
        lo, hi = min(yaw_range), max(yaw_range)
        us, vs = [], []
        for yaw in np.linspace(lo, hi, 7):
            u, v, _ = self._raw(yaw)
            us.append(u)
            vs.append(v)
        u = np.concatenate(us)
        v = np.concatenate(vs)
        # Percentiles, not extremes: a handful of outlying somas should not shrink it.
        u0, u1 = np.percentile(u, [0.15, 99.85])
        v0, v1 = np.percentile(v, [0.15, 99.85])
        self.scale = float(min(self.w * self.fill / max(u1 - u0, 1e-6),
                               self.h * self.fill / max(v1 - v0, 1e-6)))
        self.offset = np.array([
            self.w * 0.5 - (u0 + u1) * 0.5 * self.scale,
            self.h * 0.5 - (v0 + v1) * 0.5 * self.scale,
        ], dtype=np.float32)

    # -- activity -------------------------------------------------------------

    def decay_heat(self) -> None:
        self.heat *= self.decay
        self.tint *= self.decay
        cold = self.heat < 0.01
        self.heat[cold] = 0.0
        self.tint[cold] = 0.0

    def fire(self, idx: np.ndarray, colours: np.ndarray, gains: np.ndarray | None = None) -> None:
        """idx: brain indices that spiked. colours: matching (n, 3) RGB."""
        if len(idx) == 0:
            return
        add = np.ones(len(idx), np.float32) if gains is None else gains.astype(np.float32)
        np.add.at(self.heat, idx, add)
        np.add.at(self.tint, idx, colours * add[:, None])
        np.clip(self.heat, 0.0, 3.0, out=self.heat)

    def quiet(self) -> bool:
        return not bool((self.heat > 0.02).any())

    # -- drawing --------------------------------------------------------------

    def _raw(self, yaw_deg: float):
        """Camera-space coordinates in source units, before scale and centring."""
        yaw = np.radians(yaw_deg)
        ry = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)],
        ], dtype=np.float32)
        pts = self.pos @ (ry @ self.rot_pitch).T
        depth = pts[:, 2]
        # Gentle perspective so the nerve cord recedes instead of pancaking.
        persp = 1.0 / (1.0 + (depth / (self.span * 4.5)))
        return pts[:, 0] * persp, -pts[:, 1] * persp, depth

    def _project(self, yaw_deg: float, zoom: float):
        u, v, depth = self._raw(yaw_deg)
        # Fitted frame first, then zoom about the centre of the frame.
        px = (u * self.scale + self.offset[0] - self.w * 0.5) * zoom + self.w * 0.5
        py = (v * self.scale + self.offset[1] - self.h * 0.5) * zoom + self.h * 0.5
        near = (depth - depth.min()) / max(float(depth.max() - depth.min()), 1.0)
        return px, py, 1.0 - near  # 1 at the front, 0 at the back

    def _splat(self, buf, px, py, weights, colours, kernel):
        inside = (px >= 3) & (px < self.w - 3) & (py >= 3) & (py < self.h - 3)
        if not inside.any():
            return
        x = px[inside].astype(np.int32)
        y = py[inside].astype(np.int32)
        wgt = weights[inside].astype(np.float32)
        col = colours[inside] if colours.ndim == 2 else colours
        for dx, dy, kw in kernel:
            contrib = (wgt * kw)[:, None] * col
            np.add.at(buf, (y + dy, x + dx), contrib)

    def render(self, yaw_deg: float, zoom: float = 1.0, brightness: float = 1.0,
               bg: tuple[int, int, int] = (7, 9, 14)) -> Image.Image:
        px, py, front = self._project(yaw_deg, zoom)
        buf = np.zeros((self.h, self.w, 3), dtype=np.float32)

        # 1. The whole CNS, faintly, back to front.
        ctx = ~self.is_sim
        self._splat(buf, px[ctx], py[ctx], 0.055 + 0.10 * front[ctx],
                    CONTEXT[None, :] * np.ones((int(ctx.sum()), 1), np.float32), self.k_small)

        # 2. The simulated subgraph, a little brighter.
        sim = self.is_sim
        self._splat(buf, px[sim], py[sim], 0.10 + 0.20 * front[sim],
                    IDLE[None, :] * np.ones((int(sim.sum()), 1), np.float32), self.k_small)

        # 3. Whatever is still hot.
        hot = np.flatnonzero(self.heat > 0.02)
        if len(hot):
            rows = self.sim_row_int[hot]
            keep = self.placed[hot]
            rows, hot = rows[keep], hot[keep]
            if len(hot):
                heat = np.clip(self.heat[hot], 0.0, 2.2)
                # Mean colour of the spikes that made this neuron hot, so role colour
                # survives instead of washing out to white.
                colour = self.tint[hot] / np.maximum(heat[:, None], 1e-6)
                colour *= 255.0 / np.maximum(colour.max(axis=1, keepdims=True), 1e-6)
                weight = heat * (0.70 + 0.5 * front[rows])
                bright = heat > 0.9
                self._splat(buf, px[rows][bright], py[rows][bright], weight[bright] * 0.9,
                            colour[bright], self.k_glow)
                faint = ~bright
                self._splat(buf, px[rows][faint], py[rows][faint], weight[faint],
                            colour[faint], self.k_small)

        # 4. Bloom, then a filmic rolloff so bright cascades do not clip flat.
        img = buf * brightness
        blur = np.asarray(
            Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
            .filter(ImageFilter.GaussianBlur(7)), dtype=np.float32)
        img = img + blur * 0.42
        img = 255.0 * (1.0 - np.exp(-img / 210.0))
        img += np.array(bg, dtype=np.float32)      # match the page's background
        return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def role_colours(is_input, input_channel, is_readout, readout_channel,
                 open_channels, wall_channel, winning_channel):
    """Per-neuron (RGB, gain) for this decision. Gain makes roles read louder."""
    n = len(is_input)
    col = np.tile(SPIKE, (n, 1)).astype(np.float32)
    gain = np.full(n, 0.55, dtype=np.float32)
    if len(open_channels):
        band = is_input & np.isin(input_channel, list(open_channels))
        col[band] = INPUT_OPEN
        gain[band] = 1.5
    if wall_channel is not None:
        band = is_input & (input_channel == wall_channel)
        col[band] = INPUT_WALL
        gain[band] = 1.7
    col[is_readout] = READOUT
    gain[is_readout] = 1.4
    if winning_channel is not None:
        band = is_readout & (readout_channel == winning_channel)
        col[band] = WINNER
        gain[band] = 2.0
    return col, gain
