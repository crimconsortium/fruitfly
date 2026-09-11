"""Fetch real soma coordinates for the MaleCNS v1.0 neurons, so the brain can be drawn.

The renderer needs to place each simulated neuron somewhere real. The public
body-annotations file already carries a `somaLocation` per body, so no skeleton
download, no neuPrint token, and no mesh pipeline is required. 13 MB in, one small
parquet out.

Source (public, CC-BY):
  gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/
    body-annotations-male-cns-v1.0-minconf-0.5.feather

Output (committed):
  data/connectome/soma_xyz.parquet   body, x, y, z, superclass
  data/connectome/geometry.json      provenance and coverage

Coverage is written to geometry.json rather than assumed: not every body has a soma
in the volume, and the renderer must know how many it is actually drawing.

Usage:
  python src/fetch_geometry.py
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import requests

BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
ANNOTATIONS = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "connectome"


def download(name: str, dest: Path) -> str:
    url = f"{BASE}/{name}"
    print(f"downloading {url}")
    digest = hashlib.sha256()
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                digest.update(chunk)
    print(f"  {dest.stat().st_size / 1e6:.1f} MB")
    return digest.hexdigest()


def triples(series: pd.Series) -> np.ndarray:
    """somaLocation is a length-3 sequence or null. Anything else is dropped."""
    keep = series.apply(lambda v: hasattr(v, "__len__") and len(v) == 3)
    if not keep.any():
        raise SystemExit("FATAL: no usable somaLocation values in the annotations file.")
    return keep.to_numpy(), np.array([list(v) for v in series[keep]], dtype=np.float32)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / ANNOTATIONS
        sha = download(ANNOTATIONS, path)
        table = feather.read_table(path)
        cols = [c for c in ("bodyId", "somaLocation", "superclass", "class", "type")
                if c in table.column_names]
        if "bodyId" not in cols or "somaLocation" not in cols:
            raise SystemExit(
                "FATAL: annotations file lacks bodyId or somaLocation.\n"
                f"Columns present: {table.column_names}"
            )
        ann = table.select(cols).to_pandas()

    keep, xyz = triples(ann["somaLocation"])
    out = pd.DataFrame({
        "body": ann.loc[keep, "bodyId"].to_numpy().astype("int64"),
        "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
        "superclass": ann.loc[keep, "superclass"].astype("string").to_numpy()
        if "superclass" in ann else pd.array([None] * int(keep.sum()), dtype="string"),
    })
    out = out.drop_duplicates(subset="body").reset_index(drop=True)
    out.to_parquet(OUT / "soma_xyz.parquet", index=False)

    simulated = None
    neurons_path = OUT / "neurons.parquet"
    if neurons_path.exists():
        sim = pd.read_parquet(neurons_path, columns=["body", "is_readout"])
        placed = sim["body"].astype("int64").isin(set(out["body"]))
        readout = sim["is_readout"].fillna(False).to_numpy().astype(bool)
        simulated = {
            "n_simulated": int(len(sim)),
            "n_simulated_placed": int(placed.sum()),
            "n_readout": int(readout.sum()),
            "n_readout_placed": int((placed.to_numpy() & readout).sum()),
        }

    manifest = {
        "dataset": "male-cns:v1.0",
        "source_url": f"{BASE}/{ANNOTATIONS}",
        "source_sha256": sha,
        "licence": "CC-BY",
        "field": "somaLocation",
        "n_bodies_in_file": int(len(ann)),
        "n_with_soma": int(len(out)),
        "bounds": {
            "x": [float(out["x"].min()), float(out["x"].max())],
            "y": [float(out["y"].min()), float(out["y"].max())],
            "z": [float(out["z"].min()), float(out["z"].max())],
        },
        "simulated_coverage": simulated,
    }
    (OUT / "geometry.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"{len(out):,} somas -> data/connectome/soma_xyz.parquet")
    if simulated:
        print(f"simulated neurons placed: {simulated['n_simulated_placed']:,} of "
              f"{simulated['n_simulated']:,}; readout placed: "
              f"{simulated['n_readout_placed']:,} of {simulated['n_readout']:,}")


if __name__ == "__main__":
    main()
