"""Climate layers: download, clip to the study extent, prune, sample."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.windows import from_bounds

from .config import (
    BIO_NAMES,
    GRID_DEG,
    INTERIM,
    RAW,
    WORLDCLIM_BIO_URL,
    WORLDCLIM_ELEV_URL,
)


def ensure_worldclim(raw_dir: Path = RAW, cache_dir: Path | None = None) -> dict[str, Path]:
    """Download and unpack WorldClim 2.1 at 10 arc minutes if not already present."""
    cache_dir = cache_dir or (INTERIM / "worldclim")
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    wanted = {f"wc2.1_10m_bio_{i}.tif" for i in range(1, 20)} | {"wc2.1_10m_elev.tif"}
    existing = {p.name for p in cache_dir.glob("*.tif")}
    if wanted <= existing:
        return _layer_paths(cache_dir)

    for url, name in (
        (WORLDCLIM_BIO_URL, "wc_bio.zip"),
        (WORLDCLIM_ELEV_URL, "wc_elev.zip"),
    ):
        archive = raw_dir / name
        if not archive.exists() or archive.stat().st_size == 0:
            with requests.get(url, stream=True, timeout=600) as response:
                response.raise_for_status()
                with archive.open("wb") as handle:
                    for block in response.iter_content(chunk_size=1 << 20):
                        handle.write(block)
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.namelist():
                target = cache_dir / Path(member).name
                if target.exists():
                    continue
                with bundle.open(member) as source, target.open("wb") as sink:
                    sink.write(source.read())

    return _layer_paths(cache_dir)


def _layer_paths(cache_dir: Path) -> dict[str, Path]:
    layers: dict[str, Path] = {}
    for index in range(1, 20):
        token = f"BIO{index:02d}"
        layers[token] = cache_dir / f"wc2.1_10m_bio_{index}.tif"
    layers["ELEV"] = cache_dir / "wc2.1_10m_elev.tif"
    return layers


def variable_label(token: str) -> str:
    if token.startswith("BIO"):
        return BIO_NAMES.get(int(token[3:]), token)
    return "Elevation"


class Extent:
    """A closed climate stack clipped to one study extent, in memory.

    Attributes
    ----------
    array : (bands, rows, cols) float32, NaN where any layer is missing
    valid : (rows, cols) bool, all layers present
    transform : rasterio affine transform of the clipped window
    names : band tokens in stack order
    """

    def __init__(self, array: np.ndarray, valid: np.ndarray, transform, names: list[str]):
        self.array = array
        self.valid = valid
        self.transform = transform
        self.names = names

    @property
    def shape(self) -> tuple[int, int]:
        return self.valid.shape

    @property
    def cellsize(self) -> float:
        return float(self.transform.a)

    @property
    def origin(self) -> tuple[float, float]:
        return float(self.transform.c), float(self.transform.f)

    def cell_centres(self, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lon = self.origin[0] + (cols + 0.5) * self.cellsize
        lat = self.origin[1] - (rows + 0.5) * self.cellsize
        return lon, lat

    def lonlat_to_rc(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cols = np.floor((np.asarray(lon) - self.origin[0]) / self.cellsize).astype(int)
        rows = np.floor((self.origin[1] - np.asarray(lat)) / self.cellsize).astype(int)
        return rows, cols

    def values_at(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Sample the stack at coordinates; NaN where the point falls outside."""
        rows, cols = self.lonlat_to_rc(lon, lat)
        inside = (
            (rows >= 0)
            & (rows < self.shape[0])
            & (cols >= 0)
            & (cols < self.shape[1])
        )
        out = np.full((len(lon), len(self.names)), np.nan, dtype=np.float32)
        if inside.any():
            r = rows[inside]
            c = cols[inside]
            out[inside] = self.array[:, r, c].T
        return out

    def all_cell_coords(self) -> tuple[np.ndarray, np.ndarray]:
        rows, cols = np.nonzero(self.valid)
        return self.cell_centres(rows, cols)

    def matrix_at(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return self.array[:, rows, cols].T


def clip_stack(layers: dict[str, Path], bbox: tuple[float, float, float, float]) -> Extent:
    """Clip every layer to the extent and stack them on a common grid."""
    minx, miny, maxx, maxy = bbox
    names: list[str] = []
    arrays: list[np.ndarray] = []
    transform = None
    valid: np.ndarray | None = None

    for token, path in layers.items():
        with rasterio.open(path) as source:
            window = from_bounds(minx, miny, maxx, maxy, transform=source.transform)
            window = window.round_offsets().round_lengths()
            data = source.read(1, window=window, masked=True)
            if transform is None:
                transform = source.window_transform(window)
                valid = np.ones(data.shape, dtype=bool)
            elif data.shape != valid.shape:
                raise ValueError(
                    f"layer {token} has shape {data.shape}, expected {valid.shape}; "
                    "the WorldClim product is not on a single grid"
                )
            values = data.astype(np.float64).filled(np.nan).astype(np.float32)
            values = np.where(values < -1e30, np.nan, values)
            scale = source.scales[0] if source.scales else 1.0
            offset = source.offsets[0] if source.offsets else 0.0
            if scale != 1.0 or offset != 0.0:
                values = (values * scale + offset).astype(np.float32)
            arrays.append(values)
            names.append(token)
            valid &= np.isfinite(values)

    stack = np.stack(arrays, axis=0)
    if valid is not None:
        stack[:, ~valid] = np.nan
    return Extent(stack, valid, transform, names)


def prune_collinear(
    extent: Extent,
    threshold: float = 0.8,
    keep_always: tuple[str, ...] = (),
) -> tuple[Extent, pd.DataFrame]:
    """Drop one variable from every pair of strongly correlated variables.

    The variable kept from a correlated pair is the one with the lower mean
    absolute correlation against the rest of the set, then the one earlier in
    the stack. Correlated layers do not break a random forest but they do
    destabilise the linear decision surface of an RBF machine and they split
    importance arbitrarily, so pruning is done once and shared by all models.
    """
    valid = extent.valid
    flat = extent.array[:, valid].T
    frame = pd.DataFrame(flat, columns=extent.names)

    constant = [name for name in frame.columns if frame[name].std(ddof=0) == 0]
    frame = frame.drop(columns=constant)

    corr = frame.corr().abs()
    np.fill_diagonal(corr.values, 0.0)
    order = list(frame.columns)
    dropped: list[dict] = []

    while True:
        mean_corr = corr.mean(axis=1)
        candidate = None
        worst = 0.0
        for i, left in enumerate(order):
            if left not in corr.columns or left in keep_always:
                continue
            for right in order[i + 1 :]:
                if right not in corr.columns or right in keep_always:
                    continue
                value = corr.loc[left, right]
                if value > threshold and value > worst:
                    worst = value
                    # Drop the more redundant of the two.
                    candidate = left if mean_corr[left] >= mean_corr[right] else right
                    partner = right if candidate == left else left
        if candidate is None:
            break
        dropped.append(
            {
                "dropped": candidate,
                "kept_instead": partner,
                "abs_r": round(float(worst), 3),
            }
        )
        corr = corr.drop(index=candidate, columns=candidate)
        order = [name for name in order if name != candidate]

    kept = [name for name in extent.names if name in corr.columns]
    index = [extent.names.index(name) for name in kept]
    pruned = Extent(extent.array[index], extent.valid, extent.transform, kept)

    report = pd.DataFrame(dropped)
    if report.empty:
        report = pd.DataFrame(columns=["dropped", "kept_instead", "abs_r"])
    if constant:
        report = pd.concat(
            [
                report,
                pd.DataFrame(
                    {"dropped": constant, "kept_instead": "", "abs_r": 0.0}
                ),
            ],
            ignore_index=True,
        )
    report.attrs["kept"] = kept
    return pruned, report


def sample_background(
    extent: Extent, n: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw background points from the eligible cells of the calibration extent.

    Background is drawn from the whole extent, not from a buffer around the
    presences, because all four models see the same background and the
    comparison must not depend on a per-model calibration area.
    """
    rows, cols = np.nonzero(extent.valid)
    if len(rows) == 0:
        raise ValueError("no valid cells in the study extent")
    rng = np.random.default_rng(seed)
    take = min(n, len(rows))
    pick = rng.choice(len(rows), size=take, replace=False)
    rows, cols = rows[pick], cols[pick]
    lon, lat = extent.cell_centres(rows, cols)
    return lon, lat, extent.matrix_at(rows, cols)


def write_ascii_grids(extent: Extent, out_dir: Path) -> Path:
    """Write the clipped stack as ESRI ASCII grids, the format MaxEnt reads."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, cols = extent.shape
    xll = extent.origin[0]
    yll = extent.origin[1] - rows * extent.cellsize
    header = (
        f"ncols {cols}\nnrows {rows}\nxllcorner {xll:.10f}\nyllcorner {yll:.10f}\n"
        f"cellsize {extent.cellsize:.10f}\nNODATA_value -9999\n"
    )
    for index, token in enumerate(extent.names):
        path = out_dir / f"{token}.asc"
        if path.exists():
            continue
        band = np.where(np.isfinite(extent.array[index]), extent.array[index], -9999.0)
        with path.open("w") as handle:
            handle.write(header)
            np.savetxt(handle, band, fmt="%.4f", delimiter=" ")
    return out_dir


def read_ascii_grid(path: Path) -> tuple[np.ndarray, dict]:
    """Read an ESRI ASCII grid into an array with NaN for the nodata value."""
    header: dict[str, float] = {}
    with Path(path).open() as handle:
        for _ in range(6):
            key, value = handle.readline().split()
            header[key.lower()] = float(value)
    grid = np.loadtxt(path, skiprows=6, dtype=np.float64)
    nodata = header.get("nodata_value")
    if nodata is not None:
        grid = np.where(grid == nodata, np.nan, grid)
    return grid, header
