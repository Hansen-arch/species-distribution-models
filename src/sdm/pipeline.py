"""End-to-end orchestration for one species case."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import environment as envmod
from .config import (
    INTERIM,
    MAXENT_JAR,
    PANEL,
    RESULTS,
    SpeciesCase,
    all_cases,
    case as get_case,
)
from .evaluation import cross_validate, make_splits, summarise_folds
from .models import (
    MODEL_LABELS,
    FitContext,
    ModelFactory,
    available_specs,
)
from .occurrences import (
    fetch_occurrences,
    gbif_count,
    qc_occurrences,
    resolve_taxon,
    thin_to_cells,
)


@dataclass
class RunConfig:
    schemes: tuple[str, ...] = ("random", "spatial")
    models: tuple[str, ...] = ("svm", "rf", "maxent", "xgboost")
    max_records: int = 9000
    background: int = 3000
    max_presence: int = 2000
    n_splits: int = 5
    seed: int = 42
    learning_curve: bool = True
    learning_points: int = 3
    max_uncertainty_m: float = 10_000.0
    require_uncertainty: bool = False
    min_year: int = 1950
    correlation_threshold: float = 0.8
    maxent_memory: str = "2g"
    refresh: bool = False


def _case_dir(case: SpeciesCase) -> Path:
    path = INTERIM / case.key
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_extent(extent: envmod.Extent, path: Path) -> None:
    np.savez_compressed(
        path,
        array=extent.array,
        valid=extent.valid,
        names=np.array(extent.names, dtype=object),
        transform=np.array(extent.transform.to_gdal(), dtype=float),
    )


def _load_extent(path: Path) -> envmod.Extent:
    from rasterio import Affine

    bundle = np.load(path, allow_pickle=True)
    transform = Affine.from_gdal(*bundle["transform"])
    return envmod.Extent(
        bundle["array"], bundle["valid"], transform, list(bundle["names"])
    )


def prepare_case(case: SpeciesCase, config: RunConfig) -> dict:
    """Fetch, filter, clip, prune. The result is the shared data contract."""
    directory = _case_dir(case)
    stack_path = directory / f"stack_{int(config.correlation_threshold * 100)}.npz"

    raw = fetch_occurrences(
        case, max_records=config.max_records, cache=directory / "occurrences_raw.csv"
    )
    clean, qc_table = qc_occurrences(
        raw,
        case,
        max_uncertainty_m=config.max_uncertainty_m,
        min_year=config.min_year,
        require_uncertainty=config.require_uncertainty,
    )
    thinned = thin_to_cells(clean, envmod.GRID_DEG)

    layers = envmod.ensure_worldclim()
    if stack_path.exists() and not config.refresh:
        extent = _load_extent(stack_path)
        prune_table = (
            pd.read_csv(directory / "collinearity.csv")
            if (directory / "collinearity.csv").exists()
            else pd.DataFrame(columns=["dropped", "kept_instead", "abs_r"])
        )
    else:
        raw_extent = envmod.clip_stack(layers, case.bbox)
        extent, prune_table = envmod.prune_collinear(
            raw_extent, threshold=config.correlation_threshold
        )
        _save_extent(extent, stack_path)
        prune_table.to_csv(directory / "collinearity.csv", index=False)

    lon = thinned["decimalLongitude"].to_numpy(dtype=float)
    lat = thinned["decimalLatitude"].to_numpy(dtype=float)
    values = extent.values_at(lon, lat)
    keep = np.isfinite(values).all(axis=1)
    lon, lat, values = lon[keep], lat[keep], values[keep]
    thinned_used = thinned.loc[keep].reset_index(drop=True)

    if len(lon) > config.max_presence:
        rng = np.random.default_rng(config.seed)
        pick = rng.choice(len(lon), size=config.max_presence, replace=False)
        pick.sort()
        lon, lat, values = lon[pick], lat[pick], values[pick]
        thinned_used = thinned_used.iloc[pick].reset_index(drop=True)

    bg_lon, bg_lat, bg_values = envmod.sample_background(
        extent, config.background, seed=config.seed
    )

    X = np.vstack([values, bg_values]).astype(np.float64)
    y = np.concatenate([np.ones(len(values), dtype=int), np.zeros(len(bg_lon), dtype=int)])
    coords = np.column_stack(
        [np.concatenate([lon, bg_lon]), np.concatenate([lat, bg_lat])]
    )

    asc_dir = directory / "asc"
    envmod.write_ascii_grids(extent, asc_dir)

    print(
        f"[{case.key}] gbif {len(raw)} -> qc {len(clean)} -> cells {len(thinned)} "
        f"-> modelled {len(values)} presences, {len(bg_lon)} background, "
        f"{len(extent.names)} predictors",
        flush=True,
    )

    retained = pd.DataFrame(
        [
            {"stage": "gbif_records_returned", "records": len(raw)},
            {"stage": "passing_qc_gates", "records": len(clean)},
            {"stage": "one_record_per_grid_cell", "records": len(thinned)},
            {"stage": "with_climate_values", "records": len(thinned_used)},
            {"stage": "used_in_model", "records": len(values)},
        ]
    )

    return {
        "case": case,
        "extent": extent,
        "prune_table": prune_table,
        "qc_table": qc_table,
        "retention": retained,
        "presence_frame": thinned_used,
        "X": X,
        "y": y,
        "coords": coords,
        "asc_dir": asc_dir,
        "directory": directory,
    }


def _area_fraction(model, X, y, coords, extent, n_valid_cells: int, threshold: float) -> float:
    lon_all, lat_all = extent.all_cell_coords()
    rows, cols = extent.lonlat_to_rc(lon_all, lat_all)
    X_all = extent.matrix_at(rows, cols)
    scores = model.score(X_all, np.column_stack([lon_all, lat_all]))
    finite = np.isfinite(scores)
    if not finite.any():
        return float("nan")
    return float(np.mean(scores[finite] >= threshold))


def run_case(case: SpeciesCase, config: RunConfig) -> dict:
    """Run every model on one species and return the comparison rows."""
    data = prepare_case(case, config)
    extent: envmod.Extent = data["extent"]
    X, y, coords = data["X"], data["y"], data["coords"]
    n_valid_cells = int(extent.valid.sum())
    directory = data["directory"]

    context = FitContext(
        extent_asc_dir=data["asc_dir"],
        species_token=case.token,
        work_dir=directory / "maxent_work",
        grid_shape=extent.shape,
        transform=extent.transform,
        n_bands=len(extent.names),
        band_names=list(extent.names),
    )

    specs = available_specs(list(config.models))
    rows: list[dict] = []
    fold_rows: list[dict] = []

    for scheme in config.schemes:
        splits = make_splits(
            y,
            coords,
            case.bbox,
            scheme=scheme,
            n_splits=config.n_splits,
            seed=config.seed,
        )
        for spec in specs:
            factory = ModelFactory(spec=spec, context=context, seed=config.seed)
            folds, elapsed = cross_validate(
                factory.build, X, y, coords, splits, calibrate=True
            )
            stats = summarise_folds(folds)
            print(
                f"  {case.key:24s} {scheme:8s} {MODEL_LABELS[spec]:14s} "
                f"auc {stats['auc_mean']:.3f} tss {stats['tss_mean']:+.3f} "
                f"folds {stats['n_folds']} {elapsed:.1f}s",
                flush=True,
            )
            for fold in folds:
                fold_rows.append(
                    {
                        "species": case.key,
                        "model": MODEL_LABELS[spec],
                        "cv": scheme,
                        "fold": fold.fold,
                        "auc": fold.auc,
                        "tss": fold.tss,
                        "tss_trainopt": fold.tss_trainopt,
                        "threshold": fold.threshold,
                        "sensitivity": fold.sensitivity,
                        "specificity": fold.specificity,
                    }
                )

            area = float("nan")
            if scheme == "spatial" and len(folds) > 0:
                full = factory.build()
                full.fit(X, y, coords)
                # Threshold comes from the folds, each of which estimated it on a
                # held-out calibration split, so the area figure is not tuned on
                # the data the model was fitted to.
                fold_thresholds = [
                    fold.threshold for fold in folds if np.isfinite(fold.threshold)
                ]
                threshold = (
                    float(np.median(fold_thresholds)) if fold_thresholds else float("nan")
                )
                area = _area_fraction(
                    full, X, y, coords, extent, n_valid_cells, threshold
                )

            rows.append(
                {
                    "species": case.key,
                    "scientific_name": case.scientific_name,
                    "regime": case.regime,
                    "model": MODEL_LABELS[spec],
                    "cv": scheme,
                    "auc_mean": stats["auc_mean"],
                    "auc_sd": stats["auc_sd"],
                    "tss_mean": stats["tss_mean"],
                    "tss_sd": stats["tss_sd"],
                    "tss_trainopt_mean": stats["tss_trainopt_mean"],
                    "tss_trainopt_sd": stats["tss_trainopt_sd"],
                    "n_folds": stats["n_folds"],
                    "n_presence": int(y.sum()),
                    "n_background": int((y == 0).sum()),
                    "n_predictors": len(extent.names),
                    "suitable_area_frac": area,
                    "runtime_s": round(elapsed, 2),
                }
            )

    curve_rows: list[dict] = []
    if config.learning_curve and y.sum() > 0:
        curve_rows = _learning_curve(
            case, data, config, context, specs
        )

    return {
        "case": case,
        "rows": rows,
        "folds": fold_rows,
        "curve": curve_rows,
        "data": data,
    }


def _learning_curve(case, data, config, context, specs) -> list[dict]:
    """Accuracy against training-set size, holding the background fixed.

    The point is to show how each algorithm degrades when presences are scarce,
    which is the decision the table is meant to support.
    """
    X, y, coords = data["X"], data["y"], data["coords"]
    extent: envmod.Extent = data["extent"]
    presence_X = X[y == 1]
    presence_coords = coords[y == 1]
    bg_X = X[y == 0]
    bg_coords = coords[y == 0]

    n_presence = len(presence_X)
    sizes = [size for size in (50, 100, 250, 500) if 0 < size < n_presence]
    if n_presence < 50:
        sizes = [max(4, n_presence // 2)]
    sizes = sorted({min(size, n_presence) for size in sizes + [n_presence]})
    sizes = sizes[: config.learning_points + 1]

    from .evaluation import best_tss_threshold  # local import keeps module graph simple

    rows: list[dict] = []
    rng = np.random.default_rng(config.seed)
    for size in sizes:
        if size >= n_presence:
            pick = np.arange(n_presence)
        else:
            pick = np.sort(rng.choice(n_presence, size=size, replace=False))
        Xs = np.vstack([presence_X[pick], bg_X])
        ys = np.concatenate(
            [np.ones(len(pick), dtype=int), np.zeros(len(bg_X), dtype=int)]
        )
        cs = np.vstack([presence_coords[pick], bg_coords])
        splits = make_splits(
            ys,
            cs,
            case.bbox,
            scheme="spatial",
            n_splits=config.n_splits,
            seed=config.seed,
        )
        for spec in specs:
            factory = ModelFactory(spec=spec, context=context, seed=config.seed)
            folds, _ = cross_validate(
                factory.build, Xs, ys, cs, splits, calibrate=False
            )
            stats = summarise_folds(folds)
            rows.append(
                {
                    "species": case.key,
                    "model": MODEL_LABELS[spec],
                    "n_presence": int(size),
                    "auc_mean": stats["auc_mean"],
                    "auc_sd": stats["auc_sd"],
                    "tss_mean": stats["tss_mean"],
                    "n_folds": stats["n_folds"],
                }
            )
    return rows


def _write_frames(outdir, all_rows, all_folds, all_curve, qc_tables, retention_tables, prune_tables):
    """Write whatever has been computed so far.

    A panel of three species is a long run, and a kill in the middle of the
    third should not cost the first two. Tables are refreshed after every
    species for that reason.
    """
    comparison = pd.DataFrame(all_rows)
    folds = pd.DataFrame(all_folds)
    curve = pd.DataFrame(all_curve)
    qc_all = pd.concat(qc_tables, ignore_index=True) if qc_tables else pd.DataFrame()
    retention_all = (
        pd.concat(retention_tables, ignore_index=True) if retention_tables else pd.DataFrame()
    )
    prune_all = (
        pd.concat(prune_tables, ignore_index=True)
        if prune_tables
        else pd.DataFrame(columns=["species", "dropped", "kept_instead", "abs_r"])
    )
    comparison.to_csv(outdir / "comparison_table.csv", index=False)
    folds.to_csv(outdir / "fold_metrics.csv", index=False)
    curve.to_csv(outdir / "learning_curve.csv", index=False)
    qc_all.to_csv(outdir / "quality_gates.csv", index=False)
    retention_all.to_csv(outdir / "record_retention.csv", index=False)
    prune_all.to_csv(outdir / "collinearity_pruning.csv", index=False)
    return {
        "comparison": comparison,
        "folds": folds,
        "curve": curve,
        "qc": qc_all,
        "retention": retention_all,
        "pruning": prune_all,
    }


def run_panel(keys: list[str], config: RunConfig, outdir: Path = RESULTS) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    cases = [get_case(key) for key in keys] if keys else all_cases()

    all_rows: list[dict] = []
    all_folds: list[dict] = []
    all_curve: list[dict] = []
    qc_tables: list[pd.DataFrame] = []
    retention_tables: list[pd.DataFrame] = []
    prune_tables: list[pd.DataFrame] = []

    for case in cases:
        try:
            result = run_case(case, config)
        except Exception as error:  # keep the remaining species alive
            print(f"[{case.key}] failed: {type(error).__name__}: {error}", flush=True)
            continue
        all_rows.extend(result["rows"])
        all_folds.extend(result["folds"])
        all_curve.extend(result["curve"])

        qc = result["data"]["qc_table"].copy()
        qc.insert(0, "species", case.key)
        qc_tables.append(qc)

        retention = result["data"]["retention"].copy()
        retention.insert(0, "species", case.key)
        retention_tables.append(retention)

        prune = result["data"]["prune_table"].copy()
        if not prune.empty:
            prune.insert(0, "species", case.key)
            prune_tables.append(prune)

        _write_frames(
            outdir, all_rows, all_folds, all_curve, qc_tables, retention_tables, prune_tables
        )

    frames = _write_frames(
        outdir, all_rows, all_folds, all_curve, qc_tables, retention_tables, prune_tables
    )
    comparison = frames["comparison"]
    folds = frames["folds"]
    curve = frames["curve"]
    qc_all = frames["qc"]
    retention_all = frames["retention"]
    prune_all = frames["pruning"]

    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config": {
            "schemes": list(config.schemes),
            "models": list(config.models),
            "max_records": config.max_records,
            "background": config.background,
            "max_presence": config.max_presence,
            "n_splits": config.n_splits,
            "seed": config.seed,
            "correlation_threshold": config.correlation_threshold,
            "max_uncertainty_m": config.max_uncertainty_m,
            "min_year": config.min_year,
        },
        "species": [
            {
                "key": case.key,
                "scientific_name": case.scientific_name,
                "bbox": list(case.bbox),
                "gbif_count_with_coordinates": gbif_count(case),
            }
            for case in cases
        ],
        "maxent_jar": str(MAXENT_JAR),
    }
    (outdir / "provenance.json").write_text(json.dumps(provenance, indent=2))

    return {
        "comparison": comparison,
        "folds": folds,
        "curve": curve,
        "qc": qc_all,
        "retention": retention_all,
        "pruning": prune_all,
        "provenance": provenance,
    }
