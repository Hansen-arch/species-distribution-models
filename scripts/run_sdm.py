#!/usr/bin/env python3
"""Run the SDM model comparison.

Usage
-----
    python scripts/run_sdm.py --species all
    python scripts/run_sdm.py --species nepenthes-rajah --models maxent,rf
    python scripts/run_sdm.py --species all --fast

The script writes tables and figures under ``results/``. It requires the
``geo`` conda environment, see README.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sdm.config import PANEL, RESULTS  # noqa: E402
from sdm.models import MODEL_ORDER, available_specs  # noqa: E402
from sdm.pipeline import RunConfig, run_panel  # noqa: E402
from sdm.report import write_all  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--species",
        default="all",
        help="comma separated case keys, or 'all'",
    )
    parser.add_argument(
        "--models",
        default=",".join(MODEL_ORDER),
        help=f"comma separated subset of {','.join(MODEL_ORDER)}",
    )
    parser.add_argument(
        "--schemes",
        default="random,spatial",
        help="validation schemes: random, spatial",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=9000,
        help="GBIF records to pull; deep paging past ~10000 offsets is unreliable",
    )
    parser.add_argument("--background", type=int, default=2000)
    parser.add_argument("--max-presence", type=int, default=2000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-uncertainty-m",
        type=float,
        default=10_000.0,
        help="coordinate uncertainty ceiling, 10000 m is the standard SDM profile",
    )
    parser.add_argument(
        "--require-uncertainty",
        action="store_true",
        help="strict profile: drop records that carry no coordinate uncertainty value",
    )
    parser.add_argument("--min-year", type=int, default=1950)
    parser.add_argument("--correlation-threshold", type=float, default=0.8)
    parser.add_argument("--maxent-memory", default="2g")
    parser.add_argument("--no-learning-curve", action="store_true")
    parser.add_argument("--fast", action="store_true", help="smaller background and no learning curve")
    parser.add_argument("--refresh", action="store_true", help="rebuild cached rasters and occurrence pulls")
    parser.add_argument("--outdir", default=str(RESULTS))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    keys = list(PANEL) if args.species == "all" else [k.strip() for k in args.species.split(",") if k.strip()]
    unknown = [key for key in keys if key not in PANEL]
    if unknown:
        print(f"unknown species case(s): {unknown}", file=sys.stderr)
        return 2

    models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    valid = available_specs(list(models))
    missing = sorted(set(models) - set(valid))
    if missing:
        print(f"skipping unavailable models: {missing}", file=sys.stderr)
    if not valid:
        print("no runnable models selected", file=sys.stderr)
        return 2

    config = RunConfig(
        schemes=tuple(s.strip() for s in args.schemes.split(",") if s.strip()),
        models=tuple(valid),
        max_records=args.max_records,
        background=600 if args.fast else args.background,
        max_presence=args.max_presence,
        n_splits=args.folds,
        seed=args.seed,
        learning_curve=not (args.no_learning_curve or args.fast),
        max_uncertainty_m=args.max_uncertainty_m,
        require_uncertainty=args.require_uncertainty,
        min_year=args.min_year,
        correlation_threshold=args.correlation_threshold,
        maxent_memory=args.maxent_memory,
        refresh=args.refresh,
    )

    outdir = Path(args.outdir)
    print(f"panel: {keys}")
    print(f"models: {valid}")
    print(f"schemes: {config.schemes}")
    results = run_panel(keys, config, outdir)
    artefacts = write_all(results, outdir)

    comparison = results["comparison"]
    spatial = comparison[comparison["cv"] == "spatial"]
    print()
    print(spatial[["species", "model", "auc_mean", "tss_mean", "suitable_area_frac", "runtime_s"]]
          .to_string(index=False, float_format=lambda v: f"{v:0.3f}"))
    print()
    print(f"tables: {outdir}")
    print(f"figures: {artefacts['figures']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
