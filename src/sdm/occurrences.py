"""GBIF retrieval and the occurrence quality gates the models are built on.

The gate set follows the SDM fitness-for-use profile used by the GBIF
data-quality literature: valid coordinates, an explicit uncertainty ceiling,
a recent-enough record, an accepted basis of record, no fatal geospatial issue
flag, and one record per grid cell. Every gate reports the records it removed
so the retention table can be defended in a methods section.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import (
    ACCEPTED_BASIS,
    CRITICAL_GBIF_ISSUES,
    SpeciesCase,
)

GBIF_SEARCH = "https://api.gbif.org/v1/occurrence/search"
GBIF_MATCH = "https://api.gbif.org/v1/species/match"

FIELDS = [
    "key",
    "scientificName",
    "decimalLatitude",
    "decimalLongitude",
    "coordinateUncertaintyInMeters",
    "year",
    "basisOfRecord",
    "issues",
    "countryCode",
    "datasetKey",
    "occurrenceStatus",
    "taxonKey",
]


def resolve_taxon(scientific_name: str) -> dict:
    """Match a name against the GBIF backbone and return the match payload."""
    response = requests.get(
        GBIF_MATCH, params={"name": scientific_name, "strict": "false"}, timeout=60
    )
    response.raise_for_status()
    return response.json()


def gbif_count(case: SpeciesCase) -> int:
    params = {
        "scientificName": case.scientific_name,
        "hasCoordinate": "true",
        "hasGeospatialIssue": "false",
        "occurrenceStatus": "PRESENT",
        "limit": 0,
    }
    response = requests.get(GBIF_SEARCH, params=params, timeout=60)
    response.raise_for_status()
    return int(response.json().get("count", 0))


def fetch_occurrences(
    case: SpeciesCase,
    max_records: int = 9000,
    chunk: int = 300,
    cache: Path | None = None,
    workers: int = 4,
) -> pd.DataFrame:
    """Page the GBIF occurrence API up to ``max_records``.

    Pages are independent, so they are fetched in parallel; a 12 000 record pull
    then costs a few seconds instead of a few minutes. The search API returns
    records in key order rather than a random order, so a capped pull is not a
    random sample of the full index. For a 10 arc minute grid this matters
    little, because thinning to cells discards the redundancy the ordering
    concentrates. The cap is recorded in the provenance file so the pull is not
    mistaken for an exhaustive download.
    """
    if cache is not None and cache.exists():
        return pd.read_csv(cache)

    total = min(gbif_count(case), max_records)
    offsets = list(range(0, total, chunk))
    short_pages: list[int] = []

    def pull(offset: int) -> list[dict]:
        params = {
            "scientificName": case.scientific_name,
            "hasCoordinate": "true",
            "hasGeospatialIssue": "false",
            "occurrenceStatus": "PRESENT",
            "limit": chunk,
            "offset": offset,
        }
        for attempt in range(2):
            try:
                response = requests.get(
                    GBIF_SEARCH,
                    params=params,
                    timeout=45,
                    headers={"User-Agent": "sdm-benchmark/1.0 (research use)"},
                )
                response.raise_for_status()
                payload = response.json()
                return [
                    {field: row.get(field) for field in FIELDS}
                    for row in payload.get("results", [])
                ]
            except requests.RequestException:
                if attempt == 1:
                    short_pages.append(offset)
                    return []
                time.sleep(2)
        return []

    if offsets:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pages = list(pool.map(pull, offsets))
        records = [row for page in pages for row in page]
    else:
        records = []

    if short_pages:
        print(
            f"warning: {len(short_pages)} GBIF page(s) failed for {case.key} "
            f"(deep paging), retained {len(records)} records",
            flush=True,
        )

    frame = pd.DataFrame.from_records(records)
    if not frame.empty:
        frame["issues"] = frame["issues"].apply(
            lambda value: "|".join(value) if isinstance(value, list) else (value or "")
        )
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache, index=False)
    return frame


def _record_gate(name: str, before: int, after: int, note: str) -> dict:
    return {
        "gate": name,
        "records_in": before,
        "records_out": after,
        "records_removed": before - after,
        "retained_pct": round(100.0 * after / before, 1) if before else np.nan,
        "rule": note,
    }


def qc_occurrences(
    frame: pd.DataFrame,
    case: SpeciesCase,
    max_uncertainty_m: float = 10_000.0,
    min_year: int = 1950,
    require_uncertainty: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the SDM quality gates in a fixed order and report each one.

    Returns the surviving records and one row per gate with the count in, the
    count out and the rule that was applied.
    """
    gates: list[dict] = []
    work = frame.copy()
    work["decimalLatitude"] = pd.to_numeric(work["decimalLatitude"], errors="coerce")
    work["decimalLongitude"] = pd.to_numeric(work["decimalLongitude"], errors="coerce")
    work["coordinateUncertaintyInMeters"] = pd.to_numeric(
        work["coordinateUncertaintyInMeters"], errors="coerce"
    )
    work["year"] = pd.to_numeric(work["year"], errors="coerce")

    before = len(work)
    work = work.dropna(subset=["decimalLatitude", "decimalLongitude"])
    work = work[
        work["decimalLatitude"].between(-90, 90)
        & work["decimalLongitude"].between(-180, 180)
        & ~((work["decimalLatitude"] == 0) & (work["decimalLongitude"] == 0))
    ]
    gates.append(
        _record_gate(
            "coordinate_valid",
            before,
            len(work),
            "latitude and longitude present, in range, not the null island pair",
        )
    )

    minx, miny, maxx, maxy = case.bbox
    before = len(work)
    work = work[
        work["decimalLongitude"].between(minx, maxx)
        & work["decimalLatitude"].between(miny, maxy)
    ]
    gates.append(
        _record_gate(
            "inside_study_extent",
            before,
            len(work),
            f"inside the calibration extent {case.bbox}",
        )
    )

    before = len(work)
    work = work[
        ~work["issues"].fillna("").apply(
            lambda value: any(
                flag in str(value).split("|") for flag in CRITICAL_GBIF_ISSUES
            )
        )
    ]
    gates.append(
        _record_gate(
            "no_fatal_gbif_issue",
            before,
            len(work),
            "excludes zero, swapped, negated, out-of-range and country-mismatched coordinates",
        )
    )

    before = len(work)
    work = work[work["basisOfRecord"].isin(ACCEPTED_BASIS)]
    gates.append(
        _record_gate(
            "basis_of_record_accepted",
            before,
            len(work),
            "specimens, observations and machine observations only",
        )
    )

    before = len(work)
    work = work[work["year"].notna() & (work["year"] >= min_year)]
    gates.append(
        _record_gate(
            "year_recent",
            before,
            len(work),
            f"recorded in {min_year} or later, matching the climate normal period",
        )
    )

    before = len(work)
    uncertainty = work["coordinateUncertaintyInMeters"]
    missing = int(uncertainty.isna().sum())
    if require_uncertainty:
        keep = uncertainty.notna() & (uncertainty <= max_uncertainty_m)
        rule = (
            f"stated uncertainty required and at most {max_uncertainty_m / 1000:.0f} km "
            f"({missing} records without a value dropped)"
        )
    else:
        # A missing field is an absent measurement, not a measurement of poor
        # precision. Dropping those records would delete almost every herbarium
        # record of a narrow endemic, so they are kept and the count is disclosed.
        keep = uncertainty.isna() | (uncertainty <= max_uncertainty_m)
        rule = (
            f"stated uncertainty at most {max_uncertainty_m / 1000:.0f} km; "
            f"{missing} records carry no value and are retained under the standard profile"
        )
    work = work[keep]
    gates.append(_record_gate("coordinate_uncertainty", before, len(work), rule))

    before = len(work)
    work = work.drop_duplicates(subset=["decimalLatitude", "decimalLongitude"])
    gates.append(
        _record_gate(
            "duplicate_coordinates",
            before,
            len(work),
            "one record per unique coordinate pair",
        )
    )

    work = work.reset_index(drop=True)
    return work, pd.DataFrame(gates)


def thin_to_cells(
    frame: pd.DataFrame,
    cell_deg: float,
    lon_col: str = "decimalLongitude",
    lat_col: str = "decimalLatitude",
) -> pd.DataFrame:
    """Keep one record per raster cell, preferring the best located record.

    Reading a coarse climate grid at several near-identical points adds no
    information and inflates apparent sample size, so the thinning is part of
    the data contract rather than a preprocessing detail.
    """
    work = frame.copy()
    work["_cell_lon"] = np.floor(work[lon_col] / cell_deg).astype(int)
    work["_cell_lat"] = np.floor(work[lat_col] / cell_deg).astype(int)
    work = work.sort_values(
        ["coordinateUncertaintyInMeters", "year"], ascending=[True, False]
    )
    work = work.drop_duplicates(subset=["_cell_lon", "_cell_lat"], keep="first")
    return work.drop(columns=["_cell_lon", "_cell_lat"]).reset_index(drop=True)
