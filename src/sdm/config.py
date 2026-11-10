"""Paths, constants and the species panel for the SDM benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
VENDOR = ROOT / "vendor"
MAXENT_JAR = VENDOR / "maxent.jar"

# WorldClim 2.1 bioclim at 10 arc minutes, the coarsest standard product.
GRID_DEG = 10.0 / 60.0
CRS = "EPSG:4326"

WORLDCLIM_BIO_URL = (
    "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_bio.zip"
)
WORLDCLIM_ELEV_URL = (
    "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_elev.zip"
)

BIO_NAMES = {
    1: "Annual mean temperature",
    2: "Mean diurnal range",
    3: "Isothermality",
    4: "Temperature seasonality",
    5: "Max temperature of warmest month",
    6: "Min temperature of coldest month",
    7: "Temperature annual range",
    8: "Mean temperature of wettest quarter",
    9: "Mean temperature of driest quarter",
    10: "Mean temperature of warmest quarter",
    11: "Mean temperature of coldest quarter",
    12: "Annual precipitation",
    13: "Precipitation of wettest month",
    14: "Precipitation of driest month",
    15: "Precipitation seasonality",
    16: "Precipitation of wettest quarter",
    17: "Precipitation of driest quarter",
    18: "Precipitation of warmest quarter",
    19: "Precipitation of coldest quarter",
}

# GBIF issue flags that make a record unusable for a correlative model. These
# mirror the fatal checks in the GBIF data-quality vocabulary rather than
# relying on a subjective reading of the coordinates.
CRITICAL_GBIF_ISSUES = {
    "ZERO_COORDINATE",
    "COORDINATE_OUT_OF_RANGE",
    "COORDINATE_INVALID",
    "COUNTRY_COORDINATE_MISMATCH",
    "PRESUMED_SWAPPED_COORDINATE",
    "PRESUMED_NEGATED_LAT",
    "PRESUMED_NEGATED_LON",
    "GEODETIC_DATUM_INVALID",
    "FOOTPRINT_WKT_INVALID",
}

ACCEPTED_BASIS = {
    "PRESERVED_SPECIMEN",
    "HUMAN_OBSERVATION",
    "OBSERVATION",
    "MACHINE_OBSERVATION",
    "MATERIAL_SAMPLE",
    "LIVING_SPECIMEN",
}


@dataclass(frozen=True)
class SpeciesCase:
    """One species, one study extent, one kind of data situation."""

    key: str
    scientific_name: str
    bbox: tuple[float, float, float, float]  # min_lon, min_lat, max_lon, max_lat
    regime: str
    rationale: str

    @property
    def token(self) -> str:
        """Filesystem and MaxEnt safe species label."""
        return self.key.replace("-", "_")

    @property
    def extent_deg2(self) -> float:
        minx, miny, maxx, maxy = self.bbox
        return (maxx - minx) * (maxy - miny)


PANEL: dict[str, SpeciesCase] = {
    "quercus-robur": SpeciesCase(
        key="quercus-robur",
        scientific_name="Quercus robur",
        bbox=(-12.0, 34.0, 50.0, 71.0),
        regime="high effort, broad niche",
        rationale=(
            "Pedunculate oak. Over a million georeferenced records, a continuous "
            "European range and a wide climatic envelope. This is the case where "
            "flexible learners have enough signal to fit shape without overfitting."
        ),
    ),
    "ursus-arctos": SpeciesCase(
        key="ursus-arctos",
        scientific_name="Ursus arctos",
        bbox=(-12.0, 34.0, 50.0, 71.0),
        regime="moderate effort, fragmented range",
        rationale=(
            "Brown bear. Tens of thousands of records concentrated in a handful of "
            "populations, with large intervening gaps. Same extent as the oak so the "
            "background pool is directly comparable, but the presences are clustered."
        ),
    ),
    "welwitschia-mirabilis": SpeciesCase(
        key="welwitschia-mirabilis",
        scientific_name="Welwitschia mirabilis",
        bbox=(10.0, -30.0, 18.0, -16.0),
        regime="low effort, narrow endemic",
        rationale=(
            "Namib desert gymnosperm restricted to the coastal fog belt of Namibia and "
            "southern Angola. Under a thousand records, a narrow climatic envelope and a "
            "small accessible area, which is the regime where maximum entropy is usually "
            "the recommended default and where high-variance learners have least to work with."
        ),
    ),
}


def case(key: str) -> SpeciesCase:
    if key not in PANEL:
        raise KeyError(f"unknown species case {key!r}; choose from {sorted(PANEL)}")
    return PANEL[key]


def all_cases() -> list[SpeciesCase]:
    return [PANEL[k] for k in PANEL]
