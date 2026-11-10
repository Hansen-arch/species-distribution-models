"""The four candidate algorithms behind one small interface.

Every model exposes ``fit(X, y, coords)`` and ``score(X, coords)``. The score
only has to be monotone in suitability, because AUC is rank based and TSS uses
a threshold derived from the training scores. That is why the support vector
machine uses its decision function instead of paying for a Platt calibration,
and why MaxEnt is read from its cloglog grid.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import MAXENT_JAR

try:  # optional, the benchmark runs without it
    from xgboost import XGBClassifier

    HAVE_XGBOOST = True
except Exception:  # pragma: no cover - depends on the environment
    HAVE_XGBOOST = False


@dataclass
class FitContext:
    """Everything a model needs besides the design matrix."""

    extent_asc_dir: Path
    species_token: str
    work_dir: Path
    grid_shape: tuple[int, int]
    transform: object
    n_bands: int
    band_names: list[str] = field(default_factory=list)


class SDMModel:
    name = "base"
    family = ""
    hyperparameters = ""

    def fit(self, X: np.ndarray, y: np.ndarray, coords: np.ndarray) -> "SDMModel":
        raise NotImplementedError

    def score(self, X: np.ndarray, coords: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class SVMRBF(SDMModel):
    """RBF kernel machine on standardised predictors.

    C and gamma are fixed rather than tuned: tuning them per fold against the
    held-out block would leak information into the comparison. C = 10 and
    gamma = 'scale' are the usual starting point for presence and background
    data on a few thousand rows.
    """

    name = "SVM (RBF)"
    family = "kernel / max-margin"
    hyperparameters = "C=10, gamma=scale, standardised predictors"

    def __init__(self, C: float = 10.0, gamma: str = "scale", seed: int = 0):
        self.C = C
        self.gamma = gamma
        self.seed = seed
        self.pipeline: Pipeline | None = None

    def fit(self, X, y, coords):
        self.pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "svc",
                    SVC(
                        C=self.C,
                        gamma=self.gamma,
                        kernel="rbf",
                        probability=False,
                        cache_size=1000,
                        random_state=self.seed,
                    ),
                ),
            ]
        )
        self.pipeline.fit(X, y)
        return self

    def score(self, X, coords):
        assert self.pipeline is not None
        return self.pipeline.decision_function(X)


class RandomForestModel(SDMModel):
    """Bagged trees with leaf-level probability estimates."""

    name = "Random forest"
    family = "bagged trees"
    hyperparameters = "500 trees, min_samples_leaf=2, sqrt features, balanced"

    def __init__(self, n_estimators: int = 500, seed: int = 0):
        self.n_estimators = n_estimators
        self.seed = seed
        self.model: RandomForestClassifier | None = None

    def fit(self, X, y, coords):
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=self.seed,
        )
        self.model.fit(X, y)
        return self

    def score(self, X, coords):
        assert self.model is not None
        return self.model.predict_proba(X)[:, 1]


class XGBoostModel(SDMModel):
    """Gradient boosted trees with shallow interaction depth."""

    name = "XGBoost"
    family = "boosted trees"
    hyperparameters = "400 rounds, depth 4, lr 0.05, subsample 0.8"

    def __init__(self, n_estimators: int = 400, max_depth: int = 4, seed: int = 0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed
        self.model = None

    def fit(self, X, y, coords):
        if not HAVE_XGBOOST:
            raise RuntimeError("xgboost is not installed in this interpreter")
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            min_child_weight=1,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=self.seed,
        )
        self.model.fit(X, y)
        return self

    def score(self, X, coords):
        assert self.model is not None
        return self.model.predict_proba(X)[:, 1]


class MaxEntModel(SDMModel):
    """Maximum entropy, run through the reference Java implementation.

    Background is supplied as SWD rows rather than left to MaxEnt's own
    sampling, so every model in the table sees the identical background set.
    That deliberately handicaps MaxEnt on one axis (its default 10 000 random
    background draws are often kinder) and removes it on another (the other
    three models would otherwise face a different pseudo-absence sample).
    """

    name = "MaxEnt"
    family = "maximum entropy"
    hyperparameters = "auto features, beta 1, cloglog output"

    def __init__(
        self,
        context: FitContext,
        jar: Path = MAXENT_JAR,
        memory: str = "2g",
        max_iterations: int = 500,
    ):
        self.context = context
        self.jar = Path(jar)
        self.memory = memory
        self.max_iterations = max_iterations
        self.out_dir: Path | None = None
        self.grid: np.ndarray | None = None

    def fit(self, X, y, coords):
        if not self.jar.exists():
            raise FileNotFoundError(
                f"maxent.jar not found at {self.jar}; see the README for the download step"
            )
        if not shutil.which("java"):
            raise RuntimeError("java is required for MaxEnt and is not on PATH")

        run_dir = self.context.work_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        out_dir = run_dir / "output"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

        bands = list(self.context.band_names)
        if len(bands) != self.context.n_bands:
            raise ValueError(
                f"{len(bands)} band names for {self.context.n_bands} predictors"
            )
        presence = pd.DataFrame(coords[y == 1], columns=["x", "y"])
        background = pd.DataFrame(coords[y == 0], columns=["x", "y"])
        presence["species"] = self.context.species_token
        background["species"] = "background"
        for index, band in enumerate(bands):
            presence[band] = X[y == 1, index]
            background[band] = X[y == 0, index]

        presence_path = run_dir / "presence.csv"
        background_path = run_dir / "background.csv"
        columns = ["species", "x", "y"] + bands
        presence[columns].to_csv(presence_path, index=False)
        background[columns].to_csv(background_path, index=False)

        command = [
            "java",
            f"-mx{self.memory}",
            "-jar",
            str(self.jar),
            f"samplesfile={presence_path}",
            f"environmentallayers={background_path}",
            f"outputdirectory={out_dir}",
            f"projectionlayers={self.context.extent_asc_dir}",
            "autorun",
            "nowarnings",
            "nopictures",
            "noaskoverwrite",
            "nojackknife",
            "norandomseed",
            "nowriteclampgrid",
            "nowritemess",
            "outputformat=cloglog",
            "outputfiletype=asc",
            f"maximumiterations={self.max_iterations}",
        ]
        # MaxEnt builds a Swing window even in batch mode. Run it under a
        # virtual display whenever xvfb is available: a Colab VM may advertise
        # DISPLAY=:0 without an X server behind it, so testing DISPLAY alone
        # is not enough to tell whether a real display exists.
        launcher = shutil.which("xvfb-run")
        if launcher:
            command = [launcher, "-a"] + command

        result = subprocess.run(
            command, capture_output=True, text=True, timeout=1800
        )
        grid_path = self._find_grid(out_dir)
        if grid_path is None:
            produced = sorted(p.name for p in out_dir.glob("*"))
            raise RuntimeError(
                "MaxEnt produced no projection grid.\n"
                f"files written: {produced}\n"
                f"stdout:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-2000:]}"
            )
        from .environment import read_ascii_grid

        grid, _ = read_ascii_grid(grid_path)
        self.grid = grid
        self.out_dir = out_dir
        return self

    def _find_grid(self, out_dir: Path) -> Path | None:
        """Locate the projected grid.

        MaxEnt names the projection after the species, but appends the output
        file type when training data came from an SWD file, so the result is
        either ``species.asc`` or ``species_asc.asc``. Clamping and MESS grids
        are disabled in the command, otherwise they would need excluding here.
        """
        token = self.context.species_token
        exact = out_dir / f"{token}.asc"
        if exact.exists():
            return exact
        matches = sorted(p for p in out_dir.glob(f"{token}*.asc"))
        return matches[0] if matches else None

    def score(self, X, coords):
        if self.grid is None:
            raise RuntimeError("score called before fit")
        rows, cols = self._row_col(coords[:, 0], coords[:, 1])
        inside = (
            (rows >= 0)
            & (rows < self.grid.shape[0])
            & (cols >= 0)
            & (cols < self.grid.shape[1])
        )
        out = np.full(len(coords), np.nan)
        out[inside] = self.grid[rows[inside], cols[inside]]
        return out

    def _row_col(self, lon, lat):
        cellsize = float(self.context.transform.a)
        origin_x = float(self.context.transform.c)
        origin_y = float(self.context.transform.f)
        cols = np.floor((np.asarray(lon) - origin_x) / cellsize).astype(int)
        rows = np.floor((origin_y - np.asarray(lat)) / cellsize).astype(int)
        return rows, cols


@dataclass
class ModelFactory:
    """Builds a fresh, unfitted model for one fold."""

    spec: str
    context: FitContext | None = None
    seed: int = 0

    def build(self) -> SDMModel:
        if self.spec == "svm":
            return SVMRBF(seed=self.seed)
        if self.spec == "rf":
            return RandomForestModel(seed=self.seed)
        if self.spec == "maxent":
            if self.context is None:
                raise ValueError("MaxEnt needs a FitContext with the ASCII grid directory")
            return MaxEntModel(self.context)
        if self.spec == "xgboost":
            return XGBoostModel(seed=self.seed)
        raise KeyError(f"unknown model spec {self.spec!r}")


MODEL_ORDER = ["svm", "rf", "maxent", "xgboost"]

MODEL_LABELS = {
    "svm": "SVM (RBF)",
    "rf": "Random forest",
    "maxent": "MaxEnt",
    "xgboost": "XGBoost",
}


def available_specs(requested: list[str] | None = None) -> list[str]:
    specs = list(MODEL_ORDER)
    if requested:
        specs = [spec for spec in specs if spec in set(requested)]
    if not HAVE_XGBOOST:
        specs = [spec for spec in specs if spec != "xgboost"]
    return specs
