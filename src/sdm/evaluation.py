"""Shared evaluation: same folds, same metric, same threshold rule for all models."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, StratifiedShuffleSplit

from .models import SDMModel


@dataclass
class FoldResult:
    fold: int
    auc: float
    tss: float
    tss_trainopt: float
    threshold: float
    sensitivity: float
    specificity: float


def spatial_blocks(
    coords: np.ndarray,
    bbox: tuple[float, float, float, float],
    n_blocks: int = 5,
) -> np.ndarray:
    """Assign every point to a square block of the calibration extent.

    Holding out whole blocks, rather than random points, is what stops a model
    from being rewarded for interpolating between neighbouring records. The
    difference between the random and blocked scores is itself a result: it
    measures how much of each model's accuracy comes from spatial
    autocorrelation.
    """
    minx, miny, maxx, maxy = bbox
    lon = coords[:, 0]
    lat = coords[:, 1]
    width = max(maxx - minx, 1e-6)
    height = max(maxy - miny, 1e-6)
    col = np.clip(((lon - minx) / width * n_blocks).astype(int), 0, n_blocks - 1)
    row = np.clip(((maxy - lat) / height * n_blocks).astype(int), 0, n_blocks - 1)
    return row * n_blocks + col


def make_splits(
    y: np.ndarray,
    coords: np.ndarray,
    bbox: tuple[float, float, float, float],
    scheme: str = "spatial",
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return index pairs shared by every model for a given scheme."""
    if scheme == "random":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(len(y)), y))

    if scheme != "spatial":
        raise ValueError(f"unknown validation scheme {scheme!r}")

    groups = spatial_blocks(coords, bbox)
    n_groups = len(np.unique(groups))
    splits = min(n_splits, n_groups)
    if splits < 2:
        raise ValueError("the study extent produced fewer than two spatial blocks")
    splitter = GroupKFold(n_splits=splits)
    return list(splitter.split(np.zeros(len(y)), y, groups=groups))


def tss_at(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    if not np.isfinite(threshold):
        return {"sensitivity": np.nan, "specificity": np.nan, "tss": np.nan}
    predicted = score >= threshold
    positives = y == 1
    negatives = ~positives
    tp = np.sum(predicted & positives)
    fn = np.sum(~predicted & positives)
    tn = np.sum(~predicted & negatives)
    fp = np.sum(predicted & negatives)
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tss": float(sensitivity + specificity - 1),
    }


def p10_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Threshold at the 10th percentile of scores observed at training presences.

    This is the "10 percentile training presence" rule used in SDM practice: a
    model is required to call at least 90 percent of the presences it was
    trained on suitable. It costs nothing to estimate and, unlike an optimised
    threshold, it cannot be gamed by a learner whose in-sample predictions are
    saturated at zero and one.
    """
    positives = score[y == 1]
    positives = positives[np.isfinite(positives)]
    if len(positives) == 0:
        return float("nan")
    return float(np.percentile(positives, 10))


def best_tss_threshold(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """Pick the threshold that maximises training TSS.

    Reported as a secondary column only. An unpruned forest fits its training
    points almost perfectly, so this threshold sits near the saturated end of
    its own score range and transfers badly to held-out data; using it as the
    headline would penalise flexible learners for a threshold artefact rather
    than for their discrimination.
    """
    finite = np.isfinite(score)
    if finite.sum() < 2:
        return float("nan"), float("nan")
    candidates = np.unique(np.quantile(score[finite], np.linspace(0.0, 1.0, 201)))
    best_thr, best_tss = float(candidates[0]), -np.inf
    for threshold in candidates:
        value = tss_at(y, score, float(threshold))["tss"]
        if np.isfinite(value) and value > best_tss:
            best_tss, best_thr = value, float(threshold)
    return best_thr, float(best_tss)


def calibration_split(
    y_train: np.ndarray, seed: int, test_size: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out part of the training fold purely to place the threshold.

    A threshold estimated on the data a model was fitted to is optimistic for
    every learner and catastrophically so for deep trees, whose in-sample
    scores saturate near zero and one. Estimating it on a held-out slice of the
    training fold costs one extra fit and keeps the test fold untouched.
    """
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    placeholder = np.zeros((len(y_train), 1))
    return next(splitter.split(placeholder, y_train))


def cross_validate(
    build: Callable[[], SDMModel],
    X: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    min_train_positives: int = 5,
    calibrate: bool = True,
) -> tuple[list[FoldResult], float]:
    """Fit and score one model across the shared folds.

    Returns the per-fold results and the wall clock time for the whole run.
    Folds that cannot produce both classes are skipped and reported as such,
    never silently scored as zero.

    With ``calibrate=False`` only AUC is produced, for callers that need the
    ranking metric alone (the learning curve) and should not pay for a second
    fit per fold.
    """
    folds: list[FoldResult] = []
    started = time.perf_counter()
    for index, (train_idx, test_idx) in enumerate(splits):
        y_train = y[train_idx]
        y_test = y[test_idx]
        if y_train.sum() < min_train_positives or len(np.unique(y_test)) < 2:
            continue


        try:
            threshold = float("nan")
            threshold_opt = float("nan")
            if calibrate:
                fit_part, cal_part = calibration_split(y_train, seed=index)
                if (
                    y_train[fit_part].sum() >= min_train_positives
                    and len(np.unique(y_train[cal_part])) == 2
                ):
                    calibration_model = build()
                    calibration_model.fit(
                        X[train_idx][fit_part],
                        y_train[fit_part],
                        coords[train_idx][fit_part],
                    )
                    calibration_scores = calibration_model.score(
                        X[train_idx][cal_part], coords[train_idx][cal_part]
                    )
                    threshold = p10_threshold(y_train[cal_part], calibration_scores)
                    threshold_opt, _ = best_tss_threshold(
                        y_train[cal_part], calibration_scores
                    )

            model = build()
            model.fit(X[train_idx], y_train, coords[train_idx])
            score_test = model.score(X[test_idx], coords[test_idx])
        except Exception as error:  # a failed fold is a gap, not a crash
            print(
                f"    fold {index} skipped: {type(error).__name__}: {error}",
                flush=True,
            )
            continue
        if not np.isfinite(score_test).any():
            continue
        metrics = tss_at(y_test, score_test, threshold)
        tss_trainopt_test = tss_at(y_test, score_test, threshold_opt)["tss"]
        valid = np.isfinite(score_test)
        auc = (
            roc_auc_score(y_test[valid], score_test[valid])
            if len(np.unique(y_test[valid])) == 2
            else np.nan
        )
        folds.append(
            FoldResult(
                fold=index,
                auc=float(auc),
                tss=metrics["tss"],
                tss_trainopt=float(tss_trainopt_test),
                threshold=threshold,
                sensitivity=metrics["sensitivity"],
                specificity=metrics["specificity"],
            )
        )
    elapsed = time.perf_counter() - started
    return folds, elapsed


def summarise_folds(folds: list[FoldResult]) -> dict:
    if not folds:
        return {
            "auc_mean": np.nan,
            "auc_sd": np.nan,
            "tss_mean": np.nan,
            "tss_sd": np.nan,
            "tss_trainopt_mean": np.nan,
            "tss_trainopt_sd": np.nan,
            "n_folds": 0,
        }
    def mean_sd(values: np.ndarray) -> tuple[float, float]:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return float("nan"), float("nan")
        mean = float(finite.mean())
        sd = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
        return mean, sd

    aucs = np.array([fold.auc for fold in folds], dtype=float)
    tsss = np.array([fold.tss for fold in folds], dtype=float)
    trainopt = np.array([fold.tss_trainopt for fold in folds], dtype=float)
    auc_mean, auc_sd = mean_sd(aucs)
    tss_mean, tss_sd = mean_sd(tsss)
    trainopt_mean, trainopt_sd = mean_sd(trainopt)
    return {
        "auc_mean": auc_mean,
        "auc_sd": auc_sd,
        "tss_mean": tss_mean,
        "tss_sd": tss_sd,
        "tss_trainopt_mean": trainopt_mean,
        "tss_trainopt_sd": trainopt_sd,
        "n_folds": len(folds),
    }
