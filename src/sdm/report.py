"""Tables, figures and the evidence summary that closes the comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURES
from .models import MODEL_LABELS

MODEL_COLORS = {
    "SVM (RBF)": "#3C6E9F",
    "Random forest": "#2F7D5D",
    "MaxEnt": "#B0632B",
    "XGBoost": "#7A4E8A",
}

MODEL_ORDER = ["SVM (RBF)", "Random forest", "MaxEnt", "XGBoost"]


def _ordered_models(frame: pd.DataFrame) -> list[str]:
    present = set(frame["model"].unique())
    return [name for name in MODEL_ORDER if name in present]


def plot_metrics(comparison: pd.DataFrame, path: Path = FIGURES / "metrics.png") -> Path:
    """AUC and TSS per species, grouped by model, spatial blocks only."""
    data = comparison[comparison["cv"] == "spatial"]
    species = list(dict.fromkeys(data["species"]))
    models = _ordered_models(data)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=False)
    width = 0.8 / max(len(models), 1)
    x = np.arange(len(species))

    for ax, metric, label in (
        (axes[0], "auc_mean", "AUC (spatial block CV)"),
        (axes[1], "tss_mean", "TSS (spatial block CV)"),
    ):
        sd_column = metric.replace("_mean", "_sd")
        for index, model in enumerate(models):
            lookup = (
                data[data["model"] == model].set_index("species").reindex(species)
            )
            values = lookup[metric].to_numpy(dtype=float)
            errors = lookup[sd_column].to_numpy(dtype=float)
            errors = np.where(np.isfinite(errors), errors, 0.0)
            offset = (index - (len(models) - 1) / 2) * width
            ax.bar(
                x + offset,
                values,
                width=width * 0.92,
                yerr=errors,
                capsize=2,
                color=MODEL_COLORS.get(model, "#666666"),
                label=model,
                error_kw={"elinewidth": 0.8, "ecolor": "#333333"},
            )
        ax.set_xticks(x)
        ax.set_xticklabels([name.replace("-", " ") for name in species], fontsize=9)
        ax.set_ylabel(label)
        # Limits follow the data: a fixed floor would hide a genuinely weak model.
        low = np.nanmin(values - errors) if np.isfinite(values).any() else 0.0
        ax.set_ylim(min(0.35, max(0.0, low - 0.05)), 1.0)
        ax.grid(axis="y", linewidth=0.4, alpha=0.35)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=len(models), frameon=False, fontsize=8
    )
    fig.suptitle(
        "Model performance under spatial block cross-validation, error bars are one SD over folds",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_learning(curve: pd.DataFrame, path: Path = FIGURES / "learning_curve.png") -> Path:
    if curve.empty:
        return path
    species = list(dict.fromkeys(curve["species"]))
    models = _ordered_models(curve)
    fig, axes = plt.subplots(1, len(species), figsize=(4.1 * len(species), 3.8), squeeze=False)

    for ax, name in zip(axes[0], species):
        block = curve[curve["species"] == name]
        for model in models:
            line = block[block["model"] == model].sort_values("n_presence")
            if line.empty:
                continue
            ax.plot(
                line["n_presence"],
                line["auc_mean"],
                marker="o",
                markersize=4,
                linewidth=1.4,
                color=MODEL_COLORS.get(model, "#666666"),
                label=model,
            )
            ax.fill_between(
                line["n_presence"],
                line["auc_mean"] - line["auc_sd"].fillna(0),
                line["auc_mean"] + line["auc_sd"].fillna(0),
                color=MODEL_COLORS.get(model, "#666666"),
                alpha=0.12,
                linewidth=0,
            )
        ax.set_xscale("log")
        ax.set_title(name.replace("-", " "), fontsize=9)
        ax.set_xlabel("presences used for training")
        ax.set_ylabel("AUC (spatial block CV)")
        ax.grid(linewidth=0.4, alpha=0.35)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(models), frameon=False, fontsize=8)
    fig.suptitle("Sample-size sensitivity, background held constant", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def evidence_table(comparison: pd.DataFrame, curve: pd.DataFrame) -> pd.DataFrame:
    """Turn the run into the quantities the algorithm-choice argument rests on."""
    rows: list[dict] = []
    spatial = comparison[comparison["cv"] == "spatial"]
    random_cv = comparison[comparison["cv"] == "random"]

    for model in _ordered_models(spatial):
        block = spatial[spatial["model"] == model]
        rnd = random_cv[random_cv["model"] == model]
        merged = block.merge(rnd, on=["species", "model"], suffixes=("_spatial", "_random"))
        rows.append(
            {
                "model": model,
                "mean_auc_spatial": round(float(block["auc_mean"].mean()), 3),
                "mean_tss_spatial": round(float(block["tss_mean"].mean()), 3),
                "mean_auc_random": round(float(rnd["auc_mean"].mean()), 3)
                if not rnd.empty
                else np.nan,
                "auc_inflation_random_minus_spatial": round(
                    float((merged["auc_mean_random"] - merged["auc_mean_spatial"]).mean()), 3
                )
                if not merged.empty
                else np.nan,
                "mean_area_frac": round(float(block["suitable_area_frac"].mean()), 3),
                "mean_runtime_s": round(float(block["runtime_s"].mean()), 1),
            }
        )
    table = pd.DataFrame(rows)

    if not curve.empty:
        small = curve.sort_values("n_presence").groupby("model").first()
        large = curve.sort_values("n_presence").groupby("model").last()
        table["auc_at_smallest_n"] = table["model"].map(
            {model: round(float(row["auc_mean"]), 3) for model, row in small.iterrows()}
        )
        table["auc_at_largest_n"] = table["model"].map(
            {model: round(float(row["auc_mean"]), 3) for model, row in large.iterrows()}
        )
        table["auc_gain_with_n"] = (
            table["auc_at_largest_n"] - table["auc_at_smallest_n"]
        ).round(3)

    # Rank by spatial AUC, 1 is best, averaged across species.
    ranks = (
        spatial.groupby(["species", "model"])["auc_mean"]
        .mean()
        .reset_index()
        .assign(rank=lambda frame: frame.groupby("species")["auc_mean"].rank(ascending=False))
    )
    table["mean_rank_auc"] = table["model"].map(
        ranks.groupby("model")["rank"].mean().round(1)
    )
    return table.sort_values("mean_auc_spatial", ascending=False).reset_index(drop=True)


def write_evidence(comparison: pd.DataFrame, curve: pd.DataFrame, path: Path) -> pd.DataFrame:
    table = evidence_table(comparison, curve)
    table.to_csv(path.with_suffix(".csv"), index=False)

    lines = ["# Evidence summary", ""]
    lines.append(
        "All values are means over the species panel under spatial block "
        "cross-validation, computed by `scripts/run_sdm.py`. Regenerate with "
        "`python scripts/run_sdm.py --species all`."
    )
    lines.append("")
    lines.append(table.to_markdown(index=False))
    lines.append("")

    best = table.iloc[0]
    lines.append(f"- Highest mean spatial AUC: {best['model']} ({best['mean_auc_spatial']:.3f}).")
    if table["auc_inflation_random_minus_spatial"].notna().any():
        worst_gap = table.loc[table["auc_inflation_random_minus_spatial"].idxmax()]
        lines.append(
            "- Largest random-to-blocked AUC drop: "
            f"{worst_gap['model']} ({worst_gap['auc_inflation_random_minus_spatial']:+.3f}), "
            "which is the share of its accuracy that random folds hand it for free."
        )
    if "auc_gain_with_n" in table.columns and table["auc_gain_with_n"].notna().any():
        fastest = table.loc[table["auc_gain_with_n"].idxmax()]
        lines.append(
            "- Steepest gain from more presences: "
            f"{fastest['model']} ({fastest['auc_gain_with_n']:+.3f} AUC from the smallest to the largest training set)."
        )
    lines.append("")
    path.write_text("\n".join(lines))
    return table


def write_all(results: dict, outdir: Path) -> dict:
    comparison = results["comparison"]
    curve = results["curve"]
    paths = {
        "metrics": plot_metrics(comparison, outdir / "figures" / "metrics.png"),
        "learning": plot_learning(curve, outdir / "figures" / "learning_curve.png"),
    }
    evidence = write_evidence(comparison, curve, outdir / "algorithm_fit.md")
    return {"figures": paths, "evidence": evidence}
