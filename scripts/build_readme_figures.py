"""Regenerate the README figures and supporting tables from cached CME data.

Run from the repository root after notebook 02:
    python scripts/build_readme_figures.py
No data is downloaded. Notebook 07 rebuilds the standardized panel, notebook
10 supplies descriptive geometry and execution comparisons, and the shared
timing module supplies the chronological forecast evaluation.
"""
from pathlib import Path
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / "matplotlib-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nbformat
from nbclient import NotebookClient


def execute_notebook(path):
    notebook = nbformat.read(path, as_version=4)
    # Saved IDE display outputs can contain nonstandard fields (jetTransient).
    # Fresh execution needs only source cells; leave the source file untouched.
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    nbformat.validate(notebook)
    NotebookClient(notebook, timeout=1800, kernel_name="python3",
                   allow_errors=False,
                   resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()


def main():
    figures = ROOT / "docs" / "figures"
    tables = ROOT / "docs" / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    for name in ("07_build_standardized_roll_panel.ipynb",
                 "10_analyze_crossover_geometry_and_costs.ipynb"):
        print(f"Running {name}", flush=True)
        execute_notebook(ROOT / "notebooks" / name)
    subprocess.run([sys.executable, "-m", "src.timing_model"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-m", "src.timing_grid_search"], cwd=ROOT, check=True)

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    panel = pd.read_parquet(ROOT / "data/processed/zn_roll_panel.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, column, title, color in zip(
            axes, ("next_volume_share", "next_oi_share"),
            ("Trading volume", "Open interest"), ("#2563a6", "#cb6733")):
        for _, group in panel.groupby("roll_id"):
            ax.plot(group.bdays_to_fnd, group[column] * 100,
                    color=color, alpha=.12, linewidth=.8)
        q = panel.groupby("bdays_to_fnd")[column].quantile([.25, .5, .75]).unstack() * 100
        ax.fill_between(q.index, q[.25], q[.75], color=color, alpha=.18, label="Middle 50% of rolls")
        ax.plot(q.index, q[.5], color=color, linewidth=2.5, label="Median")
        ax.axhline(50, color="gray", linestyle="--", linewidth=.8)
        ax.axvline(0, color="gray", linestyle=":", linewidth=.8)
        ax.set(title=title, xlabel="Panel observations relative to supplied FND",
               xlim=(-12, 3), ylim=(0, 100))
        ax.grid(alpha=.15)
        ax.legend(fontsize=9, loc="upper left")
    axes[0].set_ylabel("Next contract's share (%)")
    fig.suptitle(f"Liquidity moves in stages | {panel.roll_id.nunique()} quarterly ZN rolls, 2020–2026")
    fig.tight_layout()
    fig.savefig(figures / "liquidity_migration.png", dpi=180)
    plt.close(fig)

    timing_dir = ROOT / "results/issue-20"
    summary = pd.read_csv(timing_dir / "summary.csv")
    labels = {"historical_median": "Historical median", "ols_asof": "Corrected OLS",
              "ridge_reduced_10": "Ridge: volume, α=10",
              "ridge_full_10": "Ridge: six features, α=10",
              "ridge_full_100": "Ridge: six features, α=100",
              "selected": "Training-only selection"}
    ordered = summary.set_index("model").loc[list(labels)]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(list(labels.values()), ordered.mae,
                   color=["#2563a6", "#9eabb8", "#9eabb8", "#9eabb8", "#9eabb8", "#2563a6"])
    ax.bar_label(bars, fmt="%.3f", padding=5)
    ax.invert_yaxis()
    ax.set(xlabel="Mean absolute error (panel observations; lower is better)",
           title="A simple calendar forecast survives the model comparison",
           xlim=(0, ordered.mae.max() * 1.2))
    fig.tight_layout()
    fig.savefig(figures / "crossover_forecasts.png", dpi=180)
    plt.close(fig)

    geometry_dir = ROOT / "results/crossover_geometry"
    geometry = pd.read_csv(geometry_dir / "roll_level_geometry.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.linspace(-5, 5, 201)
    for j, (measure, label, color) in enumerate(
            (("volume", "Volume", "#2563a6"), ("oi", "Open interest", "#cb6733"))):
        slopes = geometry[f"k_sharpness_{measure}"].dropna()
        for k in slopes:
            axes[0].plot(x, 100 / (1 + np.exp(-k * x)), color=color, alpha=.10)
        axes[0].plot(x, 100 / (1 + np.exp(-slopes.median() * x)),
                     color=color, linewidth=2.5, label=label)
        widths = 2 * np.log(4) / slopes
        axes[1].scatter(j + np.linspace(-.12, .12, len(widths)), widths,
                        color=color, alpha=.6, s=25)
        axes[1].plot([j-.22, j+.22], [widths.median()]*2, color="black", linewidth=2)
        axes[1].annotate(f"Median {widths.median():.2f}", (j+.25, widths.median()), fontsize=9)
    axes[0].axhspan(20, 80, color="gray", alpha=.07)
    axes[0].axvline(0, color="gray", linestyle=":")
    axes[0].set(xlabel="Panel observations from each fitted midpoint",
                ylabel="Fitted next-contract share (%)", title="Centered logistic fits: descriptive, full-window")
    axes[0].legend()
    axes[1].set(xticks=[0, 1], xticklabels=["Volume", "Open interest"], xlim=(-.5, 1.9),
                ylabel="20%–80% transition width (panel observations)",
                title="Every dot is one quarterly roll")
    for ax in axes:
        ax.grid(alpha=.15)
    fig.suptitle("How sharp is crossover? The central transition takes roughly 3–4 observations")
    fig.tight_layout()
    fig.savefig(figures / "crossover_sharpness.png", dpi=180)
    plt.close(fig)

    grid_dir = timing_dir / "grid-search"
    grid = pd.read_csv(grid_dir / "summary.csv")
    alphas = (.1, 1, 10, 100, 1000)
    feature_sets = ("volume", "open_interest", "full")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    max_mae = grid.loc[grid.model.str.startswith("ridge_"), "mae"].max()
    for ax, cutoff in zip(axes, (-10, -5)):
        scores = grid.loc[grid.decision_cutoff == cutoff].set_index("model").mae
        values = np.array([[scores[f"ridge_{features}_a{alpha:g}"] for alpha in alphas]
                           for features in feature_sets])
        im = ax.imshow(values, cmap="Blues", vmin=0, vmax=max_mae, aspect="auto")
        for (row, col), value in np.ndenumerate(values):
            ax.text(col, row, f"{value:.3f}", ha="center", va="center",
                    color="white" if value > max_mae * .6 else "black")
        ax.set(xticks=range(5), xticklabels=[f"{a:g}" for a in alphas],
               yticks=range(3), yticklabels=["Volume: 2 features", "OI: 2 features", "All 6 features"],
               xlabel="Ridge strength α (more shrinkage →)",
               title=f"Cutoff {cutoff} | baseline MAE {scores['historical_median']:.3f}")
    fig.colorbar(im, ax=axes, label="Diagnostic held-out MAE (panel observations)", shrink=.8)
    fig.suptitle("Feature-set diagnostics | nested selection keeps the baseline in all 19 rolls at both cutoffs")
    fig.savefig(figures / "feature_selection.png", dpi=180)
    plt.close(fig)

    event = pd.read_csv(geometry_dir / "event_time_liquidity_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    event_metrics = (
        ("spread_volume_share", "Calendar-spread activity", "Share of observed roll activity"),
        ("spread_half_range_cost_proxy", "Daily half-range proxy", "Quoted price points"),
        ("spread_roll_half_spread_cost_est", "Roll-style half-spread estimate", "Quoted price points"),
        ("spread_impact_per_100k", "Price-movement / volume proxy", "Price points per 100,000 contracts"),
    )
    for ax, (metric, title, ylabel) in zip(axes.flat, event_metrics):
        s = event.loc[event.metric == metric].sort_values("event_time")
        ax.fill_between(s.event_time, s.q25, s.q75, color="#2563a6", alpha=.18, label="Middle 50%")
        ax.plot(s.event_time, s['median'], color="#2563a6", marker="o", label="Median")
        ax.axvline(0, color="gray", linestyle="--", label="Realized volume crossover")
        ax.set(title=f"{title} | n={s['count'].min()}–{s['count'].max()} per offset",
               ylabel=ylabel, xlabel="Panel observations from realized volume crossover")
        ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Conditions around crossover | daily proxies, not observed execution costs")
    fig.tight_layout()
    fig.savefig(figures / "crossover_event_conditions.png", dpi=180)
    plt.close(fig)

    benchmarks = pd.read_csv(geometry_dir / "benchmark_roll_conditions.csv")
    rules = ["fixed_-5", "fixed_-3", "realized_volume_crossover", "fixed_-1"]
    names = ["Fixed −5", "Fixed −3", "Realized\ncrossover", "Fixed −1"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, title, ylabel in zip(
            axes, ("spread_half_range_cost_proxy", "spread_volume_share"),
            ("Daily range proxy: lower is better", "Calendar-spread activity share"),
            ("Median half-range (quoted price points)", "Median share of observed roll activity")):
        paired = benchmarks.pivot(index="roll_id", columns="rule", values=metric).reindex(columns=rules).dropna()
        for _, row in paired.iterrows():
            ax.plot(range(4), row, color="#9eabb8", alpha=.15, linewidth=.7)
        ax.plot(range(4), paired.median(), color="#2563a6", marker="o", linewidth=2.5)
        ax.set(xticks=range(4), xticklabels=names, title=f"{title}\n{len(paired)} matching rolls", ylabel=ylabel)
        ax.grid(axis="y", alpha=.15)
    fig.suptitle("Crossover is a descriptive milestone; earlier schedules can look more favorable")
    fig.tight_layout()
    fig.savefig(figures / "execution_conditions.png", dpi=180)
    plt.close(fig)

    for source, target in (
            (timing_dir / "summary.csv", "timing_summary.csv"),
            (geometry_dir / "roll_level_geometry.csv", "roll_geometry.csv"),
            (geometry_dir / "benchmark_roll_conditions.csv", "execution_conditions.csv"),
            (geometry_dir / "event_time_liquidity_summary.csv", "event_time_conditions.csv"),
            (grid_dir / "summary.csv", "feature_grid_summary.csv"),
            (grid_dir / "selected_models.csv", "feature_grid_selections.csv")):
        shutil.copyfile(source, tables / target)
    print("Saved README figures to docs/figures/ and supporting tables to docs/tables/")


if __name__ == "__main__":
    main()
