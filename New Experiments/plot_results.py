"""
plot_results.py
===============
Load experiment CSVs and produce publication-ready plots.

When CSVs contain multiple trials (a `trial` column), curves show the
trial-averaged mean with a shaded +/-1 std-dev band.

Usage
-----
  python plot_results.py
  python plot_results.py --datasets MNIST MovieLens20M
  python plot_results.py --show
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# Visual style
# ─────────────────────────────────────────────────────────────

ALG_STYLE = {
    "SphericalHRD":  {"color": "#1f77b4", "lw": 2.5, "ls": "-",   "label": "Spherical HRD"},
    "GrassmannHRD":  {"color": "#2ca02c", "lw": 2.5, "ls": "-",   "label": "Grassmannian HRD"},
    "FantopeOGD":    {"color": "#9467bd", "lw": 2.5, "ls": "-",   "label": "Fantope OGD"},
    "OfflineOptimum":{"color": "#d62728", "lw": 2.0, "ls": "--",  "label": "Offline Optimum"},
    "StreamingSVD":  {"color": "#ff7f0e", "lw": 2.0, "ls": "--",  "label": "Streaming SVD"},
    "FTRL":          {"color": "#8c564b", "lw": 2.0, "ls": "-.",  "label": "FTRL"},
    "BadNet":        {"color": "#222222", "lw": 2.0, "ls": ":",   "label": "Fixed Baseline"},
}

SKIP_ALGS = {"StreamingSVD"}

# ─────────────────────────────────────────────────────────────
# CSV discovery and loading
# ─────────────────────────────────────────────────────────────

def discover_results(results_dir: Path):
    found = []
    for path in sorted(results_dir.glob("*__*__k*.csv")):
        stem = path.stem
        parts = stem.split("__")
        if len(parts) != 3:
            continue
        ds, alg, k_str = parts
        if not k_str.startswith("k"):
            continue
        try:
            k = int(k_str[1:])
        except ValueError:
            continue
        found.append((ds, alg, k, path))
    return found


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sort_values(["trial", "step"] if "trial" in df.columns else ["step"])
    return df.reset_index(drop=True)


def average_trials(df: pd.DataFrame):
    """
    Average instantaneous_loss and cumulative_loss across trials.

    Returns (mean_df, std_df). If no `trial` column, std_df is None
    and mean_df is the original dataframe.
    """
    if "trial" not in df.columns:
        return df, None

    grp = df.groupby("step")
    mean_df = grp[["instantaneous_loss", "cumulative_loss", "n_leaves"]].mean().reset_index()
    std_df  = grp[["instantaneous_loss", "cumulative_loss"]].std(ddof=1).reset_index()

    for col in ["dataset", "algorithm", "k", "d"]:
        if col in df.columns:
            mean_df[col] = df[col].iloc[0]

    return mean_df, std_df


# ─────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────

def _shade(ax, x, mean, std, color, alpha=0.25):
    if std is not None:
        ax.fill_between(x, mean - std, mean + std, alpha=alpha, color=color, linewidth=0)


def plot_cumulative(dataset, k, alg_mean, alg_std, out_dir, show=False, fontsize=18,
                    show_bands=True):
    fig, ax = plt.subplots(figsize=(9, 5))

    for alg, mean_df in sorted(alg_mean.items()):
        style = ALG_STYLE.get(alg, {"color": "black", "lw": 1.5, "ls": "-", "label": alg})
        std_df = alg_std.get(alg)
        x = mean_df["step"]
        y = mean_df["cumulative_loss"]
        ax.plot(x, y, color=style["color"], lw=style["lw"],
                linestyle=style["ls"], label=style["label"])
        if show_bands and std_df is not None:
            _shade(ax, x, y, std_df["cumulative_loss"].values, style["color"])

    ax.set_xlabel("Time Step", fontsize=fontsize)
    ax.set_ylabel("Cumulative Loss", fontsize=fontsize)
    ax.set_title(f"{dataset}, k = {k}", fontsize=fontsize)
    ax.legend(fontsize=fontsize - 4, loc="upper left")
    ax.tick_params(labelsize=fontsize - 4)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if show:
        plt.show()
    else:
        fname = out_dir / f"{dataset}__k{k}__cumulative.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  Saved: {fname}")
    plt.close(fig)


def plot_instantaneous(dataset, k, alg_mean, alg_std, out_dir, show=False,
                       fontsize=18, window=20, show_bands=True):
    fig, ax = plt.subplots(figsize=(9, 5))

    for alg, mean_df in sorted(alg_mean.items()):
        style = ALG_STYLE.get(alg, {"color": "black", "lw": 1.5, "ls": "-", "label": alg})
        std_df = alg_std.get(alg)
        x = mean_df["step"]
        y = mean_df["instantaneous_loss"].rolling(window, min_periods=1).mean()
        ax.plot(x, y, color=style["color"], lw=style["lw"],
                linestyle=style["ls"], label=style["label"], alpha=0.8)
        if show_bands and std_df is not None:
            s = std_df["instantaneous_loss"].rolling(window, min_periods=1).mean()
            _shade(ax, x, y, s, style["color"])

    ax.set_xlabel("Time Step", fontsize=fontsize)
    ax.set_ylabel(f"Inst. Loss (rolling {window})", fontsize=fontsize)
    ax.set_title(f"{dataset}, k = {k} (smoothed)", fontsize=fontsize)
    ax.legend(fontsize=fontsize - 4, loc="upper right")
    ax.tick_params(labelsize=fontsize - 4)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if show:
        plt.show()
    else:
        fname = out_dir / f"{dataset}__k{k}__instantaneous.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  Saved: {fname}")
    plt.close(fig)


def plot_cumulative_row(dataset, k_list, out_dir, show=False, fontsize=16, show_bands=True):
    """One figure, one subplot per k, cumulative loss."""
    n = len(k_list)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for i, (ax, (k, alg_mean, alg_std)) in enumerate(zip(axes, k_list)):
        for alg, mean_df in sorted(alg_mean.items()):
            style = ALG_STYLE.get(alg, {"color": "black", "lw": 1.5, "ls": "-", "label": alg})
            std_df = alg_std.get(alg)
            x = mean_df["step"]
            y = mean_df["cumulative_loss"]
            ax.plot(x, y, color=style["color"], lw=style["lw"],
                    linestyle=style["ls"], label=style["label"])
            if show_bands and std_df is not None:
                _shade(ax, x, y, std_df["cumulative_loss"].values, style["color"])
        ax.set_xlabel("Time Step", fontsize=fontsize)
        ax.set_title(f"k = {k}", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize - 2)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_ylabel("Cumulative Loss", fontsize=fontsize)

    axes[-1].legend(fontsize=fontsize - 2, loc="upper left")
    fig.suptitle(dataset, fontsize=fontsize + 2, y=1.02)
    fig.tight_layout()

    if show:
        plt.show()
    else:
        fname = out_dir / f"{dataset}__cumulative_row.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  Saved: {fname}")
    plt.close(fig)


def print_summary_table(all_records):
    if not all_records:
        print("No results to summarise.")
        return

    rows = []
    for ds, alg, k, mean_df, std_df in all_records:
        if mean_df.empty:
            continue
        final_cum = mean_df["cumulative_loss"].iloc[-1]
        n_trials  = "n/a" if std_df is None else int(
            (std_df["cumulative_loss"].notna()).sum() + 1
        )
        rows.append({"Dataset": ds, "Algorithm": alg, "k": k,
                     "Steps": len(mean_df), "FinalCumLoss(mean)": final_cum})
    if not rows:
        return

    tbl = pd.DataFrame(rows).sort_values(["Dataset", "k", "FinalCumLoss(mean)"])
    print("\n" + "=" * 72)
    print("RESULTS SUMMARY  (averaged across trials)")
    print("=" * 72)
    print(tbl.to_string(index=False, float_format="{:.4f}".format))


# ─────────────────────────────────────────────────────────────
# Main plotting driver
# ─────────────────────────────────────────────────────────────

def plot_all(results_dir="results", out_dir="figures",
             datasets=None, algorithms=None, k_values=None,
             show=False, fontsize=18, show_bands=True):
    results_dir = Path(results_dir)
    out_dir_p   = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    entries = discover_results(results_dir)
    if not entries:
        print(f"No result CSVs found in {results_dir}.")
        return

    entries = [(d, a, k, p) for d, a, k, p in entries if a not in SKIP_ALGS]
    if datasets:
        entries = [(d, a, k, p) for d, a, k, p in entries if d in datasets]
    if algorithms:
        entries = [(d, a, k, p) for d, a, k, p in entries if a in algorithms]
    if k_values:
        entries = [(d, a, k, p) for d, a, k, p in entries if k in k_values]

    print(f"Found {len(entries)} result file(s).")

    all_records = []
    for ds, alg, k, path in entries:
        try:
            df = load_csv(path)
            mean_df, std_df = average_trials(df)
            all_records.append((ds, alg, k, mean_df, std_df))
        except Exception as e:
            print(f"  [WARN] Could not load {path}: {e}")

    print_summary_table(all_records)

    # Group by (dataset, k) -> {alg: (mean_df, std_df)}
    groups = defaultdict(lambda: (dict(), dict()))
    for ds, alg, k, mean_df, std_df in all_records:
        groups[(ds, k)][0][alg] = mean_df
        groups[(ds, k)][1][alg] = std_df

    print(f"\nGenerating plots in {out_dir_p}/")
    for (ds, k), (alg_mean, alg_std) in sorted(groups.items()):
        if not alg_mean:
            continue
        plot_cumulative(ds, k, alg_mean, alg_std, out_dir_p, show=show, fontsize=fontsize, show_bands=show_bands)
        plot_instantaneous(ds, k, alg_mean, alg_std, out_dir_p, show=show, fontsize=fontsize, show_bands=show_bands)

    # Row plots: all k values side by side per dataset
    by_dataset = defaultdict(list)
    for (ds, k), (alg_mean, alg_std) in sorted(groups.items()):
        if alg_mean:
            by_dataset[ds].append((k, alg_mean, alg_std))
    for ds, k_list in sorted(by_dataset.items()):
        if len(k_list) > 1:
            plot_cumulative_row(ds, k_list, out_dir_p, show=show, fontsize=fontsize - 2, show_bands=show_bands)

    print("Plotting complete.")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Plot online LRA benchmark results")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--output_dir",  default="figures")
    p.add_argument("--datasets",    nargs="+", default=None)
    p.add_argument("--algorithms",  nargs="+", default=None)
    p.add_argument("--k_values",    nargs="+", type=int, default=None)
    p.add_argument("--show",        action="store_true")
    p.add_argument("--no_bands",    action="store_true", help="Disable shaded error bands")
    p.add_argument("--fontsize",    type=int, default=18)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    plot_all(
        results_dir=args.results_dir,
        out_dir=args.output_dir,
        datasets=args.datasets,
        algorithms=args.algorithms,
        k_values=args.k_values,
        show=args.show,
        fontsize=args.fontsize,
        show_bands=not args.no_bands,
    )
