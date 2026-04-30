"""
plot_results.py
===============
Load experiment CSVs and produce publication-ready plots.

Usage
-----
  # Plot everything in results/
  python plot_results.py

  # Plot specific subset
  python plot_results.py --datasets MNIST CreditCard \\
                         --algorithms GrassmannHRD FantopeOGD StreamingSVD \\
                         --output_dir figures/

  # Show instead of saving
  python plot_results.py --show
"""

import argparse
import sys
from pathlib import Path

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
    "BadNet":        {"color": "#7f7f7f", "lw": 1.5, "ls": ":",   "label": "Fixed Baseline"},
}


# ─────────────────────────────────────────────────────────────
# CSV discovery
# ─────────────────────────────────────────────────────────────

def discover_results(results_dir: Path):
    """
    Scan results_dir for CSVs named <dataset>__<algorithm>__k<K>.csv.
    Returns a list of (dataset, algorithm, k, path) tuples.
    """
    found = []
    for path in sorted(results_dir.glob("*__*__k*.csv")):
        stem = path.stem           # e.g. "MNIST__GrassmannHRD__k10"
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
    df = df.sort_values("step").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────

def plot_cumulative(dataset, k, alg_dfs, out_dir, show=False, fontsize=18):
    """One figure: cumulative loss of all algorithms on dataset/k."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for alg, df in sorted(alg_dfs.items()):
        style = ALG_STYLE.get(alg, {"color": "black", "lw": 1.5, "ls": "-",
                                    "label": alg})
        ax.plot(df["step"], df["cumulative_loss"],
                color=style["color"], lw=style["lw"],
                linestyle=style["ls"], label=style["label"])

    ax.set_xlabel("Time Step", fontsize=fontsize)
    ax.set_ylabel("Cumulative Loss", fontsize=fontsize)
    ax.set_title(f"{dataset}  —  k={k}", fontsize=fontsize)
    ax.legend(fontsize=fontsize - 4, loc="upper left")
    ax.tick_params(labelsize=fontsize - 4)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if show:
        plt.show()
    else:
        fname = out_dir / f"{dataset}__k{k}__cumulative.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        print(f"  Saved: {fname}")
    plt.close(fig)


def plot_instantaneous(dataset, k, alg_dfs, out_dir, show=False, fontsize=18,
                       window=20):
    """One figure: smoothed instantaneous loss."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for alg, df in sorted(alg_dfs.items()):
        style = ALG_STYLE.get(alg, {"color": "black", "lw": 1.5, "ls": "-",
                                    "label": alg})
        y = df["instantaneous_loss"].rolling(window, min_periods=1).mean()
        ax.plot(df["step"], y,
                color=style["color"], lw=style["lw"],
                linestyle=style["ls"], label=style["label"], alpha=0.8)

    ax.set_xlabel("Time Step", fontsize=fontsize)
    ax.set_ylabel(f"Inst. Loss (rolling {window})", fontsize=fontsize)
    ax.set_title(f"{dataset}  —  k={k}  (smoothed)", fontsize=fontsize)
    ax.legend(fontsize=fontsize - 4, loc="upper right")
    ax.tick_params(labelsize=fontsize - 4)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if show:
        plt.show()
    else:
        fname = out_dir / f"{dataset}__k{k}__instantaneous.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        print(f"  Saved: {fname}")
    plt.close(fig)


def print_summary_table(all_records):
    """Print a table of final cumulative losses."""
    if not all_records:
        print("No results to summarise.")
        return

    import pandas as pd
    rows = []
    for ds, alg, k, df in all_records:
        if df.empty:
            continue
        final_cum = df["cumulative_loss"].iloc[-1]
        n_steps = len(df)
        rows.append({"Dataset": ds, "Algorithm": alg, "k": k,
                     "Steps": n_steps, "FinalCumLoss": final_cum})
    if not rows:
        return

    tbl = pd.DataFrame(rows).sort_values(["Dataset", "k", "FinalCumLoss"])
    print("\n" + "=" * 72)
    print("RESULTS SUMMARY")
    print("=" * 72)
    print(tbl.to_string(index=False, float_format="{:.4f}".format))

    # Improvement over BadNet per (dataset, k)
    print("\n" + "=" * 72)
    print("IMPROVEMENT vs. Fixed Baseline (BadNet), if available")
    print("=" * 72)
    tbl2 = tbl.copy()
    base = tbl2[tbl2["Algorithm"] == "BadNet"][["Dataset", "k", "FinalCumLoss"]]
    base = base.rename(columns={"FinalCumLoss": "BadNetLoss"})
    merged = tbl2[tbl2["Algorithm"] != "BadNet"].merge(base, on=["Dataset", "k"], how="left")
    merged["Improvement%"] = (
        (merged["BadNetLoss"] - merged["FinalCumLoss"]) / merged["BadNetLoss"] * 100
    )
    merged = merged.dropna(subset=["Improvement%"])
    if not merged.empty:
        print(merged[["Dataset", "Algorithm", "k", "FinalCumLoss", "Improvement%"]]
              .sort_values(["Dataset", "k", "Improvement%"], ascending=[True, True, False])
              .to_string(index=False, float_format="{:.2f}".format))


# ─────────────────────────────────────────────────────────────
# Main plotting driver
# ─────────────────────────────────────────────────────────────

def plot_all(results_dir="results", out_dir="figures",
             datasets=None, algorithms=None, k_values=None,
             show=False, fontsize=18):
    results_dir = Path(results_dir)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    entries = discover_results(results_dir)
    if not entries:
        print(f"No result CSVs found in {results_dir}.")
        return

    # Filter
    if datasets:
        entries = [(d, a, k, p) for d, a, k, p in entries if d in datasets]
    if algorithms:
        entries = [(d, a, k, p) for d, a, k, p in entries if a in algorithms]
    if k_values:
        entries = [(d, a, k, p) for d, a, k, p in entries if k in k_values]

    print(f"Found {len(entries)} result file(s).")

    # Load all dataframes
    all_records = []
    for ds, alg, k, path in entries:
        try:
            df = load_csv(path)
            all_records.append((ds, alg, k, df))
        except Exception as e:
            print(f"  [WARN] Could not load {path}: {e}")

    print_summary_table(all_records)

    # Group by (dataset, k)
    from collections import defaultdict
    groups = defaultdict(dict)    # (dataset, k) -> {alg: df}
    for ds, alg, k, df in all_records:
        groups[(ds, k)][alg] = df

    print(f"\nGenerating plots in {out_dir_p}/")
    for (ds, k), alg_dfs in sorted(groups.items()):
        if not alg_dfs:
            continue
        plot_cumulative(ds, k, alg_dfs, out_dir_p, show=show, fontsize=fontsize)
        plot_instantaneous(ds, k, alg_dfs, out_dir_p, show=show, fontsize=fontsize)

    print("Plotting complete.")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Plot online LRA benchmark results")
    p.add_argument("--results_dir", default="results",
                   help="Directory containing result CSVs")
    p.add_argument("--output_dir",  default="figures",
                   help="Directory to save figures")
    p.add_argument("--datasets",    nargs="+", default=None)
    p.add_argument("--algorithms",  nargs="+", default=None)
    p.add_argument("--k_values",    nargs="+", type=int, default=None)
    p.add_argument("--show",        action="store_true",
                   help="Display figures interactively instead of saving")
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
    )