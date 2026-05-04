"""
run_experiment.py
=================
Experiment runner for online low-rank approximation benchmarks.

Design principles
-----------------
- Each (dataset, algorithm, k) triple is fully independent: results are written
  to a separate CSV so a crash never loses completed work.
- Adaptive/incremental recording: rows are flushed to disk every `save_every`
  steps, so partial runs are always usable.
- Resume support: if a CSV for a run already exists, the runner detects the last
  completed step and continues from there without replaying from scratch.
- Skip logic: completely finished runs are skipped unless --force is passed.

Usage
-----
  # Run everything with defaults
  python run_experiment.py

  # Run specific datasets and algorithms
  python run_experiment.py --datasets MNIST CreditCard \\
                           --algorithms GrassmannHRD FantopeOGD OfflineOptimum \\
                           --k_values 10 15

  # Run MovieLens 20M from the Kaggle/GroupLens zip
  python run_experiment.py --datasets MovieLens20M \\
                           --movielens_path archive.zip \\
                           --algorithms GrassmannHRD FantopeOGD OfflineOptimum \\
                           --k_values 10 15 20

  # List what would run without running it
  python run_experiment.py --dry_run

  # Re-run even if CSV already exists
  python run_experiment.py --force

  # Add optional algorithms (SphericalHRD needs final_research_c.py; BadNet is fixed baseline)
  python run_experiment.py --algorithms GrassmannHRD BadNet SphericalHRD
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from algorithms import build_algorithm, ALL_ALGORITHMS, OPTIONAL_ALGORITHMS
from datasets import load_dataset, ALL_DATASETS


# ─────────────────────────────────────────────────────────────
# Default experiment grid
# ─────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "SyntheticOptimal": {
        "T": 1000, "d": 5,
        "k_values": [2, 5, 10],
        "dataset_kwargs": {},
    },
    "SyntheticClustered": {
        "T": 1000, "d": 5,
        "k_values": [2, 5, 10],
        "dataset_kwargs": {},
    },
    "MNIST": {
        "T": 500, "d": 50,
        "k_values": [10, 15, 20],
        "dataset_kwargs": {"d_reduced": 50},
    },
    "CreditCard": {
        "T": 500, "d": 28,
        "k_values": [10, 15, 20],
        "dataset_kwargs": {},
    },
    "MovieLens20M": {
        "T": 500, "d": 50,
        "k_values": [10, 15, 20],
        "dataset_kwargs": {
            "n_samples": 500,
            "n_movies": 1000,
            "d_reduced": 50,
            "min_ratings_per_user": 20,
            "center_ratings": True,
        },
    },
}

# Algorithms that need the full data array at construction time
NEEDS_FULL_DATA = {"OfflineOptimum"}

# Algorithm-specific hyperparameter overrides (on top of (d, k))
ALG_KWARGS = {
    "GrassmannHRD":  {"eta": 0.5, "n_min": 10, "n_max": 100, "epsilon_hrd": 0.1},
    "FantopeOGD":    {"init_steps": 50},
    "OfflineOptimum":{},
    "StreamingSVD":  {},
    "FTRL":          {"reg": 1.0},
    # Optional algorithms
    "SphericalHRD":  {"eta": 0.5, "n_min": 20, "n_max": 100, "epsilon_hrd": 0.1},
    "BadNet":        {},
}


# ─────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "dataset", "algorithm", "k", "d", "step",
    "instantaneous_loss", "cumulative_loss", "n_leaves",
    "elapsed_s",
]


def _csv_path(out_dir: Path, dataset: str, algorithm: str, k: int) -> Path:
    return out_dir / f"{dataset}__{algorithm}__k{k}.csv"


def _open_csv(path: Path, append: bool):
    """Open CSV for writing (or appending). Returns (file_handle, writer, start_step)."""
    if append and path.exists():
        last_step = 0
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    last_step = max(last_step, int(row["step"]))
                except (KeyError, ValueError):
                    pass
        fh = open(path, "a", newline="")
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        return fh, writer, last_step
    else:
        fh = open(path, "w", newline="")
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        return fh, writer, 0


# ─────────────────────────────────────────────────────────────
# Core single-run function
# ─────────────────────────────────────────────────────────────

def run_single(
    dataset_name: str,
    alg_name: str,
    k: int,
    X: np.ndarray,
    meta: dict,
    out_dir: Path,
    force: bool = False,
    save_every: int = 50,
    resume: bool = True,
) -> dict:
    """
    Stream X through `alg_name` with target rank k.
    Writes one CSV row per step; flushes every `save_every` rows.
    Returns a summary dict.
    """
    csv_path = _csv_path(out_dir, dataset_name, alg_name, k)
    d = X.shape[1]
    T = len(X)

    # ── Skip / resume logic ──────────────────────────────────
    start_step = 0
    if not force and csv_path.exists():
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            last_step = max(int(r["step"]) for r in rows)
            if last_step >= T - 1:
                print(f"    [SKIP] {dataset_name}/{alg_name}/k={k} — already complete")
                return {"skipped": True}
            if resume:
                start_step = last_step + 1
                print(f"    [RESUME] {dataset_name}/{alg_name}/k={k} "
                      f"from step {start_step}")

    print(f"    Running {dataset_name} | {alg_name} | k={k} | d={d} | T={T} …",
          flush=True)

    # ── Build algorithm ──────────────────────────────────────
    alg_kw = dict(ALG_KWARGS.get(alg_name, {}))
    if alg_name == "FantopeOGD":
        alg_kw.setdefault("T_est", T)

    try:
        if alg_name in NEEDS_FULL_DATA:
            alg = build_algorithm(alg_name, d, k, data=X, **alg_kw)
        else:
            alg = build_algorithm(alg_name, d, k, **alg_kw)
    except ImportError as e:
        print(f"    [SKIP] {alg_name}: {e}")
        return {"skipped": True, "reason": str(e)}

    # Fast-forward if resuming (replay without saving)
    if start_step > 0:
        print(f"      Fast-forwarding {start_step} steps …", end=" ", flush=True)
        for i in range(start_step):
            alg.step(X[i])
        print("done", flush=True)

    # ── Open CSV ─────────────────────────────────────────────
    fh, writer, _ = _open_csv(csv_path, append=(start_step > 0))
    t_start = time.perf_counter()
    buffer = []

    try:
        for step in range(start_step, T):
            x = X[step]
            loss    = alg.step(x)
            cum     = alg.cum_loss[-1]
            n_lv    = getattr(alg, "n_leaves", 1)
            elapsed = time.perf_counter() - t_start

            buffer.append({
                "dataset":            dataset_name,
                "algorithm":          alg_name,
                "k":                  k,
                "d":                  d,
                "step":               step,
                "instantaneous_loss": f"{loss:.6f}",
                "cumulative_loss":    f"{cum:.6f}",
                "n_leaves":           n_lv,
                "elapsed_s":          f"{elapsed:.3f}",
            })

            if len(buffer) >= save_every or step == T - 1:
                writer.writerows(buffer)
                fh.flush()
                buffer.clear()

            if (step + 1) % 100 == 0 or step == T - 1:
                print(f"      step {step+1}/{T}  cum_loss={cum:.3f}", flush=True)

    finally:
        fh.close()

    total_t = time.perf_counter() - t_start
    final_cum = alg.cum_loss[-1]
    print(f"      Done in {total_t:.1f}s — final cum_loss={final_cum:.4f}")
    return {
        "dataset":        dataset_name,
        "algorithm":      alg_name,
        "k":              k,
        "final_cum_loss": final_cum,
        "elapsed_s":      total_t,
    }


# ─────────────────────────────────────────────────────────────
# Top-level orchestrator
# ─────────────────────────────────────────────────────────────

def run_experiments(
    datasets=None,
    algorithms=None,
    k_override=None,
    output_dir="results",
    force=False,
    dry_run=False,
    save_every=50,
    creditcard_path="creditcard.csv",
    movielens_path="ml-20m.zip",
    resume=True,
):
    """
    Run the full benchmark grid (or any subset).

    Parameters
    ----------
    datasets        : list of dataset names (default: all in ALL_DATASETS)
    algorithms      : list of algorithm names (default: all in ALL_ALGORITHMS)
    k_override      : override k values for all datasets
    output_dir      : directory for CSV results
    force           : re-run even if a completed CSV exists
    dry_run         : just print planned runs, do not execute
    save_every      : flush CSV to disk every N steps
    creditcard_path : path to creditcard.csv (Kaggle)
    movielens_path  : path to MovieLens 20M zip/folder/rating.csv
    resume          : continue partial runs from last completed step
    """
    datasets   = datasets   or ALL_DATASETS
    algorithms = algorithms or ALL_ALGORITHMS
    out_dir    = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build task list
    tasks = []
    for ds in datasets:
        if ds not in DEFAULT_CONFIG:
            print(f"[WARNING] Unknown dataset '{ds}' — skipping")
            continue
        cfg    = DEFAULT_CONFIG[ds]
        k_vals = k_override if k_override else cfg["k_values"]
        for alg in algorithms:
            for k in k_vals:
                tasks.append((ds, alg, k, cfg))

    print(f"Planned {len(tasks)} run(s).")
    if dry_run:
        for ds, alg, k, _ in tasks:
            csv_path  = _csv_path(out_dir, ds, alg, k)
            status    = "DONE" if csv_path.exists() else "TODO"
            print(f"  [{status}] {ds} | {alg} | k={k}")
        return

    # Load datasets (cache per name)
    loaded = {}
    for ds, alg, k, cfg in tasks:
        if ds in loaded:
            continue
        print(f"\nLoading dataset: {ds} …")
        try:
            dk = dict(cfg.get("dataset_kwargs", {}))
            if ds == "CreditCard":
                dk.setdefault("path", creditcard_path)
                dk.setdefault("n_samples", cfg.get("T", 500))
            elif ds == "MovieLens20M":
                dk.setdefault("path", movielens_path)
                dk.setdefault("n_samples", cfg.get("T", 500))
            elif ds == "MNIST":
                dk.setdefault("n_samples", cfg.get("T", 500))
            elif ds in ("SyntheticOptimal", "SyntheticClustered"):
                dk.setdefault("T", cfg.get("T", 1000))
                dk.setdefault("d", cfg.get("d", 5))
            X, meta = load_dataset(ds, **dk)
            print(f"  {X.shape[0]} samples x {X.shape[1]} features")
            loaded[ds] = (X, meta)
        except Exception as e:
            print(f"  [ERROR] Could not load {ds}: {e}")
            loaded[ds] = None

    # Run each task
    summaries = []
    for i, (ds, alg, k, cfg) in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] {ds} | {alg} | k={k}")
        if loaded.get(ds) is None:
            print("  [SKIP] dataset not available")
            continue
        X, meta = loaded[ds]
        try:
            summary = run_single(
                dataset_name=ds,
                alg_name=alg,
                k=k,
                X=X,
                meta=meta,
                out_dir=out_dir,
                force=force,
                save_every=save_every,
                resume=resume,
            )
            summaries.append(summary)
        except Exception as e:
            import traceback
            print(f"  [ERROR] {e}")
            traceback.print_exc()

    # Print final summary table
    completed = [s for s in summaries if not s.get("skipped")]
    if completed:
        print("\n" + "=" * 72)
        print("EXPERIMENT SUMMARY")
        print("=" * 72)
        print(f"{'Dataset':<22} {'Algorithm':<16} {'k':>3}  {'FinalLoss':>12}  {'Time(s)':>8}")
        print("-" * 72)
        for s in completed:
            print(f"{s['dataset']:<22} {s['algorithm']:<16} {s['k']:>3}  "
                  f"{s['final_cum_loss']:>12.4f}  {s['elapsed_s']:>8.1f}")

    return summaries


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def _parse_args():
    all_alg_choices = ALL_ALGORITHMS + OPTIONAL_ALGORITHMS
    p = argparse.ArgumentParser(
        description="Online Low-Rank Approximation Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--datasets",   nargs="+", default=None,
                   choices=ALL_DATASETS,
                   help=f"Datasets to run (default: all). Choices: {ALL_DATASETS}")
    p.add_argument("--algorithms", nargs="+", default=None,
                   choices=all_alg_choices,
                   help=(f"Algorithms to run (default: {ALL_ALGORITHMS}). "
                         f"Optional extras: {OPTIONAL_ALGORITHMS}"))
    p.add_argument("--k_values",   nargs="+", type=int, default=None,
                   help="Override k values for all datasets")
    p.add_argument("--output_dir", default="results",
                   help="Directory for CSV results (default: results/)")
    p.add_argument("--creditcard_path", default="creditcard.csv",
                   help="Path to creditcard.csv (download from Kaggle)")
    p.add_argument("--movielens_path", default="ml-20m.zip",
                   help="Path to MovieLens 20M zip, extracted folder, or rating.csv/ratings.csv")
    p.add_argument("--force",      action="store_true",
                   help="Re-run even if a completed CSV already exists")
    p.add_argument("--no_resume",  action="store_true",
                   help="Do not resume partial runs (restart from scratch)")
    p.add_argument("--dry_run",    action="store_true",
                   help="Print planned tasks without running them")
    p.add_argument("--save_every", type=int, default=50,
                   help="Flush CSV to disk every N steps (default: 50)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_experiments(
        datasets=args.datasets,
        algorithms=args.algorithms,
        k_override=args.k_values,
        output_dir=args.output_dir,
        force=args.force,
        dry_run=args.dry_run,
        save_every=args.save_every,
        creditcard_path=args.creditcard_path,
        movielens_path=args.movielens_path,
        resume=not args.no_resume,
    )