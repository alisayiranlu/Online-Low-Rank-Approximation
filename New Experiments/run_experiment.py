"""
run_experiment.py
=================
Experiment runner for online low-rank approximation benchmarks.

Design principles
-----------------
- Each (dataset, algorithm, k) triple writes to one CSV; multiple trials are
  stored as separate rows distinguished by the `trial` column.
- Rows are flushed every `save_every` steps so partial runs are usable.
- Resume support: completed trials are detected from the CSV and skipped.
- Skip logic: all n_trials complete -> skip unless --force is passed.

Usage
-----
  # Run MNIST and MovieLens20M with 10 trials each (defaults)
  python run_experiment.py

  # Fewer trials, specific datasets
  python run_experiment.py --n_trials 5 --datasets MNIST

  # MovieLens path (zip or extracted folder)
  python run_experiment.py --movielens_path ml-20m.zip

  # Re-run from scratch
  python run_experiment.py --force

  # Dry run
  python run_experiment.py --dry_run
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
    "MNIST": {
        "T": 5000, "d": 50,
        "k_values": [10, 15, 20],
        "dataset_kwargs": {"d_reduced": 50},
    },
    "MovieLens20M": {
        "T": 2000, "d": 50,
        "k_values": [10, 15, 20],
        "dataset_kwargs": {"d_reduced": 50},
    },
}

NEEDS_FULL_DATA = {"OfflineOptimum"}

ALG_KWARGS = {
    "GrassmannHRD":  {"eta": 0.5, "n_min": 10, "n_max": 100, "epsilon_hrd": 0.1},
    "FantopeOGD":    {"init_steps": 50},
    "OfflineOptimum":{},
    "StreamingSVD":  {},
    "FTRL":          {"reg": 1.0},
    "SphericalHRD":  {"eta": 0.5, "n_min": 20, "n_max": 100, "epsilon_hrd": 0.1},
    "BadNet":        {},
}


# ─────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "dataset", "algorithm", "k", "d", "trial", "step",
    "instantaneous_loss", "cumulative_loss", "n_leaves",
    "elapsed_s",
]


def _csv_path(out_dir: Path, dataset: str, algorithm: str, k: int) -> Path:
    return out_dir / f"{dataset}__{algorithm}__k{k}.csv"


def _get_completed_trials(path: Path, T: int) -> set:
    """Return set of trial indices whose step sequence reaches T-1."""
    if not path.exists():
        return set()
    trial_max_step = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if "trial" not in (reader.fieldnames or []):
            return set()
        for row in reader:
            try:
                t = int(row["trial"])
                s = int(row["step"])
                if t not in trial_max_step or s > trial_max_step[t]:
                    trial_max_step[t] = s
            except (KeyError, ValueError):
                pass
    return {t for t, max_s in trial_max_step.items() if max_s >= T - 1}


# ─────────────────────────────────────────────────────────────
# Core run function
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
    n_trials: int = 10,
) -> dict:
    """
    Stream shuffled copies of X through `alg_name` for n_trials independent
    trials. Each trial shuffles X with seed=trial_idx so results are
    reproducible. Writes one CSV row per (trial, step).
    """
    csv_path = _csv_path(out_dir, dataset_name, alg_name, k)
    d = X.shape[1]
    T = len(X)

    # ── Skip / resume logic ──────────────────────────────────
    completed = set()
    if not force:
        if csv_path.exists():
            with open(csv_path, newline="") as f:
                fields = csv.DictReader(f).fieldnames or []
            if "trial" not in fields:
                print(f"    [SKIP] {dataset_name}/{alg_name}/k={k} "
                      f"-- old single-trial format; use --force to re-run")
                return {"skipped": True, "reason": "old format"}
        completed = _get_completed_trials(csv_path, T)

    trials_needed = [i for i in range(n_trials) if i not in completed]

    if not trials_needed:
        print(f"    [SKIP] {dataset_name}/{alg_name}/k={k} "
              f"-- all {n_trials} trials complete")
        return {"skipped": True}

    print(f"    Running {dataset_name} | {alg_name} | k={k} | d={d} | T={T} "
          f"| trials {trials_needed[0]}-{trials_needed[-1]} ...", flush=True)

    # ── Open CSV ─────────────────────────────────────────────
    new_file = force or not csv_path.exists()
    fh = open(csv_path, "w" if new_file else "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    if new_file:
        writer.writeheader()

    alg_kw = dict(ALG_KWARGS.get(alg_name, {}))
    if alg_name == "FantopeOGD":
        alg_kw.setdefault("T_est", T)

    t_wall_start = time.perf_counter()

    try:
        for trial_idx in trials_needed:
            rng = np.random.default_rng(trial_idx)
            X_trial = X[rng.permutation(T)]

            try:
                if alg_name in NEEDS_FULL_DATA:
                    alg = build_algorithm(alg_name, d, k, data=X_trial, **alg_kw)
                else:
                    alg = build_algorithm(alg_name, d, k, **alg_kw)
            except ImportError as e:
                print(f"    [SKIP] {alg_name}: {e}")
                fh.close()
                return {"skipped": True, "reason": str(e)}

            t_trial = time.perf_counter()
            buffer = []

            for step in range(T):
                x = X_trial[step]
                loss    = alg.step(x)
                cum     = alg.cum_loss[-1]
                n_lv    = getattr(alg, "n_leaves", 1)
                elapsed = time.perf_counter() - t_trial

                buffer.append({
                    "dataset":            dataset_name,
                    "algorithm":          alg_name,
                    "k":                  k,
                    "d":                  d,
                    "trial":              trial_idx,
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

                if (step + 1) % 1000 == 0 or step == T - 1:
                    print(f"      trial {trial_idx+1}/{n_trials}  "
                          f"step {step+1}/{T}  cum={cum:.3f}", flush=True)

            trial_t = time.perf_counter() - t_trial
            print(f"      Trial {trial_idx} done in {trial_t:.1f}s "
                  f"-- final cum_loss={alg.cum_loss[-1]:.4f}")

    finally:
        fh.close()

    total_t = time.perf_counter() - t_wall_start
    print(f"    All trials done in {total_t:.1f}s")
    return {
        "dataset":   dataset_name,
        "algorithm": alg_name,
        "k":         k,
        "elapsed_s": total_t,
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
    n_trials=10,
    movielens_path="ml-20m.zip",
):
    datasets   = datasets   or list(DEFAULT_CONFIG.keys())
    algorithms = algorithms or ALL_ALGORITHMS
    out_dir    = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for ds in datasets:
        if ds not in DEFAULT_CONFIG:
            print(f"[WARNING] Unknown dataset '{ds}' -- skipping")
            continue
        cfg    = DEFAULT_CONFIG[ds]
        k_vals = k_override if k_override else cfg["k_values"]
        for alg in algorithms:
            for k in k_vals:
                tasks.append((ds, alg, k, cfg))

    print(f"Planned {len(tasks)} run(s) x {n_trials} trial(s) each.")
    if dry_run:
        for ds, alg, k, _ in tasks:
            csv_path = _csv_path(out_dir, ds, alg, k)
            done     = _get_completed_trials(csv_path, DEFAULT_CONFIG[ds]["T"])
            status   = f"DONE({len(done)}/{n_trials})" if done else "TODO"
            print(f"  [{status}] {ds} | {alg} | k={k}")
        return

    # Load datasets (cache per name)
    loaded = {}
    for ds, alg, k, cfg in tasks:
        if ds in loaded:
            continue
        print(f"\nLoading dataset: {ds} ...")
        try:
            dk = dict(cfg.get("dataset_kwargs", {}))
            if ds == "MNIST":
                dk.setdefault("n_samples", cfg.get("T", 5000))
            elif ds == "MovieLens20M":
                dk.setdefault("n_samples", cfg.get("T", 2000))
                dk.setdefault("path", movielens_path)
            X, meta = load_dataset(ds, **dk)
            print(f"  {X.shape[0]} samples x {X.shape[1]} features")
            loaded[ds] = (X, meta)
        except Exception as e:
            print(f"  [ERROR] Could not load {ds}: {e}")
            loaded[ds] = None

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
                n_trials=n_trials,
            )
            summaries.append(summary)
        except Exception as e:
            import traceback
            print(f"  [ERROR] {e}")
            traceback.print_exc()

    completed = [s for s in summaries if not s.get("skipped")]
    if completed:
        print("\n" + "=" * 72)
        print("EXPERIMENT SUMMARY")
        print("=" * 72)
        print(f"{'Dataset':<22} {'Algorithm':<16} {'k':>3}  {'Time(s)':>8}")
        print("-" * 72)
        for s in completed:
            print(f"{s['dataset']:<22} {s['algorithm']:<16} {s['k']:>3}  "
                  f"{s['elapsed_s']:>8.1f}")

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
    p.add_argument("--datasets",       nargs="+", default=None,
                   choices=ALL_DATASETS)
    p.add_argument("--algorithms",     nargs="+", default=None,
                   choices=all_alg_choices)
    p.add_argument("--k_values",       nargs="+", type=int, default=None)
    p.add_argument("--output_dir",     default="results")
    p.add_argument("--movielens_path", default="ml-20m.zip",
                   help="Path to MovieLens 20M zip or extracted folder")
    p.add_argument("--n_trials",       type=int, default=10,
                   help="Number of random-shuffle trials to average (default: 10)")
    p.add_argument("--force",          action="store_true")
    p.add_argument("--dry_run",        action="store_true")
    p.add_argument("--save_every",     type=int, default=50)
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
        n_trials=args.n_trials,
        movielens_path=args.movielens_path,
    )
