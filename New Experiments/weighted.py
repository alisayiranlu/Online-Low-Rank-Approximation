"""
weighted_experiment.py
=======================
Small-scale *synthetic weighted* experiments for online weighted low-rank
approximation (WLRA).

Generative model
-----------------
A fixed true rank-k subspace U* in R^d generates clean signals
    s_t = U* a_t ,           a_t ~ N(0, I_k) .
Each coordinate i is then corrupted by *heteroscedastic* Gaussian noise
    x_t = s_t + n_t ,        n_{t,i} ~ N(0, sigma_i^2) ,
where a "reliable" group of coordinates has small noise sigma_lo and an
"unreliable" group has large noise sigma_hi.  The heterogeneity ratio is
    rho = sigma_hi / sigma_lo .

The statistically-correct confidence weights are the inverse noise variances,
normalised so the largest weight is 1:
    w_i = sigma_min^2 / sigma_i^2   in (0, 1] .

With these weights the *population* weighted optimum is exactly U* (inverse-
variance / GLS weighting is the ML estimator for a diagonal-noise factor
model).  The *unweighted* top-k eigenvectors of E[x x^T], however, are pulled
toward the high-variance noise coordinates whenever sigma_hi is large.  So the
weighted and unweighted comparators genuinely differ, and rho controls how
much.  At rho = 1 the weights are uniform and WLRA reduces exactly to PCA.

A missing-data variant (canonical WLRA setting) is also included: unreliable
coordinates are frequently unobserved, missing entries are zero-filled, and
the observation mask is used as the 0/1 weight.

What is compared
----------------
All methods are evaluated on the *same* true weighted objective
    l_t(U) = || W_t^{1/2} (I - P_U) x_t ||^2 ,
regardless of what they internally optimise, together with two method-agnostic
recovery metrics:
  * subspace error   d_G(U_t, U*)      (chordal Grassmannian distance to truth)
  * clean-signal loss || (I - P_{U_t}) s_hat_t ||^2   (fit to the noise-free signal)

Algorithms:
  * Fantope OGD (Weighted)    -- paper's poly-time WLRA algorithm, true weights
  * Fantope OGD (Unweighted)  -- identical algorithm with W_t = I (control)
  * Streaming SVD             -- standard unweighted online PCA baseline
  * True subspace U*          -- oracle reference (subspace error = 0)

Only the anchor reduction is switched off here (m = d) so the comparison
isolates the effect of weighting in the online update rather than anchor
estimation; d is small so this costs nothing.

Outputs (written to weighted_results/ and weighted_figures/)
------------------------------------------------------------
  W1  streaming comparison at fixed rho          -> per-step CSV + 3-panel figure
  W2  heterogeneity sweep over rho               -> summary CSV + sweep figure
  W3  missing-data streaming comparison          -> per-step CSV + 3-panel figure
  a plain-text summary table printed to stdout

Run:
    python weighted_experiment.py
    python weighted_experiment.py --quick     # fewer trials / shorter stream
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algorithms import FantopeOGDAlg, StreamingSVDAlg, _grass_dist


# ─────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────

def weighted_proj_loss(U, x, w):
    """True WLRA loss  || W^{1/2} (I - P_U) x ||^2  for orthonormal d×k basis U."""
    if U is None:
        return float(np.sum(w * x * x))
    r = x - U @ (U.T @ x)
    return float(np.sum(w * r * r))


def clean_proj_loss(U, s):
    """Reconstruction loss of the *clean* signal:  || (I - P_U) s ||^2 ."""
    if U is None:
        return float(np.dot(s, s))
    r = s - U @ (U.T @ s)
    return float(np.dot(r, r))


def subspace_error(U, U_true):
    """Chordal Grassmannian distance d_G(U, U*) in [0, sqrt(k)] (0 = exact)."""
    if U is None:
        return np.nan
    return _grass_dist(U, U_true)


# ─────────────────────────────────────────────────────────────
# Synthetic weighted data generators
# ─────────────────────────────────────────────────────────────

def _random_orthonormal(d, k, rng):
    A = rng.standard_normal((d, k))
    Q, _ = np.linalg.qr(A)
    return Q[:, :k]


def _confined_subspace(d, k, n_rel, seed=12345):
    """
    Random rank-k orthonormal subspace whose support is the first `n_rel`
    (reliable) coordinates; rows in the unreliable block are exactly zero.
    """
    rng = np.random.default_rng(seed)
    Q = _random_orthonormal(n_rel, k, rng)   # k orthonormal cols in R^{n_rel}
    U = np.zeros((d, k))
    U[:n_rel, :] = Q
    return U


def make_heteroscedastic_stream(T, d, k, rho, sigma_lo=0.10,
                                rel_frac=0.5, U_true=None, seed=0):
    """
    Heteroscedastic weighted stream (heterogeneous sensor reliability).

    The rank-k signal is carried by a block of `reliable` coordinates with
    small noise sigma_lo; the remaining `unreliable` coordinates carry NO
    signal and only high-variance noise sigma_hi = rho * sigma_lo.  This is
    the classic regime in which unweighted PCA is fooled by the high-variance
    noise axes, while inverse-variance weighting (the correct confidence
    weights) suppresses them.  At rho = 1 the noise is homogeneous, the
    weights are uniform, and WLRA reduces exactly to PCA.

    Returns dict with:
      X       (T,d)  unit-normalised observations
      W       (T,d)  per-coordinate inverse-variance weights in (0,1]
      S       (T,d)  clean signal in the SAME normalised coordinates as X
      U_true  (d,k)  true generating subspace (supported on reliable block)
      w_row   (d,)   the (time-invariant) weight vector
    """
    rng = np.random.default_rng(seed)
    n_rel = max(k, int(round(rel_frac * d)))
    if U_true is None:
        U_true = _confined_subspace(d, k, n_rel)

    sigma = np.full(d, sigma_lo * rho)      # unreliable coords: high noise
    sigma[:n_rel] = sigma_lo                # reliable coords: low noise
    # Inverse-variance weights, normalised so max weight = 1.
    inv_var = 1.0 / (sigma ** 2)
    w_row = inv_var / inv_var.max()

    A = rng.standard_normal((T, k))
    S_raw = A @ U_true.T                     # signal, supported on reliable block
    N = rng.standard_normal((T, d)) * sigma[None, :]
    X_raw = S_raw + N

    norms = np.linalg.norm(X_raw, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    X = X_raw / norms
    S = S_raw / norms                        # clean signal in normalised frame
    W = np.tile(w_row, (T, 1))

    return {"X": X, "W": W, "S": S, "U_true": U_true, "w_row": w_row,
            "sigma": sigma}


def make_missing_stream(T, d, k, p_rel=0.95, p_unrel=0.2, sigma=0.05,
                        rel_frac=0.5, w_missing=1e-3, U_true=None, seed=0):
    """
    Missing-data weighted stream (canonical WLRA / matrix-completion setting).

    Reliable coordinates are observed w.p. p_rel, unreliable ones w.p. p_unrel.
    Missing entries are zero-filled; the 0/1 observation mask is used as the
    weight (missing coords get a tiny weight w_missing so W is well conditioned).
    """
    rng = np.random.default_rng(seed)
    if U_true is None:
        U_true = _random_orthonormal(d, k, np.random.default_rng(12345))

    n_rel = max(1, int(round(rel_frac * d)))
    p = np.full(d, p_unrel)
    p[:n_rel] = p_rel

    A = rng.standard_normal((T, k))
    S_full = A @ U_true.T + sigma * rng.standard_normal((T, d))

    mask = (rng.random((T, d)) < p[None, :]).astype(float)   # 1 = observed
    X_obs = S_full * mask                                     # zero-filled

    # Normalise by the observed-part norm (what an online method actually sees).
    norms = np.linalg.norm(X_obs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    X = X_obs / norms
    S = S_full / norms                       # clean full signal, same frame
    W = np.where(mask > 0, 1.0, w_missing)

    return {"X": X, "W": W, "S": S, "U_true": U_true, "mask": mask, "p": p}


# ─────────────────────────────────────────────────────────────
# Streaming drivers
# ─────────────────────────────────────────────────────────────

def _played_basis(alg):
    """Orthonormal d×k basis the algorithm is *currently playing* (pre-update)."""
    if isinstance(alg, FantopeOGDAlg):
        if not alg._initialized:
            return None
        return alg.get_basis()
    if isinstance(alg, StreamingSVDAlg):
        return alg.basis
    raise TypeError(type(alg))


def run_stream(alg, data, use_weights):
    """
    Stream `data` through `alg`, recording the three metrics at every step,
    evaluated on the basis the algorithm plays *before* its update.

    Returns dict of per-step arrays (length T): 'wloss', 'suberr', 'cleanloss'.
    """
    X, W, S, U_true = data["X"], data["W"], data["S"], data["U_true"]
    T = X.shape[0]
    wloss = np.full(T, np.nan)
    suberr = np.full(T, np.nan)
    cleanloss = np.full(T, np.nan)

    for t in range(T):
        x, w, s = X[t], W[t], S[t]
        U = _played_basis(alg)
        wloss[t] = weighted_proj_loss(U, x, w)
        suberr[t] = subspace_error(U, U_true)
        cleanloss[t] = clean_proj_loss(U, s)

        if isinstance(alg, FantopeOGDAlg):
            alg.step(x, W_diag=(w if use_weights else np.ones_like(w)))
        else:
            alg.step(x)          # StreamingSVD: unweighted online PCA
    return {"wloss": wloss, "suberr": suberr, "cleanloss": cleanloss}


def _make_alg(name, d, k, T):
    if name == "Fantope-Weighted" or name == "Fantope-Unweighted":
        # m = d  -> full anchor, isolates the weighting effect (no anchor loss).
        return FantopeOGDAlg(d, k, m=d, init_steps=50, T_est=T)
    if name == "StreamingSVD":
        return StreamingSVDAlg(d, k)
    raise ValueError(name)


STREAM_ALGS = ["Fantope-Weighted", "Fantope-Unweighted", "StreamingSVD"]


def cumulative(a):
    """Cumulative sum that treats leading NaNs (warmup) as 0 contribution."""
    return np.nancumsum(a)


# ─────────────────────────────────────────────────────────────
# W1 / W3 : per-step streaming comparison
# ─────────────────────────────────────────────────────────────

def run_streaming_comparison(make_stream, stream_kwargs, T, d, k,
                             n_trials, tag):
    """
    Average the three per-step metrics over n_trials for every algorithm.
    Returns nested dict: results[alg][metric] = (mean(T,), std(T,)).
    """
    per_alg = {a: {"wloss": [], "suberr": [], "cleanloss": [],
                   "cum_wloss": [], "cum_cleanloss": []}
               for a in STREAM_ALGS}
    per_alg["TrueSubspace"] = {"wloss": [], "suberr": [], "cleanloss": [],
                               "cum_wloss": [], "cum_cleanloss": []}

    for trial in range(n_trials):
        data = make_stream(T=T, d=d, k=k, seed=trial, **stream_kwargs)
        U_true = data["U_true"]

        # Oracle: true subspace, fixed.
        Xt, Wt, St = data["X"], data["W"], data["S"]
        orc_w = np.array([weighted_proj_loss(U_true, Xt[t], Wt[t]) for t in range(T)])
        orc_c = np.array([clean_proj_loss(U_true, St[t]) for t in range(T)])
        per_alg["TrueSubspace"]["wloss"].append(orc_w)
        per_alg["TrueSubspace"]["suberr"].append(np.zeros(T))
        per_alg["TrueSubspace"]["cleanloss"].append(orc_c)
        per_alg["TrueSubspace"]["cum_wloss"].append(np.cumsum(orc_w))
        per_alg["TrueSubspace"]["cum_cleanloss"].append(np.cumsum(orc_c))

        for alg_name in STREAM_ALGS:
            alg = _make_alg(alg_name, d, k, T)
            use_w = (alg_name == "Fantope-Weighted")
            m = run_stream(alg, data, use_weights=use_w)
            per_alg[alg_name]["wloss"].append(m["wloss"])
            per_alg[alg_name]["suberr"].append(m["suberr"])
            per_alg[alg_name]["cleanloss"].append(m["cleanloss"])
            per_alg[alg_name]["cum_wloss"].append(cumulative(m["wloss"]))
            per_alg[alg_name]["cum_cleanloss"].append(cumulative(m["cleanloss"]))

    results = {}
    for alg_name, md in per_alg.items():
        results[alg_name] = {}
        for metric, runs in md.items():
            arr = np.vstack(runs)
            results[alg_name][metric] = (np.nanmean(arr, axis=0),
                                         np.nanstd(arr, axis=0))
    return results


# ─────────────────────────────────────────────────────────────
# W2 : heterogeneity sweep
# ─────────────────────────────────────────────────────────────

def run_heterogeneity_sweep(rho_values, T, d, k, n_trials, sigma_lo=0.05,
                            rel_frac=0.5):
    """
    For each rho, measure the *final* subspace error and mean clean-signal loss
    of the weighted vs unweighted Fantope method (and the PCA baseline).

    Returns dict keyed by algorithm -> dict of arrays over rho:
        'suberr_mean','suberr_std','clean_mean','clean_std'.
    """
    algs = ["Fantope-Weighted", "Fantope-Unweighted", "StreamingSVD"]
    out = {a: {"suberr": [], "suberr_std": [], "clean": [], "clean_std": []}
           for a in algs}

    for rho in rho_values:
        trial_sub = {a: [] for a in algs}
        trial_cln = {a: [] for a in algs}
        for trial in range(n_trials):
            data = make_heteroscedastic_stream(
                T=T, d=d, k=k, rho=rho, sigma_lo=sigma_lo,
                rel_frac=rel_frac, seed=trial)
            for alg_name in algs:
                alg = _make_alg(alg_name, d, k, T)
                use_w = (alg_name == "Fantope-Weighted")
                m = run_stream(alg, data, use_weights=use_w)
                # Final-quality: average over the last 20% of the stream.
                tail = slice(int(0.8 * T), T)
                trial_sub[alg_name].append(np.nanmean(m["suberr"][tail]))
                trial_cln[alg_name].append(np.nanmean(m["cleanloss"][tail]))
        for alg_name in algs:
            out[alg_name]["suberr"].append(np.mean(trial_sub[alg_name]))
            out[alg_name]["suberr_std"].append(np.std(trial_sub[alg_name]))
            out[alg_name]["clean"].append(np.mean(trial_cln[alg_name]))
            out[alg_name]["clean_std"].append(np.std(trial_cln[alg_name]))
    for alg_name in algs:
        for key in out[alg_name]:
            out[alg_name][key] = np.array(out[alg_name][key])
    return out


# ─────────────────────────────────────────────────────────────
# Plotting  (matches the paper's plot_results.py style)
# ─────────────────────────────────────────────────────────────

STYLE = {
    "Fantope-Weighted":   {"color": "#9467bd", "lw": 2.6, "ls": "-",  "label": "Fantope OGD (Weighted)"},
    "Fantope-Unweighted": {"color": "#2ca02c", "lw": 2.4, "ls": "--", "label": "Fantope OGD (Unweighted)"},
    "StreamingSVD":       {"color": "#ff7f0e", "lw": 2.2, "ls": "-.", "label": "Online PCA (Streaming SVD)"},
    "TrueSubspace":       {"color": "#d62728", "lw": 1.8, "ls": ":",  "label": "True subspace (oracle)"},
}
PLOT_ORDER = ["Fantope-Weighted", "Fantope-Unweighted", "StreamingSVD", "TrueSubspace"]


def _rolling(a, w=40):
    """Causal rolling mean, NaN-aware (for the OGD warm-up region)."""
    a = np.asarray(a, float)
    out = np.full_like(a, np.nan)
    for i in range(len(a)):
        lo = max(0, i - w + 1)
        seg = a[lo:i + 1]
        seg = seg[~np.isnan(seg)]
        if seg.size:
            out[i] = seg.mean()
    return out


def _plot_metric(ax, results, metric, ylabel, title, fontsize=15,
                 include=None, bands=True, roll=40):
    include = include or PLOT_ORDER
    for alg in PLOT_ORDER:
        if alg not in results or alg not in include:
            continue
        if metric not in results[alg]:
            continue
        mean, std = results[alg][metric]
        mean = _rolling(mean, roll)
        std = _rolling(std, roll)
        x = np.arange(len(mean))
        s = STYLE[alg]
        ax.plot(x, mean, color=s["color"], lw=s["lw"], ls=s["ls"], label=s["label"])
        if bands:
            ax.fill_between(x, mean - std, mean + std, color=s["color"],
                            alpha=0.15, linewidth=0)
    ax.set_xlabel("Time step", fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=fontsize - 3)


def plot_streaming(results, out_path, suptitle):
    """Steady-state (rolling instantaneous) view of the three metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
    _plot_metric(axes[0], results, "suberr",
                 r"$d_G(U_t,\,U^\star)$", "Subspace recovery error",
                 include=["Fantope-Weighted", "Fantope-Unweighted", "StreamingSVD"])
    _plot_metric(axes[1], results, "cleanloss",
                 "Clean-signal loss (rolling)", "Clean-signal reconstruction")
    _plot_metric(axes[2], results, "wloss",
                 "Weighted loss (rolling)", "Weighted objective")
    axes[0].legend(fontsize=11, loc="upper right")
    fig.suptitle(suptitle, fontsize=17, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_sweep(sweep, rho_values, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    algs = ["Fantope-Weighted", "Fantope-Unweighted", "StreamingSVD"]

    for alg in algs:
        s = STYLE[alg]
        axes[0].errorbar(rho_values, sweep[alg]["suberr"],
                         yerr=sweep[alg]["suberr_std"], color=s["color"],
                         lw=s["lw"], ls=s["ls"], marker="o", ms=5,
                         capsize=3, label=s["label"])
        axes[1].errorbar(rho_values, sweep[alg]["clean"],
                         yerr=sweep[alg]["clean_std"], color=s["color"],
                         lw=s["lw"], ls=s["ls"], marker="o", ms=5,
                         capsize=3, label=s["label"])

    axes[0].axvline(1.0, color="gray", ls=":", lw=1.2)
    axes[0].annotate("uniform weights\n(WLRA = PCA)", xy=(1.0, axes[0].get_ylim()[1]),
                     xytext=(1.15, 0.82), textcoords="axes fraction",
                     fontsize=10, color="gray")
    axes[0].set_xlabel(r"Noise heterogeneity  $\rho=\sigma_{\rm hi}/\sigma_{\rm lo}$", fontsize=15)
    axes[0].set_ylabel(r"Final subspace error $d_G(U_t,U^\star)$", fontsize=15)
    axes[0].set_title("Subspace recovery vs. heterogeneity", fontsize=15)
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=11)

    axes[1].axvline(1.0, color="gray", ls=":", lw=1.2)
    axes[1].set_xlabel(r"Noise heterogeneity  $\rho=\sigma_{\rm hi}/\sigma_{\rm lo}$", fontsize=15)
    axes[1].set_ylabel("Final clean-signal loss (per step)", fontsize=15)
    axes[1].set_title("Clean reconstruction vs. heterogeneity", fontsize=15)
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=11)

    fig.suptitle("Weighting helps more as coordinate importance becomes heterogeneous",
                 fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
# CSV writers
# ─────────────────────────────────────────────────────────────

def save_streaming_csv(results, path, every=10):
    T = len(next(iter(results.values()))["suberr"][0])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "step", "cum_wloss_mean",
                    "suberr_mean", "suberr_std",
                    "cum_cleanloss_mean"])
        for alg, md in results.items():
            cw = md["cum_wloss"][0]
            se_m, se_s = md["suberr"]
            cc = md["cum_cleanloss"][0]
            for t in range(0, T, every):
                w.writerow([alg, t, f"{cw[t]:.6f}", f"{se_m[t]:.6f}",
                            f"{se_s[t]:.6f}", f"{cc[t]:.6f}"])
    print(f"  Saved: {path}")


def save_sweep_csv(sweep, rho_values, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "rho", "final_suberr_mean", "final_suberr_std",
                    "final_cleanloss_mean", "final_cleanloss_std"])
        for alg, md in sweep.items():
            for i, rho in enumerate(rho_values):
                w.writerow([alg, rho, f"{md['suberr'][i]:.6f}",
                            f"{md['suberr_std'][i]:.6f}",
                            f"{md['clean'][i]:.6f}", f"{md['clean_std'][i]:.6f}"])
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────
# Summary printing
# ─────────────────────────────────────────────────────────────

def print_final_table(name, results):
    """Steady-state quality: metrics averaged over the final 20% of the stream."""
    print(f"\n  Steady-state metrics [{name}] (mean over trials, last 20% of stream):")
    print(f"  {'Algorithm':<28}{'weighted loss':>15}{'subspace err':>15}{'clean loss':>13}")
    print("  " + "-" * 71)
    for alg in PLOT_ORDER:
        if alg not in results:
            continue
        wl = results[alg]["wloss"][0]
        se = results[alg]["suberr"][0]
        cl = results[alg]["cleanloss"][0]
        tail = slice(int(0.8 * len(se)), len(se))
        print(f"  {STYLE[alg]['label']:<28}"
              f"{np.nanmean(wl[tail]):>15.4f}"
              f"{np.nanmean(se[tail]):>15.4f}"
              f"{np.nanmean(cl[tail]):>13.4f}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d", type=int, default=20)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--T", type=int, default=2000)
    ap.add_argument("--n_trials", type=int, default=10)
    ap.add_argument("--rho", type=float, default=8.0,
                    help="heterogeneity ratio for the fixed streaming runs W1")
    ap.add_argument("--results_dir", default="weighted_results")
    ap.add_argument("--figures_dir", default="weighted_figures")
    ap.add_argument("--quick", action="store_true",
                    help="fast smoke run (T=600, 3 trials, coarse sweep)")
    ap.add_argument("--only", nargs="+", default=["W1", "W2", "W3"],
                    choices=["W1", "W2", "W3"],
                    help="run only a subset of the experiments")
    args = ap.parse_args()

    if args.quick:
        args.T, args.n_trials = 600, 3

    d, k, T, n_trials = args.d, args.k, args.T, args.n_trials
    res_dir = Path(args.results_dir); res_dir.mkdir(exist_ok=True)
    fig_dir = Path(args.figures_dir); fig_dir.mkdir(exist_ok=True)

    print("=" * 72)
    print("SYNTHETIC WEIGHTED WLRA EXPERIMENTS")
    print(f"d={d}  k={k}  T={T}  trials={n_trials}")
    print("=" * 72)
    t0 = time.perf_counter()

    # ── W1: heteroscedastic streaming comparison ────────────────
    if "W1" in args.only:
        print(f"\n[W1] Heteroscedastic stream (rho = {args.rho}) ...")
        res_w1 = run_streaming_comparison(
            make_heteroscedastic_stream,
            {"rho": args.rho, "sigma_lo": 0.10, "rel_frac": 0.5},
            T, d, k, n_trials, tag="hetero")
        save_streaming_csv(res_w1, res_dir / "W1_heteroscedastic.csv")
        plot_streaming(res_w1, fig_dir / "W1_heteroscedastic.png",
                       f"Heteroscedastic weighted stream  (d={d}, k={k}, "
                       rf"$\rho$={args.rho}, {n_trials} trials)")
        print_final_table("W1 heteroscedastic", res_w1)

    # ── W2: heterogeneity sweep ────────────────────────────────
    if "W2" in args.only:
        rho_values = ([1.0, 2.0, 4.0, 8.0] if args.quick
                      else [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
        print(f"\n[W2] Heterogeneity sweep over rho = {rho_values} ...")
        sweep = run_heterogeneity_sweep(rho_values, T, d, k, n_trials)
        save_sweep_csv(sweep, rho_values, res_dir / "W2_sweep.csv")
        plot_sweep(sweep, rho_values, fig_dir / "W2_sweep.png")
        print("\n  Sweep (final subspace error d_G(U_t,U*)):")
        print(f"  {'rho':>6} | {'Weighted':>10} {'Unweighted':>12} {'OnlinePCA':>11}")
        for i, rho in enumerate(rho_values):
            print(f"  {rho:>6.1f} | {sweep['Fantope-Weighted']['suberr'][i]:>10.4f} "
                  f"{sweep['Fantope-Unweighted']['suberr'][i]:>12.4f} "
                  f"{sweep['StreamingSVD']['suberr'][i]:>11.4f}")

    # ── W3: missing-data streaming comparison ──────────────────
    if "W3" in args.only:
        print(f"\n[W3] Missing-data stream (p_rel=0.95, p_unrel=0.2) ...")
        res_w3 = run_streaming_comparison(
            make_missing_stream,
            {"p_rel": 0.95, "p_unrel": 0.2, "sigma": 0.05, "rel_frac": 0.5},
            T, d, k, n_trials, tag="missing")
        save_streaming_csv(res_w3, res_dir / "W3_missing.csv")
        plot_streaming(res_w3, fig_dir / "W3_missing.png",
                       f"Missing-data weighted stream  (d={d}, k={k}, "
                       f"{n_trials} trials)")
        print_final_table("W3 missing-data", res_w3)

    dt = time.perf_counter() - t0
    print("\n" + "=" * 72)
    print(f"Done in {dt:.1f}s.  Results -> {res_dir}/   Figures -> {fig_dir}/")
    print("=" * 72)


if __name__ == "__main__":
    main()