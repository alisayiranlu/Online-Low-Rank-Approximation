"""
datasets.py
===========
Functions to load and preprocess all experimental datasets.

Every function returns:
    X : np.ndarray, shape (n_samples, d)   — unit-norm rows
    meta : dict                             — dataset metadata

Available loaders
-----------------
  load_synthetic_optimal(T, d)     — points concentrated in a 2-D subspace
  load_synthetic_clustered(T, d)   — three-cluster data on the sphere
  load_mnist(n_samples, d_reduced) — MNIST reduced via TruncatedSVD
  load_creditcard(path, n_samples) — Credit Card Fraud (PCA features V1-V28)
"""

import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# Shared helper
# ─────────────────────────────────────────────────────────────

def _normalize_rows(X, eps=1e-12):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    return X / norms


# ─────────────────────────────────────────────────────────────
# Synthetic: Optimal 2-D subspace
# ─────────────────────────────────────────────────────────────

def load_synthetic_optimal(T=1000, d=5, seed=42):
    """
    Points lie in the span(e_1, e_2) subspace embedded in R^d.
    Optimal rank-2 basis can perfectly reconstruct every point.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((T, d))
    X[:, 0] = rng.standard_normal(T)
    X[:, 1] = rng.standard_normal(T)
    X = _normalize_rows(X)
    return X, {"name": "SyntheticOptimal", "d": d, "T": T,
               "description": "Points in 2-D subspace"}


# ─────────────────────────────────────────────────────────────
# Synthetic: Clustered data
# ─────────────────────────────────────────────────────────────

def load_synthetic_clustered(T=1000, d=5, n_clusters=3, noise=0.3, seed=42):
    """
    Three cluster centres sampled uniformly on S^{d-1}.
    Each point = normalise(centre + N(0, noise^2 I)).
    """
    rng = np.random.default_rng(seed)
    centers = _normalize_rows(rng.standard_normal((n_clusters, d)))
    X = np.zeros((T, d))
    for t in range(T):
        c = centers[t % n_clusters]
        x = c + noise * rng.standard_normal(d)
        X[t] = x / max(np.linalg.norm(x), 1e-12)
    return X, {"name": "SyntheticClustered", "d": d, "T": T, "n_clusters": n_clusters,
               "description": "3-cluster spherical data"}


# ─────────────────────────────────────────────────────────────
# MNIST
# ─────────────────────────────────────────────────────────────

def load_mnist(n_samples=500, d_reduced=50, seed=42):
    """
    MNIST 784-dim → TruncatedSVD to d_reduced dims → unit-normalised rows.
    Downloads via sklearn.datasets.fetch_openml (cached after first call).
    """
    from sklearn.datasets import fetch_openml
    from sklearn.decomposition import TruncatedSVD

    print("  Loading MNIST (may download on first run)…", flush=True)
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X_raw = mnist["data"].astype(float)

    svd = TruncatedSVD(n_components=d_reduced, random_state=seed)
    X_red = svd.fit_transform(X_raw)
    X_red = _normalize_rows(X_red)

    n = min(n_samples, len(X_red))
    X = X_red[:n]
    return X, {"name": "MNIST", "d": d_reduced, "T": n,
               "description": f"MNIST TruncSVD d={d_reduced}"}


# ─────────────────────────────────────────────────────────────
# Credit Card Fraud
# ─────────────────────────────────────────────────────────────

def load_creditcard(path="creditcard.csv", n_samples=500):
    """
    Credit Card Fraud dataset (Kaggle).  Keeps V1-V28 (PCA features),
    drops Time, Amount, Class.  Rows unit-normalised.
    """
    import csv

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Credit card dataset not found at '{path}'.\n"
            "Download 'creditcard.csv' from Kaggle:\n"
            "  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "and place it in the working directory."
        )

    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Keep columns 1..28 (V1-V28), drop Time(0), Amount(-2), Class(-1)
        cols_drop = {0, len(header) - 2, len(header) - 1}
        keep = [i for i in range(len(header)) if i not in cols_drop]
        for row in reader:
            rows.append([float(row[i]) for i in keep])
            if len(rows) >= n_samples:
                break

    X = np.array(rows, dtype=float)
    X = _normalize_rows(X)
    d = X.shape[1]
    return X, {"name": "CreditCard", "d": d, "T": len(X),
               "description": "Credit Card Fraud V1-V28 (PCA features)"}


# ─────────────────────────────────────────────────────────────
# Dataset registry
# ─────────────────────────────────────────────────────────────

ALL_DATASETS = ["SyntheticOptimal", "SyntheticClustered", "MNIST", "CreditCard"]


def load_dataset(name, **kwargs):
    """
    Unified loader.  Extra kwargs are forwarded to the specific function.
    Returns (X, meta).
    """
    if name == "SyntheticOptimal":
        return load_synthetic_optimal(**kwargs)
    if name == "SyntheticClustered":
        return load_synthetic_clustered(**kwargs)
    if name == "MNIST":
        return load_mnist(**kwargs)
    if name == "CreditCard":
        return load_creditcard(**kwargs)
    raise ValueError(f"Unknown dataset: {name}.  "
                     f"Available: {ALL_DATASETS}")