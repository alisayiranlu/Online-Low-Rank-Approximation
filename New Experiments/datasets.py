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
  load_movielens_20m(path, n_samples) — MovieLens 20M user-rating rows
"""

import numpy as np
from pathlib import Path
from contextlib import contextmanager
import zipfile


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
# MovieLens 20M
# ─────────────────────────────────────────────────────────────

def _find_movielens_ratings_source(path):
    """
    Resolve MovieLens ratings source.

    Accepts:
      - a Kaggle/GroupLens zip file, e.g. archive.zip or ml-20m.zip
      - an extracted directory containing rating.csv or ratings.csv
      - a direct path to rating.csv/ratings.csv

    Kaggle's MovieLens 20M commonly uses rating.csv, while the official
    GroupLens ml-20m.zip uses ratings.csv.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"MovieLens ratings file not found at '{path}'.\n"
            "Pass --movielens_path pointing to your Kaggle zip, extracted folder, "
            "or rating.csv/ratings.csv."
        )

    if path.is_dir():
        candidates = []
        for name in ("rating.csv", "ratings.csv"):
            candidates.extend(path.rglob(name))
        if not candidates:
            raise FileNotFoundError(
                f"Could not find rating.csv or ratings.csv under '{path}'."
            )
        return ("file", candidates[0], None)

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            candidates = [
                n for n in names
                if n.lower().endswith(("/rating.csv", "/ratings.csv"))
                or n.lower() in ("rating.csv", "ratings.csv")
            ]
        if not candidates:
            raise FileNotFoundError(
                f"Could not find rating.csv or ratings.csv inside '{path}'."
            )
        # Prefer ratings/rating files over similarly named metadata.
        candidates = sorted(candidates, key=lambda n: (len(Path(n).name), n))
        return ("zip", path, candidates[0])

    return ("file", path, None)


@contextmanager
def _open_movielens_ratings(path):
    kind, src, member = _find_movielens_ratings_source(path)
    if kind == "zip":
        with zipfile.ZipFile(src) as zf:
            with zf.open(member) as f:
                yield f
    else:
        with open(src, "rb") as f:
            yield f


def _movielens_chunks(path, chunksize=1_000_000):
    """Yield pandas chunks with userId, movieId, rating columns."""
    import pandas as pd

    with _open_movielens_ratings(path) as f:
        for chunk in pd.read_csv(
            f,
            usecols=["userId", "movieId", "rating"],
            dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
            chunksize=chunksize,
        ):
            yield chunk


def load_movielens_20m(
    path="ml-20m.zip",
    n_samples=500,
    n_movies=1000,
    d_reduced=50,
    min_ratings_per_user=20,
    center_ratings=True,
    rating_center=3.0,
    seed=42,
    chunksize=1_000_000,
):
    """
    MovieLens 20M → dense user-by-movie matrix → optional TruncatedSVD → unit rows.

    This loader is intentionally a *subsampled* MovieLens experiment.  The full
    MovieLens 20M matrix is far too large and sparse for the dense algorithms in
    this codebase, so we:

      1. keep the n_movies most frequently rated movies,
      2. keep the n_samples users with the most ratings among those movies,
      3. build a dense user × movie matrix with missing entries set to zero,
      4. optionally center observed ratings by rating_center, so zero roughly
         means "unobserved / neutral", and
      5. reduce to d_reduced dimensions using TruncatedSVD.

    The returned rows can then be streamed exactly like MNIST/CreditCard rows.
    """
    import pandas as pd
    from sklearn.decomposition import TruncatedSVD

    path = Path(path)
    print(f"  Loading MovieLens 20M from {path} …", flush=True)

    # Pass 1: choose the most frequently rated movies.
    movie_counts = None
    total_ratings = 0
    for chunk in _movielens_chunks(path, chunksize=chunksize):
        vc = chunk["movieId"].value_counts()
        movie_counts = vc if movie_counts is None else movie_counts.add(vc, fill_value=0)
        total_ratings += len(chunk)
    top_movies = movie_counts.sort_values(ascending=False).head(n_movies).index.astype(int).tolist()
    top_movie_set = set(top_movies)

    # Pass 2: choose users who have the most ratings among those movies.
    user_counts = None
    for chunk in _movielens_chunks(path, chunksize=chunksize):
        chunk = chunk[chunk["movieId"].isin(top_movie_set)]
        if chunk.empty:
            continue
        vc = chunk["userId"].value_counts()
        user_counts = vc if user_counts is None else user_counts.add(vc, fill_value=0)

    if user_counts is None or user_counts.empty:
        raise ValueError("No MovieLens ratings remained after filtering top movies.")

    eligible = user_counts[user_counts >= min_ratings_per_user]
    if len(eligible) < n_samples:
        eligible = user_counts
    top_users = eligible.sort_values(ascending=False).head(n_samples).index.astype(int).tolist()

    movie_to_col = {movie_id: j for j, movie_id in enumerate(top_movies)}
    user_to_row = {user_id: i for i, user_id in enumerate(top_users)}
    X = np.zeros((len(top_users), len(top_movies)), dtype=np.float32)

    # Pass 3: build the selected dense submatrix.
    for chunk in _movielens_chunks(path, chunksize=chunksize):
        chunk = chunk[
            chunk["movieId"].isin(top_movie_set)
            & chunk["userId"].isin(user_to_row)
        ]
        if chunk.empty:
            continue
        vals = chunk["rating"].to_numpy(dtype=np.float32)
        if center_ratings:
            vals = vals - np.float32(rating_center)
        rows = chunk["userId"].map(user_to_row).to_numpy()
        cols = chunk["movieId"].map(movie_to_col).to_numpy()
        X[rows, cols] = vals

    # Drop rows that somehow stayed all zero after centering/filtering.
    nonzero = np.linalg.norm(X, axis=1) > 1e-12
    X = X[nonzero]

    if d_reduced is not None and d_reduced > 0 and X.shape[1] > d_reduced:
        n_comp = min(d_reduced, X.shape[0] - 1, X.shape[1] - 1)
        if n_comp <= 0:
            raise ValueError("Not enough nonzero MovieLens rows for TruncatedSVD.")
        svd = TruncatedSVD(n_components=n_comp, random_state=seed)
        X = svd.fit_transform(X)
        d_out = n_comp
        svd_note = f"TruncSVD d={n_comp} from top {len(top_movies)} movies"
    else:
        d_out = X.shape[1]
        svd_note = f"dense top-{len(top_movies)} movie matrix"

    X = _normalize_rows(X)
    return X, {
        "name": "MovieLens20M",
        "d": d_out,
        "T": len(X),
        "n_movies_raw": 27278,
        "n_ratings_raw_approx": total_ratings,
        "n_movies_used": len(top_movies),
        "n_users_used": len(X),
        "description": f"MovieLens 20M user-rating rows; {svd_note}",
    }


# ─────────────────────────────────────────────────────────────
# Dataset registry
# ─────────────────────────────────────────────────────────────

ALL_DATASETS = ["SyntheticOptimal", "SyntheticClustered", "MNIST", "CreditCard", "MovieLens20M"]


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
    if name == "MovieLens20M":
        return load_movielens_20m(**kwargs)
    raise ValueError(f"Unknown dataset: {name}.  "
                     f"Available: {ALL_DATASETS}")