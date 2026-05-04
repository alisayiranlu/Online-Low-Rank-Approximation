"""
algorithms.py
=============
All online low-rank approximation algorithms used in the experiments.

Available algorithms
--------------------
  GrassmannHRD   : Grassmannian HRD + MWUA  (ICML paper §3.4-3.6)
  FantopeOGD     : Online GD over the Fantope (WLRA Improvement paper)
  OfflineOptimum : Oracle SVD of the full dataset (lower-bound baseline)
  StreamingSVD   : Follow-the-Leader / incremental PCA baseline
  FTRL           : Follow-the-Regularized-Leader baseline
  SphericalHRD   : Spherical HRD + MWUA (optional; needs final_research_c.py)
  BadNet         : Fixed random subspace (legacy comparison only)

Build any algorithm via:
    build_algorithm(name, d, k, data=..., **kwargs)
"""

import numpy as np
import math
import itertools
from collections import deque


# ─────────────────────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────────────────────

def _normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v / n if n >= eps else v.copy()


def projection_loss(U, x):
    """Squared residual ||x - U U^T x||^2.  U can be a d×k ndarray or list of vectors."""
    if isinstance(U, list):
        if not U:
            return float(np.dot(x, x))
        U = np.column_stack(U)
    if U.size == 0:
        return float(np.dot(x, x))
    proj = U @ (U.T @ x)
    r = x - proj
    return float(np.dot(r, r))


def _fantope_project(M, k):
    """Project symmetric matrix M onto Fantope F_k(m): eigenvalues in [0,1], sum = k."""
    M = (M + M.T) / 2
    eigvals, eigvecs = np.linalg.eigh(M)
    clipped = _simplex_clip(eigvals, k)
    return eigvecs @ np.diag(clipped) @ eigvecs.T


def _simplex_clip(v, k):
    """Project v onto {λ ∈ [0,1]^n : Σλ_i = k} via bisection on Lagrange shift θ."""
    lo, hi = float(v.min()) - 1.0, float(v.max())
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if np.clip(v - mid, 0.0, 1.0).sum() < k:
            hi = mid
        else:
            lo = mid
    return np.clip(v - 0.5 * (lo + hi), 0.0, 1.0)


# ─────────────────────────────────────────────────────────────
# 1.  GRASSMANNIAN HRD  (ICML paper §3.4 – 3.6)
# ─────────────────────────────────────────────────────────────

def _grass_dist(U, V):
    """Chordal Grassmannian distance d_G(U,V) = sqrt(k - ||U^T V||_F^2)."""
    sv = np.linalg.svd(U.T @ V, compute_uv=False)
    sv = np.clip(sv, 0.0, 1.0)
    return float(np.sqrt(max(0.0, U.shape[1] - float(np.sum(sv ** 2)))))


def _lift_k(x, k):
    """Embed unit vector x into Gr(k,d) by appending k-1 orthogonal standard directions."""
    d = len(x)
    basis = [_normalize(x)]
    for i in range(d):
        if len(basis) >= k:
            break
        e = np.zeros(d)
        e[i] = 1.0
        v = e.copy()
        for b in basis:
            v -= np.dot(v, b) * b
        if np.linalg.norm(v) > 1e-10:
            basis.append(_normalize(v))
    return np.column_stack(basis[:k]) if len(basis) >= k else np.eye(d, k)


def _perturb_subspace(U, scale):
    """Random geodesic perturbation of subspace U on the Grassmannian by magnitude ~scale."""
    d, k = U.shape
    Z = np.random.randn(d, k)
    Z -= U @ (U.T @ Z)           # project to tangent space T_U Gr(k,d)
    fn = np.linalg.norm(Z, 'fro')
    if fn < 1e-10:
        return U.copy()
    Z = Z * (scale / fn)
    try:
        Ud, s, Vt = np.linalg.svd(Z, full_matrices=False)
        W = U @ Vt.T @ np.diag(np.cos(s)) + Ud @ np.diag(np.sin(s))
        W = W @ Vt
        Q, _ = np.linalg.qr(W)
        return Q[:, :k]
    except Exception:
        return U.copy()


def _update_center_online(center, x_lifted, n, step=0.1):
    """Incremental Fréchet mean on the Grassmannian via a tangent-space step."""
    if n == 1:
        return x_lifted.copy()
    alpha = min(step, 1.0 / n)
    V_perp = x_lifted - center @ (center.T @ x_lifted)
    fn = np.linalg.norm(V_perp, 'fro')
    if fn < 1e-10:
        return center
    try:
        Ud, s_perp, _ = np.linalg.svd(V_perp, full_matrices=False)
        _, cos_th, _ = np.linalg.svd(center.T @ x_lifted, compute_uv=True)
        cos_th = np.clip(cos_th[:len(s_perp)], 0, 1)
        theta = np.arccos(cos_th)
        Delta = Ud[:, :len(theta)] @ np.diag(theta)
        Ud2, s2, Vt2 = np.linalg.svd(alpha * Delta, full_matrices=False)
        W = center @ Vt2.T @ np.diag(np.cos(s2)) + Ud2 @ np.diag(np.sin(s2))
        W = W @ Vt2
        Q, _ = np.linalg.qr(W)
        return Q[:, :center.shape[1]]
    except Exception:
        return center


class _GrassNode:
    _ctr = itertools.count()

    def __init__(self, center, radius, depth=0):
        self.id = next(_GrassNode._ctr)
        self.center = center        # d × k orthonormal matrix
        self.radius = radius        # Grassmannian chordal-ball radius
        self.depth = depth
        self.is_leaf = True
        self.children = []
        self.n = 0
        self.buf = deque(maxlen=200)


class GrassmannHRDAlg:
    """
    Grassmannian HRD + Mass-Tree Multiplicative Weights (MTMW).

    Each leaf of the adaptive binary tree is a chordal ball B_G(U_R, Δ_R) on
    Gr(k,d) with an explicit center subspace.  Multiplicative weights (MWUA)
    govern expert selection; the prediction at each step is the expected
    projection loss under the current weight distribution over leaves.

    Parameters
    ----------
    d            : ambient dimension
    k            : target rank
    epsilon_hrd  : HRD resolution parameter ε_{hrd}
    eta          : MWUA learning rate
    n_min        : minimum leaf samples before a split may occur
    n_max        : maximum leaf samples (triggers forced split)
    """

    name = "GrassmannHRD"

    def __init__(self, d, k, epsilon_hrd=0.1, eta=0.5, n_min=10, n_max=100):
        self.d = d
        self.k = k
        self.epsilon_hrd = epsilon_hrd
        self.eta = eta
        self.n_min = n_min
        self.n_max = n_max
        self.t = 0

        root = _GrassNode(center=np.eye(d, k), radius=math.sqrt(k), depth=0)
        self.root = root
        self.leaves = {root.id: root}
        self.weights = {root.id: 1.0}
        self.cum_loss = [0.0]

    def _delta_t(self):
        return self.epsilon_hrd / max(1, self.t) ** 1.5

    def _refinement_ok(self, node, x_lifted):
        r = _grass_dist(node.center, x_lifted)
        thresh = max(self.epsilon_hrd * r / 2.0, self._delta_t())
        return node.radius <= thresh

    def _route(self, x_lifted):
        node = self.root
        while not node.is_leaf:
            node = min(node.children, key=lambda c: _grass_dist(c.center, x_lifted))
        return node

    def _split(self, node):
        new_r = node.radius / 2.0
        ch1 = _GrassNode(_perturb_subspace(node.center, node.radius * 0.35),
                         new_r, node.depth + 1)
        ch2 = _GrassNode(_perturb_subspace(node.center, node.radius * 0.35),
                         new_r, node.depth + 1)
        node.children = [ch1, ch2]
        node.is_leaf = False

        del self.leaves[node.id]
        w = self.weights.pop(node.id, 1.0)
        self.weights[ch1.id] = w / 2      # uniform prior (MTMW mass split)
        self.weights[ch2.id] = w / 2
        self.leaves[ch1.id] = ch1
        self.leaves[ch2.id] = ch2

        for u in node.buf:
            ul = _lift_k(u, self.k)
            best = min(node.children, key=lambda c: _grass_dist(c.center, ul))
            best.n += 1
            best.buf.append(u)
            best.center = _update_center_online(best.center, ul, best.n)
        node.buf.clear()
        return [ch1, ch2]

    def step(self, x):
        self.t += 1
        x = _normalize(x)
        x_lifted = _lift_k(x, self.k)

        for nid in self.leaves:
            if nid not in self.weights:
                self.weights[nid] = 1.0 / max(1, len(self.leaves))

        leaf_ids = list(self.leaves.keys())
        leaf_losses = {nid: projection_loss(self.leaves[nid].center, x)
                       for nid in leaf_ids}

        total_w = sum(self.weights.get(nid, 0.0) for nid in leaf_ids)
        if total_w <= 0:
            total_w = len(leaf_ids)
            for nid in leaf_ids:
                self.weights[nid] = 1.0
        agg = sum(self.weights.get(nid, 0.0) / total_w * leaf_losses[nid]
                  for nid in leaf_ids)
        self.cum_loss.append(self.cum_loss[-1] + agg)

        # HRD update
        leaf = self._route(x_lifted)
        leaf.n += 1
        leaf.buf.append(x)
        leaf.center = _update_center_online(leaf.center, x_lifted, leaf.n)

        to_check = [leaf]
        while to_check:
            nd = to_check.pop()
            if (nd.is_leaf and nd.n >= self.n_min
                    and not self._refinement_ok(nd, x_lifted)):
                to_check.extend(self._split(nd))

        # MWUA update
        for nid in [nid for nid in list(self.weights) if nid not in self.leaves]:
            del self.weights[nid]
        for nid in leaf_ids:
            if nid in self.weights and nid in leaf_losses:
                self.weights[nid] *= math.exp(-self.eta * leaf_losses[nid])
        for nid in self.leaves:
            if nid not in self.weights:
                self.weights[nid] = 1.0 / max(1, len(self.leaves))
        total_w = sum(self.weights.values())
        if total_w > 0:
            for nid in self.weights:
                self.weights[nid] /= total_w

        return float(agg)

    @property
    def n_leaves(self):
        return len(self.leaves)


# ─────────────────────────────────────────────────────────────
# 2.  FANTOPE OGD  (WLRA Improvement paper)
# ─────────────────────────────────────────────────────────────

class FantopeOGDAlg:
    """
    Online Gradient Descent over the Fantope F_k(m).

    Implements the fully polynomial-time algorithm from
    "Fully Polynomial-Time Online WLRA via Convex Relaxation and
    Exact Loss Decoupling".

    An anchor subspace V ∈ R^{d×m} is estimated from the first `init_steps`
    data points via truncated SVD, then fixed for the rest of the stream.

    Parameters
    ----------
    d           : ambient dimension
    k           : target rank
    m           : anchor dimension (default: min(d, 3k))
    eta         : OGD step size (default: analytically optimal √(2k)/(4W√T))
    init_steps  : number of initial points used to estimate the anchor
    T_est       : estimated stream length (used for default step-size)
    """

    name = "FantopeOGD"

    def __init__(self, d, k, m=None, eta=None, init_steps=50, T_est=1000):
        self.d = d
        self.k = k
        self.m = m if m is not None else min(d, max(k, min(3 * k, d)))
        self.W_bound = 1.0

        T = max(1, T_est)
        self.eta = eta if eta is not None else (
            math.sqrt(2 * k) / (4 * self.W_bound * math.sqrt(T))
        )

        self.init_steps = min(init_steps, max(10, self.m))
        self._buf = []
        self._initialized = False
        self.V = None
        self.P = None
        self.cum_loss = [0.0]
        self.t = 0

    def _init_anchor(self):
        X = np.array(self._buf)
        U, _, _ = np.linalg.svd(X.T @ X, full_matrices=False)
        m = min(self.m, U.shape[1])
        self.V = U[:, :m]
        if m < self.m:
            extra = np.random.randn(self.d, self.m - m)
            for j in range(extra.shape[1]):
                v = extra[:, j]
                for col in range(self.V.shape[1]):
                    v -= np.dot(v, self.V[:, col]) * self.V[:, col]
                nrm = np.linalg.norm(v)
                if nrm > 1e-10:
                    self.V = np.column_stack([self.V, v / nrm])
                if self.V.shape[1] >= self.m:
                    break
        self.V = self.V[:, :self.m]
        self.P = (self.k / self.m) * np.eye(self.m)
        self._initialized = True

    def step(self, x, W_diag=None):
        self.t += 1
        x = np.asarray(x, dtype=float)

        if not self._initialized:
            self._buf.append(x.copy())
            loss = float(np.dot(x, x))
            self.cum_loss.append(self.cum_loss[-1] + loss)
            if len(self._buf) >= self.init_steps:
                self._init_anchor()
            return float(loss)

        if W_diag is None:
            W_diag = np.ones(self.d)

        # Sufficient statistics (Theorem 3.1)
        x_bar   = self.V.T @ x
        z_bar   = self.V.T @ (W_diag * x)
        W_tilde = self.V.T @ (W_diag[:, None] * self.V)
        X_bar   = np.outer(x_bar, x_bar)

        P_full   = self.V @ self.P @ self.V.T
        residual = x - P_full @ x
        loss     = float((W_diag * residual) @ residual)
        self.cum_loss.append(self.cum_loss[-1] + loss)

        # Analytical gradient (Theorem 4.1)
        C_t  = np.outer(z_bar, x_bar) + np.outer(x_bar, z_bar)
        grad = -C_t + (W_tilde @ self.P @ X_bar + X_bar @ self.P @ W_tilde)

        # OGD step + Fantope projection
        M      = self.P - self.eta * grad
        self.P = _fantope_project(M, self.k)

        return float(loss)

    def get_basis(self):
        """Extract rank-k orthonormal basis from current fractional projection."""
        P_full = self.V @ self.P @ self.V.T
        P_full = (P_full + P_full.T) / 2
        eigvals, eigvecs = np.linalg.eigh(P_full)
        idx = np.argsort(eigvals)[::-1]
        return eigvecs[:, idx[:self.k]]

    @property
    def n_leaves(self):
        return 1


# ─────────────────────────────────────────────────────────────
# 3.  OFFLINE OPTIMUM  (oracle lower bound)
# ─────────────────────────────────────────────────────────────

class OfflineOptimumAlg:
    """
    Oracle baseline. Computes the globally optimal rank-k subspace via SVD of
    the entire data matrix passed at construction, then uses that fixed basis.
    Provides a lower bound on achievable cumulative loss.

    Parameters
    ----------
    d    : ambient dimension
    k    : target rank
    data : array-like (n, d) — the full data sequence (all T vectors)
    """

    name = "OfflineOptimum"

    def __init__(self, d, k, data):
        self.d = d
        self.k = k
        X = np.array(data, dtype=float)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        self.basis = Vt[:k].T       # d × k, orthonormal
        self.cum_loss = [0.0]

    def step(self, x):
        loss = projection_loss(self.basis, x)
        self.cum_loss.append(self.cum_loss[-1] + loss)
        return float(loss)

    @property
    def n_leaves(self):
        return 1


# ─────────────────────────────────────────────────────────────
# 4.  STREAMING SVD  (Follow-the-Leader / incremental PCA)
# ─────────────────────────────────────────────────────────────

class StreamingSVDAlg:
    """
    Follow-the-Leader baseline. At time t predicts with the top-k eigenvectors
    of Σ_{t-1} = Σ_{s<t} x_s x_s^T (best rank-k subspace seen so far).

    Parameters
    ----------
    d : ambient dimension
    k : target rank
    """

    name = "StreamingSVD"

    def __init__(self, d, k):
        self.d = d
        self.k = k
        self.basis = np.eye(d, k)
        self.cov   = np.zeros((d, d), dtype=float)
        self.n     = 0
        self.cum_loss = [0.0]

    def step(self, x):
        loss = projection_loss(self.basis, x)
        self.cum_loss.append(self.cum_loss[-1] + loss)

        x = np.asarray(x, dtype=float)
        self.n += 1
        self.cov += np.outer(x, x)

        eigvals, eigvecs = np.linalg.eigh(self.cov)
        idx = np.argsort(eigvals)[::-1]
        self.basis = eigvecs[:, idx[:self.k]]

        return float(loss)

    @property
    def n_leaves(self):
        return 1


# ─────────────────────────────────────────────────────────────
# 5.  FTRL  (Follow-the-Regularized-Leader)
# ─────────────────────────────────────────────────────────────

class FTRLAlg:
    """
    Follow-the-Regularized-Leader baseline.

    Solves at each step:
        P_t = argmax_{P ∈ F_k(d)} Tr(P · Σ_t) − (λ_t/2) ||P − (k/d)I||_F²
    where Σ_t = Σ_{s≤t} x_s x_s^T and λ_t = reg / (t+1).

    Parameters
    ----------
    d   : ambient dimension
    k   : target rank
    reg : regularisation strength (default 1.0)
    """

    name = "FTRL"

    def __init__(self, d, k, reg=1.0):
        self.d   = d
        self.k   = k
        self.reg = reg
        self.S   = np.zeros((d, d), dtype=float)
        self.basis = np.eye(d, k)
        self.t   = 0
        self.cum_loss = [0.0]

    def step(self, x):
        loss = projection_loss(self.basis, x)
        self.cum_loss.append(self.cum_loss[-1] + loss)

        x = np.asarray(x, dtype=float)
        self.t += 1
        self.S += np.outer(x, x)

        lam = self.reg / (self.t + 1)
        M   = self.S + lam * np.eye(self.d)
        eigvals, eigvecs = np.linalg.eigh(M)
        idx = np.argsort(eigvals)[::-1]
        self.basis = eigvecs[:, idx[:self.k]]

        return float(loss)

    @property
    def n_leaves(self):
        return 1


# ─────────────────────────────────────────────────────────────
# 6.  SPHERICAL HRD  (optional — requires final_research_c.py)
# ─────────────────────────────────────────────────────────────

class SphericalHRDAlg:
    """
    Spherical HRD + MWUA.  Only available when final_research_c.py is present.
    """

    name = "SphericalHRD"

    def __init__(self, d, k, d_split=None, eta=0.5, n_min=20, n_max=100,
                 epsilon_hrd=0.1, candidate_pool_size=12, max_experts=300):
        try:
            from final_research_c import SphericalHRD, ExpertMWUA
        except ImportError:
            raise ImportError(
                "SphericalHRD requires 'final_research_c.py' in the Python path."
            )
        if d_split is None:
            d_split = min(max(4, d // 3), d - 1)
        hrd = SphericalHRD(d=d, d_split=d_split, k_expert=k,
                           n_min=n_min, epsilon_hrd=epsilon_hrd, n_max_leaf=n_max)
        self._mw = ExpertMWUA(hrd, eta=eta, k_expert=k,
                              candidate_pool_size=candidate_pool_size,
                              max_experts=max_experts, combined_basis_dim=k,
                              random_seed=0)
        self._hrd = hrd
        self.cum_loss = [0.0]

    def step(self, x):
        loss, _, _ = self._mw.step(x)
        self.cum_loss = self._mw.cum_loss
        return float(loss)

    @property
    def n_leaves(self):
        return len(self._hrd.leaves)


# ─────────────────────────────────────────────────────────────
# 7.  BAD-NET  (fixed random baseline — legacy)
# ─────────────────────────────────────────────────────────────

class BadNetAlg:
    """Fixed random subspace baseline kept for legacy comparison."""

    name = "BadNet"

    def __init__(self, d, k, seed=42):
        rng = np.random.default_rng(seed)
        vecs = []
        for i in range(k):
            coords = np.array(
                [(j + 1) / 3.0 * (1 if rng.random() > 0.5 else -1) for j in range(d)]
            )
            vecs.append(_normalize(coords))
        self.basis = vecs
        self.cum_loss = [0.0]

    def step(self, x):
        loss = projection_loss(self.basis, x)
        self.cum_loss.append(self.cum_loss[-1] + loss)
        return float(loss)

    @property
    def n_leaves(self):
        return 1


# ─────────────────────────────────────────────────────────────
# ALGORITHM REGISTRY
# ─────────────────────────────────────────────────────────────

# Primary algorithms used in paper experiments (run by default)
ALL_ALGORITHMS = [
    "GrassmannHRD",
    "FantopeOGD",
    "OfflineOptimum",
    "FTRL",
    "BadNet",
]

# Optional extras — not run by default
OPTIONAL_ALGORITHMS = [
    "SphericalHRD",   # requires final_research_c.py
    "BadNet",         # fixed random baseline
]


def build_algorithm(name, d, k, data=None, **kwargs):
    """
    Factory. Returns a fully constructed algorithm instance.

    Parameters
    ----------
    name   : algorithm name (see ALL_ALGORITHMS / OPTIONAL_ALGORITHMS)
    d      : ambient dimension
    k      : target rank
    data   : (n, d) array — required only for OfflineOptimum
    **kwargs : forwarded to the algorithm constructor
    """
    if name == "GrassmannHRD":
        return GrassmannHRDAlg(d, k, **kwargs)
    if name == "FantopeOGD":
        return FantopeOGDAlg(d, k, **kwargs)
    if name == "OfflineOptimum":
        if data is None:
            raise ValueError("OfflineOptimum requires the `data` argument.")
        return OfflineOptimumAlg(d, k, data, **kwargs)
    if name == "StreamingSVD":
        return StreamingSVDAlg(d, k, **kwargs)
    if name == "FTRL":
        return FTRLAlg(d, k, **kwargs)
    if name == "SphericalHRD":
        return SphericalHRDAlg(d, k, **kwargs)
    if name == "BadNet":
        return BadNetAlg(d, k, **kwargs)
    raise ValueError(
        f"Unknown algorithm: '{name}'.  "
        f"Available: {ALL_ALGORITHMS + OPTIONAL_ALGORITHMS}"
    )