import numpy as np
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import math
import itertools
import random

# ---------------------------
# Utility Functions
# ---------------------------
def normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        return v.copy()
    return v / n

def random_gaussian_vector(k):
    return np.random.normal(size=(k,))

def angular_distance(p, q):
    """Angular distance between two unit vectors on sphere."""
    dot_product = np.clip(np.dot(p, q), -1.0, 1.0)
    return np.arccos(dot_product)

# ---------------------------
# Node (region) class with coreset (reservoir sampling)
# ---------------------------
class Node:
    _id_iter = itertools.count()
    def __init__(self, k, parent=None, depth=0, coreset_capacity=50):
        self.id = next(Node._id_iter)
        self.is_leaf = True
        self.V = None               # k x d splitting frame (columns are normals)
        self.b = None               # bias vector (length d): b_j = -v_j . c
        self.children = dict()      # mask -> Node

        # leaf stats
        self.n = 0                  # total points seen in this region (not coreset size)
        # note: we no longer maintain s as sum of all points (to avoid storing full data)
        self.Rlen = 0.0
        self.r_bar = 0.0
        self.c = None               # centroid (unit vector) computed from coreset
        self.min_dot_est = 1.0
        self.residuals = deque(maxlen=50)
        self.expert_basis = None    # list of orthonormal basis vectors (columns) for low-rank approx
        # coreset: small sampled representative set (reservoir)
        self.coreset = []           # list of unit vectors
        self.coreset_capacity = coreset_capacity
        self.parent = parent
        self.depth = depth
        self.angular_diameter = 0.0
        self.k = k

    def __repr__(self):
        return f"Node(id={self.id},leaf={self.is_leaf},n={self.n},depth={self.depth})"

    def add_point_to_coreset(self, u):
        """
        Reservoir sampling: maintain at most coreset_capacity items uniformly sampled
        from all points seen in this node. self.n should be incremented before calling.
        """
        # reservoir sampling: with probability cap/n replace random index
        cap = self.coreset_capacity
        if cap <= 0:
            return
        if len(self.coreset) < cap:
            self.coreset.append(u.copy())
        else:
            # generate integer in [1, self.n]
            j = random.randint(1, max(1, self.n))
            if j <= cap:
                idx = random.randint(0, cap - 1)
                self.coreset[idx] = u.copy()

    def recompute_centroid_from_coreset(self):
        """
        Compute centroid c from the current coreset (mean of coreset vectors), then normalize.
        If coreset empty, leave c unchanged (or set to default).
        """
        if len(self.coreset) == 0:
            # fallback: if node has parent centroid, use that; otherwise unit e1
            if self.parent is not None and self.parent.c is not None:
                self.c = self.parent.c.copy()
            else:
                vec = np.zeros(self.k)
                vec[0] = 1.0
                self.c = vec
            return self.c
        # use simple average of coreset (unweighted). Reservoir samples approximate the mean.
        M = np.stack(self.coreset, axis=0)  # m x k
        mean = np.mean(M, axis=0)
        if np.linalg.norm(mean) < 1e-12:
            # degenerate, pick first coreset vector
            self.c = normalize(self.coreset[0])
        else:
            self.c = normalize(mean)
        # update Rlen and r_bar as approximate proxies
        self.Rlen = np.linalg.norm(mean) * len(self.coreset)
        self.r_bar = (np.linalg.norm(mean) / max(1, len(self.coreset))) if len(self.coreset) > 0 else 0.0
        return self.c

# ---------------------------
# HRD functions
# ---------------------------

def get_child_index(u, V, b=None):
    """
    Return bitmask index for u relative to orthonormal frame V (k x d),
    using hyperplanes v_j \cdot x + b_j = 0 if b provided; otherwise v_j \cdot x >= 0.
    Bits: i-th bit = 1 if sign is positive, 0 if negative.
    """
    mask = 0
    if V is None or V.size == 0:
        return mask
    d = V.shape[1]
    if b is None:
        # default: hyperplanes pass through origin
        for i in range(d):
            if np.dot(V[:, i], u) >= 0:
                mask |= (1 << i)
    else:
        for i in range(d):
            if (np.dot(V[:, i], u) + float(b[i])) >= 0:
                mask |= (1 << i)
    return mask

def gram_schmidt(V):
    """Orthonormalize columns of V using Gram-Schmidt, return matrix with orthonormal columns."""
    if V is None:
        return np.zeros((0,0))
    if V.size == 0:
        return np.zeros((V.shape[0], 0))
    U = []
    for v in V.T:
        w = v.copy()
        for u in U:
            w = w - np.dot(u, w) * u
        normw = np.linalg.norm(w)
        if normw > 1e-12:
            U.append(w / normw)
    if len(U) == 0:
        return np.zeros((V.shape[0], 0))
    return np.column_stack(U)

def sample_splitting_frame(c, d, k, depth, points=None, used_directions=None):
    n = depth
    # CASE 1: Sufficient coordinate directions remain
    if n * d <= (k - 1):
        # pick next d coordinate axes (0-based indexing)
        start = n * d
        axes = []
        for i in range(start, start + d):
            if i >= k:
                break
            e = np.zeros(k)
            e[i] = 1.0
            axes.append(e)
        if len(axes) > 0:
            V = np.column_stack(axes)
        else:
            V = np.zeros((k, 0))
        # If we somehow got fewer than d axes (shouldn't happen with condition), pad with orthogonals
        if V.shape[1] < d:
            # find extra orthonormal directions orthogonal to existing ones
            extra = d - V.shape[1]
            if extra > 0:
                W = np.random.randn(k, extra)
                if c is not None:
                    for j in range(extra):
                        W[:, j] = W[:, j] - np.dot(W[:, j], c) * c
                Uextra = gram_schmidt(W)
                if Uextra.size > 0:
                    V = np.column_stack([V, Uextra[:, :max(0, min(extra, Uextra.shape[1]))]])
        return V[:, :min(d, V.shape[1])]

    # CASE 2: coordinate directions exhausted -> use centroid and random Gaussian
    if c is None:
        # worst-case fallback: pick some unit vector
        c = np.zeros(k)
        c[0] = 1.0
    c = normalize(c)
    # Gaussian random matrix
    W = np.random.randn(k, d * 2)  # sample extra for robustness
    # project columns onto tangent space at c
    for j in range(W.shape[1]):
        W[:, j] = W[:, j] - np.dot(W[:, j], c) * c
    # Gram-Schmidt orthonormalization
    V = gram_schmidt(W)
    if V.shape[1] < d:
        # augment by sampling more random tangent vectors and orthonormalize again
        needed = d - V.shape[1]
        W2 = np.random.randn(k, needed * 2)
        for j in range(W2.shape[1]):
            W2[:, j] = W2[:, j] - np.dot(W2[:, j], c) * c
        V_full = gram_schmidt(np.column_stack([V, W2]) if V.size else W2)
        # ensure we have d columns (if still fewer, return what we have)
        if V_full.shape[1] >= d:
            V = V_full[:, :d]
        else:
            V = V_full
    return V[:, :d]

# ---------------------------
# Expert construction helpers (Refactored)
# ---------------------------

def build_basis_from_centroids(selected_centroids):
    """
    Given a list/array of centroids (each a unit vector), construct an orthonormal basis
    from them using Gram-Schmidt on the selected vectors. Returns list of column vectors.
    If Gram-Schmidt produces fewer than provided, returns what it can.
    """
    if len(selected_centroids) == 0:
        return []
    M = np.column_stack([normalize(c) for c in selected_centroids])
    M_orth = gram_schmidt(M)
    # return list of column vectors
    basis_list = [M_orth[:, i].copy() for i in range(M_orth.shape[1])]
    return basis_list

def build_expert_from_region_centroids(all_centroids, target_idx, r):
    """
    Build an r-dimensional basis for the leaf (expert) whose centroid is at index target_idx
    by selecting that centroid plus the r-1 nearest other centroids (angular distance).
    all_centroids: list or array of shape (m, k)
    target_idx: index in all_centroids to center around
    r: desired basis dimension (number of centroids to pick)
    Returns: list of orthonormal column vectors (length up to r)
    """
    m = len(all_centroids)
    if m == 0:
        return []
    # if only one centroid available or r==1, return that centroid
    if r == 1 or m == 1:
        return [normalize(all_centroids[target_idx])]

    # compute angular distances to others
    target = all_centroids[target_idx]
    dists = []
    for i, c in enumerate(all_centroids):
        if i == target_idx:
            continue
        dists.append((angular_distance(target, c), i))
    # sort by angular distance (ascending)
    dists.sort(key=lambda x: x[0])
    # pick top (r-1) nearest (or fewer if not enough)
    nearest_indices = [idx for (_, idx) in dists[:max(0, min(len(dists), r - 1))]]
    selected = [normalize(target)] + [normalize(all_centroids[i]) for i in nearest_indices]
    # orthonormalize selected to produce basis
    basis = build_basis_from_centroids(selected)
    # if Gram-Schmidt returned fewer than r due to linear dependence, return what we have
    return basis

# ---------------------------
# Tree HRD (uses node.coreset as sample coreset)
# ---------------------------

class SphericalHRD:
    def __init__(self, k, d_split, r_expert,
                 n_min, epsilon_hrd, epsilon_exp=1e-3, n_max_leaf=500,
                 used_directions=None, coreset_capacity=50):
        self.k = k
        self.d = min(d_split, k)  # cannot exceed k
        self.r_expert = r_expert  # number of centroids per expert (basis size)
        self.n_min = n_min
        self.epsilon_hrd = epsilon_hrd
        self.epsilon_exp = epsilon_exp
        self.n_max_leaf = n_max_leaf
        self.used_directions = used_directions
        self.coreset_capacity = coreset_capacity
        # root node
        self.root = Node(k=k, parent=None, depth=0, coreset_capacity=self.coreset_capacity)
        # list (dict) of current leaf nodes by id
        self.leaves = {self.root.id: self.root}
        # maintain expert registry (leaf.id -> expert basis)
        self.expert_map = dict()
        # track total points processed
        self.t = 0

    def route(self, u):
        node = self.root
        while not node.is_leaf:
            mask = get_child_index(u, node.V, node.b)
            if mask not in node.children:
                # lazy-create child
                child = Node(k=self.k, parent=node, depth=node.depth + 1, coreset_capacity=self.coreset_capacity)
                node.children[mask] = child
                self.leaves[child.id] = child
            node = node.children[mask]
        return node

    def update_leaf(self, node, u):
        # increment total count for the node and update coreset (reservoir)
        node.n += 1
        node.add_point_to_coreset(u)
        node.recompute_centroid_from_coreset()
        # update angular/proxy stats
        node.min_dot_est = min(node.min_dot_est, float(np.dot(node.c, u))) if node.c is not None else node.min_dot_est
        # keep residuals for diagnostics (distance to centroid)
        if node.c is not None:
            node.residuals.append(np.sum((u - (node.c * np.dot(u, node.c))) ** 2))
        # note: we don't maintain full-data sums here (memory-bounded)

    def should_split(self, node, x_t):
        if node.n >= self.n_max_leaf:
            return True

        if node.n < self.n_min:
            return False

        # compute min angular distance from new point x_t to any point in the node.coreset
        min_angular_dist = float('inf')
        for p in list(node.coreset):
            dist = angular_distance(p, x_t)
            min_angular_dist = min(min_angular_dist, dist)

        if min_angular_dist == float('inf'):  # no points in region
            min_angular_dist = 0.0

        r = min_angular_dist
        delta_t = self.epsilon_hrd / (2 * (self.t + 1) ** 3)  # add 1 to avoid div by 0
        decision = max(self.epsilon_hrd * r / 2, delta_t)

        return node.angular_diameter > decision

    def update_angular_diameter(self, node):
        """Compute max angular distance between any two coreset points in node.coreset."""
        if len(node.coreset) < 2:
            node.angular_diameter = 0.0
            return

        max_dist = 0.0
        buffer_list = list(node.coreset)
        for i in range(len(buffer_list)):
            for j in range(i + 1, len(buffer_list)):
                dist = angular_distance(buffer_list[i], buffer_list[j])
                max_dist = max(max_dist, dist)

        node.angular_diameter = max_dist

    def split_leaf(self, node):
        """Split node into up to 2^d children using d orthogonal hyperplanes v_j·(x - c)=0."""
        # prepare centroid
        if node.c is None:
            # cannot split well without centroid; use arbitrary
            node.c = np.zeros(self.k)
            node.c[0] = 1.0
        # sample splitting frame V
        V = sample_splitting_frame(node.c, self.d, self.k, node.depth, points=list(node.coreset),
                                   used_directions=self.used_directions)
        # ensure V has exactly d columns (if fewer, pad using orthonormalization trick)
        if V.shape[1] < self.d:
            # try to augment by sampling random tangent directions and orthonormalize
            extra_needed = self.d - V.shape[1]
            W = np.random.randn(self.k, extra_needed * 2)
            for j in range(W.shape[1]):
                W[:, j] = W[:, j] - np.dot(W[:, j], node.c) * node.c
            V_aug = gram_schmidt(np.column_stack([V, W]) if V.size else W)
            if V_aug.shape[1] >= self.d:
                V = V_aug[:, :self.d]
            else:
                V = V_aug

        # compute bias b_j = -v_j . c so hyperplane is v_j·x + b_j = 0  <=> v_j·(x - c)=0
        b = - (V.T @ node.c) if V.size else np.array([])

        node.V = V
        node.b = b
        node.is_leaf = False
        # create children mapping on demand
        node.children = dict()
        # Remove parent from leaves and children will be added on demand
        if node.id in self.leaves:
            del self.leaves[node.id]
        
        # redistribute coreset points to child regions, only create regions that are necessary
        for u in list(node.coreset):
            mask = get_child_index(u, V, b)
            if mask not in node.children:
                child = Node(k=self.k, parent=node, depth=node.depth + 1, coreset_capacity=self.coreset_capacity)
                node.children[mask] = child
                self.leaves[child.id] = child
            child = node.children[mask]
            # update child using coreset point (treat as if it's a new observation)
            child.n += 1
            child.add_point_to_coreset(u)
            child.recompute_centroid_from_coreset()
        
        # clear parent coreset (keep stats maybe)
        node.coreset.clear()
        return list(node.children.values())

    def _construct_experts_from_region_centroids(self):
        """
        Build expert bases for each leaf region using r centroids selected among all leaf centroids.
        For each leaf p: choose p.c plus its r-1 nearest other leaf centroids (by angular distance),
        orthonormalize them -> leaf.expert_basis.
        """
        # collect centroids for all leaves that have at least one point (from coreset)
        leaf_nodes = [nd for nd in self.leaves.values() if nd.c is not None]
        if len(leaf_nodes) == 0:
            return
        centroids = [nd.c for nd in leaf_nodes]
        # for each leaf, build its basis
        for idx, nd in enumerate(leaf_nodes):
            basis = build_expert_from_region_centroids(centroids, idx, self.r_expert)
            nd.expert_basis = basis
            # also update mapping for convenience
            self.expert_map[nd.id] = basis

    def process_point(self, x):
        """
        Process a single point x (raw vector) IE: normalize to unit sphere, route, update,
        possibly split; return the leaf node in which x was stored.
        """
        self.t += 1
        u = normalize(x)
        node = self.route(u)
        # update only via coreset (no full-data sums)
        self.update_leaf(node, u)
        # update angular diameter for this node (and possibly ancestors if desired)
        self.update_angular_diameter(node)
        # splitting loop (could split many times along subtree)
        to_check = [node]
        while to_check:
            nd = to_check.pop()
            if nd.is_leaf and self.should_split(nd, u):
                children = self.split_leaf(nd)
                # check newly created children for immediate splits
                # compute their angular diameters
                for ch in children:
                    self.update_angular_diameter(ch)
                to_check.extend(children)
        # AFTER processing point and any splits, (re)construct experts from current region centroids
        self._construct_experts_from_region_centroids()
        return node

# ---------------------------
# Helpers for selecting top-r regions and forming combined basis
# ---------------------------

def compute_region_loss_for_point(node, x):
    """
    Compute instantaneous loss for a region (node) on point x.
    Prefer using node.expert_basis if present, otherwise fall back to centroid projection.
    """
    if node.expert_basis:
        return projection_loss(node.expert_basis, x)
    # fallback: if node.c exists, project onto single-centroid direction
    if node.c is not None:
        return projection_loss([node.c], x)
    # if nothing, treat as full loss (||x||^2)
    return float(np.dot(x, x))

def build_combined_basis_from_region_ids(hrd, region_ids, max_dim=None):
    """
    Collects centroids / expert basis vectors from selected region ids, orthonormalizes them
    and returns a list of orthonormal column vectors. If max_dim provided, truncate to that dim.
    """
    vecs = []
    for rid in region_ids:
        node = hrd.leaves.get(rid, None)
        if node is None:
            continue
        # Prefer node.expert_basis (list of orthonormal vectors). If present, extend with them.
        if node.expert_basis and len(node.expert_basis) > 0:
            vecs.extend([v.copy() for v in node.expert_basis])
        elif node.c is not None:
            vecs.append(node.c.copy())
    if len(vecs) == 0:
        return []
    M = np.column_stack([normalize(v) for v in vecs])
    M_orth = gram_schmidt(M)
    if max_dim is not None and M_orth.size > 0:
        M_orth = M_orth[:, :max_dim]
    # return as list of column vectors
    return [M_orth[:, i].copy() for i in range(M_orth.shape[1])] if M_orth.size > 0 else []

# ---------------------------
# Expert-based MWUA: experts are sets of r leaf regions
# ---------------------------

class ExpertMWUA:
    def __init__(self, hrd: SphericalHRD, eta=0.1, r_expert=2,
                 candidate_pool_size=10, max_experts=200, combined_basis_dim=None,
                 random_seed=None):
        """
        hrd: the SphericalHRD instance
        eta: learning rate for multiplicative weight updates
        r_expert: number of regions per expert (size of each expert tuple)
        candidate_pool_size: how many top regions to consider when forming candidate experts
        max_experts: maximum number of experts to maintain (cap for efficiency)
        combined_basis_dim: dimension to truncate combined orthonormal basis to
        """
        self.hrd = hrd
        self.eta = float(eta)
        self.r = max(1, int(r_expert))
        self.candidate_pool_size = max(1, int(candidate_pool_size))
        self.max_experts = max(1, int(max_experts))
        self.combined_basis_dim = combined_basis_dim
        self.random = random.Random(random_seed)

        # experts: dict mapping expert_id (tuple of region ids sorted) -> weight
        self.expert_weights = dict()
        # store most-recent round loss history (for debugging/plotting)
        self.cum_loss = [0.0]
        self.last_chosen_expert = None

    def _prune_stale_experts(self):
        """Remove experts containing leaves that no longer exist."""
        current_leaf_ids = set(self.hrd.leaves.keys())
        stale = [eid for eid in self.expert_weights if not set(eid).issubset(current_leaf_ids)]
        for eid in stale:
            del self.expert_weights[eid]

    def _ensure_region_priors(self):
        """
        Return a normalized prior over current leaves (dict leaf_id -> prior mass).
        Uses leaf.n (count) or 1.0 if zero to provide minimal mass.
        """
        priors = {}
        for node in self.hrd.leaves.values():
            priors[node.id] = max(1.0, float(node.n))
        total = sum(priors.values())
        if total <= 0:
            # fallback uniform
            nleaves = max(1, len(priors))
            for k in priors:
                priors[k] = 1.0 / nleaves
            return priors
        for k in priors:
            priors[k] /= total
        return priors

    def _build_candidate_pool(self, priors):
        """
        Choose a candidate pool of leaf ids to consider forming experts from.
        We pick the top `candidate_pool_size` leaves by prior mass.
        """
        if not priors:
            return []
        items = sorted(priors.items(), key=lambda kv: kv[1], reverse=True)
        pool = [kv[0] for kv in items[:self.candidate_pool_size]]
        return pool

    def _construct_candidate_experts(self, pool, priors):
        """
        Construct a set of candidate expert tuples (sorted tuples of leaf ids).
        If the pool size is small, enumerate all combos; otherwise, choose the most promising combos
        by product-of-priors ranking and then add random sampled combos to fill up to max_experts.
        """
        pool = list(pool)
        if len(pool) < self.r:
            combos = set()
            for comb in itertools.combinations_with_replacement(pool, self.r):
                combos.add(tuple(sorted(comb)))
            return list(combos)[:self.max_experts]

        # enumerate all combinations if feasible
        total_combos = math.comb(len(pool), self.r)
        if total_combos <= self.max_experts:
            combos = [tuple(sorted(c)) for c in itertools.combinations(pool, self.r)]
            return combos

        # Too many combos: choose top combos by product-of-priors
        # compute scores for all combos but only store top_k by score
        scored = []
        # To avoid enumerating all of them (which could still be large), enumerate combinations but cut early
        # We'll enumerate combos in lexicographic order but maintain a shortlist of top by product score.
        # If pool is reasonably small (e.g., <= 25), this is fine.
        if len(pool) <= 25:
            for comb in itertools.combinations(pool, self.r):
                prod = 1.0
                for pid in comb:
                    prod *= priors.get(pid, 1e-12)
                scored.append((prod, tuple(sorted(comb))))
            scored.sort(key=lambda x: x[0], reverse=True)
            top_by_score = [c for (_, c) in scored[:self.max_experts]]
            # if we still need more, sample random combos
            if len(top_by_score) < self.max_experts:
                needed = self.max_experts - len(top_by_score)
                sampled = set(top_by_score)
                attempts = 0
                while len(sampled) < self.max_experts and attempts < self.max_experts * 5:
                    comb = tuple(sorted(self.random.sample(pool, self.r)))
                    sampled.add(comb)
                    attempts += 1
                top_by_score = list(sampled)
            return top_by_score[:self.max_experts]
        else:
            # pool large: use heuristic - take top-`q` elements of pool and enumerate, plus random sampling
            q = min(len(pool), max(self.candidate_pool_size, 20))
            top_q = pool[:q]
            combos = []
            for comb in itertools.combinations(top_q, self.r):
                combos.append(tuple(sorted(comb)))
                if len(combos) >= self.max_experts // 2:
                    break
            # then add random sampled combos from pool to reach max_experts
            sampled = set(combos)
            attempts = 0
            while len(sampled) < self.max_experts and attempts < self.max_experts * 10:
                comb = tuple(sorted(self.random.sample(pool, self.r)))
                sampled.add(comb)
                attempts += 1
            return list(sampled)[:self.max_experts]

    def _initialize_new_experts(self, candidate_experts, priors):
        """
        Ensure expert_weights contains entries for candidate_experts. Initialize new experts
        from product of priors of constituent regions (or small positive epsilon).
        """
        eps = 1e-12
        # initialize any missing experts
        for e in candidate_experts:
            if e not in self.expert_weights:
                prod_prior = 1.0
                for pid in e:
                    prod_prior *= max(eps, priors.get(pid, eps))
                self.expert_weights[e] = prod_prior
        # normalize
        total = sum(self.expert_weights.values())
        if total > 0:
            for e in list(self.expert_weights.keys()):
                self.expert_weights[e] /= total
        else:
            # fallback uniform
            m = max(1, len(self.expert_weights))
            for e in list(self.expert_weights.keys()):
                self.expert_weights[e] = 1.0 / m

    def step(self, x):
        """
        1) Update HRD with x (may create leaves / experts)
        2) Prune stale experts
        3) Build candidate pool and candidate experts
        4) Initialize any new experts (weights)
        5) Compute per-expert losses and aggregate loss
        6) Update expert weights multiplicatively
        7) Return chosen expert (argmax weight) and its combined basis
        """
        # 1) Update HRD
        _ = self.hrd.process_point(x)

        # 2) Prune stale experts
        self._prune_stale_experts()

        # 3) Build priors and candidate pool
        priors = self._ensure_region_priors()
        pool = self._build_candidate_pool(priors)

        # 4) Construct candidate experts (list of tuples)
        candidate_experts = self._construct_candidate_experts(pool, priors)

        # 5) Initialize new experts with product-of-priors prior
        self._initialize_new_experts(candidate_experts, priors)

        # 6) Compute per-expert losses
        losses = {}
        for e in list(self.expert_weights.keys()):
            # ensure expert is still valid
            valid = True
            for pid in e:
                if pid not in self.hrd.leaves:
                    valid = False
                    break
            if not valid:
                # will be pruned next round
                losses[e] = 0.0
                continue
            # build combined basis for this expert (from the regions in e)
            basis = build_combined_basis_from_region_ids(self.hrd, e, max_dim=self.combined_basis_dim)
            losses[e] = projection_loss(basis, x)

        # 7) Aggregate loss under current expert distribution
        es = list(losses.keys())
        w_arr = np.array([self.expert_weights.get(e, 0.0) for e in es], dtype=float)
        if w_arr.sum() <= 0:
            w_arr = np.ones_like(w_arr) / len(w_arr)
        else:
            w_arr = w_arr / w_arr.sum()
        l_arr = np.array([losses[e] for e in es], dtype=float)
        agg_loss = float(np.dot(w_arr, l_arr))
        self.cum_loss.append(self.cum_loss[-1] + agg_loss)

        # 8) Multiplicative weights update for each expert
        for e in es:
            prev = max(self.expert_weights.get(e, 1e-12), 1e-12)
            self.expert_weights[e] = prev * math.exp(-self.eta * losses[e])

        # Renormalize expert weights
        total = sum(self.expert_weights.values())
        if total > 0:
            for e in list(self.expert_weights.keys()):
                self.expert_weights[e] /= total
        else:
            # fallback uniform
            m = max(1, len(self.expert_weights))
            for e in list(self.expert_weights.keys()):
                self.expert_weights[e] = 1.0 / m

        # 9) Select best expert (highest weight) as chosen representation
        if len(self.expert_weights) == 0:
            chosen = None
            combined_basis = []
        else:
            chosen = max(self.expert_weights.items(), key=lambda kv: kv[1])[0]
            combined_basis = build_combined_basis_from_region_ids(self.hrd, chosen, max_dim=self.combined_basis_dim)
        self.last_chosen_expert = chosen

        return agg_loss, chosen, combined_basis

def projection_loss(basis, x):
    """
    Squared projection loss of x onto span(basis). Basis is a list of orthonormal column vectors.
    If basis is empty, loss is ||x||^2.
    """
    if not basis:
        return float(np.dot(x, x))  # squared norm if no basis (project to zero)
    B = np.column_stack(basis)  # k x r (orthonormal columns expected)
    coeffs = B.T @ x
    proj = B @ coeffs
    return float(np.sum((x - proj) ** 2))

# ---------------------------
# Example usage / demo
# ---------------------------
def main():
    np.random.seed(0)
    random.seed(0)
    k = 5
    T = 600
    # generate streaming points concentrated around two directions (so HRD will refine)
    centers = [normalize(np.array([1.0] + [0.0] * (k - 1))), normalize(np.array([0.0, 1.0] + [0.0] * (k - 2)))]
    x_sequence = []
    for t in range(T):
        if np.random.rand() < 0.6:
            base = centers[0]
        else:
            base = centers[1]
        v = base + 0.2 * np.random.randn(k)
        x_sequence.append(normalize(v))

    # Create HRD and ExpertMWUA
    hrd = SphericalHRD(k=k, d_split=4, r_expert=2,
                       n_min=10, epsilon_hrd=0.2, n_max_leaf=200, coreset_capacity=50)
    # ExpertMWUA: each expert is a set of r_expert regions (we choose r_expert=2 for demo)
    mw = ExpertMWUA(hrd, eta=0.5, r_expert=2,
                    candidate_pool_size=12, max_experts=300, combined_basis_dim=3, random_seed=0)

    # Run online
    agg_losses = []

    for i, x in enumerate(x_sequence):
        agg_loss, chosen_expert, comb_basis = mw.step(x)
        # occasionally print progress
        if i % 100 == 0:
            print(f"t={i}, agg_loss={agg_loss:.4f}, #leaves={len(hrd.leaves)}, #experts={len(mw.expert_weights)}")
        agg_losses.append(agg_loss)

    # Print final stats
    print("Final chosen expert (region ids):", mw.last_chosen_expert)
    print("Combined basis dim:", 0 if comb_basis is None else len(comb_basis))
    print("Total leaves:", len(hrd.leaves))
    sorted_weights = sorted(mw.expert_weights.items(), key=lambda kv: kv[1], reverse=True)[:10]
    print("Top expert weights (top 10):")
    for eid, w in sorted_weights:
        print("expert", eid, "weight", w)
    # Plot cumulative loss
    plt.plot(mw.cum_loss[1:], label='Cumulative Expert-MWUA loss')
    plt.xlabel('t')
    plt.ylabel('Cumulative loss')
    plt.title('Expert-MWUA over Spherical HRD (demo) — experts = r-region sets (coreset centers)')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()