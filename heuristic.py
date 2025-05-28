import numpy as np
import pandas as pd
import time
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# Auto adjust group parameters to ensure m*S_min <= n <= m*S_max.
# If n > m*S_max: prioritize increasing m; if prefer_fix_m=False, increase S_max instead.
# If n < m*S_min: decrease S_min accordingly (minimum 1).
# ──────────────────────────────────────────────────────────────
def auto_adjust_group_params(n, m_init, S_min_init, S_max_init,
                             prefer_fix_m=True):
    import math
    m, S_min, S_max = m_init, S_min_init, S_max_init

    if n > m * S_max:
        if prefer_fix_m:
            m = math.ceil(n / S_max)
        else:
            S_max = math.ceil(n / m)
    if n < m * S_min:
        S_min = max(1, math.floor(n / m))

    return m, S_min, S_max

# ──────────────────────────────────────────────────────────────
# Load participant votes data as sparse matrix and compute agreement/dissimilarity.
# Also extract engagement E, sentiment S, comment indicator C.
# ──────────────────────────────────────────────────────────────
def load_data_sparse(pv_path: str, cm_path: str):
    pv = pd.read_csv(pv_path)
    cm = pd.read_csv(cm_path)  # Currently unused

    participants = pv['participant'].tolist()
    n = len(participants)

    vote_cols = [c for c in pv.columns if c.isdigit()]
    votes_sparse = csr_matrix(pv[vote_cols].fillna(0).values)

    # Agreement matrix based on cosine similarity of vote vectors
    A = cosine_similarity(votes_sparse)
    D = 1 - A  # Dissimilarity matrix

    E = (pv['n-comments'] + pv['n-votes']).values        # Engagement measure
    S = np.sign(pv['n-agree'] - pv['n-disagree']).astype(int)  # Sentiment (-1,0,1)
    C = (pv['n-comments'] > 0).astype(int)               # Comment presence indicator

    return participants, n, A, D, E, S, C

# ──────────────────────────────────────────────────────────────
# Calibrate safety thresholds delta (max intra-group dissimilarity)
# and eta (minimum engagement per group)
# ──────────────────────────────────────────────────────────────
def calibrate_params(D: np.ndarray, E: np.ndarray,
                     S_max: int, m: int,
                     q: float = 0.85, eta_ratio: float = 0.7):
    pair_max = S_max * (S_max - 1) / 2
    upper_d = np.quantile(D[np.triu_indices_from(D, k=1)], q)
    delta = upper_d * pair_max

    eta = eta_ratio * E.sum() / m
    return delta, eta

# ──────────────────────────────────────────────────────────────
# Heuristic greedy method to assign participants to groups
# optimizing diversity minus engagement imbalance.
# Includes careful enforcement of group size limits.
# ──────────────────────────────────────────────────────────────
def heuristic_greedy_composite(n, m, D, E, S_min, S_max,
                               lambda_1=1.0, lambda_2=1.0,
                               max_iter: int = 2000):

    # Initial random assignment
    assign = np.random.choice(m, size=n)
    counts = np.bincount(assign, minlength=m)

    # Fix group sizes deterministically: transfer from largest to smallest groups
    def balance_groups(assign, counts):
        for _ in range(2000):  # avoid infinite loop
            too_small = np.where(counts < S_min)[0]
            too_large = np.where(counts > S_max)[0]
            if len(too_small) == 0 and len(too_large) == 0:
                break
            for j in too_small:
                # Move one participant from a too-large group to j
                donors = [g for g in too_large if counts[g] > S_max]
                if not donors:
                    break  # no donors available
                donor = donors[0]
                candidates = np.where(assign == donor)[0]
                if len(candidates) == 0:
                    continue
                i = candidates[0]
                assign[i] = j
                counts[donor] -= 1
                counts[j] += 1
        return assign, counts

    assign, counts = balance_groups(assign, counts)

    # Objective function: diversity - imbalance penalty
    def obj(a):
        diversity = 0.0
        imbalance = 0.0
        e_avg = E.sum() / m
        for j in range(m):
            idx = np.where(a == j)[0]
            if len(idx) < 2:
                continue
            dij = D[np.ix_(idx, idx)]
            diversity += dij[np.triu_indices_from(dij, k=1)].sum()
            e_sum = E[idx].sum()
            imbalance += abs(e_sum - e_avg)
        return lambda_1 * diversity - lambda_2 * imbalance

    best = assign.copy()
    best_val = obj(best)

    for _ in tqdm(range(max_iter), desc="Optimizing groups"):
        i = np.random.randint(n)
        old_group = best[i]
        new_group = np.random.randint(m)

        # Skip if no change or would violate group size constraints
        if new_group == old_group:
            continue
        if counts[new_group] >= S_max or counts[old_group] <= S_min:
            continue

        cand = best.copy()
        cand[i] = new_group

        cand_val = obj(cand)
        if cand_val > best_val:
            best = cand
            best_val = cand_val
            counts[old_group] -= 1
            counts[new_group] += 1

    return best, best_val

# ──────────────────────────────────────────────────────────────
# Compute metrics: intra-group agreement, inter-group polarization,
# and variance of engagement across groups.
# Vectorized implementation for efficiency.
# ──────────────────────────────────────────────────────────────
def compute_metrics(assign, A, D, E):
    m = assign.max() + 1
    intra_agreement = 0.0
    inter_polarization = 0.0

    for j in range(m):
        idx = np.where(assign == j)[0]
        if len(idx) > 1:
            intra_agreement += A[np.ix_(idx, idx)].sum()

    for j in range(m):
        for k in range(j + 1, m):
            idx_j = np.where(assign == j)[0]
            idx_k = np.where(assign == k)[0]
            if len(idx_j) > 0 and len(idx_k) > 0:
                inter_polarization += D[np.ix_(idx_j, idx_k)].sum()

    engagement_sums = np.array([E[assign == j].sum() for j in range(m)])
    engagement_var = np.var(engagement_sums)

    return dict(intra_agreement=intra_agreement,
                inter_polarization=inter_polarization,
                engagement_var=engagement_var)

# ──────────────────────────────────────────────────────────────
# Main experiment runner — loads data, adjusts parameters,
# runs heuristic assignment and reports metrics.
# ──────────────────────────────────────────────────────────────
def run_experiments(pv_path='participants-votes.csv',
                    cm_path='comments.csv'):

    # 1. Load data
    participants, n, A, D, E, S, C = load_data_sparse(pv_path, cm_path)

    # 2. Adjust number of groups and size limits
    m_init, S_min_init, S_max_init = 2, 5, 15
    m, S_min, S_max = auto_adjust_group_params(
        n, m_init, S_min_init, S_max_init, prefer_fix_m=True)
    print(f"[Info] n={n}, m={m}, S_min={S_min}, S_max={S_max}")

    # 3. Calibrate delta and eta
    delta, eta = calibrate_params(D, E, S_max, m)
    print(f"[Info] delta={delta:.4f}, eta={eta:.4f}")

    # 4. Set lambda parameters (can be tuned)
    lambda_1, lambda_2 = 0.8, 1.2

    # 5. Define heuristic method
    methods = {
        'Heuristic_Composite': lambda: heuristic_greedy_composite(
            n, m, D, E, S_min, S_max, lambda_1, lambda_2)
    }

    # 6. Run experiments and summarize
    records = []
    for name, fn in methods.items():
        print(f"\n===== {name} =====")
        t0 = time.perf_counter()
        assign, obj_val = fn()
        elapsed = time.perf_counter() - t0
        metrics = compute_metrics(assign, A, D, E)
        metrics['time_sec'] = elapsed
        metrics['objective'] = obj_val
        records.append((name, metrics))
        print(f"{name} finished in {elapsed:.2f}s, metrics → {metrics}")

    summary = pd.DataFrame({k: v for k, v in records}).T
    print("\n===== Summary =====")
    print(summary)
    return summary

# Solve instance
def solve_heuristic_instance(n: int,
                             D: np.ndarray,
                             E: np.ndarray,
                             smax: int,
                             lambda1: float = 0.8,
                             lambda2: float = 1.2,
                             max_iter: int = 2000,
                             seed: int | None = None
) -> tuple[np.ndarray, float, int, int, int]:
    """
    Run the heuristic greedy composite algorithm on a generated instance.

    Args:
        n: number of participants
        D: (n×n) dissimilarity matrix
        E: (n,) engagement vector
        smax: maximum group size bound
        lambda1, lambda2: objective weights
        max_iter: number of local search iterations
        seed: optional random seed for reproducibility

    Returns:
        assign: array of length n with group assignments [0..m-1]
        obj_val: objective value of returned assignment
        m: number of groups
        S_min: minimum group size
        S_max: maximum group size (equals smax)
    """
    # 1. set RNG
    if seed is not None:
        np.random.seed(seed)

    # 2. auto adjust group parameters
    m_init = math.ceil(n / smax)
    S_min_init = max(1, math.floor(n / m_init))
    S_max_init = smax
    m, S_min, S_max = auto_adjust_group_params(
        n, m_init, S_min_init, S_max_init, prefer_fix_m=True
    )

    # 3. optional: calibrate thresholds (not used by heuristic_greedy_composite)
    _delta, _eta = calibrate_params(D, E, S_max, m)

    # 4. run heuristic
    assign, obj_val = heuristic_greedy_composite(
        n=n,
        m=m,
        D=D,
        E=E,
        S_min=S_min,
        S_max=S_max,
        lambda_1=lambda1,
        lambda_2=lambda2,
        max_iter=max_iter
    )

    return assign, obj_val, m, S_min, S_max

# ──────────────────────────────────────────────────────────────
# Command line interface
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    run_experiments('participants-votes.csv', 'comments.csv')
