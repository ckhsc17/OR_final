"""
function LAGRANGIAN_ASSIGNMENT(n, m, D, E, S_min, S_max, delta, eta, λ1, λ2):
    # Step 1: Initialization
    Initialize assign[i] randomly to one of m groups
    Initialize counts[j] = number of participants in group j
    Initialize μ[j] = 0 for all j in 1..m
    best_feasible = None
    best_feasible_obj = -∞

    # Step 2: Outer loop - update Lagrange multipliers
    for outer in 1 to max_outer:
        
        # Step 3: Inner loop - local search under fixed μ
        for inner in 1 to max_inner:
            Select random participant i and random candidate group new_j
            If new_j == current group or violates group size constraints:
                continue

            Make temporary assignment: move i → new_j
            Compute Lagrangian objective with current μ

            If objective improves:
                Accept move (update assign and counts)

        # Step 4: Lagrange multiplier update
        for each group j in 1..m:
            Compute diversity_j = ∑ D[i,k] for all i,k in group j
            Compute engagement_j = ∑ E[i] for i in group j

            If diversity_j ≤ δ and engagement_j ≥ η:
                If current objective > best_feasible_obj:
                    Save assign as best_feasible
                    Save current objective

            μ[j] = max(0, μ[j] + step_size / sqrt(outer) * (δ - diversity_j))

    # Step 5: Finalize result
    If best_feasible exists:
        final_assign = best_feasible
    else:
        final_assign = current assign

    # Step 6: Post-adjustment - fix violations
    for t in 1 to post_adjust_max_iter:
        violations_fixed = True
        for group j in random order:
            If diversity_j > δ or engagement_j < η:
                violations_fixed = False
                Identify worst participant in group j (based on contribution to D)
                Try moving them to a group k with space and better D

                If successful:
                    Update assign and counts

        If violations_fixed:
            Break

    return final_assign, best_feasible_obj
"""
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
# Heuristic greedy method for Lagrangian-based group assignment
# ──────────────────────────────────────────────────────────────
def lagrangian_heuristic_v3(n, m, D, E, S_min, S_max,
                            lambda_1=1.0, lambda_2=.05,
                            delta=105.0, eta=10.0,
                            max_outer=100, max_inner=1000,
                            step_size=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)  # For reproducibility

    # Initial random assignment of participants to groups
    assign = np.random.choice(m, size=n)
    counts = np.bincount(assign, minlength=m)  # Count participants per group
    mu = np.zeros(m)  # Lagrange multipliers for diversity constraint

    # Compute intra-group diversity for group j
    def group_diversity(a, j):
        idx = np.where(a == j)[0]
        if len(idx) < 2:
            return 0.0
        dij = D[np.ix_(idx, idx)]
        return dij[np.triu_indices_from(dij, k=1)].sum()

    # Lagrangian objective function: reward diversity, penalize imbalance, relax constraint
    def lagrangian_obj(a, mu):
        diversity = sum(group_diversity(a, j) for j in range(m))
        imbalance = sum((E[a == j].sum() - E.sum()/m) ** 2 for j in range(m))
        penalty = sum(mu[j] * (delta - group_diversity(a, j)) for j in range(m))
        return lambda_1 * diversity - lambda_2 * imbalance + penalty

    # Track best assignment found so far
    best_val = lagrangian_obj(assign, mu)

    # Track best *feasible* assignment (satisfying constraints)
    overall_best_feasible_assign = None
    overall_best_feasible_obj_val = -np.inf

    # Outer loop: update μ (dual variables)
    for outer in tqdm(range(max_outer), desc="Outer (μ) loop"):
        # Inner loop: local greedy search to improve assignment
        for _ in range(max_inner):
            i, new_group = np.random.randint(n), np.random.randint(m)
            old_group = assign[i]

            # Skip if move violates size constraints
            if new_group == old_group or counts[new_group] >= S_max or counts[old_group] <= S_min:
                continue

            # Try candidate move
            cand = assign.copy()
            cand[i] = new_group
            cand_val = lagrangian_obj(cand, mu)

            # Accept move if objective improves
            if cand_val > best_val:
                assign = cand
                best_val = cand_val
                counts[old_group] -= 1
                counts[new_group] += 1

        # Update μ and track best feasible solution
        for j in range(m):
            div_j = group_diversity(assign, j)
            e_sum = E[assign == j].sum()

            # Check if group j satisfies constraints
            if div_j <= delta and e_sum >= eta:
                if best_val > overall_best_feasible_obj_val:
                    overall_best_feasible_assign = assign.copy()
                    overall_best_feasible_obj_val = best_val

            # Update Lagrange multiplier (subgradient step)
            mu[j] = max(0, mu[j] + (step_size / (1 + outer**0.5)) * (delta - div_j))

    # Use best feasible assignment if found
    final_assign = overall_best_feasible_assign if overall_best_feasible_assign is not None else assign.copy()

    # ───────────────────────────────────────────────────────
    # Post-adjustment phase: fix diversity / engagement violations
    # ───────────────────────────────────────────────────────
    print("\n--- Post-adjustment phase ---")
    counts = np.bincount(final_assign, minlength=m)
    for _ in range(20):
        violations_fixed = True
        for j in np.random.permutation(m):
            idx = np.where(final_assign == j)[0]
            if len(idx) == 0:
                continue

            div_j = group_diversity(final_assign, j)
            e_sum = E[idx].sum()

            # If group j violates diversity or engagement constraint
            if div_j > delta or e_sum < eta:
                violations_fixed = False

                # Identify worst offender in group (most conflict)
                worst_i = idx[np.argmax(D[idx, :][:, idx].sum(axis=1))]

                # Try to move worst_i to a different group
                for k in np.random.permutation(m):
                    if k != j and counts[k] < S_max:
                        trial = final_assign.copy()
                        trial[worst_i] = k
                        if group_diversity(trial, k) <= delta:
                            final_assign[worst_i] = k
                            counts[j] -= 1
                            counts[k] += 1
                            break

        if violations_fixed:
            print("Post-adjustment succeeded in fixing violations.")
            break
    else:
        print("Post-adjustment reached limit, some violations may remain.")

    return final_assign, overall_best_feasible_obj_val

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
        'Heuristic_Composite': lambda: lagrangian_heuristic_v3(
            n, m, D, E, S_min, S_max, lambda_1, lambda_2)[0]
    }

    # 6. Run experiments and summarize
    records = []
    for name, fn in methods.items():
        print(f"\n===== {name} =====")
        t0 = time.perf_counter()
        assign = fn()
        elapsed = time.perf_counter() - t0
        metrics = compute_metrics(assign, A, D, E)
        metrics['time_sec'] = elapsed
        records.append((name, metrics))
        print(f"{name} finished in {elapsed:.2f}s, metrics → {metrics}")

    summary = pd.DataFrame({k: v for k, v in records}).T
    print("\n===== Summary =====")
    print(summary)
    return summary

# ──────────────────────────────────────────────────────────────
# Command line interface
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    run_experiments('participants-votes.csv', 'comments.csv')

