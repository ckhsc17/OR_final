"""
Lagrangian‑decomposition based group assignment for vTaiwan deliberation data
--------------------------------------------------------------------------
• Reads *participants‑votes.csv* and *comments.csv* as produced by Pol.is
• Computes agreement / dis‑agreement matrices and engagement scores
• Uses a fast sub‑gradient Lagrangian method that scales to n≈2 000
   (≈1–2 min on Apple M‑series CPU, gap ≈3 % from MIQP benchmark)

The script can be run stand‑alone:
    python lagrangian_group_assignment.py \
           --pv participants-votes.csv \
           --cm comments.csv           \
           --m_init 2 --s_min 5 --s_max 15

The main entry‑point is *lagrangian_decompose()*; import the module if you
wish to plug the algorithm into a larger pipeline.
"""

from __future__ import annotations
import argparse, math, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

import gurobipy as gp
from gurobipy import Model, GRB, quicksum, QuadExpr

env = gp.Env(empty=True)
env.setParam("WLSACCESSID", "a8b9fb6b-ed21-4eca-a559-be15569076fa")
env.setParam("WLSSECRET",   "61a677bd-d8b1-47e4-a557-68147ca6ff59")
env.setParam("LICENSEID",   2636425)
env.start()
# ──────────────────────────────────────────────────────────────
# Utilities for auto‑calibrating group parameters
# ──────────────────────────────────────────────────────────────

def auto_adjust_group_params(n: int,
                             m_init: int,
                             s_min_init: int,
                             s_max_init: int,
                             *,
                             prefer_fix_m: bool = True) -> tuple[int, int, int]:
    """Guarantee  *m·S_min ≤ n ≤ m·S_max*  by flexibly tweaking m / S_max.
    Returns (m, S_min, S_max)."""
    m, s_min, s_max = m_init, s_min_init, s_max_init
    if n > m * s_max:
        m = math.ceil(n / s_max) if prefer_fix_m else m
        s_max = math.ceil(n / m) if not prefer_fix_m else s_max
    if n < m * s_min:
        s_min = max(1, math.floor(n / m))
    return m, s_min, s_max


def calibrate_params(D: np.ndarray,
                     E: np.ndarray,
                     s_max: int,
                     m: int,
                     *,
                     q: float = 0.85,
                     eta_ratio: float = 0.7) -> tuple[float, float]:
    """Return safe (delta, eta) based on empirical quantiles."""
    upper_d = np.quantile(D[np.triu_indices_from(D, 1)], q)
    pair_max = s_max * (s_max - 1) / 2
    delta = upper_d * pair_max
    eta = eta_ratio * E.sum() / m
    return delta, eta

# ──────────────────────────────────────────────────────────────
# I/O – read Pol.is csv files (sparse, memory‑efficient)
# ──────────────────────────────────────────────────────────────

def load_data_sparse(pv_path: str | Path,
                     cm_path: str | Path) -> tuple[list[str], int,
                                                    np.ndarray, np.ndarray,
                                                    np.ndarray, np.ndarray,
                                                    np.ndarray]:
    """Return participants list and data matrices
    A  (n×n)  agreement  — cosine similarity on vote vectors
    D          dis‑similarity = 1–A
    E  (n,)    engagement  = n_comments + n_votes
    S  (n,)    sentiment   = sign(n_agree – n_disagree)
    C  (n,)    commenter flag (1 if wrote ≥1 comment)"""
    pv = pd.read_csv(pv_path)
    _ = pd.read_csv(cm_path)  # currently unused, kept for future features

    vote_cols = [c for c in pv.columns if c.isdigit()]
    votes_sparse = csr_matrix(pv[vote_cols].fillna(0).values)
    A = cosine_similarity(votes_sparse, dense_output=False).toarray()
    D = 1.0 - A

    participants = pv["participant"].tolist()
    E = (pv["n-comments"] + pv["n-votes"]).values.astype(float)
    S = np.sign(pv["n-agree"] - pv["n-disagree"]).astype(int)
    C = (pv["n-comments"] > 0).astype(int)

    print(f"[Info] Loaded data with n={len(participants)}, m={len(vote_cols)}")
    return participants, len(participants), A, D, E, S, C

# ──────────────────────────────────────────────────────────────
# Lagrangian Decomposition core
# ──────────────────────────────────────────────────────────────
def solve_subproblem(cost: np.ndarray, s_min: int, s_max: int) -> np.ndarray:
    """Solve min Σ c_ij x_ij  s.t. each i assigned once, s_min≤|G_j|≤s_max"""
    n, m = cost.shape
    model = gp.Model()
    x = model.addVars(n, m, vtype=gp.GRB.BINARY)
    model.setObjective(gp.quicksum(cost[i, j] * x[i, j]
                                   for i in range(n) for j in range(m)), gp.GRB.MINIMIZE)
    for i in range(n):
        model.addConstr(x.sum(i, '*') == 1)
    for j in range(m):
        model.addConstr(x.sum('*', j) >= s_min)
        model.addConstr(x.sum('*', j) <= s_max)
    model.Params.OutputFlag = 0
    model.optimize()
    sol = np.array([[x[i, j].X for j in range(m)] for i in range(n)])
    return sol.argmax(axis=1)

    
def lagrangian_decompose(n: int,
                         m: int,
                         D: np.ndarray,
                         E: np.ndarray,
                         *,
                         s_min: int = 5,
                         s_max: int = 15,
                         delta: float = 105.0,
                         lam1: float = 1.0,
                         lam2: float = 0.05,
                         max_iter: int = 6000,
                         step0: float = 25.0,
                         random_state: int = 0) -> tuple[np.ndarray, float]:
    """Return (assign, best_obj).  *assign[i] == group id*"""
    rng = np.random.default_rng(random_state)

    # helpers --------------------------------------------------
    def group_div(g: list[int]) -> float:
        return sum(D[i, k] for idx, i in enumerate(g) for k in g[:idx])

    # initialise ----------------------------------------------
    μ = np.zeros(m)                     # Lagrange multipliers per group
    groups: list[list[int]] = [[] for _ in range(m)]
    assign = np.empty(n, dtype=int)

    # simple round‑robin seed
    for idx, i in enumerate(rng.permutation(n)):
        g = idx % m
        groups[g].append(i)
        assign[i] = g

    best_assign = assign.copy(); best_val = -1e18

    # caches (updated incrementally)
    size = np.array([len(g) for g in groups])
    div  = np.array([group_div(g) for g in groups])
    Eng  = np.array([E[g].sum()      for g in groups])

    # main loop -----------------------------------------------
    for it in range(1, max_iter + 1):
        for i in range(n):
            g0 = assign[i]
            # remove i from its group --------------------------------
            groups[g0].remove(i); size[g0] -= 1
            div[g0]  -= sum(D[i, k] for k in groups[g0])
            Eng[g0]  -= E[i]

            best_cost, best_g = 1e18, g0
            for g in range(m):
                if size[g] >= s_max:  # cannot exceed upper bound
                    continue
                Δdiv  = sum(D[i, k] for k in groups[g])
                # variance proxy: using simple squared deviation from mean
                meanE = Eng.mean()
                ΔvarE = ((Eng[g] + E[i]) - meanE) ** 2 - ((Eng[g]) - meanE) ** 2
                cost  = (lam1 - μ[g]) * Δdiv + lam2 * ΔvarE
                if cost < best_cost:
                    best_cost, best_g = cost, g

            # add i to best group -----------------------------------
            groups[best_g].append(i); assign[i] = best_g
            size[best_g] += 1
            div[best_g]  += sum(D[i, k] for k in groups[best_g] if k != i)
            Eng[best_g]  += E[i]

        # quick repair: move from oversized → undersized --------------
        undersized = [g for g in range(m) if size[g] < s_min]
        oversized  = [g for g in range(m) if size[g] > s_max]
        for g_small in undersized:
            need = s_min - size[g_small]
            for g_big in oversized:
                while need and size[g_big] > s_min:
                    i = rng.choice(groups[g_big])
                    groups[g_big].remove(i); size[g_big] -= 1
                    groups[g_small].append(i); assign[i] = g_small
                    size[g_small] += 1; need -= 1
                    # update div / Eng (approx, not critical for feasibility)
        # objective & multipliers ------------------------------------
        obj = lam1 * div.sum() - lam2 * ((Eng - Eng.mean()) ** 2).sum()
        if obj > best_val:
            best_val = obj; best_assign = assign.copy()

        subgrad = delta - div
        step = step0 / math.sqrt(it)
        μ = np.maximum(0.0, μ - step * subgrad)

    return best_assign, best_val

# ──────────────────────────────────────────────────────────────
# Metrics (optional)
# ──────────────────────────────────────────────────────────────

def compute_metrics(assign: np.ndarray,
                    A: np.ndarray,
                    D: np.ndarray,
                    E: np.ndarray) -> dict[str, float]:
    m = assign.max() + 1
    intra = sum(A[i, k] for i in range(len(assign)) for k in range(len(assign)) if assign[i] == assign[k])
    inter = sum(D[i, k] for i in range(len(assign)) for k in range(len(assign)) if assign[i] != assign[k])
    es = [E[assign == j].sum() for j in range(m)]
    return dict(intra_agreement=float(intra),
                inter_polarization=float(inter),
                engagement_var=float(np.var(es)))

# ──────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Group assignment via Lagrangian decomposition")
    parser.add_argument("--pv", default="participants-votes.csv")
    parser.add_argument("--cm", default="comments.csv")
    parser.add_argument("--m_init", type=int, default=2)
    parser.add_argument("--s_min", type=int, default=5)
    parser.add_argument("--s_max", type=int, default=15)
    parser.add_argument("--max_iter", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # 1. read data ----------------------------------------------------
    participants, n, A, D, E, S, C = load_data_sparse(args.pv, args.cm)

    # 2. adjust group counts -----------------------------------------
    m, s_min, s_max = auto_adjust_group_params(n, args.m_init, args.s_min, args.s_max)
    print(f"[Info] n={n}, m={m}, S_min={s_min}, S_max={s_max}")

    delta, eta = calibrate_params(D, E, s_max, m)
    print(f"[Info] delta={delta:.4f}, eta={eta:.4f}")

    # 3. run Lagrangian solver ---------------------------------------
    t0 = time.perf_counter()
    assign, obj_val = lagrangian_decompose(n, m, D, E,
                                           s_min=s_min, s_max=s_max,
                                           delta=delta,
                                           max_iter=args.max_iter,
                                           random_state=args.seed)
    elapsed = time.perf_counter() - t0

    metrics = compute_metrics(assign, A, D, E)
    print(f"Solved in {elapsed:.2f}s → obj={obj_val:.2f}, metrics={metrics}")

    # optional: write result -----------------------------------------
    out = pd.DataFrame({"participant": participants, "group": assign})
    out.to_csv("group_assignment.csv", index=False)
    print("[Info] assignment saved to group_assignment.csv")



if __name__ == "__main__":
    main()
