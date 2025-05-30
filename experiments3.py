import pandas as pd
import numpy as np
import time
from gurobipy import Model, GRB, quicksum, QuadExpr

# --- 1. Data Loading & Matrix Construction ---
def load_data(pv_path, cm_path):
    pv = pd.read_csv(pv_path)
    cm = pd.read_csv(cm_path)
    participants = pv['participant'].tolist()
    n = len(participants)
    # Votes matrix
    vote_cols = [c for c in pv.columns if c.isdigit()]
    votes = pv[vote_cols].fillna(0).values
    norms = np.linalg.norm(votes, axis=1)
    A = (votes @ votes.T) / (norms[:, None] * norms[None, :] + 1e-9) # cosine similarity
    D = 1 - A
    E = (pv['n-comments'] + pv['n-votes']).values
    S = np.sign(pv['n-agree'] - pv['n-disagree']).astype(int)
    C = (pv['n-comments'] > 0).astype(int)
    return participants, n, A, D, E, S, C

# --- 2. Gurobi IP Solver with logging & timing ---
def solve_ip(n, m, D, E, S_min, S_max, delta, eta, lambda1=1.0, lambda2=0.1):
    t0 = time.perf_counter()
    print("[IP-V3] Building model...")
    model = Model()
    x = model.addVars(n, m, vtype=GRB.BINARY, name="x")

    # --- Objective Function ---
    diversity = QuadExpr()
    for j in range(m):
        for i in range(n):
            for k in range(i + 1, n):
                diversity.add(D[i, k] * x[i, j] * x[k, j])

    avgE = E.sum() / m
    varE = QuadExpr()
    for j in range(m):
        lin = quicksum(E[i] * x[i, j] for i in range(n)) - avgE
        varE.add(lin * lin)

    model.setObjective(lambda1 * diversity - lambda2 * varE, GRB.MAXIMIZE)

    # --- Constraints ---

    # (1) Unique Assignment
    for i in range(n):
        model.addConstr(x.sum(i, '*') == 1, name=f"assign_{i}")

    # (2) Group Size Bounds + Odd Cardinality
    y = model.addVars(m, vtype=GRB.INTEGER, lb=0, name="y")  # For odd check
    for j in range(m):
        size = x.sum('*', j)
        model.addConstr(size >= S_min, name=f"size_min_{j}")
        model.addConstr(size <= S_max, name=f"size_max_{j}")
        model.addConstr(size - 1 == 2 * y[j], name=f"odd_size_{j}")

    # (3) Intra-group Diversity Upper Bound
    for j in range(m):
        div = QuadExpr()
        for i in range(n):
            for k in range(i + 1, n):
                div.add(D[i, k] * x[i, j] * x[k, j])
        model.addQConstr(div <= delta, name=f"diversity_max_{j}")

    # (4) Minimum Group Engagement
    for j in range(m):
        eng = quicksum(E[i] * x[i, j] for i in range(n))
        model.addConstr(eng >= eta, name=f"engagement_min_{j}")

    print(f"[IP-V3] Model built: {model.NumVars} vars, {model.NumConstrs} constrs")
    print("[IP-V3] Starting optimization...")
    model.Params.OutputFlag = 0
    model.optimize()
    print(f"[IP-V3] Optimization complete. ObjVal = {model.ObjVal:.4f}")
    sol = np.array([[x[i, j].X for j in range(m)] for i in range(n)])
    assign = sol.argmax(axis=1)
    t1 = time.perf_counter()
    return assign, t1 - t0


# --- 3. LP Relaxation + Rounding with logging & timing ---
def solve_lp_rounding(n, m, D, E, S_min, S_max, delta, eta, lambda1=1.0, lambda2=0.1):
    t0 = time.perf_counter()
    print("[LP-V3] Building model...")
    model = Model()
    x = model.addVars(n, m, lb=0, ub=1, name="x")

    # Objective
    diversity = QuadExpr()
    for j in range(m):
        for i in range(n):
            for k in range(i + 1, n):
                diversity.add(D[i, k] * x[i, j] * x[k, j])
    avgE = E.sum() / m
    varE = QuadExpr()
    for j in range(m):
        lin = quicksum(E[i] * x[i, j] for i in range(n)) - avgE
        varE.add(lin * lin)

    model.setObjective(lambda1 * diversity - lambda2 * varE, GRB.MAXIMIZE)

    # Constraints
    for i in range(n):
        model.addConstr(x.sum(i, '*') == 1, name=f"assign_{i}")
    for j in range(m):
        size = x.sum('*', j)
        model.addConstr(size >= S_min, name=f"size_min_{j}")
        model.addConstr(size <= S_max, name=f"size_max_{j}")
        div = QuadExpr()
        for i in range(n):
            for k in range(i + 1, n):
                div.add(D[i, k] * x[i, j] * x[k, j])
        model.addQConstr(div <= delta, name=f"diversity_max_{j}")
        eng = quicksum(E[i] * x[i, j] for i in range(n))
        model.addConstr(eng >= eta, name=f"engagement_min_{j}")

    print(f"[LP-V3] Model built: {model.NumVars} vars, {model.NumConstrs} constrs")
    print("[LP-V3] Starting optimization...")
    model.Params.OutputFlag = 0
    model.optimize()
    print(f"[LP-V3] Optimization complete. ObjVal = {model.ObjVal:.4f}")
    sol = np.array([[x[i, j].X for j in range(m)] for i in range(n)])
    assign = sol.argmax(axis=1)
    t1 = time.perf_counter()
    return assign, t1 - t0


# --- 4. Greedy Local Search Heuristic with logging & timing ---
def heuristic_greedy(n, m, A, D, E, S, C, S_min, S_max):
    t0 = time.perf_counter()
    print("[Heuristic] Starting greedy assignment...")
    assign = np.random.choice(m, size=n)
    counts = np.bincount(assign, minlength=m)
    # Repair for size constraints
    for j in range(m):
        while counts[j] < S_min:
            i = np.random.choice(np.where(counts[assign] > S_min)[0])
            counts[assign[i]] -= 1
            assign[i] = j
            counts[j] += 1
        while counts[j] > S_max:
            i = np.random.choice(np.where(assign == j)[0])
            newj = np.random.choice([g for g in range(m) if counts[g] < S_max])
            assign[i] = newj
            counts[j] -= 1; counts[newj] += 1
    print("[Heuristic] Initial feasible assignment done.")
    # Local search
    def obj_val(a):
        val = 0.0
        for j in range(m):
            idxs = np.where(a == j)[0]
            for u in idxs:
                for v in idxs:
                    val += A[u, v]
        return val
    improved = True
    while improved:
        improved = False
        base = obj_val(assign)
        for i in range(n):
            orig = assign[i]
            for j in range(m):
                if j == orig: continue
                candidate = assign.copy()
                candidate[i] = j
                if obj_val(candidate) > base:
                    assign = candidate
                    improved = True
                    break
            if improved: break
    print("[Heuristic] Local search complete.")
    t1 = time.perf_counter()
    return assign, t1 - t0

# --- 5. Metrics Calculation ---
def compute_metrics(assign, A, D, E):
    m = assign.max() + 1
    intra = sum(A[i, k] for i in range(len(assign)) for k in range(len(assign)) if assign[i] == assign[k])
    inter = sum(D[i, k] for i in range(len(assign)) for k in range(len(assign)) if assign[i] != assign[k])
    es = [E[assign == j].sum() for j in range(m)]
    varE = np.var(es)
    return {'intra_agreement': intra, 'inter_polarization': inter, 'engagement_var': varE}

# --- 6. Experiment Runner with timing comparison ---
def run_experiments(pv_path, cm_path):
    participants, n, A, D, E, S, C = load_data(pv_path, cm_path)
    # Settings
    m, S_min, S_max, delta, theta = 5, 5, 15, 10.0, 1
    methods = {
        'IP':      lambda: solve_ip(n, m, A, D, E, S, C, S_min, S_max, delta, theta),
        'LP':      lambda: solve_lp_rounding(n, m, A, D, E, S, C, S_min, S_max, delta, theta),
        'Heuristic': lambda: heuristic_greedy(n, m, A, D, E, S, C, S_min, S_max)
    }
    records = []
    for name, fn in methods.items():
        print(f"\n===== Running {name} =====")
        assign, elapsed = fn()
        print(f"[{name}] Elapsed time: {elapsed:.2f} sec")
        metrics = compute_metrics(assign, A, D, E)
        metrics['time_sec'] = elapsed
        records.append((name, metrics))
    df = pd.DataFrame({name: vals for name, vals in records}).T
    print("\n===== Summary =====")
    print(df)
    return df

if __name__ == '__main__':
    run_experiments('participants-votes.csv', 'comments.csv')
