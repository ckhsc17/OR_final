import numpy as np
import pandas as pd
from gurobipy import Model, GRB, quicksum
import gurobipy as gp
from pathlib import Path

# ─── Gurobi 雲端授權環境 ───
env = gp.Env(empty=True)
env.setParam("WLSACCESSID", "a8b9fb6b-ed21-4eca-a559-be15569076fa")
env.setParam("WLSSECRET",   "61a677bd-d8b1-47e4-a557-68147ca6ff59")
env.setParam("LICENSEID",   2636425)
env.start()

# ─── Data loading ───
def load_data(pv_path: str, cm_path: str):
    pv = pd.read_csv(pv_path)
    vote_cols = [c for c in pv.columns if c.isdigit()]
    votes = pv[vote_cols].fillna(0).values
    participants = pv['participant'].tolist()

    from scipy.sparse import csr_matrix
    from sklearn.metrics.pairwise import cosine_similarity

    votes_sparse = csr_matrix(votes)
    A = cosine_similarity(votes_sparse)
    D = 1 - A
    E = (pv["n-comments"] + pv["n-votes"]).values
    S = np.sign(pv["n-agree"] - pv["n-disagree"]).astype(int)
    return participants, D, A, E, S

# ─── Solver for v4 ───
def solve_v4(D, E, S, m: int, s_min: int, s_max: int, lambda1=1.0, lambda2=0.1, lambda3=0.1):
    print(f"[Stage] Building Gurobi model (n={len(E)}, m={m})...")  # ← 加這行
    n = len(E)
    model = Model(env=env)
    x = model.addVars(n, m, vtype=GRB.BINARY, name="x")

    # y_jl = 1 if group j has size l
    y = model.addVars(m, s_max + 1, vtype=GRB.BINARY, name="y")

    # zj: engagement deviation from average
    z = model.addVars(m, vtype=GRB.CONTINUOUS, name="z")

    # dj: stakeholder polarity deviation
    d = model.addVars(m, vtype=GRB.CONTINUOUS, name="d")

    # group engagement and sentiment sums
    Ej = {}
    Sj = {}
    for j in range(m):
        Ej[j] = quicksum(E[i] * x[i, j] for i in range(n))
        Sj[j] = quicksum(S[i] * x[i, j] for i in range(n))

    avgE = E.sum() / m

    # Objective: maximize diversity - engagement variance - sentiment imbalance
    diversity = quicksum(D[i, k] * x[i, j] * x[k, j]
                         for j in range(m) for i in range(n) for k in range(i+1, n))
    model.setObjective(
        lambda1 * diversity
        - lambda2 * quicksum(z[j] for j in range(m))
        - lambda3 * quicksum(d[j] for j in range(m)),
        GRB.MAXIMIZE
    )

    # Constraints
    for i in range(n):
        model.addConstr(x.sum(i, '*') == 1)

    for j in range(m):
        size = x.sum('*', j)

        # link y[j, l] to size
        model.addConstr(quicksum(y[j, l] for l in range(s_min, s_max + 1)) == 1)
        model.addConstr(size == quicksum(l * y[j, l] for l in range(s_min, s_max + 1)))

        # engagement deviation
        model.addConstr(Ej[j] - avgE <= z[j])
        model.addConstr(avgE - Ej[j] <= z[j])

        # polarity deviation
        model.addConstr(Sj[j] <= d[j])
        model.addConstr(-Sj[j] <= d[j])

    model.optimize()
    if model.Status == GRB.OPTIMAL:
        sol = np.array([[x[i, j].X for j in range(m)] for i in range(n)])
        return sol.argmax(axis=1), model.ObjVal
    else:
        raise RuntimeError("Gurobi failed with status {}".format(model.Status))

# ─── CLI Run ───
if __name__ == "__main__":
    pv = "participants-votes.csv"
    cm = "comments.csv"
    participants, D, A, E, S = load_data(pv, cm)

    n = len(participants)
    m, s_min, s_max = 2, 5, 300  # Change as needed, m is the number of groups, s_min and s_max are the min and max group sizes
    print("[Stage] Solving optimization problem using v4 model...")
    assign, obj_val = solve_v4(D, E, S, m, s_min, s_max)
    df = pd.DataFrame({"participant": participants, "group": assign})
    df.to_csv("group_assignment_v4.csv", index=False)
    print("[Done] Objective value:", obj_val)
