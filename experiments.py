import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum, QuadExpr

# --- 1. Data Loading & Matrix Construction ---
def load_data(pv_path, cm_path):
    pv = pd.read_csv(pv_path)
    cm = pd.read_csv(cm_path)
    participants = pv['participant'].tolist()
    id2idx = {pid: idx for idx, pid in enumerate(participants)}
    n = len(participants)
    # Votes matrix
    vote_cols = [c for c in pv.columns if c.isdigit()]
    votes = pv[vote_cols].fillna(0).values
    norms = np.linalg.norm(votes, axis=1)
    A = (votes @ votes.T) / (norms[:,None]*norms[None,:] + 1e-9)
    D = 1 - A
    E = (pv['n-comments'] + pv['n-votes']).values
    S = np.sign(pv['n-agree'] - pv['n-disagree']).astype(int)
    C = (pv['n-comments'] > 0).astype(int)
    return participants, n, A, D, E, S, C

# --- 2. Gurobi IP Solver ---
def solve_ip(n, m, A, D, E, S, C, S_min, S_max, delta, theta,
             w1=1.0, w2=0.5, w3=0.1):
    model = Model()
    x = model.addVars(n, m, vtype=GRB.BINARY)
    # Objective: agreement - w2*polarization - w3*variance(E)
    agreement = QuadExpr()
    for j in range(m):
        for i in range(n):
            for k in range(i+1, n):
                agreement.add(A[i,k]*x[i,j]*x[k,j])
    polar = QuadExpr()
    for j1 in range(m):
        for j2 in range(j1+1, m):
            for i in range(n):
                for k in range(n):
                    polar.add(D[i,k]*x[i,j1]*x[k,j2])
    avgE = E.sum()/m
    varE = QuadExpr()
    for j in range(m):
        tmp = quicksum(E[i]*x[i,j] for i in range(n)) - avgE
        varE.add(tmp*tmp)
    model.setObjective(w1*agreement - w2*polar - w3*varE, GRB.MAXIMIZE)
    # Constraints
    # each i -> exactly one j
    for i in range(n): model.addConstr(x.sum(i,'*') == 1)
    # size and oddity
    y = model.addVars(m, vtype=GRB.INTEGER, lb=0)
    for j in range(m):
        size = x.sum('*', j)
        model.addConstr(size >= S_min)
        model.addConstr(size <= S_max)
        model.addConstr(size - 1 == 2*y[j])
    # diversity and commentator & sentiment
    for j in range(m):
        div = QuadExpr()
        for i in range(n):
            for k in range(i+1, n): div.add(D[i,k]*x[i,j]*x[k,j])
        model.addQConstr(div >= delta)
        model.addConstr(quicksum(C[i]*x[i,j] for i in range(n)) >= 1)
        model.addConstr(quicksum((S[i]==1)*x[i,j] for i in range(n)) >= 1)
        model.addConstr(quicksum((S[i]==-1)*x[i,j] for i in range(n)) >= 1)
    # solve
    model.Params.OutputFlag = 0
    model.optimize()
    sol = np.array([[x[i,j].X for j in range(m)] for i in range(n)])
    assign = sol.argmax(axis=1)
    return assign

# --- 3. LP Relaxation + Rounding ---
def solve_lp_rounding(n, m, A, D, E, S, C, S_min, S_max, delta, theta,
                      w1=1.0, w2=0.5, w3=0.1):
    model = Model()
    x = model.addVars(n, m, lb=0, ub=1)
    # same objective formulation as IP
    agreement, polar, varE = QuadExpr(), QuadExpr(), QuadExpr()
    for j in range(m):
        for i in range(n):
            for k in range(i+1, n): agreement.add(A[i,k]*x[i,j]*x[k,j])
    for j1 in range(m):
        for j2 in range(j1+1, m):
            for i in range(n):
                for k in range(n): polar.add(D[i,k]*x[i,j1]*x[k,j2])
    avgE = E.sum()/m
    for j in range(m):
        tmp = quicksum(E[i]*x[i,j] for i in range(n)) - avgE
        varE.add(tmp*tmp)
    model.setObjective(w1*agreement - w2*polar - w3*varE, GRB.MAXIMIZE)
    # continuous constraints (skip diversity/commentator for speed)
    for i in range(n): model.addConstr(x.sum(i,'*') == 1)
    for j in range(m):
        size = x.sum('*', j)
        model.addConstr(size >= S_min)
        model.addConstr(size <= S_max)
    model.Params.OutputFlag = 0
    model.optimize()
    sol = np.array([[x[i,j].X for j in range(m)] for i in range(n)])
    # rounding: assign i to group with highest fractional value
    assign = sol.argmax(axis=1)
    return assign

# --- 4. Greedy Local Search Heuristic ---
def heuristic_greedy(n, m, A, D, E, S, C, S_min, S_max):
    # initialize randomly satisfying size/sentiment/commentator
    assign = np.random.choice(m, size=n)
    # simple repair to satisfy size
    counts = np.bincount(assign, minlength=m)
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
    # local search: one-swap
    def obj_val(assign):
        val = 0.0
        # agreement
        for j in range(m):
            idxs = np.where(assign==j)[0]
            for u in idxs:
                for v in idxs:
                    val += A[u,v]
        # penalty terms omitted for brevity
        return val
    improved = True
    while improved:
        improved = False
        base = obj_val(assign)
        for i in range(n):
            curr = assign[i]
            for j in range(m):
                if j==curr: continue
                new_assign = assign.copy(); new_assign[i] = j
                if obj_val(new_assign) > base:
                    assign = new_assign; improved = True; break
            if improved: break
    return assign

# --- 5. Metrics Calculation ---
def compute_metrics(assign, A, D, E):
    m = assign.max()+1
    # intra-agreement
    intra = sum(A[i,k] for i in range(len(assign)) for k in range(len(assign)) if assign[i]==assign[k])
    # inter-polarization
    inter = sum(D[i,k] for i in range(len(assign)) for k in range(len(assign)) if assign[i]!=assign[k])
    # engagement variance
    es = [E[assign==j].sum() for j in range(m)]
    varE = np.var(es)
    return {'intra_agreement': intra, 'inter_polarization': inter, 'engagement_var': varE}

# --- 6. Experiment Runner ---
def run_experiments(pv_path, cm_path):
    parts, n, A, D, E, S, C = load_data(pv_path, cm_path)
    # settings
    m, S_min, S_max, delta, theta = 5, 5, 15, 10.0, 1
    methods = {
        #'IP': lambda: solve_ip(n,m,A,D,E,S,C,S_min,S_max,delta,theta),
        #'LP_round': lambda: solve_lp_rounding(n,m,A,D,E,S,C,S_min,S_max,delta,theta),
        'Heuristic': lambda: heuristic_greedy(n,m,A,D,E,S,C,S_min,S_max)
    }
    results = {}
    for name, fn in methods.items():
        assign = fn()
        metrics = compute_metrics(assign, A, D, E)
        results[name] = metrics
    df = pd.DataFrame(results).T
    print(df)
    return df

if __name__ == '__main__':
    run_experiments('participants-votes.csv', 'comments.csv')
