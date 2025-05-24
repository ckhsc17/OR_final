import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum, QuadExpr

# --- 1. Load data ---
pv = pd.read_csv('participants-votes.csv')
cm = pd.read_csv('comments.csv')

# Participant list and index mapping
participants = pv['participant'].tolist()
n = len(participants)
id2idx = {pid: idx for idx,pid in enumerate(participants)}

# --- 2. Preprocess parameters ---
# 2.1 Vote vectors (rows: participants, cols: comment IDs)
vote_cols = [col for col in pv.columns if col.isdigit()]
votes = pv[vote_cols].fillna(0).values  # shape: (n, num_comments)

# 2.2 Agreement matrix A (cosine similarity)
norms = np.linalg.norm(votes, axis=1)
A_mat = (votes @ votes.T) / (norms[:,None] * norms[None,:] + 1e-6)

# 2.3 Diversity matrix D = 1 - A
D_mat = 1 - A_mat

# 2.4 Engagement score E (sum of comments + votes)
E_vec = (pv['n-comments'] + pv['n-votes']).values

# 2.5 Sentiment score S: sign(n-agree - n-disagree)
S_vec = np.sign(pv['n-agree'] - pv['n-disagree']).astype(int)

# 2.6 Commentator indicator C: 1 if participant authored >=1 comment
C_vec = (pv['n-comments'] > 0).astype(int)

# --- 3. Model parameters ---
m = n // 7            # number of groups (adjustable)
S_min, S_max = 5, 9
delta = 10.0     # intra-group diversity threshold
theta = 1        # sentiment balance tolerance

# Weights for composite objective
theta1, lambda2, lambda3 = 1.0, 0.5, 0.1

# --- 4. Build Gurobi model ---
model = Model('vTaiwan_group_assignment')
model.Params.OutputFlag = 1

# 4.1 Decision variables x[i,j] = 1 if participant i -> group j
x = {}
for i in range(n):
    for j in range(m):
        x[i,j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

# 4.2 Auxiliary integer vars for odd-size constraint: size_j - 1 = 2*y_j
y = {j: model.addVar(vtype=GRB.INTEGER, lb=0, name=f"y_{j}") for j in range(m)}
model.update()

# --- 5. Objective construction ---
# 5.1 Maximize intra-group agreement
agreement = QuadExpr()
for j in range(m):
    for i in range(n):
        for k in range(i+1, n):
            agreement.add(A_mat[i,k] * x[i,j] * x[k,j])

# 5.2 Minimize inter-group polarization
polar = QuadExpr()
for j1 in range(m):
    for j2 in range(j1+1, m):
        for i in range(n):
            for k in range(n):
                polar.add(D_mat[i,k] * x[i,j1] * x[k,j2])

# 5.3 Engagement fairness via variance (approx.)
avg_E = E_vec.sum() / m
var_expr = QuadExpr()
for j in range(m):
    lin = quicksum(E_vec[i] * x[i,j] for i in range(n)) - avg_E
    var_expr.add(lin * lin)

# Composite objective: max{ agreement - lambda2*polar - lambda3*var_expr }
obj = theta1 * agreement - lambda2 * polar - lambda3 * var_expr
model.setObjective(obj, GRB.MAXIMIZE)

# --- 6. Constraints ---
# 6.1 Unique assignment
for i in range(n):
    model.addConstr(quicksum(x[i,j] for j in range(m)) == 1, name=f"assign_{i}")

# 6.2 Group size limits & odd cardinality
for j in range(m):
    size_j = quicksum(x[i,j] for i in range(n))
    model.addConstr(size_j >= S_min, name=f"size_min_{j}")
    model.addConstr(size_j <= S_max, name=f"size_max_{j}")
    model.addConstr(size_j - 1 == 2 * y[j], name=f"odd_size_{j}")

# 6.3 Intra-group diversity constraint
for j in range(m):
    div_expr = QuadExpr()
    for i in range(n):
        for k in range(i+1, n):
            div_expr.add(D_mat[i,k] * x[i,j] * x[k,j])
    model.addQConstr(div_expr >= delta, name=f"diversity_{j}")

# 6.4 At least one commentator per group
for j in range(m):
    model.addConstr(quicksum(C_vec[i] * x[i,j] for i in range(n)) >= 1,
                    name=f"commentator_{j}")

# 6.5 Sentiment balance: >=1 agree & >=1 disagree
for j in range(m):
    model.addConstr(quicksum((S_vec[i] == 1) * x[i,j] for i in range(n)) >= 1,
                    name=f"sent_agree_{j}")
    model.addConstr(quicksum((S_vec[i] == -1) * x[i,j] for i in range(n)) >= 1,
                    name=f"sent_disagree_{j}")

# --- 7. Optimize and extract ---
model.optimize()

# Build group assignments
groups = {j: [] for j in range(m)}
for i in range(n):
    for j in range(m):
        if x[i,j].X > 0.5:
            groups[j].append(participants[i])

# Display result
for j, members in groups.items():
    print(f"Group {j+1} ({len(members)}): {members}")
