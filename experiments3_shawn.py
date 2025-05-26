import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

import gurobipy as gp
from gurobipy import Model, GRB, quicksum, QuadExpr

# ──────────────────────────────────────────────────────────────
# 0. Gurobi 雲端授權環境（依個人帳號修改）
# ──────────────────────────────────────────────────────────────
env = gp.Env(empty=True)
env.setParam("WLSACCESSID", "a8b9fb6b-ed21-4eca-a559-be15569076fa")
env.setParam("WLSSECRET",   "61a677bd-d8b1-47e4-a557-68147ca6ff59")
env.setParam("LICENSEID",   2636425)
env.start()


# ──────────────────────────────────────────────────────────────
# (A) 自動調組數 / 上限  ─ 放在檔案最前面即可
# ──────────────────────────────────────────────────────────────
def auto_adjust_group_params(n, m_init, S_min_init, S_max_init,
                             prefer_fix_m=True):
    """
    確保 m*S_min <= n <= m*S_max。
    若 n > m*S_max：優先增加 m；若 prefer_fix_m=False，則擴大 S_max。
    若 n < m*S_min：縮小 S_min。
    """
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
# 1. 資料讀取（稀疏矩陣版本）
# ──────────────────────────────────────────────────────────────
def load_data_sparse(pv_path: str, cm_path: str):
    pv = pd.read_csv(pv_path)
    cm = pd.read_csv(cm_path)              # 目前程式碼未用到，可保留後續擴充
    participants = pv['participant'].tolist()
    n = len(participants)

    vote_cols = [c for c in pv.columns if c.isdigit()]
    votes_sparse = csr_matrix(pv[vote_cols].fillna(0).values)
    A = cosine_similarity(votes_sparse)    # agreement
    D = 1 - A                              # dissimilarity

    E = (pv['n-comments'] + pv['n-votes']).values
    S = np.sign(pv['n-agree'] - pv['n-disagree']).astype(int)
    C = (pv['n-comments'] > 0).astype(int)
    return participants, n, A, D, E, S, C

# ──────────────────────────────────────────────────────────────
# 2. 根據資料動態標定 delta (多樣性上限) 與 eta (最低參與度)
# ──────────────────────────────────────────────────────────────
def calibrate_params(D: np.ndarray, E: np.ndarray,
                     S_max: int, m: int,
                     q: float = 0.85, eta_ratio: float = 0.7):
    """回傳 (delta, eta) 兩個安全參數"""
    pair_max = S_max * (S_max - 1) / 2
    upper_d = np.quantile(D[np.triu_indices_from(D, k=1)], q)
    delta = upper_d * pair_max

    eta = eta_ratio * E.sum() / m
    return delta, eta

# ──────────────────────────────────────────────────────────────
# 3-1. Integer Programming (IP) 解
# ──────────────────────────────────────────────────────────────
def solve_ip(n, m, D, E, S_min, S_max, delta, eta,
             lambda1=1.0, lambda2=0.1):
    model = Model(env=env)
    x = model.addVars(n, m, vtype=GRB.BINARY, name="x")

    # ── 目標函數 ───────────────────────────────────────────
    diversity = quicksum(
        D[i, k] * x[i, j] * x[k, j]
        for j in range(m)
        for i in range(n) for k in range(i + 1, n)
    )
    avgE = E.sum() / m
    varE = quicksum(
        (quicksum(E[i] * x[i, j] for i in range(n)) - avgE) *
        (quicksum(E[i] * x[i, j] for i in range(n)) - avgE)
        for j in range(m)
    )
    model.setObjective(lambda1 * diversity - lambda2 * varE, GRB.MAXIMIZE)

    # ── 約束式 ─────────────────────────────────────────────
    for i in range(n):
        model.addConstr(x.sum(i, '*') == 1)

    y = model.addVars(m, vtype=GRB.INTEGER, lb=0, name="y")
    for j in range(m):
        size = x.sum('*', j)
        model.addConstr(size >= S_min)
        model.addConstr(size <= S_max)
        model.addConstr(size - 1 == 2 * y[j])
        div = quicksum(
            D[i, k] * x[i, j] * x[k, j]
            for i in range(n) for k in range(i + 1, n)
        )
        model.addQConstr(div <= delta)
        eng = quicksum(E[i] * x[i, j] for i in range(n))
        # model.addConstr(eng >= eta)

    model.Params.OutputFlag = 0
    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        print("Model is infeasible. Computing IIS...")
        model.computeIIS()
        model.write("model.ilp")  # Save IIS to a file for inspection
        raise RuntimeError("IP infeasible. IIS written to 'model.ilp'.")

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError("IP infeasible (status {})".format(model.Status))

    sol = np.array([[x[i, j].X for j in range(m)] for i in range(n)])
    return sol.argmax(axis=1), model.ObjVal

# ──────────────────────────────────────────────────────────────
# 3-2. LP Relaxation + Rounding
# ──────────────────────────────────────────────────────────────
def solve_lp_rounding(n, m, D, E, S_min, S_max, delta, eta,
                      lambda1=1.0, lambda2=0.1):
    model = Model(env=env)
    x = model.addVars(n, m, lb=0, ub=1, name="x")

    # 目標函數同上
    diversity = quicksum(
        D[i, k] * x[i, j] * x[k, j]
        for j in range(m)
        for i in range(n) for k in range(i + 1, n)
    )
    avgE = E.sum() / m
    varE = quicksum(
        (quicksum(E[i] * x[i, j] for i in range(n)) - avgE) *
        (quicksum(E[i] * x[i, j] for i in range(n)) - avgE)
        for j in range(m)
    )
    model.setObjective(lambda1 * diversity - lambda2 * varE, GRB.MAXIMIZE)

    # 約束式（無奇數強制）
    for i in range(n):
        model.addConstr(x.sum(i, '*') == 1)
    for j in range(m):
        size = x.sum('*', j)
        model.addConstr(size >= S_min)
        model.addConstr(size <= S_max)
        div = quicksum(
            D[i, k] * x[i, j] * x[k, j]
            for i in range(n) for k in range(i + 1, n)
        )
        model.addQConstr(div <= delta)
        eng = quicksum(E[i] * x[i, j] for i in range(n))
        # model.addConstr(eng >= eta)

    model.Params.OutputFlag = 0
    model.optimize()
    if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        raise RuntimeError("LP infeasible (status {})".format(model.Status))

    # 簡單最大值 round（可改進為 max-prob 或隨機取樣）
    frac = np.array([[x[i, j].X for j in range(m)] for i in range(n)])
    assign = frac.argmax(axis=1)
    return assign, model.ObjVal

# ──────────────────────────────────────────────────────────────
# 3-3. Heuristic：隨機啟發 + local search
# ──────────────────────────────────────────────────────────────
def heuristic_greedy(n, m, A, D, E, S, C, S_min, S_max,
                     max_iter: int = 200):
    # 隨機初始分派
    assign = np.random.choice(m, size=n)
    counts = np.bincount(assign, minlength=m)

    # 修復組別大小
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
            counts[j] -= 1
            counts[newj] += 1

    # ⟨簡易⟩ 目標：最大化組內 agreement 總和
    def obj(a):
        val = 0.0
        for j in range(m):
            idx = np.where(a == j)[0]
            val += A[np.ix_(idx, idx)].sum()
        return val

    best = assign.copy()
    best_val = obj(best)
    for _ in range(max_iter):
        i = np.random.randint(n)
        cand = best.copy()
        cand[i] = np.random.randint(m)
        if counts[cand[i]] >= S_max or counts[best[i]] <= S_min:
            continue
        cand_val = obj(cand)
        if cand_val > best_val:
            best, best_val = cand, cand_val
    return best, best_val

# ──────────────────────────────────────────────────────────────
# 4. 指標計算
# ──────────────────────────────────────────────────────────────
def compute_metrics(assign, A, D, E):
    m = assign.max() + 1
    intra = sum(A[i, k] for i in range(len(assign))
                           for k in range(len(assign))
                           if assign[i] == assign[k])
    inter = sum(D[i, k] for i in range(len(assign))
                           for k in range(len(assign))
                           if assign[i] != assign[k])
    es = [E[assign == j].sum() for j in range(m)]
    return dict(intra_agreement=intra,
                inter_polarization=inter,
                engagement_var=np.var(es))

# ──────────────────────────────────────────────────────────────
# 5. 主要實驗流程（修正版）
# ──────────────────────────────────────────────────────────────
def run_experiments(pv_path='participants-votes.csv',
                    cm_path='comments.csv'):

    # ① 先讀資料 → 才拿得到 n、A、D、E …
    participants, n, A, D, E, S, C = load_data_sparse(pv_path, cm_path)

    # ② 自動調組數 / 上限 (不要在這之後再覆寫!)
    m_init, S_min_init, S_max_init = 2, 5, 15
    m, S_min, S_max = auto_adjust_group_params(
        n, m_init, S_min_init, S_max_init, prefer_fix_m=True)
    print(f"[Info] n={n}, m={m}, S_min={S_min}, S_max={S_max}")

    # ③ 依調整後的 m、S_max 重新算 δ、η
    delta, eta = calibrate_params(D, E, S_max, m)
    print(f"[Info] delta={delta:.4f}, eta={eta:.4f}")

    # ④ 定義各方法
    methods = {
        # 'Heuristic': lambda: heuristic_greedy(
        #     n, m, A, D, E, S, C, S_min, S_max)[0],
        'IP': lambda: solve_ip(
            n, m, D, E, S_min, S_max, delta, eta)[0],
        'LP': lambda: solve_lp_rounding(
            n, m, D, E, S_min, S_max, delta, eta)[0]
    }

    # ⑤ 跑實驗 & 摘要
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
# 6. CLI
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    run_experiments('participants-votes.csv', 'comments.csv')
