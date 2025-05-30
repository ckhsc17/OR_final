import time
import math
import numpy as np
import pandas as pd

from generator2 import generate_instance
from heuristic import solve_heuristic_instance
# 重命名原本的 Lagrangian function
from lagarange_exp3 import solve_lagrangian_instance as solve_ip_lagrangian_instance
# 占位：之後實作 linear relaxation + Lagrange
# from lp_lagrangian import solve_lp_lagrangian_instance

# 計算 optimality gap (%)
def compute_gap(obj_val: float, reference_val: float) -> float:
    if reference_val == 0:
        return 0.0
    return (reference_val - obj_val) / abs(reference_val) * 100.0

# 計算每組 engagement total 的標準差，用於衡量平衡度
def compute_engagement_std(assign: np.ndarray, E: np.ndarray) -> float:
    m = int(assign.max() + 1)
    sums = [E[assign == j].sum() for j in range(m)]
    return float(np.std(sums))

# 目標函數，與 heuristic 中相同
def compute_objective(assign: np.ndarray,
                      D: np.ndarray,
                      E: np.ndarray,
                      lambda1: float,
                      lambda2: float) -> float:
    diversity = 0.0
    imbalance = 0.0
    m = int(assign.max() + 1)
    e_avg = E.sum() / m

    for j in range(m):
        idx = np.where(assign == j)[0]
        k = len(idx)
        if k > 1:
            dij = D[np.ix_(idx, idx)]
            triu = np.triu_indices(k, k=1)
            diversity += dij[triu].sum()
        e_sum = E[idx].sum()
        imbalance += abs(e_sum - e_avg)

    return lambda1 * diversity - lambda2 * imbalance


def main():
    results = []
    scenario_id = 0

    # 超參數
    lambda1_h, lambda2_h = 1.0, 0.0001
    lambda1_lr, lambda2_lr = 1.0, 0.0001
    max_iter_h = 2000
    max_iter_lr = 60

    for n in [50, 300, 1000]:
        for smax in [15, 30, 50]:
            for seed in range(3):
                inst = generate_instance(
                    n=n, smax=smax, r=3,
                    diversity_mode='clustered',
                    engagement_mode='normal',
                    random_seed=seed
                )
                D = inst['diversity']
                E = inst['engagement']

                # Heuristic 方法
                t0 = time.perf_counter()
                assign_h, obj_h, m, S_min, S_max = solve_heuristic_instance(
                    n, D, E, smax, lambda1_h, lambda2_h, max_iter_h
                )
                t_h = time.perf_counter() - t0

                # IP + Lagrangian Relaxation 方法
                t0 = time.perf_counter()
                assign_ip_lag, obj_ip_lag, _, _, _ = solve_ip_lagrangian_instance(
                    n, D, E, smax,
                    lam1=lambda1_lr, lam2=lambda2_lr,
                    max_iter=max_iter_lr, seed=seed
                )
                t_ip_lag = time.perf_counter() - t0

                # Naive baseline 方法（原 IP stub）
                t0 = time.perf_counter()
                assign_naive = np.arange(n) % m
                obj_naive = compute_objective(assign_naive, D, E, lambda1_lr, lambda2_lr)
                t_naive = time.perf_counter() - t0

                # (後續) LP Relaxation + Lagrangian 比較 - 占位呼叫
                # t0 = time.perf_counter()
                # assign_lp_lag, obj_lp_lag, _, _, _ = solve_lp_lagrangian_instance(...)
                # t_lp_lag = time.perf_counter() - t0

                # 計算 gaps 與 stds
                gap_naive = compute_gap(obj_naive, obj_ip_lag)
                gap_ip_lag = compute_gap(obj_ip_lag, obj_ip_lag)
                gap_h = compute_gap(obj_h, obj_ip_lag)
                # gap_lp_lag = compute_gap(obj_lp_lag, obj_ip_lag)

                std_naive = compute_engagement_std(assign_naive, E)
                std_ip_lag = compute_engagement_std(assign_ip_lag, E)
                std_h = compute_engagement_std(assign_h, E)
                # std_lp_lag = compute_engagement_std(assign_lp_lag, E)

                results.append({
                    'Scenario': scenario_id,
                    'n': n,
                    'smax': smax,
                    'seed': seed,
                    'Naive Gap (%)': gap_naive,
                    'Naive Std': std_naive,
                    'Naive Time (s)': t_naive,
                    'IP_Lagrangian Gap (%)': gap_ip_lag,
                    'IP_Lagrangian Std': std_ip_lag,
                    'IP_Lagrangian Time (s)': t_ip_lag,
                    'Heuristic Gap (%)': gap_h,
                    'Heuristic Std': std_h,
                    'Heuristic Time (s)': t_h,
                    # 'LP_Lagrangian Gap (%)': gap_lp_lag,
                    # 'LP_Lagrangian Std': std_lp_lag,
                    # 'LP_Lagrangian Time (s)': t_lp_lag
                })
                scenario_id += 1

    df = pd.DataFrame(results)
    summary = df.groupby('Scenario').agg({
        'Naive Gap (%)': ['mean', 'std'],
        'IP_Lagrangian Gap (%)': ['mean', 'std'],
        'Heuristic Gap (%)': ['mean', 'std'],
        'Naive Time (s)': 'mean',
        'IP_Lagrangian Time (s)': 'mean',
        'Heuristic Time (s)': 'mean'
    })

    print("===== Experiment Summary =====")
    print(summary)
    df.to_csv('experiment_summary.csv', index=False)

if __name__ == '__main__':
    main()
