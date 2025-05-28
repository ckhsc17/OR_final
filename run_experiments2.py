import time
import math
import numpy as np
import pandas as pd

from generator2 import generate_instance
from heuristic import solve_heuristic_instance
from lagarange_exp3 import solve_lagrangian_instance

# 計算 optimality gap (%)
def compute_gap(obj_val: float, optimal_val: float) -> float:
    if optimal_val == 0:
        return 0.0
    return (optimal_val - obj_val) / abs(optimal_val) * 100.0

# 計算每組 engagement total 的標準差，用於衡量平衡度
def compute_engagement_std(assign: np.ndarray, E: np.ndarray) -> float:
    m = int(assign.max() + 1)
    sums = [E[assign == j].sum() for j in range(m)]
    return float(np.std(sums))

# 目標函數，與 heuristic 中相同
def compute_objective(assign: np.ndarray,
                      D: np.ndarray,
                      E: np.ndarray,
                      lambda1: float = 1.0,
                      lambda2: float = 1.0) -> float:
    diversity = 0.0
    imbalance = 0.0
    m = int(assign.max() + 1)
    e_avg = E.sum() / m

    for j in range(m):
        idx = np.where(assign == j)[0]
        k = len(idx)
        if k > 1:
            dij = D[np.ix_(idx, idx)]
            # sum of upper-triangle
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
            for seed in range(3):  # 三次重複
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
                print("===== Heuristic Result =====")
                print(assign_h)
                print(obj_h)


                # Lagrangian Relaxation 方法
                t0 = time.perf_counter()
                assign_lr, obj_lr, _, _, _ = solve_lagrangian_instance(
                    n, D, E, smax,
                    lam1=lambda1_lr, lam2=lambda2_lr,
                    max_iter=max_iter_lr, seed=seed
                )
                t_lr = time.perf_counter() - t0
                print("===== Lagrangian Relaxation Result =====")
                print(assign_lr)
                print(obj_lr)



                # Naive (IP stub) 方法：round-robin
                t0 = time.perf_counter()
                assign_ip = np.arange(n) % m
                obj_ip = compute_objective(assign_ip, D, E, lambda1_lr, lambda2_lr)
                t_ip = time.perf_counter() - t0
                print("===== IP Stub Result =====")
                print(assign_ip)
                print(obj_ip)



                # 計算 gaps 與 stds
                gap_ip = compute_gap(obj_ip, obj_lr)
                gap_lr = compute_gap(obj_lr, obj_lr)  # 理論上 = 0
                gap_h = compute_gap(obj_h, obj_lr)

                std_ip = compute_engagement_std(assign_ip, E)
                std_lr = compute_engagement_std(assign_lr, E)
                std_h = compute_engagement_std(assign_h, E)

                results.append({
                    'Scenario': scenario_id,
                    'n': n,
                    'smax': smax,
                    'seed': seed,
                    'IP Gap (%)': gap_ip,
                    'IP Std (%)': std_ip,
                    'IP Time (s)': t_ip,
                    'LR Gap (%)': gap_lr,
                    'LR Std (%)': std_lr,
                    'LR Time (s)': t_lr,
                    'Heuristic Gap (%)': gap_h,
                    'Heuristic Std (%)': std_h,
                    'Heuristic Time (s)': t_h
                })
                scenario_id += 1

    df = pd.DataFrame(results)
    # 彙整統計：平均 + 標準差
    summary = df.groupby('Scenario').agg({
        'IP Gap (%)': ['mean', 'std'],
        'LR Gap (%)': ['mean', 'std'],
        'Heuristic Gap (%)': ['mean', 'std'],
        'IP Time (s)': 'mean',
        'LR Time (s)': 'mean',
        'Heuristic Time (s)': 'mean'
    })

    print("===== Experiment Summary =====")
    print(summary)
    df.to_csv('experiment_summary.csv', index=False)


if __name__ == '__main__':
    main()