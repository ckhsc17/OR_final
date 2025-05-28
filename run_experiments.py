# run_experiments.py

import pandas as pd
from generator2 import generate_instance
from heuristic import run_experiments as run_heuristic_experiments
from lagarange_exp3 import run_experiments as run_lagrangian_experiments

def main():
    # Step 1: Generate instances
    print("Generating instances...")
    scenarios = []
    for n in [50, 300, 1000]:
        for smax in [15, 30, 50]:
            for i in range(3):  # 3 random runs per scenario
                scenario = generate_instance(n=n, smax=smax, r=3,
                                             diversity_mode='clustered',
                                             engagement_mode='normal',
                                             random_seed=i)
                scenarios.append({
                    'n': n,
                    'smax': smax,
                    'instance_id': i,
                    'engagement_mean': scenario['engagement'].mean(),
                    'diversity_mean': scenario['diversity'].mean()
                })

    df_scenarios = pd.DataFrame(scenarios)
    print("Generated Scenario Summary:")
    print(df_scenarios)

    # Step 2: Run heuristic experiments
    print("\nRunning heuristic experiments...")
    heuristic_summary = run_heuristic_experiments()
    print("Heuristic Experiment Summary:")
    print(heuristic_summary)

    # Step 3: Run Lagrangian experiments
    print("\nRunning Lagrangian experiments...")
    lagrangian_summary = run_lagrangian_experiments()
    print("Lagrangian Experiment Summary:")
    print(lagrangian_summary)

if __name__ == '__main__':
    main()