import numpy as np
import pandas as pd
import random
from itertools import combinations
from scipy.spatial.distance import pdist, squareform

def generate_instance(n, smax, r, diversity_mode='uniform', engagement_mode='normal', random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    # Stakeholder assignment (uniform)
    stakeholder_ids = list(range(r))
    s = np.random.choice(stakeholder_ids, size=n)

    # Engagement score
    if engagement_mode == 'normal':
        E = np.clip(np.random.normal(loc=5, scale=2, size=n), 0, None)
    elif engagement_mode == 'uniform':
        E = np.random.uniform(2, 8, size=n)
    else:
        raise ValueError("Unknown engagement_mode")

    # Diversity score matrix D(i,k)
    if diversity_mode == 'uniform':
        latent_pos = np.random.uniform(0, 1, size=(n, 2))
    elif diversity_mode == 'clustered':
        centers = np.random.uniform(0, 1, size=(r, 2))
        latent_pos = np.vstack([
            np.random.normal(loc=centers[s[i]], scale=0.05, size=(1, 2))
            for i in range(n)
        ])
    else:
        raise ValueError("Unknown diversity_mode")

    D_matrix = squareform(pdist(latent_pos, metric='euclidean'))

    # Round D and normalize
    D_matrix = D_matrix / D_matrix.max()

    return {
        'n': n,
        'smax': smax,
        'r': r,
        'stakeholder': s,
        'engagement': E,
        'diversity': D_matrix
    }

if __name__ == '__main__':
    # Generate example scenario combinations
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
    #import ace_tools as tools; tools.display_dataframe_to_user(name="Generated Scenario Summary", dataframe=df_scenarios)
    print("Generated Scenario Summary")
    print(df_scenarios)

