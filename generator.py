import numpy as np
import pandas as pd

def generate_random_instance(n=200, r=3, polarization='medium', engagement_level='medium'):
    """
    Generate a random instance for the group assignment model.
    
    Parameters:
    - n: number of participants
    - r: number of stakeholder groups
    - polarization: 'low', 'medium', 'high'
    - engagement_level: 'low', 'medium', 'high'
    
    Returns:
    - participants: DataFrame with columns ['participant', 'stakeholder', 'engagement']
    - D: (n x n) diversity matrix
    - E: (n,) engagement vector
    - Ebar: target average engagement
    """
    # 1. Stakeholder assignment
    stakeholders = np.random.choice(r, size=n)
    
    # 2. Diversity matrix
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            if stakeholders[i] == stakeholders[j]:
                low, high = (0.1, 0.3) if polarization != 'high' else (0.1, 0.5)
            else:
                low, high = (0.7, 0.9) if polarization != 'low' else (0.5, 0.8)
            val = np.random.uniform(low, high)
            D[i, j] = D[j, i] = val
    
    # 3. Engagement scores
    if engagement_level == 'low':
        E = np.random.uniform(1, 5, size=n)
    elif engagement_level == 'high':
        E = np.random.uniform(20, 50, size=n)
    else:
        E = np.random.uniform(5, 20, size=n)
    
    # 4. Target average engagement
    Ebar = E.mean()
    
    # 5. Participants DataFrame
    participants = pd.DataFrame({
        'participant': np.arange(n),
        'stakeholder': stakeholders,
        'engagement': E
    })
    
    return participants, D, E, Ebar

# Example of generating multiple scenarios
def generate_multiple_instances(configs):
    """
    configs: list of dicts with keys 'n', 'r', 'polarization', 'engagement_level'
    Returns a dict of instances keyed by scenario name.
    """
    instances = {}
    for cfg in configs:
        name = f"n{cfg['n']}_r{cfg['r']}_{cfg['polarization']}_{cfg['engagement_level']}"
        instances[name] = generate_random_instance(**cfg)
    return instances

# Example configuration grid
configs = [
    {'n':100, 'r':2, 'polarization':'low',    'engagement_level':'medium'},
    {'n':100, 'r':2, 'polarization':'high',   'engagement_level':'medium'},
    {'n':200, 'r':3, 'polarization':'medium', 'engagement_level':'low'},
    {'n':200, 'r':3, 'polarization':'medium', 'engagement_level':'high'},
]

instances = generate_multiple_instances(configs)
# instances now holds participants, D, E, Ebar for each scenario
