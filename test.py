import pandas as pd
import numpy as np

# --- 1. Load data ---
pv = pd.read_csv('participants-votes.csv')
cm = pd.read_csv('comments.csv')

# Count how many participants authored > 2 comments in participants-votes.csv
def count_participants_with_comments(pv, threshold=2):
    return (pv['n-comments'] > threshold).sum()

print(count_participants_with_comments(pv, threshold=2))
print(count_participants_with_comments(pv, threshold=1))