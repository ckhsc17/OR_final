import pandas as pd

# Load the experiment summary CSV file
file_path = "experiment_summary_new.csv"
df = pd.read_csv(file_path)

# Display the structure of the DataFrame for analysis
df.head()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set up visual style
sns.set(style="whitegrid")
plt.figure(figsize=(14, 6))

# Melt for comparison plot
df_melted = df.melt(
    id_vars=["Scenario", "n", "smax", "seed"],
    value_vars=["Naive Gap (%)", "IP_Lagrangian Gap (%)", "Heuristic Gap (%)", "LP_Lagrangian Gap (%)"],
    var_name="Method",
    value_name="Gap (%)"
)

# Convert negative gaps to positive (maximize objective → better = higher)
df_melted["Gap (%)"] = -df_melted["Gap (%)"]

# Plot boxplot for gaps across different methods
sns.boxplot(x="Method", y="Gap (%)", data=df_melted)
plt.title("Optimality Gap Comparison Across Methods (Positive is Better)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()