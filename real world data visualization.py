from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

pv = pd.read_csv(r"participants_votes_all_features.csv")
ga = pd.read_csv(r"group_assignment.csv")

df = pd.merge(pv, ga, on='participant', how='left')

# vote data
vote_cols = [c for c in pv.columns if c.isdigit()]
votes = pv[vote_cols].fillna(0).values

# PCA
pca = PCA(n_components=2)
votes_pca = pca.fit_transform(votes)

pca_df = pd.DataFrame(votes_pca, columns=['PC1', 'PC2'])
pca_df['group'] = df['group']

pca_df2 = pd.DataFrame(votes_pca, columns=['PC1', 'PC2'])
pca_df2['kmeans_cluster'] = df['kmeans_cluster']

# plot PCA 1
plt.figure(figsize=(10, 8))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='group', palette='tab10', s=100)
plt.title('PCA Plot by Group')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Group')
plt.show()

# plot PCA 2
plt.figure(figsize=(10, 8))
sns.scatterplot(data=pca_df2, x='PC1', y='PC2', hue='kmeans_cluster', palette='tab10', s=100)
plt.title('PCA Plot by Group-id')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Stakeholder Cluster ID')
plt.show()

group_counts = df.groupby(['group', 'kmeans_cluster']).size().unstack(fill_value=0)

# plot stacked bar chart
group_counts.plot(kind='bar', stacked=True, figsize=(10, 7), colormap='tab20')
plt.title('Stakeholder Cluster Composition by Group')
plt.xlabel('Group')
plt.ylabel('Count')
plt.legend(title='Stakeholder Cluster ID', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()