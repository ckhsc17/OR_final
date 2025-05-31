import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


comments_df = pd.read_csv("comments_with_sentiment.csv")
participants_votes_df = pd.read_csv("participants-votes.csv")

# # print basic info about the dataframes
# print("Comments DataFrame Info:")
# print(comments_df.info())
# print(comments_df.head(5))
# print("\nParticipants Votes DataFrame Info:")
# print(participants_votes_df.info())
# print(participants_votes_df.head(5))

# missing_group_ids = participants_votes_df[participants_votes_df['group-id'].isna()]
# print(f"Participants missing group-id: {len(missing_group_ids)}")
# print(missing_group_ids[['participant']])

# Load the data
comments_df = pd.read_csv("comments_with_sentiment.csv")
participants_votes_df = pd.read_csv("participants-votes.csv")

# STEP 1: Create a mapping from comment-id to sentiment
sentiment_map = comments_df.set_index('comment-id')['sentiment'].to_dict()

# STEP 2: Identify actual comment vote columns (these should match comment-ids)
vote_cols = [col for col in participants_votes_df.columns if col.isdigit() and int(col) in sentiment_map]

# STEP 3: Build the feature dataframe: start by copying base features from participants_votes_df
base_cols = ['participant', 'n-comments', 'n-votes', 'n-agree', 'n-disagree']
features = participants_votes_df[base_cols].copy()

# STEP 4: Map comment IDs to their sentiment
comment_sentiments = [sentiment_map[int(col)] for col in vote_cols]

# STEP 5: Extract just the vote matrix
vote_matrix = participants_votes_df[vote_cols].copy()

# STEP 6: For each sentiment, count agree/disagree/pass votes
for sentiment in ['positive', 'neutral', 'negative']:
    sentiment_cols = [col for col, s in zip(vote_cols, comment_sentiments) if s == sentiment]
    sub_matrix = vote_matrix[sentiment_cols]

    features[f'agree_{sentiment}'] = (sub_matrix == 1).sum(axis=1)
    features[f'disagree_{sentiment}'] = (sub_matrix == -1).sum(axis=1)
    features[f'pass_{sentiment}'] = (sub_matrix == 0).sum(axis=1)

# print(features)

# More features:
features['total_votes_cast'] = (
    features[['agree_positive', 'agree_neutral', 'agree_negative',
              'disagree_positive', 'disagree_neutral', 'disagree_negative',
              'pass_positive', 'pass_neutral', 'pass_negative']].sum(axis=1)
)


def vote_entropy(row):
    counts = np.array([
        row['agree_positive'], row['agree_neutral'], row['agree_negative'],
        row['disagree_positive'], row['disagree_neutral'], row['disagree_negative'],
        row['pass_positive'], row['pass_neutral'], row['pass_negative']
    ])
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    return -np.sum([p * np.log2(p) for p in probs if p > 0])

features['vote_entropy'] = features.apply(vote_entropy, axis=1)

features['fraction_agree'] = (
    features[['agree_positive', 'agree_neutral', 'agree_negative']].sum(axis=1) /
    features['total_votes_cast'].replace(0, np.nan)
)

features['fraction_pass'] = (
    features[['pass_positive', 'pass_neutral', 'pass_negative']].sum(axis=1) /
    features['total_votes_cast'].replace(0, np.nan)
)

def sentiment_bias(row):
    sentiment_totals = [
        row['agree_positive'] + row['disagree_positive'] + row['pass_positive'],
        row['agree_neutral'] + row['disagree_neutral'] + row['pass_neutral'],
        row['agree_negative'] + row['disagree_negative'] + row['pass_negative'],
    ]
    total = sum(sentiment_totals)
    if total == 0:
        return 0.0
    proportions = [s / total for s in sentiment_totals]
    return max(proportions) - min(proportions)

features['sentiment_bias'] = features.apply(sentiment_bias, axis=1)

#print(features[['participant', 'total_votes_cast', 'vote_entropy', 'fraction_agree', 'fraction_pass', 'sentiment_bias']].head())

########
# Use PCA to reduce the 197 comment-vote columns into 3 principal components

# Extract the vote matrix (last 197 columns are votes)
vote_matrix = participants_votes_df.iloc[:, -197:]

# Fill NaNs (not voted) with 0
vote_matrix_filled = vote_matrix[vote_cols].fillna(0)

# Add all vote columns directly to the features DataFrame
features = participants_votes_df[['participant']].copy()
features = pd.concat([features, vote_matrix_filled], axis=1)

# Standardize
scaler = StandardScaler()
X_features_scaled = scaler.fit_transform(features)


########
# Clustering
########

# Elbow method to find optimal k
inertia = []
K_range = range(2, 15)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_features_scaled)
    inertia.append(km.inertia_)

# Plot the elbow
plt.figure(figsize=(8, 4))
plt.plot(K_range, inertia, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia (within-cluster sum of squares)')
plt.title('Elbow Method For Optimal k')
plt.grid(True)
plt.show()

#############
#
# After visually choosing the elbow point (e.g., k = 5) SET k
#
#############

optimal_k = 3

# KMeans clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
features['kmeans_cluster'] = kmeans.fit_predict(X_features_scaled)

# Agglomerative clustering
# agg = AgglomerativeClustering(n_clusters=optimal_k)
# features['agglo_cluster'] = agg.fit_predict(X_features_scaled)

# Save the updated participants_votes_df with cluster labels
features.to_csv("participants_votes_all_features.csv",
                             index=False)
print("participants_votes_all_features: ", participants_votes_df.head())

# Visualize the clusters

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_features_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1],
            c=features['kmeans_cluster'], alpha=0.7, edgecolor='k',
            cmap='viridis')
plt.title("Participant Clusters (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label='Cluster ID')
plt.show()


# Reduce to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_features_scaled)

# Add PCA coordinates to features
features['PC1'] = X_pca[:, 0]
features['PC2'] = X_pca[:, 1]

# Merge group-id info if not already present
if 'group-id' not in features.columns:
    features = features.merge(participants_votes_df[['participant', 'group-id']],
                              on='participant', how='left')

plt.figure(figsize=(8, 6))

# Base scatter plot: participants colored by cluster
scatter = plt.scatter(features['PC1'], features['PC2'],
                      c=features['kmeans_cluster'], cmap='viridis', alpha=0.7)

# Group overlays
group1 = features[features['group-id'] == 1]
group0 = features[features['group-id'] == 0]

# Hollow red circles (group-id = 1)
plt.scatter(group0['PC1'], group0['PC2'],
            color='red', marker='+',
            label='pol.is class A', alpha=0.3)

# Hollow blue circles (group-id = 0)
plt.scatter(group1['PC1'], group1['PC2'],
            color='black', marker='x',
            label='pol.is class B', alpha=0.3)

# Labels, colorbar, and legend
plt.title("Participant Clusters (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(scatter, label='Cluster ID')
plt.legend()
plt.tight_layout()
plt.show()