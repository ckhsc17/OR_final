import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt

# 1. Load comments
cm = pd.read_csv('comments.csv')
texts = cm['comment-body'].fillna('').tolist()

# 2. Character n-gram vectorization (2-4 grams) as a proxy for Chinese tokenization
vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2,4), max_features=500)
X = vectorizer.fit_transform(texts)

# 3. LDA topic modeling
n_topics = 5
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=10, random_state=42)
lda.fit(X)

# Display top n-grams per topic
n_top = 10
feature_names = vectorizer.get_feature_names_out()
topics = []
for idx, topic in enumerate(lda.components_):
    top_features = [feature_names[i] for i in topic.argsort()[:-n_top-1:-1]]
    topics.append(top_features)
topics_df = pd.DataFrame(topics, index=[f"Topic {i+1}" for i in range(n_topics)])
import ace_tools as tools; tools.display_dataframe_to_user(name="Top n-grams per LDA Topic", dataframe=topics_df)

# 4. Simple sentiment analysis via vote counts
# Label as Positive if agrees > disagrees, Negative if disagrees > agrees, else Neutral
sentiment = cm.apply(lambda row: 'Positive' if row['agrees'] > row['disagrees'] 
                     else ('Negative' if row['disagrees'] > row['agrees'] else 'Neutral'), axis=1)
sent_counts = sentiment.value_counts().reindex(['Positive', 'Neutral', 'Negative'], fill_value=0)

# 5. Plot sentiment distribution
plt.figure()
sent_counts.plot(kind='bar')
plt.title("Sentiment Distribution Based on Vote Counts")
plt.xlabel("Sentiment")
plt.ylabel("Number of Comments")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
