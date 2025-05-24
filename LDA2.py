import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt

# 1. Load comments
cm = pd.read_csv('comments.csv')
texts = cm['comment-body'].fillna('').tolist()

# 2. Chinese tokenization with jieba + stopword filtering
# Define a basic Chinese stopword list (customize as needed)
stopwords = {'的', '了', '是', '在', '和', '也', '就', '都', '而', '及', '與', '著', '等', '但', '或', '又', '：', '，', '。', '、'}

# Perform jieba tokenization and join tokens by space
docs = [' '.join([tok for tok in jieba.lcut(text) if tok not in stopwords and tok.strip()]) for text in texts]

# 3. Vectorization using jieba tokens
vectorizer = CountVectorizer(
    tokenizer=lambda s: s.split(),
    stop_words=None,
    max_features=500
)
X = vectorizer.fit_transform(docs)

# 4. LDA topic modeling
n_topics = 5
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=10, random_state=42)
lda.fit(X)

# Display top words per topic
n_top = 10
feature_names = vectorizer.get_feature_names_out()
topics = []
for idx, topic in enumerate(lda.components_):
    top_features = [feature_names[i] for i in topic.argsort()[:-n_top-1:-1]]
    topics.append(top_features)
topics_df = pd.DataFrame(topics, index=[f"Topic {i+1}" for i in range(n_topics)])
import ace_tools as tools; tools.display_dataframe_to_user(name="Top Keywords per LDA Topic (jieba)", dataframe=topics_df)

# 5. Simple sentiment analysis via vote counts
# Label as Positive if agrees > disagrees, Negative if disagrees > agrees, else Neutral
sentiment = cm.apply(
    lambda row: 'Positive' if row['agrees'] > row['disagrees']
                else ('Negative' if row['disagrees'] > row['agrees'] else 'Neutral'),
    axis=1
)
sent_counts = sentiment.value_counts().reindex(['Positive', 'Neutral', 'Negative'], fill_value=0)

# Plot sentiment distribution
plt.figure()
sent_counts.plot(kind='bar')
plt.title("Sentiment Distribution Based on Vote Counts")
plt.xlabel("Sentiment")
plt.ylabel("Number of Comments")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
