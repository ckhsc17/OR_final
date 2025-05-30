from transformers import pipeline
import pandas as pd

# Load comments
df = pd.read_csv("comments.csv")

# Load multilingual sentiment model with Chinese support
sentiment_pipeline = pipeline("sentiment-analysis",
                              model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

# Predict sentiment
def classify_sentiment(text):
    try:
        result = sentiment_pipeline(text[:512])  # Truncate for safety
        return result[0]['label']
    except:
        return "undetermined"


df['sentiment'] = df['comment-body'].apply(classify_sentiment)

print(df[['comment-body', 'sentiment']].head())

print(df['sentiment'].value_counts())
# Save results to a new CSV file
df.to_csv("comments_with_sentiment.csv", index=False)
