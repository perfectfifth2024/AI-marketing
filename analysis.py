from textblob import TextBlob
import pandas as pd

# Load social media data
data = pd.read_csv("social_media.csv")

# Perform sentiment analysis
data['sentiment'] = data['comments'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Label sentiment (positive, neutral, negative)
data['sentiment_category'] = data['sentiment'].apply(lambda x: "positive" if x > 0.2 else "neutral" if x > -0.2 else "negative")

# Save results
data.to_csv("user_sentiment.csv", index=False)
print("Sentiment analysis complete!")
