import pandas as pd

# Load datasets (simulated example)
web_data = pd.read_csv("web_engagement.csv")  # Browsing history, page views
social_data = pd.read_csv("social_media.csv")  # Likes, shares, comments
transaction_data = pd.read_csv("transaction_history.csv")  # Property inquiries, purchases

# Merge datasets on user ID
user_data = web_data.merge(social_data, on="user_id").merge(transaction_data, on="user_id")

# Save combined dataset
user_data.to_csv("combined_user_data.csv", index=False)
print("Data fusion complete!")
