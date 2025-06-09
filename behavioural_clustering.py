from sklearn.cluster import KMeans
import pandas as pd

# Load user data
data = pd.read_csv("combined_user_data.csv")

# Selecting relevant features
features = data[['page_views', 'likes', 'shares', 'inquiries']]  

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['segment'] = kmeans.fit_predict(features)

# Save results
data.to_csv("segmented_users.csv", index=False)
print("User segmentation complete!")
