import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load segmented data
data = pd.read_csv("segmented_users.csv")

# Define features and target
X = data[['page_views', 'likes', 'shares', 'segment']]
y = data['inquiries'] > 5  # Example threshold for interest

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate performance
y_pred = model.predict(X_test)
print(f"Model accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
