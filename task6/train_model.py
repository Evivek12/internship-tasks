import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("fake_news.csv")

# Keep only required columns
df = df[['text', 'label']]

# Remove missing values
df = df.dropna()

# Features and Labels
X = df['text']
y = df['label']

# Convert text to numerical features
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

X = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model and vectorizer
joblib.dump(model, "news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model Saved Successfully!")