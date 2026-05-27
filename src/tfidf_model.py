import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Load datasets
train_df = pd.read_csv("data/train_processed.csv")

test_df = pd.read_csv("data/test_processed.csv")

print("=" * 60)
print("TF-IDF Hallucination Detection Model")
print("=" * 60)

# Features and labels
X_train = train_df["text"]

y_train = train_df["label"]

X_test = test_df["text"]

y_test = test_df["label"]

# Build ML pipeline
pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
    ),
])

print("\nTraining model...")

# Train model
pipeline.fit(X_train, y_train)

print("Model training completed")

# Predictions
y_pred = pipeline.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(round(accuracy, 4))

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(
    pipeline,
    "models/tfidf_logistic_model.pkl"
)

print("\nModel saved successfully")

print("=" * 60)